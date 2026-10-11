import json
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

def stats(x, prefix):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_rms": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_slope": np.nan,
            f"{prefix}_change": np.nan,
        }
    return {
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_std": np.std(x),
        f"{prefix}_min": np.min(x),
        f"{prefix}_max": np.max(x),
        f"{prefix}_range": np.ptp(x),
        f"{prefix}_median": np.median(x),
        f"{prefix}_rms": np.sqrt(np.mean(x ** 2)),
        f"{prefix}_q25": np.percentile(x, 25),
        f"{prefix}_q75": np.percentile(x, 75),
        f"{prefix}_slope": np.polyfit(np.arange(len(x)), x, 1)[0],
        f"{prefix}_change": x[-1] - x[0],
    }

def make_features(ppg, hr, pulse, spo2):
    features = {}
    for x, prefix in [(spo2, "spo2"), (hr, "hr"), (pulse, "pulse"), (ppg, "ppg")]:
        features.update(stats(x, prefix))
        
    for name, x in [("spo2", spo2), ("hr", hr), ("pulse", pulse)]:
        x = np.asarray(x, dtype=float)
        for i in [5, 10, 20, 30]:
            if len(x) >= i:
                features[f"{name}_change_{i}s"] = x[-1] - x[-i]
            else:
                features[f"{name}_change_{i}s"] = np.nan
                
    valid = np.isfinite(hr) & np.isfinite(spo2)
    if valid.sum() >= 2:
        features["hr_spo2_corr"] = np.corrcoef(hr[valid], spo2[valid])[0, 1]
    else:
        features["hr_spo2_corr"] = np.nan
        
    if len(ppg) >= 2:
        diff = np.diff(ppg)
        features["ppg_diff_mean"] = np.mean(diff)
        features["ppg_diff_std"] = np.std(diff)
        features["ppg_energy"] = np.mean(ppg ** 2)
    else:
        for key in ["ppg_diff_mean", "ppg_diff_std", "ppg_energy"]:
            features[key] = np.nan
            
    return features


class SpO2Forecaster:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "vital_anomaly" / "models"
            
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / "thermora_spo2_forecaster.json"
        
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "horizons": {
                    "30": "Persistence",
                    "60": "Persistence"
                },
                "features": []
            }
            
        self.models = {}
        for horizon, method in self.config["horizons"].items():
            if method not in ["Persistence", "LinearTrend", "ClippedLinearTrend", "EWMATrend"]:
                model_file = self.models_dir / f"thermora_spo2_predictor_{horizon}s.pkl"
                if model_file.exists():
                    self.models[horizon] = joblib.load(model_file)
                else:
                    self.config["horizons"][horizon] = "Persistence"
                    
    def predict(self, ppg_history, hr_history, pulse_history, spo2_history):
        if len(spo2_history) == 0:
            return None
            
        current = float(spo2_history[-1])
        features_dict = make_features(ppg_history, hr_history, pulse_history, spo2_history)
        
        slope = features_dict.get("spo2_slope", 0.0)
        change_10s = features_dict.get("spo2_change_10s", 0.0)
        
        predictions = {}
        forecast_methods = {}
        
        for horizon in ["30", "60"]:
            method = self.config["horizons"].get(horizon, "Persistence")
            h_int = int(horizon)
            
            if method == "Persistence":
                pred = current
            elif method == "LinearTrend":
                pred = current + h_int * slope
            elif method == "ClippedLinearTrend":
                pred = np.clip(current + h_int * slope, 50, 100)
            elif method == "EWMATrend":
                pred = np.clip(current + h_int * (change_10s / 10.0), 50, 100)
            else:
                if horizon in self.models:
                    model = self.models[horizon]
                    # Create DataFrame with exact feature order
                    df = pd.DataFrame([features_dict])
                    if "features" in self.config and self.config["features"]:
                        # Ensure features exist
                        for f in self.config["features"]:
                            if f not in df:
                                df[f] = np.nan
                        df = df[self.config["features"]]
                    pred = model.predict(df)[0]
                else:
                    pred = current
                    method = "Persistence (Fallback)"
                    
            predictions[horizon] = float(pred)
            forecast_methods[horizon] = method
            
        return {
            "current_spo2": current,
            "predicted_spo2_30s": predictions["30"],
            "predicted_spo2_60s": predictions["60"],
            "trend": float(slope),
            "rate_of_change": float(change_10s),
            "forecast_method": f"30s: {forecast_methods['30']} | 60s: {forecast_methods['60']}"
        }
