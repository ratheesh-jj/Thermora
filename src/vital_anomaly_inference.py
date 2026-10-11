import os
import sys
import json
import joblib
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# UNPICKLING SUPPORT
# ============================================================
class PCADetector:
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.threshold = None
        self.imputer = SimpleImputer(strategy="median")
        
    def fit(self, X):
        X_imp = self.imputer.fit_transform(X)
        X_s = self.scaler.fit_transform(X_imp)
        self.pca.fit(X_s)
        errors = np.mean((X_s - self.pca.inverse_transform(self.pca.transform(X_s)))**2, axis=1)
        self.threshold = np.percentile(errors, 95)
        return self
        
    def predict(self, X):
        X_imp = self.imputer.transform(X)
        X_s = self.scaler.transform(X_imp)
        errors = np.mean((X_s - self.pca.inverse_transform(self.pca.transform(X_s)))**2, axis=1)
        return np.where(errors > self.threshold, -1, 1)

# Ensure joblib can unpickle PCADetector if it was saved in __main__
sys.modules['__main__'].PCADetector = PCADetector

# ============================================================
# FEATURE EXTRACTORS (Copied from builder)
# ============================================================
def safe_slope(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    mask = np.isfinite(values) & np.isfinite(times)
    if mask.sum() < 2: return 0.0
    x, y = times[mask], values[mask]
    if np.ptp(x) == 0: return 0.0
    return float(np.polyfit(x, y, 1)[0])

def numeric_features(values, times, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {f"{prefix}_{k}": np.nan for k in ["latest", "mean", "median", "std", "min", "max", "range", "slope", "first_last_change"]}
        
    feats = {
        f"{prefix}_latest": float(values[-1]),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_slope": safe_slope(values, times),
        f"{prefix}_first_last_change": float(values[-1] - values[0]),
    }
    
    if prefix == "spo2":
        feats["spo2_pct_below_95"] = float(np.mean(values < 95) * 100)
        feats["spo2_pct_below_92"] = float(np.mean(values < 92) * 100)
        feats["spo2_pct_below_90"] = float(np.mean(values < 90) * 100)
    return feats

def ppg_features(ppg, time):
    ppg = np.asarray(ppg, dtype=float)
    if len(ppg) == 0: return {}
        
    diff = np.diff(ppg)
    abs_diff = np.abs(diff)
    
    feats = {
        "ppg_mean": float(np.mean(ppg)),
        "ppg_std": float(np.std(ppg)),
        "ppg_min": float(np.min(ppg)),
        "ppg_max": float(np.max(ppg)),
        "ppg_range": float(np.ptp(ppg)),
        "ppg_rms": float(np.sqrt(np.mean(ppg ** 2))),
        "ppg_median": float(np.median(ppg)),
        "ppg_p10": float(np.percentile(ppg, 10)),
        "ppg_p25": float(np.percentile(ppg, 25)),
        "ppg_p75": float(np.percentile(ppg, 75)),
        "ppg_p90": float(np.percentile(ppg, 90)),
        "ppg_slope": safe_slope(ppg, time),
        "ppg_first_last_change": float(ppg[-1] - ppg[0]) if len(ppg) > 0 else 0.0,
        "ppg_diff_abs_mean": float(np.mean(abs_diff)) if len(diff) else 0.0,
        "ppg_diff_std": float(np.std(diff)) if len(diff) else 0.0,
        "ppg_energy": float(np.mean((ppg - np.mean(ppg))**2))
    }
    
    amplitude = feats["ppg_max"] - feats["ppg_min"]
    feats["ppg_amplitude"] = amplitude
    
    if len(ppg) > 10:
        peaks, props = find_peaks(ppg, distance=int(125*0.35), prominence=max(feats["ppg_std"]*0.25, 1e-6))
        feats["ppg_pulse_amplitude"] = float(np.mean(props["prominences"])) if len(peaks) > 0 else 0.0
        if len(peaks) >= 2:
            intervals = np.diff(time[peaks])
            feats["ppg_beat_regularity"] = float(np.std(intervals))
        else:
            feats["ppg_beat_regularity"] = np.nan
    else:
        feats["ppg_pulse_amplitude"] = 0.0
        feats["ppg_beat_regularity"] = np.nan

    feats["ppg_baseline_variation"] = feats["ppg_std"]
    ac = feats["ppg_std"]
    dc = abs(feats["ppg_mean"])
    feats["ppg_ac_dc_ratio"] = float(ac/dc) if dc > 1e-6 else 0.0
    
    feats["ppg_clipping_pct"] = float(np.mean((ppg >= 0.99*feats["ppg_max"]) | (ppg <= 1.01*feats["ppg_min"])) * 100) if amplitude > 1e-6 else 100.0
    feats["ppg_flatline_pct"] = float(np.mean(abs_diff < 1e-6) * 100) if len(diff) else 100.0
    feats["ppg_zero_pct"] = float(np.mean(np.abs(ppg) < 1e-6) * 100)
    
    if len(diff) > 2:
        zc = np.sum(np.diff(np.sign(diff)) != 0)
        feats["ppg_hf_noise"] = float(zc / len(diff))
    else:
        feats["ppg_hf_noise"] = 0.0
        
    return feats

# ============================================================
# VITAL ANOMALY DETECTOR INFERENCE
# ============================================================
class VitalAnomalyDetector:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "vital_anomaly" / "models"
            
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / "thermora_vital_anomaly_model.pkl"
        self.schema_path = self.models_dir / "thermora_vital_anomaly_features.json"
        
        self.model = joblib.load(self.model_path)
        with open(self.schema_path, "r") as f:
            self.schema = json.load(f)["features"]
            
        self.reset_state()
        
    def reset_state(self):
        self.baseline_windows = []
        self.baseline_hr = None
        self.baseline_spo2 = None
        self.baseline_ppg_amp = None
        
        self.consecutive_anomalies = 0
        self.current_episode_id = None
        
    def _is_stable_window(self, feats):
        if feats.get("ppg_flatline_pct", 100) >= 5: return False
        if feats.get("hr_range", 100) >= 20: return False
        if feats.get("spo2_latest", 0) < 90: return False
        return True

    def process_window(self, ppg, ppg_time, hr, hr_time, pulse, pulse_time, spo2, spo2_time):
        # 1. Feature Extraction
        row = {}
        row.update(ppg_features(ppg, ppg_time))
        row.update(numeric_features(hr, hr_time, "hr"))
        row.update(numeric_features(spo2, spo2_time, "spo2"))
        
        # 2. Baseline logic
        if self.baseline_hr is None:
            self.baseline_windows.append(row)
            # Evaluate if we have enough to set baseline
            stable = [w for w in self.baseline_windows if self._is_stable_window(w)]
            if len(stable) >= 5 or len(self.baseline_windows) >= 10:
                use_windows = stable[:5] if len(stable) >= 5 else self.baseline_windows[:5]
                self.baseline_hr = np.mean([w["hr_mean"] for w in use_windows])
                self.baseline_spo2 = np.mean([w["spo2_mean"] for w in use_windows])
                self.baseline_ppg_amp = np.mean([w["ppg_amplitude"] for w in use_windows])
            else:
                return {"vital_status": "CALIBRATING_BASELINE"}
                
        # 3. Deviations
        row["baseline_hr"] = self.baseline_hr
        row["baseline_spo2"] = self.baseline_spo2
        row["baseline_ppg_amplitude"] = self.baseline_ppg_amp
        
        row["hr_dev_abs"] = row["hr_latest"] - self.baseline_hr
        row["hr_dev_rel"] = abs(row["hr_latest"] - self.baseline_hr) / max(self.baseline_hr, 1.0)
        row["spo2_dev_abs"] = self.baseline_spo2 - row["spo2_latest"]
        row["ppg_amp_dev_rel"] = (row["ppg_amplitude"] - self.baseline_ppg_amp) / max(self.baseline_ppg_amp, 1e-6)
        
        # 4. PPG Quality
        quality = "GOOD"
        if row["ppg_flatline_pct"] >= 10 or row["ppg_zero_pct"] >= 10 or row["ppg_clipping_pct"] >= 10:
            quality = "POOR"
        elif row["ppg_amplitude"] < 0.001 or row["ppg_amp_dev_rel"] < -0.8 or row["ppg_hf_noise"] > 0.5:
            quality = "QUESTIONABLE"
            
        # 5. Rule Engine
        ev_score = 0
        hr_str = False
        spo2_str = False
        
        if row["hr_latest"] >= 120 or row["hr_latest"] <= 50 or row["hr_dev_rel"] >= 0.25:
            ev_score += 2; hr_str = True
        elif row["hr_dev_rel"] >= 0.15:
            ev_score += 1
            
        if row["spo2_latest"] < 90 or row["spo2_dev_abs"] >= 3:
            ev_score += 2; spo2_str = True
        elif row["spo2_latest"] < 94 or row["spo2_dev_abs"] >= 2:
            ev_score += 1
            
        # 6. ML Score
        df_ml = pd.DataFrame([row])
        for f in self.schema:
            if f not in df_ml: df_ml[f] = np.nan
        X = df_ml[self.schema]
        
        ml_score_raw = self.model.predict(X)[0]
        ml_is_anomaly = (ml_score_raw == -1)
        
        if ml_is_anomaly and quality != "POOR":
            ev_score += 1
            
        # 7. Decision Fusion & Temporal Persistence
        status = "NORMAL"
        is_abnormal = False
        
        if quality == "POOR":
            status = "MEASUREMENT_UNRELIABLE"
        elif hr_str and spo2_str:
            status = "VITAL_ANOMALY"
            is_abnormal = True
        elif ev_score >= 3:
            status = "VITAL_ANOMALY"
            is_abnormal = True
        elif ev_score == 2:
            status = "HR_DEVIATION" if hr_str else "SPO2_DEVIATION"
            is_abnormal = True
        elif ev_score == 1:
            if ml_is_anomaly: status = "ML_ANOMALY_SUPPORT"
            elif hr_str: status = "HR_DEVIATION"
            elif spo2_str: status = "SPO2_DEVIATION"
            else: status = "PHYSIOLOGICAL_DEVIATION"
            
        if is_abnormal:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0
            self.current_episode_id = None
            
        if status == "VITAL_ANOMALY" and self.consecutive_anomalies < 2:
            status = "UNCONFIRMED_VITAL_ANOMALY"
            
        if status == "VITAL_ANOMALY" and self.current_episode_id is None:
            self.current_episode_id = str(uuid.uuid4())
            
        return {
            "current_hr": row.get("hr_latest", np.nan),
            "current_spo2": row.get("spo2_latest", np.nan),
            "ppg_quality": quality,
            "hr_deviation": row.get("hr_dev_rel", np.nan),
            "spo2_deviation": row.get("spo2_dev_abs", np.nan),
            "hr_trend": "INCREASING" if row.get("hr_slope", 0) > 0.5 else "STABLE",
            "spo2_trend": "DECREASING" if row.get("spo2_slope", 0) < -0.1 else "STABLE",
            "ml_anomaly_score": int(ml_score_raw),
            "ml_support": ml_is_anomaly,
            "vital_status": status,
            "episode_id": self.current_episode_id
        }
