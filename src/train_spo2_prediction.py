import os
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "vital_anomaly" / "bidmc_csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "vital_anomaly" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
INPUT_SECONDS = 30
STEP_SECONDS = 5
HORIZONS = [30, 60]
RANDOM_STATE = 42

# ============================================================
# HELPERS
# ============================================================
def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_column(df, names):
    exact = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = name.strip().lower()
        if key in exact:
            return exact[key]
    for c in df.columns:
        c_low = str(c).strip().lower()
        for name in names:
            if name.strip().lower() in c_low:
                return c
    return None

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

def nearest_value(times, values, target):
    if len(times) == 0:
        return np.nan
    idx = np.searchsorted(times, target)
    candidates = [i for i in [idx - 1, idx] if 0 <= i < len(times)]
    if not candidates:
        return np.nan
    best = min(candidates, key=lambda i: abs(times[i] - target))
    if abs(times[best] - target) > 0.75:
        return np.nan
    return values[best]

# ============================================================
# LOAD BIDMC RECORDING
# ============================================================
def load_recording(recording_id):
    signal_file = DATA_DIR / f"{recording_id}_Signals.csv"
    numeric_file = DATA_DIR / f"{recording_id}_Numerics.csv"
    
    signals = clean_columns(pd.read_csv(signal_file))
    numerics = clean_columns(pd.read_csv(numeric_file))
    
    signal_time = find_column(signals, ["Time [s]", "Time"])
    ppg_col = find_column(signals, ["PLETH"])
    
    numeric_time = find_column(numerics, ["Time [s]", "Time"])
    hr_col = find_column(numerics, ["HR"])
    pulse_col = find_column(numerics, ["PULSE"])
    spo2_col = find_column(numerics, ["SpO2", "SPO2"])
    
    required = {
        "signal_time": signal_time,
        "PPG": ppg_col,
        "numeric_time": numeric_time,
        "HR": hr_col,
        "PULSE": pulse_col,
        "SpO2": spo2_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"{recording_id}: missing {missing}")
    
    # PPG
    sig = signals[[signal_time, ppg_col]].copy()
    sig[signal_time] = pd.to_numeric(sig[signal_time], errors="coerce")
    sig[ppg_col] = pd.to_numeric(sig[ppg_col], errors="coerce")
    sig = sig.dropna(subset=[signal_time]).sort_values(signal_time).drop_duplicates(signal_time)
    ppg_time = sig[signal_time].to_numpy(float)
    ppg = sig[ppg_col].to_numpy(float)
    
    # Numerics
    num = numerics[[numeric_time, hr_col, pulse_col, spo2_col]].copy()
    for c in [numeric_time, hr_col, pulse_col, spo2_col]:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    num = num.dropna(subset=[numeric_time]).sort_values(numeric_time).drop_duplicates(numeric_time)
    num = num.set_index(numeric_time).interpolate(limit=3, limit_direction="both")
    numeric_time_values = num.index.to_numpy(float)
    
    return {
        "ppg_time": ppg_time,
        "ppg": ppg,
        "time": numeric_time_values,
        "hr": num[hr_col].to_numpy(float),
        "pulse": num[pulse_col].to_numpy(float),
        "spo2": num[spo2_col].to_numpy(float),
    }

# ============================================================
# FEATURE EXTRACTION
# ============================================================
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

# ============================================================
# BUILD WINDOWS
# ============================================================
def build_dataset(recordings):
    rows = []
    for index, recording_id in enumerate(recordings, start=1):
        print(f"[{index:02d}/{len(recordings):02d}] {recording_id}")
        data = load_recording(recording_id)
        
        ppg_time = data["ppg_time"]
        ppg = data["ppg"]
        time = data["time"]
        hr = data["hr"]
        pulse = data["pulse"]
        spo2 = data["spo2"]
        
        start = int(np.ceil(time[0])) + INPUT_SECONDS - 1
        end = int(np.floor(time[-1])) - max(HORIZONS)
        
        for t in range(start, end + 1, STEP_SECONDS):
            input_start = t - INPUT_SECONDS + 1
            mask = (time >= input_start) & (time <= t)
            if mask.sum() < 20: continue
            
            target_30 = nearest_value(time, spo2, t + 30)
            target_60 = nearest_value(time, spo2, t + 60)
            if not (np.isfinite(target_30) and np.isfinite(target_60)): continue
            
            ppg_mask = (ppg_time >= input_start) & (ppg_time <= t)
            if ppg_mask.sum() < 1000: continue
            
            features = make_features(ppg[ppg_mask], hr[mask], pulse[mask], spo2[mask])
            features.update({
                "recording_id": recording_id,
                "window_end_s": float(t),
                "current_spo2": float(spo2[mask][-1]),
                "target_spo2_30s": float(target_30),
                "target_spo2_60s": float(target_60),
            })
            rows.append(features)
    return pd.DataFrame(rows)

def calculate_metrics(y_true, prediction):
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "R2": float(r2_score(y_true, prediction)),
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("THERMORA SpO2 FUTURE PREDICTION")
    print("=" * 70)
    
    recordings = sorted(p.stem.replace("_Signals", "") for p in DATA_DIR.glob("bidmc_*_Signals.csv"))
    if not recordings:
        raise FileNotFoundError(f"No BIDMC files found:\n{DATA_DIR}")
        
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = recordings.copy()
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * 0.20)))
    test_recordings = sorted(shuffled[:test_count])
    train_recordings = sorted(shuffled[test_count:])
    
    dataset = build_dataset(recordings)
    if dataset.empty:
        raise RuntimeError("No prediction windows created.")
        
    train = dataset[dataset.recording_id.isin(train_recordings)].copy()
    test = dataset[dataset.recording_id.isin(test_recordings)].copy()
    
    metadata = {"recording_id", "window_end_s", "current_spo2", "target_spo2_30s", "target_spo2_60s"}
    feature_names = [c for c in dataset.columns if c not in metadata]
    
    X_train = train[feature_names]
    X_test = test[feature_names]
    
    builders = {
        "Ridge": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "RandomForest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1))
        ]),
        "HistGradientBoosting": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.5, random_state=42))
        ]),
    }
    
    try:
        from xgboost import XGBRegressor
        builders["XGBoost"] = lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1))
        ])
    except ImportError:
        pass

    comparison = []
    prediction_tables = []
    
    selected_methods = {}
    
    for horizon in HORIZONS:
        target = f"target_spo2_{horizon}s"
        print("\n" + "=" * 70)
        print(f"SpO2 +{horizon} SECOND PREDICTION")
        print("=" * 70)
        
        y_train = train[target].to_numpy(float)
        y_test = test[target].to_numpy(float)
        current_spo2 = test["current_spo2"].to_numpy(float)
        
        # Baselines
        spo2_slope_test = test["spo2_slope"].to_numpy(float)
        spo2_change_10s = test["spo2_change_10s"].to_numpy(float)
        
        baselines = {
            "Persistence": current_spo2,
            "LinearTrend": current_spo2 + horizon * spo2_slope_test,
            "ClippedLinearTrend": np.clip(current_spo2 + horizon * spo2_slope_test, 50, 100),
            "EWMATrend": np.clip(current_spo2 + horizon * (spo2_change_10s / 10.0), 50, 100) 
        }
        
        best_name = None
        best_mae = float("inf")
        best_prediction = None
        best_model_obj = None
        
        # Evaluate Baselines
        for b_name, b_pred in baselines.items():
            metrics = calculate_metrics(y_test, b_pred)
            comparison.append({"horizon_seconds": horizon, "model": b_name, **metrics})
            print(f"{b_name:20s} MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}")
            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                best_name = b_name
                best_prediction = b_pred
                
        persistence_mae = [c["MAE"] for c in comparison if c["model"] == "Persistence" and c["horizon_seconds"] == horizon][0]
                
        # Evaluate ML Models
        for name, builder in builders.items():
            print(f"Training {name}...")
            model = builder()
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            metrics = calculate_metrics(y_test, prediction)
            comparison.append({"horizon_seconds": horizon, "model": name, **metrics})
            print(f"{name:20s} MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}")
            
            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                best_name = name
                best_prediction = prediction
                best_model_obj = model
                
        # Selection
        selected_methods[str(horizon)] = {
            "method": best_name,
            "is_ml": best_model_obj is not None,
            "mae": best_mae
        }
        
        if best_model_obj is not None:
            model_path = MODELS_DIR / f"thermora_spo2_predictor_{horizon}s.pkl"
            joblib.dump(best_model_obj, model_path)
            imp = 100 * (persistence_mae - best_mae) / persistence_mae
            print(f"\nSELECTED: {best_name} (ML)")
            print(f"Improvement over persistence: {imp:.2f}%")
        else:
            print(f"\nSELECTED: {best_name} (Baseline)")
            print("NO ML MODEL BEATS BEST BASELINE")
            
        output = test[["recording_id", "window_end_s", "current_spo2", target]].copy()
        output[f"predicted_spo2_{horizon}s"] = best_prediction
        prediction_tables.append(output)

    # Save Configuration Artifact
    # We create thermora_spo2_forecaster.json to describe the deployed methods
    forecaster_config = {
        "version": "1.0",
        "input_seconds": INPUT_SECONDS,
        "features": feature_names,
        "horizons": {
            "30": selected_methods["30"]["method"],
            "60": selected_methods["60"]["method"]
        }
    }
    with open(MODELS_DIR / "thermora_spo2_forecaster.json", "w") as f:
        json.dump(forecaster_config, f, indent=2)

    pd.DataFrame(comparison).to_csv(RESULTS_DIR / "spo2_prediction_model_comparison.csv", index=False)
    
    summary = {
        "dataset": "BIDMC",
        "train_recordings": train_recordings,
        "test_recordings": test_recordings,
        "prediction_windows": len(dataset),
        "comparison": comparison,
        "selected_methods": selected_methods,
        "note": "THERMORA uses a persistence/trend-based short-horizon SpO2 forecast if ML models did not demonstrate lower held-out MAE than the persistence baseline."
    }
    with open(RESULTS_DIR / "spo2_prediction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SpO2 PREDICTION BUILD COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()