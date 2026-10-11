import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, confusion_matrix

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "vital_anomaly" / "processed" / "vital_windows.csv"
RESULTS_DIR = PROJECT_ROOT / "vital_anomaly" / "results"
MODELS_DIR = PROJECT_ROOT / "vital_anomaly" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ML WRAPPERS
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

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def create_proxy_labels(df):
    is_abnormal = (df["spo2_latest"] < 90) | (df["hr_latest"] >= 120) | (df["hr_latest"] <= 50)
    return is_abnormal.astype(int)

def generate_synthetic_data(df):
    # Take only clean windows (no proxy anomalies, good PPG)
    clean = df[(create_proxy_labels(df) == 0) & (df["ppg_flatline_pct"] < 5) & (df["ppg_amplitude"] > 1e-3)].copy()
    
    synthetic = []
    
    # A. SpO2 drop
    df_a = clean.copy()
    df_a["spo2_latest"] -= 4
    df_a["spo2_dev_abs"] += 4
    df_a["synthetic_type"] = "SPO2_DROP"
    df_a["severity"] = "moderate"
    synthetic.append(df_a)
    
    # B. Sustained low SpO2
    df_b = clean.copy()
    df_b["spo2_latest"] = 88
    df_b["spo2_mean"] = 88
    df_b["spo2_dev_abs"] = df_b["baseline_spo2"] - 88
    df_b["synthetic_type"] = "SUSTAINED_LOW_SPO2"
    df_b["severity"] = "high"
    synthetic.append(df_b)
    
    # C. HR increase
    df_c = clean.copy()
    df_c["hr_latest"] += 30
    df_c["hr_dev_abs"] += 30
    df_c["hr_dev_rel"] = (df_c["hr_latest"] - df_c["baseline_hr"]) / df_c["baseline_hr"]
    df_c["synthetic_type"] = "HR_INCREASE"
    df_c["severity"] = "high"
    synthetic.append(df_c)
    
    # D. HR decrease
    df_d = clean.copy()
    df_d["hr_latest"] -= 20
    df_d["hr_dev_abs"] -= 20
    df_d["hr_dev_rel"] = abs(df_d["hr_latest"] - df_d["baseline_hr"]) / df_d["baseline_hr"]
    df_d["synthetic_type"] = "HR_DECREASE"
    df_d["severity"] = "moderate"
    synthetic.append(df_d)
    
    # E. Combined HR + SpO2
    df_e = clean.copy()
    df_e["spo2_latest"] -= 4
    df_e["spo2_dev_abs"] += 4
    df_e["hr_latest"] += 30
    df_e["hr_dev_abs"] += 30
    df_e["synthetic_type"] = "COMBINED"
    df_e["severity"] = "critical"
    synthetic.append(df_e)
    
    # F. PPG amplitude reduction
    df_f = clean.copy()
    df_f["ppg_amplitude"] /= 10
    df_f["ppg_amp_dev_rel"] = (df_f["ppg_amplitude"] - df_f["baseline_ppg_amplitude"]) / df_f["baseline_ppg_amplitude"]
    df_f["synthetic_type"] = "PPG_AMP_DROP"
    df_f["severity"] = "high"
    synthetic.append(df_f)
    
    # G. PPG noise
    df_g = clean.copy()
    df_g["ppg_std"] *= 5
    df_g["ppg_diff_std"] *= 5
    df_g["synthetic_type"] = "PPG_NOISE"
    df_g["severity"] = "high"
    synthetic.append(df_g)
    
    # H. PPG flatline
    df_h = clean.copy()
    df_h["ppg_flatline_pct"] = 95
    df_h["ppg_amplitude"] = 0
    df_h["synthetic_type"] = "PPG_FLATLINE"
    df_h["severity"] = "critical"
    synthetic.append(df_h)
    
    syn_df = pd.concat(synthetic, ignore_index=True)
    syn_df["is_synthetic"] = 1
    return syn_df

def determine_ppg_quality(row):
    if row["ppg_flatline_pct"] >= 10 or row["ppg_zero_pct"] >= 10 or row["ppg_clipping_pct"] >= 10:
        return "POOR"
    if row["ppg_amplitude"] < 0.001 or row["ppg_amp_dev_rel"] < -0.8:
        return "QUESTIONABLE"
    if row["ppg_hf_noise"] > 0.5:
        return "QUESTIONABLE"
    return "GOOD"

def evaluate_evidence(row):
    score = 0
    hr_strong = False
    spo2_strong = False
    
    # HR Rules
    if row["hr_latest"] >= 120 or row["hr_latest"] <= 50 or row["hr_dev_rel"] >= 0.25:
        score += 2
        hr_strong = True
    elif row["hr_dev_rel"] >= 0.15:
        score += 1
        
    # SpO2 Rules
    if row["spo2_latest"] < 90 or row["spo2_dev_abs"] >= 3:
        score += 2
        spo2_strong = True
    elif row["spo2_latest"] < 94 or row["spo2_dev_abs"] >= 2:
        score += 1
        
    return score, hr_strong, spo2_strong

def run_hybrid_logic(df, ml_model, features):
    df = df.copy()
    X = df[features]
    df["ml_score"] = ml_model.predict(X)
    df["ml_is_anomaly"] = (df["ml_score"] == -1)
    
    results = []
    
    # Group by recording to track temporal persistence
    for rec_id, group in df.groupby("recording_id"):
        group = group.sort_values("window_start_s").copy()
        
        consecutive_anomalies = 0
        
        for idx, row in group.iterrows():
            quality = determine_ppg_quality(row)
            ev_score, hr_str, spo2_str = evaluate_evidence(row)
            
            # Add ML support
            if row["ml_is_anomaly"] and quality != "POOR":
                ev_score += 1
                
            status = "NORMAL"
            is_abnormal_window = False
            
            if quality == "POOR":
                status = "MEASUREMENT_UNRELIABLE"
            elif hr_str and spo2_str:
                status = "VITAL_ANOMALY"
                is_abnormal_window = True
            elif ev_score >= 3:
                status = "VITAL_ANOMALY"
                is_abnormal_window = True
            elif ev_score == 2:
                status = "HR_DEVIATION" if hr_str else "SPO2_DEVIATION"
                is_abnormal_window = True
            elif ev_score == 1:
                if row["ml_is_anomaly"]:
                    status = "ML_ANOMALY_SUPPORT"
                elif hr_str:
                    status = "HR_DEVIATION"
                elif spo2_str:
                    status = "SPO2_DEVIATION"
                else:
                    status = "PHYSIOLOGICAL_DEVIATION"
                    
            # Temporal persistence logic
            if is_abnormal_window:
                consecutive_anomalies += 1
            else:
                consecutive_anomalies = 0
                
            if status == "VITAL_ANOMALY" and consecutive_anomalies < 2:
                # Downgrade if not persistent
                status = "UNCONFIRMED_VITAL_ANOMALY"
                
            row["ppg_quality"] = quality
            row["evidence_score"] = ev_score
            row["vital_status"] = status
            results.append(row)
            
    return pd.DataFrame(results)

def form_episodes(df):
    episodes = []
    for rec_id, group in df.groupby("recording_id"):
        group = group.sort_values("window_start_s")
        current_episode = None
        
        for idx, row in group.iterrows():
            is_abnormal = row["vital_status"] in ["VITAL_ANOMALY", "UNCONFIRMED_VITAL_ANOMALY", "HR_DEVIATION", "SPO2_DEVIATION"]
            
            if is_abnormal:
                if current_episode is None:
                    current_episode = {
                        "recording_id": rec_id,
                        "start_time": row["window_start_s"],
                        "end_time": row["window_end_s"],
                        "windows": 1,
                        "peak_hr": row["hr_latest"],
                        "min_spo2": row["spo2_latest"],
                        "baseline_hr": row["baseline_hr"],
                        "baseline_spo2": row["baseline_spo2"],
                        "max_dev_hr": row["hr_dev_abs"],
                        "max_dev_spo2": row["spo2_dev_abs"],
                        "types": set([row["vital_status"]])
                    }
                else:
                    current_episode["end_time"] = row["window_end_s"]
                    current_episode["windows"] += 1
                    current_episode["peak_hr"] = max(current_episode["peak_hr"], row["hr_latest"])
                    current_episode["min_spo2"] = min(current_episode["min_spo2"], row["spo2_latest"])
                    current_episode["max_dev_hr"] = max(current_episode["max_dev_hr"], row["hr_dev_abs"])
                    current_episode["max_dev_spo2"] = max(current_episode["max_dev_spo2"], row["spo2_dev_abs"])
                    current_episode["types"].add(row["vital_status"])
            else:
                if current_episode is not None:
                    if "VITAL_ANOMALY" in current_episode["types"] and current_episode["windows"] >= 2:
                        ep_type = "COMBINED_ANOMALY"
                        if "SPO2_DEVIATION" not in current_episode["types"] and current_episode["min_spo2"] >= 94:
                            ep_type = "HR_ANOMALY"
                        elif "HR_DEVIATION" not in current_episode["types"] and current_episode["peak_hr"] < 110:
                            ep_type = "SPO2_ANOMALY"
                        current_episode["type"] = ep_type
                        del current_episode["types"]
                        episodes.append(current_episode)
                    current_episode = None
                    
        if current_episode is not None and "VITAL_ANOMALY" in current_episode["types"] and current_episode["windows"] >= 2:
            current_episode["type"] = "VITAL_ANOMALY"
            del current_episode["types"]
            episodes.append(current_episode)
            
    return pd.DataFrame(episodes)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("THERMORA VITAL ANOMALY - DETECTOR REWORK V5")
    print("=" * 70)
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}")
        
    dataset = pd.read_csv(DATA_FILE)
    recordings = dataset["recording_id"].unique()
    
    np.random.seed(42)
    shuffled = recordings.copy()
    np.random.shuffle(shuffled)
    
    test_count = max(1, int(round(len(shuffled) * 0.20)))
    test_recordings = sorted(shuffled[:test_count])
    train_recordings = sorted(shuffled[test_count:])
    
    train = dataset[dataset["recording_id"].isin(train_recordings)].copy()
    test = dataset[dataset["recording_id"].isin(test_recordings)].copy()
    
    metadata_cols = ["recording_id", "window_start_s", "window_end_s", "ppg_samples", "numeric_samples"]
    features = [c for c in dataset.columns if c not in metadata_cols and not c.startswith("baseline_") and not c.endswith("_dev_abs") and not c.endswith("_dev_rel")]
    
    X_train = train[features].copy()
    
    # Fill NAs
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    
    print(f"Features: {len(features)}")
    print(f"Train windows: {len(train)} (Recordings: {len(train_recordings)})")
    print(f"Test windows: {len(test)} (Recordings: {len(test_recordings)})")
    
    models = {
        "IsolationForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", IsolationForest(n_estimators=300, contamination=0.05, random_state=42))
        ]),
        "OneClassSVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", OneClassSVM(nu=0.05, kernel="rbf", gamma="scale"))
        ]),
        "EllipticEnvelope": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", EllipticEnvelope(contamination=0.05, random_state=42))
        ]),
        "PCADetector": PCADetector(n_components=0.95)
    }
    
    test_proxy_labels = create_proxy_labels(test)
    synthetic_test = generate_synthetic_data(test)
    X_test = test[features]
    X_syn = synthetic_test[features]
    
    comparison = []
    best_model_name = None
    best_model = None
    best_syn_score = -1
    
    print("\nEvaluating ML Anomaly Detectors on Proxy & Synthetic Data:")
    
    for name, model in models.items():
        try:
            model.fit(X_train)
            preds_test = model.predict(X_test)
            preds_syn = model.predict(X_syn)
            
            is_anomaly_test = (preds_test == -1)
            is_anomaly_syn = (preds_syn == -1)
            
            # Proxy eval
            proxy_recall = recall_score(test_proxy_labels, is_anomaly_test, zero_division=0)
            proxy_precision = precision_score(test_proxy_labels, is_anomaly_test, zero_division=0)
            fpr = np.mean(is_anomaly_test[test_proxy_labels == 0])
            
            # Synthetic eval
            syn_recall = np.mean(is_anomaly_syn)
            syn_types_recall = synthetic_test.copy()
            syn_types_recall["detected"] = is_anomaly_syn
            syn_breakdown = syn_types_recall.groupby("synthetic_type")["detected"].mean().to_dict()
            
            comparison.append({
                "model": name,
                "proxy_recall": float(proxy_recall),
                "proxy_precision": float(proxy_precision),
                "fpr": float(fpr),
                "syn_overall_recall": float(syn_recall),
                **{f"syn_{k}": float(v) for k, v in syn_breakdown.items()}
            })
            
            print(f"  {name:20s} | FPR: {fpr:.2f} | Proxy Rec: {proxy_recall:.2f} | Syn Rec: {syn_recall:.2f}")
            
            # Select based on lowest FPR, highest Syn Recall
            score = (1 - fpr) * 0.5 + syn_recall * 0.5
            if score > best_syn_score:
                best_syn_score = score
                best_model_name = name
                best_model = model
        except Exception as e:
            print(f"  {name} failed: {e}")
            
    print(f"\nSELECTED ML MODEL: {best_model_name}")
    
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(RESULTS_DIR / "vital_model_comparison.csv", index=False)
    
    # Run full hybrid logic on test set
    test_hybrid = run_hybrid_logic(test, best_model, features)
    syn_hybrid = run_hybrid_logic(synthetic_test, best_model, features)
    
    # Save hybrid outputs
    test_hybrid.to_csv(RESULTS_DIR / "vital_hybrid_predictions.csv", index=False)
    syn_hybrid.to_csv(RESULTS_DIR / "vital_synthetic_stress_test.csv", index=False)
    
    # Episode formation
    episodes = form_episodes(test_hybrid)
    if not episodes.empty:
        episodes.to_csv(RESULTS_DIR / "vital_episodes.csv", index=False)
        print(f"\nGenerated {len(episodes)} validated episodes on test data.")
    else:
        print("\nNo persistent episodes formed in test data (this is okay if recordings were clean).")
        
    # Save models
    joblib.dump(best_model, MODELS_DIR / "thermora_vital_anomaly_model.pkl")
    with open(MODELS_DIR / "thermora_vital_anomaly_features.json", "w") as f:
        json.dump({"features": features, "feature_count": len(features)}, f, indent=2)
        
    # Summary
    summary = {
        "dataset": "BIDMC",
        "train_recordings": len(train_recordings),
        "test_recordings": len(test_recordings),
        "total_windows": len(dataset),
        "model": best_model_name,
        "features": len(features),
        "proxy_evaluation": {
            "proxy_recall": float(comp_df[comp_df["model"]==best_model_name]["proxy_recall"].iloc[0]),
            "proxy_precision": float(comp_df[comp_df["model"]==best_model_name]["proxy_precision"].iloc[0]),
            "false_positive_rate": float(comp_df[comp_df["model"]==best_model_name]["fpr"].iloc[0]),
        },
        "synthetic_evaluation": {
            "overall_recall": float(comp_df[comp_df["model"]==best_model_name]["syn_overall_recall"].iloc[0]),
        },
        "episodes": len(episodes) if not episodes.empty else 0,
        "disclaimer": "The BIDMC dataset does not provide ground-truth labels for clinical vital-sign anomalies. Therefore, the anomaly detector is evaluated using proxy physiological thresholds and controlled synthetic perturbation tests. These results demonstrate engineering behavior and do not constitute clinical validation."
    }
    
    with open(RESULTS_DIR / "vital_anomaly_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 70)
    print("VITAL ANOMALY PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
