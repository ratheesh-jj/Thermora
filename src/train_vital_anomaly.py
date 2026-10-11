import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATA_FILE = r"vital_anomaly\processed\vital_windows.csv"

MODEL_DIR = r"vital_anomaly\models"
RESULT_DIR = r"vital_anomaly\results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Percentage of recordings reserved for completely unseen testing
TEST_RECORDING_FRACTION = 0.20

# Isolation Forest expected contamination.
# This is deliberately modest because anomaly labels do not exist.
CONTAMINATION = 0.05
WINDOW_SECONDS = 30


# ============================================================
# FEATURE GROUPS
# ============================================================

# These are features that correspond to signals available from
# the eventual MAX30102-based deployment.

FEATURES = [
    # -------------------------
    # PPG waveform
    # -------------------------
    "ppg_mean",
    "ppg_std",
    "ppg_range",
    "ppg_median",
    "ppg_rms",
    "ppg_q25",
    "ppg_q75",

    "ppg_diff_mean",
    "ppg_diff_std",
    "ppg_diff_abs_mean",
    "ppg_diff_max_abs",

    "ppg_energy",

    "ppg_peak_count",
    "ppg_peak_rate_per_sec",
    "ppg_peak_interval_mean",
    "ppg_peak_interval_std",
    "ppg_peak_interval_min",
    "ppg_peak_interval_max",

    "ppg_slope",
    "ppg_start_end_change",

    # -------------------------
    # Heart rate
    # -------------------------
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_range",
    "hr_median",
    "hr_rms",
    "hr_q25",
    "hr_q75",
    "hr_slope",
    "hr_start_end_change",

    # -------------------------
    # Pulse
    # -------------------------
    "pulse_mean",
    "pulse_std",
    "pulse_min",
    "pulse_max",
    "pulse_range",
    "pulse_median",
    "pulse_rms",
    "pulse_q25",
    "pulse_q75",
    "pulse_slope",
    "pulse_start_end_change",

    # -------------------------
    # SpO2
    # -------------------------
    "spo2_mean",
    "spo2_std",
    "spo2_min",
    "spo2_max",
    "spo2_range",
    "spo2_median",
    "spo2_rms",
    "spo2_q25",
    "spo2_q75",
    "spo2_slope",
    "spo2_start_end_change",

    # -------------------------
    # Cross-signal
    # -------------------------
    "hr_pulse_mean_difference",
]


# ============================================================
# PROXY ANOMALY LABEL
# ============================================================

def make_proxy_anomaly_labels(df):
    """
    Create a deliberately conservative physiological
    stress-test label.

    IMPORTANT:
    These are NOT ground-truth clinical labels.

    They are only used to check whether the unsupervised
    detector responds to obviously unusual observations
    in the BIDMC dataset.
    """

    labels = np.zeros(len(df), dtype=int)

    # Strong SpO2 deviation
    labels |= (df["spo2_min"] < 90).astype(int)

    # Very low oxygen saturation across the window
    labels |= (df["spo2_mean"] < 90).astype(int)

    # Very high heart rate
    labels |= (df["hr_max"] >= 120).astype(int)

    # Very low heart rate
    labels |= (df["hr_min"] <= 50).astype(int)

    return labels.astype(int)


# ============================================================
# RECORDING-LEVEL SPLIT
# ============================================================

def split_recordings(df):

    recordings = sorted(
        df["recording_id"].unique()
    )

    rng = np.random.default_rng(RANDOM_STATE)

    shuffled = recordings.copy()
    rng.shuffle(shuffled)

    n_test = max(
        1,
        int(round(len(recordings) * TEST_RECORDING_FRACTION))
    )

    test_recordings = sorted(
        shuffled[:n_test]
    )

    train_recordings = sorted(
        shuffled[n_test:]
    )

    train_df = df[
        df["recording_id"].isin(train_recordings)
    ].copy()

    test_df = df[
        df["recording_id"].isin(test_recordings)
    ].copy()

    return (
        train_df,
        test_df,
        train_recordings,
        test_recordings,
    )


# ============================================================
# CLEAN FEATURES
# ============================================================

def prepare_features(train_df, test_df):

    missing_train = [
        c for c in FEATURES
        if c not in train_df.columns
    ]

    if missing_train:
        raise ValueError(
            "Missing training features:\n"
            + "\n".join(missing_train)
        )

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    # Replace invalid values
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Median imputation based ONLY on training data
    medians = X_train.median()

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_test


# ============================================================
# ANOMALY SCORE
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(scores, dtype=float)

    low = np.percentile(scores, 5)
    high = np.percentile(scores, 95)

    if high <= low:
        return np.zeros_like(scores)

    normalized = (
        (scores - low)
        / (high - low)
    )

    return np.clip(normalized, 0, 1)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    predictions,
    scores,
    proxy_labels
):

    predicted_anomaly = (
        predictions == -1
    ).astype(int)

    proxy_positive = proxy_labels == 1
    pred_positive = predicted_anomaly == 1

    precision = precision_score(
        proxy_labels,
        predicted_anomaly,
        zero_division=0
    )

    recall = recall_score(
        proxy_labels,
        predicted_anomaly,
        zero_division=0
    )

    f1 = f1_score(
        proxy_labels,
        predicted_anomaly,
        zero_division=0
    )

    # False-positive rate among proxy-normal windows
    normal_mask = proxy_labels == 0

    if normal_mask.sum() > 0:
        false_positive_rate = float(
            np.mean(
                predicted_anomaly[normal_mask] == 1
            )
        )
    else:
        false_positive_rate = np.nan

    anomaly_rate = float(
        np.mean(predicted_anomaly)
    )

    print()
    print("-" * 60)
    print(name)
    print("-" * 60)

    print(
        f"Predicted anomaly rate: "
        f"{anomaly_rate * 100:.2f}%"
    )

    print(
        f"Proxy recall:           "
        f"{recall:.4f}"
    )

    print(
        f"Proxy precision:        "
        f"{precision:.4f}"
    )

    print(
        f"Proxy F1:               "
        f"{f1:.4f}"
    )

    print(
        f"Proxy false-positive:   "
        f"{false_positive_rate:.4f}"
    )

    print(
        f"Score mean:             "
        f"{np.mean(scores):.4f}"
    )

    return {
        "model": name,
        "anomaly_rate": anomaly_rate,
        "proxy_recall": recall,
        "proxy_precision": precision,
        "proxy_f1": f1,
        "proxy_false_positive_rate": false_positive_rate,
        "score_mean": float(np.mean(scores)),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("THERMORA VITAL ANOMALY MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print()
    print("Dataset")
    print(f"Rows:       {len(df):,}")
    print(f"Columns:    {len(df.columns)}")
    print(
        f"Recordings: "
        f"{df['recording_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Proxy labels
    # --------------------------------------------------------

    df["proxy_anomaly"] = make_proxy_anomaly_labels(df)

    print()
    print("PROXY ANOMALY STRESS TEST")
    print("-------------------------")

    print(
        f"Proxy normal:  "
        f"{(df['proxy_anomaly'] == 0).sum():,}"
    )

    print(
        f"Proxy abnormal: "
        f"{(df['proxy_anomaly'] == 1).sum():,}"
    )

    print(
        "\nIMPORTANT: Proxy labels are NOT clinical "
        "ground truth."
    )

    # --------------------------------------------------------
    # Recording-level split
    # --------------------------------------------------------

    (
        train_df,
        test_df,
        train_recordings,
        test_recordings,
    ) = split_recordings(df)

    print()
    print("RECORDING-LEVEL SPLIT")
    print("---------------------")

    print(
        f"Training recordings: "
        f"{len(train_recordings)}"
    )

    print(
        f"Testing recordings:  "
        f"{len(test_recordings)}"
    )

    print(
        f"Training windows:     "
        f"{len(train_df):,}"
    )

    print(
        f"Testing windows:      "
        f"{len(test_df):,}"
    )

    print()
    print("Test recordings:")
    print(", ".join(test_recordings))

    # --------------------------------------------------------
    # Prepare feature matrix
    # --------------------------------------------------------

    X_train, X_test = prepare_features(
        train_df,
        test_df
    )

    print()
    print(
        f"Model features: {X_train.shape[1]}"
    )

    # ========================================================
    # MODEL 1: ISOLATION FOREST
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING ISOLATION FOREST")
    print("=" * 70)

    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    isolation_forest.fit(X_train)

    if_predictions = isolation_forest.predict(
        X_test
    )

    if_raw_scores = -isolation_forest.decision_function(
        X_test
    )

    if_scores = normalize_scores(
        if_raw_scores
    )

    if_metrics = evaluate_model(
        "Isolation Forest",
        if_predictions,
        if_scores,
        test_df["proxy_anomaly"].to_numpy()
    )

    # ========================================================
    # MODEL 2: ONE-CLASS SVM
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING ONE-CLASS SVM")
    print("=" * 70)

    svm_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=CONTAMINATION,
            )
        )
    ])

    svm_model.fit(X_train)

    svm_predictions = svm_model.predict(
        X_test
    )

    svm_raw_scores = -svm_model.decision_function(
        X_test
    )

    svm_scores = normalize_scores(
        svm_raw_scores
    )

    svm_metrics = evaluate_model(
        "One-Class SVM",
        svm_predictions,
        svm_scores,
        test_df["proxy_anomaly"].to_numpy()
    )

    # ========================================================
    # COMPARE
    # ========================================================

    results = pd.DataFrame([
        if_metrics,
        svm_metrics,
    ])

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(index=False)
    )

    comparison_file = os.path.join(
        RESULT_DIR,
        "vital_anomaly_model_comparison.csv"
    )

    results.to_csv(
        comparison_file,
        index=False
    )

    # ========================================================
    # SELECT MODEL
    # ========================================================

    # Primary selection criterion:
    # F1 against the clearly documented proxy stress-test.
    #
    # Secondary consideration:
    # low false-positive rate.

    best_index = results["proxy_f1"].idxmax()

    best_name = results.loc[
        best_index,
        "model"
    ]

    if best_name == "Isolation Forest":

        selected_model = isolation_forest

        model_type = "IsolationForest"

    else:

        selected_model = svm_model

        model_type = "OneClassSVM"

    print()
    print("=" * 70)
    print("SELECTED MODEL")
    print("=" * 70)

    print(f"Model: {best_name}")

    # ========================================================
    # SAVE MODEL
    # ========================================================

    import joblib

    model_file = os.path.join(
        MODEL_DIR,
        "thermora_vital_anomaly_model.pkl"
    )

    joblib.dump(
        selected_model,
        model_file
    )

    # Save feature schema
    feature_file = os.path.join(
        MODEL_DIR,
        "thermora_vital_anomaly_features.json"
    )

    with open(
        feature_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "features": FEATURES,
                "feature_count": len(FEATURES),
                "window_seconds": WINDOW_SECONDS,
                "input_signals": [
                    "PPG",
                    "HR",
                    "PULSE",
                    "SpO2",
                ],
                "model": best_name,
                "model_type": model_type,
                "contamination": CONTAMINATION,
                "random_state": RANDOM_STATE,
                "evaluation_note": (
                    "BIDMC has no ground-truth vital "
                    "anomaly labels. Proxy labels are "
                    "used only as a stress-test."
                ),
            },
            f,
            indent=2
        )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    if best_name == "Isolation Forest":

        best_predictions = if_predictions
        best_scores = if_scores

    else:

        best_predictions = svm_predictions
        best_scores = svm_scores

    output = test_df[
        [
            "recording_id",
            "window_start_s",
            "window_end_s",
            "hr_mean",
            "hr_min",
            "hr_max",
            "spo2_mean",
            "spo2_min",
            "spo2_max",
            "proxy_anomaly",
        ]
    ].copy()

    output["anomaly_prediction"] = (
        best_predictions == -1
    ).astype(int)

    output["anomaly_score"] = best_scores

    prediction_file = os.path.join(
        RESULT_DIR,
        "vital_anomaly_test_predictions.csv"
    )

    output.to_csv(
        prediction_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "dataset_rows": int(len(df)),
        "recordings": int(
            df["recording_id"].nunique()
        ),
        "training_recordings": int(
            len(train_recordings)
        ),
        "testing_recordings": int(
            len(test_recordings)
        ),
        "training_windows": int(
            len(train_df)
        ),
        "testing_windows": int(
            len(test_df)
        ),
        "feature_count": int(
            len(FEATURES)
        ),
        "selected_model": best_name,
        "proxy_anomaly_windows": int(
            df["proxy_anomaly"].sum()
        ),
        "proxy_label_warning": (
            "Proxy labels are not clinical ground truth."
        ),
        "model_file": model_file,
        "feature_file": feature_file,
        "comparison_file": comparison_file,
        "prediction_file": prediction_file,
    }

    summary_file = os.path.join(
        RESULT_DIR,
        "vital_anomaly_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(model_file)
    print(feature_file)
    print(comparison_file)
    print(prediction_file)
    print(summary_file)

    print()
    print("=" * 70)
    print("VITAL ANOMALY TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()