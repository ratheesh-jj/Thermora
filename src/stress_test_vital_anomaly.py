import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# THERMORA VITAL ANOMALY - SYNTHETIC STRESS TEST
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "vital_anomaly" / "processed" / "vital_windows.csv"
MODEL_FILE = PROJECT_ROOT / "vital_anomaly" / "models" / "thermora_vital_anomaly_model.pkl"
FEATURE_FILE = PROJECT_ROOT / "vital_anomaly" / "models" / "thermora_vital_anomaly_features.json"

RESULT_DIR = PROJECT_ROOT / "vital_anomaly" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULT_DIR / "vital_anomaly_synthetic_stress_test.csv"
SUMMARY_FILE = RESULT_DIR / "vital_anomaly_synthetic_stress_summary.json"


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

RANDOM_SEED = 42

# Number of synthetic abnormal copies created from normal windows
COPIES_PER_ANOMALY = 2

# Minimum number of normal windows to use
MAX_NORMAL_WINDOWS = 500

# Thresholds to evaluate
THRESHOLDS = np.arange(0.05, 1.001, 0.025)


# ------------------------------------------------------------
# ANOMALY DEFINITIONS
# ------------------------------------------------------------

ANOMALY_TYPES = [
    "spo2_drop",
    "high_hr",
    "low_hr",
    "combined_hr_spo2",
    "ppg_amplitude_disturbance",
    "ppg_noise_disturbance",
]


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def clip_series(value, low, high):
    return np.clip(value, low, high)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print_header("THERMORA VITAL ANOMALY SYNTHETIC STRESS TEST")

print("\nLoading processed dataset:")
print(DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found:\n{DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"\nRows:       {len(df)}")
print(f"Columns:    {len(df.columns)}")
print(f"Recordings: {df['recording_id'].nunique()}")


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

print("\nLoading trained model:")
print(MODEL_FILE)

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_FILE}")

model = joblib.load(MODEL_FILE)

print(f"Model type: {type(model)}")


# ------------------------------------------------------------
# LOAD FEATURE SCHEMA
# ------------------------------------------------------------

if not FEATURE_FILE.exists():
    raise FileNotFoundError(f"Feature schema not found:\n{FEATURE_FILE}")

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    feature_info = json.load(f)

if isinstance(feature_info, dict):
    if "features" in feature_info:
        FEATURE_COLUMNS = feature_info["features"]
    elif "feature_names" in feature_info:
        FEATURE_COLUMNS = feature_info["feature_names"]
    else:
        raise ValueError(
            "Could not find feature list in feature JSON. "
            "Expected 'features' or 'feature_names'."
        )
else:
    FEATURE_COLUMNS = feature_info

FEATURE_COLUMNS = list(FEATURE_COLUMNS)

print(f"Expected features: {len(FEATURE_COLUMNS)}")


# ------------------------------------------------------------
# VERIFY FEATURES
# ------------------------------------------------------------

missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:
    print("\nMissing features:")
    for col in missing_features:
        print("  -", col)

    raise ValueError(
        f"{len(missing_features)} model features are missing from dataset."
    )

X_original = df[FEATURE_COLUMNS].copy()

# Convert to numeric
for col in FEATURE_COLUMNS:
    X_original[col] = pd.to_numeric(X_original[col], errors="coerce")

# Median imputation
X_original = X_original.replace([np.inf, -np.inf], np.nan)

medians = X_original.median()

X_original = X_original.fillna(medians)
X_original = X_original.fillna(0.0)

X_original = X_original.astype(np.float32)


# ------------------------------------------------------------
# IDENTIFY NORMAL WINDOWS
# ------------------------------------------------------------

print_header("SELECTING NORMAL WINDOWS")

normal_mask = (
    (df["spo2_min"] >= 92)
    & (df["spo2_mean"] >= 92)
    & (df["hr_max"] < 115)
    & (df["hr_min"] > 50)
)

normal_indices = np.where(normal_mask.to_numpy())[0]

print(f"Candidate normal windows: {len(normal_indices)}")

if len(normal_indices) == 0:
    raise RuntimeError("No suitable normal windows found.")

rng = np.random.default_rng(RANDOM_SEED)

if len(normal_indices) > MAX_NORMAL_WINDOWS:
    normal_indices = rng.choice(
        normal_indices,
        size=MAX_NORMAL_WINDOWS,
        replace=False,
    )

normal_indices = np.sort(normal_indices)

print(f"Normal windows selected: {len(normal_indices)}")


# ------------------------------------------------------------
# SYNTHETIC ANOMALY GENERATION
# ------------------------------------------------------------

print_header("GENERATING SYNTHETIC ANOMALIES")


def create_synthetic_window(row, anomaly_type, rng):
    """
    Create a modified copy of a normal feature window.

    This operates on the already extracted window-level features.
    It does NOT modify the original BIDMC dataset.
    """

    modified = row.copy()

    # --------------------------------------------------------
    # SpO2 DROP
    # --------------------------------------------------------

    if anomaly_type == "spo2_drop":

        original_mean = safe_float(modified["spo2_mean"])

        # Simulate sustained oxygen saturation decrease.
        target_mean = rng.uniform(86.0, 89.0)

        modified["spo2_mean"] = target_mean
        modified["spo2_median"] = target_mean
        modified["spo2_min"] = max(83.0, target_mean - rng.uniform(1.0, 3.0))
        modified["spo2_max"] = min(95.0, target_mean + rng.uniform(0.5, 1.5))

        modified["spo2_range"] = (
            modified["spo2_max"] - modified["spo2_min"]
        )

        modified["spo2_q25"] = max(
            modified["spo2_min"],
            target_mean - rng.uniform(0.5, 1.5)
        )

        modified["spo2_q75"] = min(
            modified["spo2_max"],
            target_mean + rng.uniform(0.2, 1.0)
        )

        modified["spo2_rms"] = np.sqrt(
            target_mean ** 2
        )

        modified["spo2_slope"] = rng.uniform(-0.15, -0.03)
        modified["spo2_start_end_change"] = rng.uniform(-4.0, -1.0)

    # --------------------------------------------------------
    # HIGH HR
    # --------------------------------------------------------

    elif anomaly_type == "high_hr":

        target_mean = rng.uniform(120.0, 135.0)

        modified["hr_mean"] = target_mean
        modified["hr_median"] = target_mean
        modified["hr_min"] = max(
            110.0,
            target_mean - rng.uniform(5.0, 12.0)
        )
        modified["hr_max"] = min(
            145.0,
            target_mean + rng.uniform(5.0, 15.0)
        )

        modified["hr_range"] = (
            modified["hr_max"] - modified["hr_min"]
        )

        modified["hr_q25"] = target_mean - rng.uniform(2.0, 6.0)
        modified["hr_q75"] = target_mean + rng.uniform(2.0, 6.0)

        modified["hr_rms"] = np.sqrt(
            target_mean ** 2
        )

        modified["hr_slope"] = rng.uniform(0.05, 0.25)
        modified["hr_start_end_change"] = rng.uniform(5.0, 15.0)

    # --------------------------------------------------------
    # LOW HR
    # --------------------------------------------------------

    elif anomaly_type == "low_hr":

        target_mean = rng.uniform(45.0, 50.0)

        modified["hr_mean"] = target_mean
        modified["hr_median"] = target_mean
        modified["hr_min"] = max(
            40.0,
            target_mean - rng.uniform(2.0, 6.0)
        )
        modified["hr_max"] = min(
            58.0,
            target_mean + rng.uniform(2.0, 6.0)
        )

        modified["hr_range"] = (
            modified["hr_max"] - modified["hr_min"]
        )

        modified["hr_q25"] = target_mean - rng.uniform(1.0, 3.0)
        modified["hr_q75"] = target_mean + rng.uniform(1.0, 3.0)

        modified["hr_rms"] = np.sqrt(
            target_mean ** 2
        )

        modified["hr_slope"] = rng.uniform(-0.20, -0.05)
        modified["hr_start_end_change"] = rng.uniform(-12.0, -4.0)

    # --------------------------------------------------------
    # COMBINED HR + SpO2
    # --------------------------------------------------------

    elif anomaly_type == "combined_hr_spo2":

        # High HR
        hr_target = rng.uniform(120.0, 135.0)

        modified["hr_mean"] = hr_target
        modified["hr_median"] = hr_target
        modified["hr_min"] = hr_target - rng.uniform(5.0, 10.0)
        modified["hr_max"] = hr_target + rng.uniform(5.0, 10.0)
        modified["hr_range"] = (
            modified["hr_max"] - modified["hr_min"]
        )
        modified["hr_q25"] = hr_target - 5.0
        modified["hr_q75"] = hr_target + 5.0
        modified["hr_rms"] = hr_target ** 2 ** 0.5
        modified["hr_slope"] = rng.uniform(0.05, 0.25)
        modified["hr_start_end_change"] = rng.uniform(5.0, 15.0)

        # SpO2 drop
        spo2_target = rng.uniform(86.0, 89.0)

        modified["spo2_mean"] = spo2_target
        modified["spo2_median"] = spo2_target
        modified["spo2_min"] = max(83.0, spo2_target - 2.0)
        modified["spo2_max"] = min(95.0, spo2_target + 1.0)
        modified["spo2_range"] = (
            modified["spo2_max"] - modified["spo2_min"]
        )
        modified["spo2_q25"] = spo2_target - 1.0
        modified["spo2_q75"] = spo2_target + 0.5
        modified["spo2_rms"] = spo2_target
        modified["spo2_slope"] = rng.uniform(-0.15, -0.03)
        modified["spo2_start_end_change"] = rng.uniform(-5.0, -1.0)

    # --------------------------------------------------------
    # PPG AMPLITUDE DISTURBANCE
    # --------------------------------------------------------

    elif anomaly_type == "ppg_amplitude_disturbance":

        # Reduce PPG amplitude substantially.
        scale = rng.uniform(0.20, 0.40)

        mean = safe_float(modified["ppg_mean"])

        modified["ppg_std"] *= scale
        modified["ppg_range"] *= scale
        modified["ppg_q25"] = mean - (
            abs(mean - safe_float(modified["ppg_q25"])) * scale
        )
        modified["ppg_q75"] = mean + (
            abs(safe_float(modified["ppg_q75"]) - mean) * scale
        )

        modified["ppg_diff_std"] *= scale
        modified["ppg_diff_abs_mean"] *= scale
        modified["ppg_diff_max_abs"] *= scale
        modified["ppg_energy"] *= scale ** 2

        modified["ppg_slope"] *= rng.uniform(2.0, 4.0)
        modified["ppg_start_end_change"] *= rng.uniform(2.0, 4.0)

    # --------------------------------------------------------
    # PPG NOISE DISTURBANCE
    # --------------------------------------------------------

    elif anomaly_type == "ppg_noise_disturbance":

        noise_factor = rng.uniform(4.0, 8.0)

        modified["ppg_std"] *= noise_factor
        modified["ppg_diff_std"] *= noise_factor
        modified["ppg_diff_abs_mean"] *= noise_factor
        modified["ppg_diff_max_abs"] *= noise_factor

        modified["ppg_range"] *= rng.uniform(2.0, 4.0)

        modified["ppg_peak_count"] *= rng.uniform(0.5, 1.5)
        modified["ppg_peak_rate_per_sec"] *= rng.uniform(0.5, 1.5)

        modified["ppg_peak_interval_std"] *= noise_factor

    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return modified


# ------------------------------------------------------------
# BUILD SYNTHETIC DATASET
# ------------------------------------------------------------

records = []

# First add normal samples
for idx in normal_indices:

    row = df.iloc[idx].copy()

    records.append({
        "source_index": int(idx),
        "recording_id": row["recording_id"],
        "anomaly_type": "normal",
        "ground_truth": 0,
        "synthetic": False,
        **{
            col: safe_float(row[col])
            for col in FEATURE_COLUMNS
        }
    })


# Then create synthetic anomalies
for anomaly_type in ANOMALY_TYPES:

    for idx in normal_indices:

        for copy_id in range(COPIES_PER_ANOMALY):

            row = df.iloc[idx].copy()

            modified = create_synthetic_window(
                row,
                anomaly_type,
                rng,
            )

            records.append({
                "source_index": int(idx),
                "recording_id": row["recording_id"],
                "anomaly_type": anomaly_type,
                "ground_truth": 1,
                "synthetic": True,
                **{
                    col: safe_float(modified[col])
                    for col in FEATURE_COLUMNS
                }
            })


synthetic_df = pd.DataFrame(records)

print(f"\nSynthetic stress-test rows: {len(synthetic_df)}")

print("\nGround-truth distribution:")
print(
    synthetic_df["ground_truth"]
    .value_counts()
    .sort_index()
)

print("\nAnomaly types:")
print(
    synthetic_df["anomaly_type"]
    .value_counts()
)


# ------------------------------------------------------------
# PREPARE MODEL MATRIX
# ------------------------------------------------------------

X_test = synthetic_df[FEATURE_COLUMNS].copy()

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.fillna(medians)
X_test = X_test.fillna(0.0)

X_test = X_test.astype(np.float32)


# ------------------------------------------------------------
# RAW ANOMALY SCORES
# ------------------------------------------------------------

print_header("RUNNING ISOLATION FOREST")

decision = model.decision_function(X_test)

# IsolationForest:
# more negative = more anomalous
raw_score = -decision

print(f"Raw score min:  {raw_score.min():.6f}")
print(f"Raw score max:  {raw_score.max():.6f}")
print(f"Raw score mean: {raw_score.mean():.6f}")


# ------------------------------------------------------------
# NORMALIZE USING TRAINING MODEL SCORE DISTRIBUTION
# ------------------------------------------------------------

# Reconstruct training data from original dataset.
#
# We use the same normal windows only to establish a stable
# score scale for this synthetic evaluation.

X_normal = X_original.iloc[normal_indices].copy()

normal_decision = model.decision_function(X_normal)
normal_raw_score = -normal_decision

low = np.percentile(normal_raw_score, 5)
high = np.percentile(normal_raw_score, 95)

if high <= low:
    high = low + 1e-9

normalized_score = (
    raw_score - low
) / (
    high - low
)

normalized_score = np.clip(
    normalized_score,
    0.0,
    1.0
)

synthetic_df["anomaly_score"] = normalized_score


# ------------------------------------------------------------
# THRESHOLD EVALUATION
# ------------------------------------------------------------

print_header("SYNTHETIC THRESHOLD CALIBRATION")

y_true = synthetic_df["ground_truth"].to_numpy()

results = []

for threshold in THRESHOLDS:

    y_pred = (
        normalized_score >= threshold
    ).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    anomaly_rate = float(np.mean(y_pred))

    results.append({
        "threshold": round(float(threshold), 4),
        "anomaly_rate": anomaly_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    })

results_df = pd.DataFrame(results)

print(
    results_df[
        [
            "threshold",
            "anomaly_rate",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# SELECT BEST THRESHOLD
# ------------------------------------------------------------

# Primary goal:
# maximize F1.
#
# If several thresholds are close, prefer lower FPR.

best_row = (
    results_df
    .sort_values(
        by=[
            "f1",
            "recall",
            "precision",
            "false_positive_rate",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )
    .iloc[0]
)


# Also identify best threshold with FPR <= 5%.
acceptable = results_df[
    results_df["false_positive_rate"] <= 0.05
]

if len(acceptable) > 0:

    best_low_fpr = (
        acceptable
        .sort_values(
            by=[
                "f1",
                "recall",
                "precision",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

else:
    best_low_fpr = None


print_header("BEST SYNTHETIC OPERATING POINT")

print(
    f"Best F1 threshold:      "
    f"{best_row['threshold']:.4f}"
)

print(
    f"Precision:              "
    f"{best_row['precision']:.4f}"
)

print(
    f"Recall:                 "
    f"{best_row['recall']:.4f}"
)

print(
    f"F1:                     "
    f"{best_row['f1']:.4f}"
)

print(
    f"False-positive rate:    "
    f"{best_row['false_positive_rate']:.4f}"
)

print(
    f"Anomaly rate:           "
    f"{best_row['anomaly_rate']:.4f}"
)


if best_low_fpr is not None:

    print("\nBEST POINT WITH FPR <= 5%")

    print(
        f"Threshold:              "
        f"{best_low_fpr['threshold']:.4f}"
    )

    print(
        f"Precision:              "
        f"{best_low_fpr['precision']:.4f}"
    )

    print(
        f"Recall:                 "
        f"{best_low_fpr['recall']:.4f}"
    )

    print(
        f"F1:                     "
        f"{best_low_fpr['f1']:.4f}"
    )

    print(
        f"False-positive rate:    "
        f"{best_low_fpr['false_positive_rate']:.4f}"
    )


# ------------------------------------------------------------
# PERFORMANCE BY ANOMALY TYPE
# ------------------------------------------------------------

print_header("PERFORMANCE BY ANOMALY TYPE")

selected_threshold = float(best_row["threshold"])

synthetic_df["prediction"] = (
    synthetic_df["anomaly_score"] >= selected_threshold
).astype(int)


type_results = []

for anomaly_type in ["normal"] + ANOMALY_TYPES:

    subset = synthetic_df[
        synthetic_df["anomaly_type"] == anomaly_type
    ]

    if anomaly_type == "normal":

        false_positive_rate = float(
            subset["prediction"].mean()
        )

        type_results.append({
            "anomaly_type": anomaly_type,
            "samples": len(subset),
            "detection_rate": np.nan,
            "false_positive_rate": false_positive_rate,
        })

    else:

        detection_rate = float(
            subset["prediction"].mean()
        )

        type_results.append({
            "anomaly_type": anomaly_type,
            "samples": len(subset),
            "detection_rate": detection_rate,
            "false_positive_rate": np.nan,
        })


type_results_df = pd.DataFrame(type_results)

print(
    type_results_df.to_string(index=False)
)


# ------------------------------------------------------------
# SCORE DISTRIBUTION SUMMARY
# ------------------------------------------------------------

print_header("ANOMALY SCORE SUMMARY")

score_summary = (
    synthetic_df
    .groupby("anomaly_type")["anomaly_score"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        median="median",
        max="max",
    )
    .reset_index()
)

print(
    score_summary.to_string(index=False)
)


# ------------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------------

# Add selected threshold to output
synthetic_df["selected_threshold"] = selected_threshold

# Save the complete synthetic test predictions.
synthetic_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# Build summary JSON
summary = {
    "dataset": "BIDMC PPG and Respiration Dataset v1.0.0",
    "test_type": "synthetic_anomaly_stress_test",
    "random_seed": RANDOM_SEED,
    "normal_windows": int(len(normal_indices)),
    "copies_per_anomaly": COPIES_PER_ANOMALY,
    "synthetic_anomaly_types": ANOMALY_TYPES,
    "total_test_rows": int(len(synthetic_df)),
    "normal_rows": int(
        np.sum(synthetic_df["ground_truth"] == 0)
    ),
    "abnormal_rows": int(
        np.sum(synthetic_df["ground_truth"] == 1)
    ),
    "selected_threshold": selected_threshold,
    "best_precision": float(best_row["precision"]),
    "best_recall": float(best_row["recall"]),
    "best_f1": float(best_row["f1"]),
    "best_false_positive_rate": float(
        best_row["false_positive_rate"]
    ),
    "best_anomaly_rate": float(
        best_row["anomaly_rate"]
    ),
    "best_low_fpr_threshold": (
        float(best_low_fpr["threshold"])
        if best_low_fpr is not None
        else None
    ),
    "best_low_fpr_precision": (
        float(best_low_fpr["precision"])
        if best_low_fpr is not None
        else None
    ),
    "best_low_fpr_recall": (
        float(best_low_fpr["recall"])
        if best_low_fpr is not None
        else None
    ),
    "best_low_fpr_f1": (
        float(best_low_fpr["f1"])
        if best_low_fpr is not None
        else None
    ),
    "best_low_fpr_false_positive_rate": (
        float(best_low_fpr["false_positive_rate"])
        if best_low_fpr is not None
        else None
    ),
    "warning": (
        "Synthetic anomalies are controlled engineering stress tests "
        "and are NOT clinical ground truth or clinical validation."
    ),
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print_header("SYNTHETIC STRESS TEST COMPLETED")

print("Output files:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\nSelected threshold:")
print(f"{selected_threshold:.4f}")

print("\nIMPORTANT:")
print(
    "Synthetic anomalies are useful for engineering validation, "
    "but they do NOT constitute clinical validation."
)

print("\nNext step:")
print(
    "Paste the complete terminal output here so we can decide "
    "whether the vital anomaly detector is ready for integration."
)