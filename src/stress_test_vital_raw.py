import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# THERMORA VITAL ANOMALY
# RAW-SIGNAL SYNTHETIC STRESS TEST
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BIDMC_DIR = PROJECT_ROOT / "vital_anomaly" / "bidmc_csv"

MODEL_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "models"
    / "thermora_vital_anomaly_model.pkl"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "models"
    / "thermora_vital_anomaly_features.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "vital_anomaly_raw_signal_stress_test.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "vital_anomaly_raw_signal_stress_summary.json"
)

RANDOM_SEED = 42

WINDOW_SECONDS = 30
NUMERIC_RATE_HZ = 1.0

# Use a subset of recordings so the experiment stays fast.
MAX_RECORDINGS = 30

# Evaluate thresholds from the actual model score scale.
THRESHOLDS = np.arange(0.05, 1.001, 0.025)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def clean_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]
    return df


def find_column(df, candidates):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def safe_float_array(values):
    return pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(dtype=float)


def interpolate(values):
    values = np.asarray(values, dtype=float)

    if np.all(~np.isfinite(values)):
        return np.zeros_like(values)

    series = pd.Series(values)

    series = (
        series
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
    )

    series = series.fillna(series.median())
    series = series.fillna(0.0)

    return series.to_numpy(dtype=float)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def ppg_features(ppg):
    ppg = interpolate(ppg)

    diff = np.diff(ppg)

    if len(diff) == 0:
        diff = np.array([0.0])

    time = np.arange(len(ppg), dtype=float)

    if len(ppg) > 1:
        slope = np.polyfit(time, ppg, 1)[0]
        start_end_change = ppg[-1] - ppg[0]
    else:
        slope = 0.0
        start_end_change = 0.0

    mean = np.mean(ppg)
    std = np.std(ppg)
    minimum = np.min(ppg)
    maximum = np.max(ppg)

    # Simple peak detection.
    # This intentionally avoids external signal-processing
    # dependencies.
    if len(ppg) >= 3:

        threshold = mean + 0.25 * std

        peaks = (
            (ppg[1:-1] > ppg[:-2])
            & (ppg[1:-1] >= ppg[2:])
            & (ppg[1:-1] > threshold)
        )

        peak_indices = np.where(peaks)[0] + 1

    else:
        peak_indices = np.array([], dtype=int)

    peak_count = len(peak_indices)

    peak_rate = (
        peak_count / WINDOW_SECONDS
        if WINDOW_SECONDS > 0
        else 0.0
    )

    if peak_count >= 2:
        intervals = np.diff(
            peak_indices
        ) / 125.0

        peak_interval_mean = np.mean(intervals)
        peak_interval_std = np.std(intervals)
        peak_interval_min = np.min(intervals)
        peak_interval_max = np.max(intervals)

    else:
        peak_interval_mean = 0.0
        peak_interval_std = 0.0
        peak_interval_min = 0.0
        peak_interval_max = 0.0

    return {
        "ppg_mean": mean,
        "ppg_std": std,
        "ppg_min": minimum,
        "ppg_max": maximum,
        "ppg_range": maximum - minimum,
        "ppg_median": np.median(ppg),
        "ppg_rms": np.sqrt(np.mean(ppg ** 2)),
        "ppg_q25": np.percentile(ppg, 25),
        "ppg_q75": np.percentile(ppg, 75),
        "ppg_diff_mean": np.mean(diff),
        "ppg_diff_std": np.std(diff),
        "ppg_diff_abs_mean": np.mean(np.abs(diff)),
        "ppg_diff_max_abs": np.max(np.abs(diff)),
        "ppg_energy": np.sum(ppg ** 2),
        "ppg_peak_count": peak_count,
        "ppg_peak_rate_per_sec": peak_rate,
        "ppg_peak_interval_mean": peak_interval_mean,
        "ppg_peak_interval_std": peak_interval_std,
        "ppg_peak_interval_min": peak_interval_min,
        "ppg_peak_interval_max": peak_interval_max,
        "ppg_slope": slope,
        "ppg_start_end_change": start_end_change,
    }


def numeric_features(values, prefix):
    values = interpolate(values)

    time = np.arange(len(values), dtype=float)

    if len(values) > 1:
        slope = np.polyfit(time, values, 1)[0]
        start_end_change = values[-1] - values[0]
    else:
        slope = 0.0
        start_end_change = 0.0

    mean = np.mean(values)

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": np.std(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_range": np.ptp(values),
        f"{prefix}_median": np.median(values),
        f"{prefix}_rms": np.sqrt(np.mean(values ** 2)),
        f"{prefix}_q25": np.percentile(values, 25),
        f"{prefix}_q75": np.percentile(values, 75),
        f"{prefix}_slope": slope,
        f"{prefix}_start_end_change": start_end_change,
    }


def extract_window_features(ppg, hr, pulse, spo2):

    features = {}

    features.update(
        ppg_features(ppg)
    )

    features.update(
        numeric_features(hr, "hr")
    )

    features.update(
        numeric_features(pulse, "pulse")
    )

    features.update(
        numeric_features(spo2, "spo2")
    )

    features["hr_pulse_mean_difference"] = (
        features["hr_mean"]
        - features["pulse_mean"]
    )

    features["hr_spo2_product"] = (
        features["hr_mean"]
        * features["spo2_mean"]
    )

    return features


# ============================================================
# RAW ANOMALY INJECTION
# ============================================================

def inject_spo2_drop(spo2, rng):

    modified = spo2.copy()

    n = len(modified)

    start = int(
        rng.uniform(
            0.35,
            0.55
        ) * n
    )

    end = min(
        n,
        start + int(
            rng.uniform(
                0.25,
                0.45
            ) * n
        )
    )

    target = rng.uniform(
        86.0,
        89.0
    )

    modified[start:end] = np.linspace(
        modified[start],
        target,
        end - start
    )

    if end < n:
        recovery_end = min(
            n,
            end + int(0.20 * n)
        )

        modified[end:recovery_end] = np.linspace(
            target,
            modified[end],
            recovery_end - end
        )

    return np.clip(
        modified,
        82.0,
        100.0
    )


def inject_high_hr(hr, rng):

    modified = hr.copy()

    n = len(modified)

    start = int(
        rng.uniform(
            0.30,
            0.50
        ) * n
    )

    end = min(
        n,
        start + int(
            rng.uniform(
                0.25,
                0.45
            ) * n
        )
    )

    target = rng.uniform(
        120.0,
        135.0
    )

    modified[start:end] = np.linspace(
        modified[start],
        target,
        end - start
    )

    return np.clip(
        modified,
        35.0,
        160.0
    )


def inject_low_hr(hr, rng):

    modified = hr.copy()

    n = len(modified)

    start = int(
        rng.uniform(
            0.30,
            0.50
        ) * n
    )

    end = min(
        n,
        start + int(
            rng.uniform(
                0.25,
                0.45
            ) * n
        )
    )

    target = rng.uniform(
        42.0,
        48.0
    )

    modified[start:end] = np.linspace(
        modified[start],
        target,
        end - start
    )

    return np.clip(
        modified,
        35.0,
        160.0
    )


def inject_combined(hr, spo2, rng):

    return (
        inject_high_hr(hr, rng),
        inject_spo2_drop(spo2, rng)
    )


def inject_ppg_amplitude(ppg, rng):

    modified = ppg.copy()

    mean = np.mean(modified)

    # Reduce pulsatile component while retaining DC level.
    modified = (
        mean
        + (modified - mean)
        * rng.uniform(
            0.15,
            0.35
        )
    )

    return modified


def inject_ppg_noise(ppg, rng):

    modified = ppg.copy()

    amplitude = np.std(modified)

    noise = rng.normal(
        0,
        amplitude * rng.uniform(
            2.5,
            5.0
        ),
        size=len(modified)
    )

    modified = modified + noise

    return modified


# ============================================================
# LOAD BIDMC
# ============================================================

print_header("LOADING BIDMC RAW DATA")

if not BIDMC_DIR.exists():
    raise FileNotFoundError(
        f"BIDMC directory not found:\n{BIDMC_DIR}"
    )

signal_files = sorted(
    BIDMC_DIR.glob(
        "bidmc_*_Signals.csv"
    )
)

numeric_files = sorted(
    BIDMC_DIR.glob(
        "bidmc_*_Numerics.csv"
    )
)

print(
    f"Signal files:  {len(signal_files)}"
)

print(
    f"Numeric files: {len(numeric_files)}"
)

if len(signal_files) == 0:
    raise RuntimeError(
        "No BIDMC signal files found."
    )

numeric_lookup = {
    f.name.replace(
        "_Numerics.csv",
        ""
    ): f
    for f in numeric_files
}

recording_ids = []

for signal_file in signal_files:

    recording_id = signal_file.stem.replace(
        "_Signals",
        ""
    )

    if recording_id in numeric_lookup:
        recording_ids.append(
            recording_id
        )

if len(recording_ids) > MAX_RECORDINGS:

    rng_select = np.random.default_rng(
        RANDOM_SEED
    )

    recording_ids = sorted(
        rng_select.choice(
            recording_ids,
            size=MAX_RECORDINGS,
            replace=False
        )
    )

print(
    f"Recordings selected: {len(recording_ids)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print_header("LOADING VITAL ANOMALY MODEL")

model = joblib.load(
    MODEL_FILE
)

print(
    f"Model type: {type(model)}"
)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_info = json.load(f)

if isinstance(feature_info, dict):

    if "features" in feature_info:
        FEATURE_COLUMNS = feature_info["features"]

    elif "feature_names" in feature_info:
        FEATURE_COLUMNS = feature_info["feature_names"]

    else:
        raise ValueError(
            "Feature JSON does not contain "
            "'features' or 'feature_names'."
        )

else:
    FEATURE_COLUMNS = feature_info

FEATURE_COLUMNS = list(
    FEATURE_COLUMNS
)

print(
    f"Expected features: {len(FEATURE_COLUMNS)}"
)


# ============================================================
# BUILD RAW-SIGNAL WINDOWS
# ============================================================

print_header("BUILDING RAW-SIGNAL WINDOWS")

rng = np.random.default_rng(
    RANDOM_SEED
)

records = []

normal_count = 0
synthetic_count = 0


for recording_id in recording_ids:

    signal_file = (
        BIDMC_DIR
        / f"{recording_id}_Signals.csv"
    )

    numeric_file = numeric_lookup[
        recording_id
    ]

    signals = clean_columns(
        pd.read_csv(signal_file)
    )

    numerics = clean_columns(
        pd.read_csv(numeric_file)
    )

    ppg_col = find_column(
        signals,
        [
            "PLETH",
            " PLETH",
        ]
    )

    hr_col = find_column(
        numerics,
        [
            "HR",
            " HR",
        ]
    )

    pulse_col = find_column(
        numerics,
        [
            "PULSE",
            " PULSE",
        ]
    )

    spo2_col = find_column(
        numerics,
        [
            "SpO2",
            " SpO2",
            "SPO2",
        ]
    )

    if ppg_col is None:
        print(
            f"Skipping {recording_id}: PPG missing"
        )
        continue

    if (
        hr_col is None
        or pulse_col is None
        or spo2_col is None
    ):
        print(
            f"Skipping {recording_id}: "
            "numeric vital columns missing"
        )
        continue

    ppg = safe_float_array(
        signals[ppg_col]
    )

    signal_time = safe_float_array(
        signals[
            find_column(
                signals,
                ["Time [s]"]
            )
        ]
    )

    hr = safe_float_array(
        numerics[hr_col]
    )

    pulse = safe_float_array(
        numerics[pulse_col]
    )

    spo2 = safe_float_array(
        numerics[spo2_col]
    )

    numeric_time_col = find_column(
        numerics,
        ["Time [s]"]
    )

    numeric_time = safe_float_array(
        numerics[numeric_time_col]
    )

    if len(signal_time) == 0:
        continue

    duration = min(
        signal_time[-1],
        numeric_time[-1]
    )

    start_times = np.arange(
        0,
        duration - WINDOW_SECONDS + 0.001,
        WINDOW_SECONDS
    )

    for start_time in start_times:

        end_time = (
            start_time
            + WINDOW_SECONDS
        )

        signal_mask = (
            (signal_time >= start_time)
            & (signal_time < end_time)
        )

        numeric_mask = (
            (numeric_time >= start_time)
            & (numeric_time < end_time)
        )

        ppg_window = ppg[
            signal_mask
        ]

        hr_window = hr[
            numeric_mask
        ]

        pulse_window = pulse[
            numeric_mask
        ]

        spo2_window = spo2[
            numeric_mask
        ]

        # Need enough data for a meaningful window.
        if len(ppg_window) < 1000:
            continue

        if len(hr_window) < 10:
            continue

        if len(spo2_window) < 10:
            continue

        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        normal_features = extract_window_features(
            ppg_window.copy(),
            hr_window.copy(),
            pulse_window.copy(),
            spo2_window.copy()
        )

        records.append({
            "recording_id": recording_id,
            "window_start_s": float(start_time),
            "window_end_s": float(end_time),
            "anomaly_type": "normal",
            "ground_truth": 0,
            **normal_features
        })

        normal_count += 1

        # ----------------------------------------------------
        # SYNTHETIC ANOMALIES
        # ----------------------------------------------------

        anomaly_payloads = {
            "spo2_drop": (
                ppg_window.copy(),
                hr_window.copy(),
                pulse_window.copy(),
                inject_spo2_drop(
                    spo2_window.copy(),
                    rng
                )
            ),

            "high_hr": (
                ppg_window.copy(),
                inject_high_hr(
                    hr_window.copy(),
                    rng
                ),
                pulse_window.copy(),
                spo2_window.copy()
            ),

            "low_hr": (
                ppg_window.copy(),
                inject_low_hr(
                    hr_window.copy(),
                    rng
                ),
                pulse_window.copy(),
                spo2_window.copy()
            ),

            "combined_hr_spo2": (
                ppg_window.copy(),
                inject_high_hr(
                    hr_window.copy(),
                    rng
            ),
            pulse_window.copy(),
            inject_spo2_drop(
                spo2_window.copy(),
                rng
            ),
        ),

            "ppg_amplitude_disturbance": (
                inject_ppg_amplitude(
                    ppg_window.copy(),
                    rng
                ),
                hr_window.copy(),
                pulse_window.copy(),
                spo2_window.copy()
            ),

            "ppg_noise_disturbance": (
                inject_ppg_noise(
                    ppg_window.copy(),
                    rng
                ),
                hr_window.copy(),
                pulse_window.copy(),
                spo2_window.copy()
            ),
        }

        for anomaly_type, payload in anomaly_payloads.items():

            features = extract_window_features(
                payload[0],
                payload[1],
                payload[2],
                payload[3]
            )

            records.append({
                "recording_id": recording_id,
                "window_start_s": float(start_time),
                "window_end_s": float(end_time),
                "anomaly_type": anomaly_type,
                "ground_truth": 1,
                **features
            })

            synthetic_count += 1


print(
    f"\nNormal windows:    {normal_count}"
)

print(
    f"Synthetic windows: {synthetic_count}"
)

print(
    f"Total windows:     {len(records)}"
)


# ============================================================
# DATAFRAME
# ============================================================

test_df = pd.DataFrame(
    records
)

if len(test_df) == 0:
    raise RuntimeError(
        "No usable raw-signal windows were created."
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

print_header("PREPARING MODEL FEATURES")

X = test_df[
    FEATURE_COLUMNS
].copy()

for col in FEATURE_COLUMNS:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Median imputation from the evaluation data is used only
# to keep the existing model interface operational.
#
# This does NOT retrain the model.
X = X.fillna(
    X.median()
)

X = X.fillna(0.0)

X = X.astype(
    np.float32
)

print(
    f"Model matrix: {X.shape}"
)


# ============================================================
# MODEL SCORING
# ============================================================

print_header("RUNNING MODEL")

decision = model.decision_function(
    X
)

raw_score = -decision

print(
    f"Raw score min:  {raw_score.min():.6f}"
)

print(
    f"Raw score max:  {raw_score.max():.6f}"
)

print(
    f"Raw score mean: {raw_score.mean():.6f}"
)


# ============================================================
# NORMAL SCORE CALIBRATION
# ============================================================

normal_mask = (
    test_df["ground_truth"] == 0
)

normal_scores = raw_score[
    normal_mask.to_numpy()
]

low = np.percentile(
    normal_scores,
    5
)

high = np.percentile(
    normal_scores,
    95
)

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

test_df[
    "anomaly_score"
] = normalized_score


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print_header(
    "RAW-SIGNAL THRESHOLD CALIBRATION"
)

y_true = test_df[
    "ground_truth"
].to_numpy()

threshold_results = []

for threshold in THRESHOLDS:

    y_pred = (
        normalized_score >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    threshold_results.append({
        "threshold": float(
            round(threshold, 4)
        ),
        "anomaly_rate": float(
            np.mean(y_pred)
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    })

threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df[
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


# ============================================================
# SELECT BEST OPERATING POINT
# ============================================================

# Prefer a threshold with FPR <= 5%.
acceptable = threshold_df[
    threshold_df[
        "false_positive_rate"
    ] <= 0.05
]

if len(acceptable) > 0:

    selected = (
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
            ]
        )
        .iloc[0]
    )

    selection_reason = (
        "Best F1 operating point among "
        "thresholds with <=5% false-positive rate."
    )

else:

    # If 5% FPR is impossible, choose the
    # lowest FPR with useful recall.
    useful = threshold_df[
        threshold_df["recall"] > 0
    ]

    if len(useful) > 0:

        selected = (
            useful
            .sort_values(
                by=[
                    "false_positive_rate",
                    "recall",
                    "f1",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ]
            )
            .iloc[0]
        )

        selection_reason = (
            "No threshold achieved <=5% FPR; "
            "selected lowest-FPR point with "
            "non-zero recall."
        )

    else:

        selected = (
            threshold_df
            .sort_values(
                by=[
                    "false_positive_rate"
                ],
                ascending=True
            )
            .iloc[0]
        )

        selection_reason = (
            "No threshold produced useful recall."
        )


selected_threshold = float(
    selected["threshold"]
)


# ============================================================
# SELECTED POINT
# ============================================================

print_header(
    "SELECTED RAW-SIGNAL OPERATING POINT"
)

print(
    f"Threshold:             "
    f"{selected_threshold:.4f}"
)

print(
    f"Anomaly rate:          "
    f"{selected['anomaly_rate']:.4f}"
)

print(
    f"Precision:             "
    f"{selected['precision']:.4f}"
)

print(
    f"Recall:                "
    f"{selected['recall']:.4f}"
)

print(
    f"F1:                    "
    f"{selected['f1']:.4f}"
)

print(
    f"False-positive rate:   "
    f"{selected['false_positive_rate']:.4f}"
)

print(
    "\nSelection reason:"
)

print(
    selection_reason
)


# ============================================================
# PERFORMANCE BY ANOMALY TYPE
# ============================================================

print_header(
    "PERFORMANCE BY ANOMALY TYPE"
)

test_df["prediction"] = (
    test_df["anomaly_score"]
    >= selected_threshold
).astype(int)

type_results = []

for anomaly_type in [
    "normal",
    "spo2_drop",
    "high_hr",
    "low_hr",
    "combined_hr_spo2",
    "ppg_amplitude_disturbance",
    "ppg_noise_disturbance",
]:

    subset = test_df[
        test_df["anomaly_type"]
        == anomaly_type
    ]

    if anomaly_type == "normal":

        fpr = float(
            subset["prediction"].mean()
        )

        type_results.append({
            "anomaly_type": anomaly_type,
            "samples": len(subset),
            "detection_rate": np.nan,
            "false_positive_rate": fpr,
        })

    else:

        detection = float(
            subset["prediction"].mean()
        )

        type_results.append({
            "anomaly_type": anomaly_type,
            "samples": len(subset),
            "detection_rate": detection,
            "false_positive_rate": np.nan,
        })


type_df = pd.DataFrame(
    type_results
)

print(
    type_df.to_string(index=False)
)


# ============================================================
# SCORE SUMMARY
# ============================================================

print_header(
    "ANOMALY SCORE SUMMARY"
)

score_summary = (
    test_df
    .groupby("anomaly_type")[
        "anomaly_score"
    ]
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
    score_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

test_df[
    [
        "recording_id",
        "window_start_s",
        "window_end_s",
        "anomaly_type",
        "ground_truth",
        "anomaly_score",
        "prediction",
    ]
    + FEATURE_COLUMNS
].to_csv(
    OUTPUT_FILE,
    index=False
)


summary = {
    "dataset": "BIDMC PPG and Respiration Dataset v1.0.0",
    "test_type": "raw_signal_synthetic_anomaly_stress_test",
    "random_seed": RANDOM_SEED,
    "recordings_used": len(recording_ids),
    "recording_ids": recording_ids,
    "normal_windows": int(normal_count),
    "synthetic_windows": int(synthetic_count),
    "total_windows": int(len(test_df)),
    "anomaly_types": [
        "spo2_drop",
        "high_hr",
        "low_hr",
        "combined_hr_spo2",
        "ppg_amplitude_disturbance",
        "ppg_noise_disturbance",
    ],
    "selected_threshold": selected_threshold,
    "precision": float(
        selected["precision"]
    ),
    "recall": float(
        selected["recall"]
    ),
    "f1": float(
        selected["f1"]
    ),
    "false_positive_rate": float(
        selected["false_positive_rate"]
    ),
    "anomaly_rate": float(
        selected["anomaly_rate"]
    ),
    "selection_reason": selection_reason,
    "warning": (
        "Synthetic raw-signal anomalies are controlled "
        "engineering tests and are NOT clinical validation."
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


# ============================================================
# COMPLETE
# ============================================================

print_header(
    "RAW-SIGNAL STRESS TEST COMPLETED"
)

print("Output files:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print(
    "\nThe original BIDMC data and trained model "
    "were not modified."
)

print(
    "\nIMPORTANT:"
)

print(
    "This evaluates controlled synthetic abnormalities "
    "after raw-signal injection and normal feature extraction. "
    "It is still not clinical validation."
)

print(
    "\nPaste the terminal output here."
)