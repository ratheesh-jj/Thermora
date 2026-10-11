import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# THERMORA HYBRID VITAL DETECTOR
# SYNTHETIC STRESS TEST
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "processed"
    / "vital_windows.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "hybrid_vital_stress_test.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "hybrid_vital_stress_summary.json"
)


# ============================================================
# PROTOTYPE THRESHOLDS
# ============================================================

SPO2_LOW_THRESHOLD = 90.0
HR_HIGH_THRESHOLD = 120.0
HR_LOW_THRESHOLD = 50.0

HR_RELATIVE_CHANGE = 0.25
SPO2_DROP_THRESHOLD = 3.0

# PPG quality thresholds
PPG_STD_MIN = 0.005
PPG_RANGE_MIN = 0.02

# Persistence
PERSISTENCE_WINDOWS = 2


# ============================================================
# HELPERS
# ============================================================

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


def modify_spo2(row, rng):

    out = row.copy()

    target = rng.uniform(
        86.0,
        89.0
    )

    out["spo2_mean"] = target
    out["spo2_median"] = target

    out["spo2_min"] = max(
        83.0,
        target - rng.uniform(1.0, 2.5)
    )

    out["spo2_max"] = min(
        95.0,
        target + rng.uniform(0.5, 1.5)
    )

    out["spo2_range"] = (
        out["spo2_max"]
        - out["spo2_min"]
    )

    out["spo2_q25"] = (
        target - rng.uniform(0.5, 1.5)
    )

    out["spo2_q75"] = (
        target + rng.uniform(0.2, 1.0)
    )

    out["spo2_slope"] = rng.uniform(
        -0.15,
        -0.03
    )

    out["spo2_start_end_change"] = rng.uniform(
        -5.0,
        -1.0
    )

    return out


def modify_high_hr(row, rng):

    out = row.copy()

    target = rng.uniform(
        120.0,
        135.0
    )

    out["hr_mean"] = target
    out["hr_median"] = target

    out["hr_min"] = max(
        110.0,
        target - rng.uniform(5.0, 10.0)
    )

    out["hr_max"] = min(
        145.0,
        target + rng.uniform(5.0, 12.0)
    )

    out["hr_range"] = (
        out["hr_max"]
        - out["hr_min"]
    )

    out["hr_q25"] = (
        target - rng.uniform(2.0, 5.0)
    )

    out["hr_q75"] = (
        target + rng.uniform(2.0, 5.0)
    )

    out["hr_slope"] = rng.uniform(
        0.05,
        0.25
    )

    out["hr_start_end_change"] = rng.uniform(
        5.0,
        15.0
    )

    return out


def modify_low_hr(row, rng):

    out = row.copy()

    target = rng.uniform(
        42.0,
        48.0
    )

    out["hr_mean"] = target
    out["hr_median"] = target

    out["hr_min"] = max(
        38.0,
        target - rng.uniform(2.0, 5.0)
    )

    out["hr_max"] = min(
        55.0,
        target + rng.uniform(2.0, 5.0)
    )

    out["hr_range"] = (
        out["hr_max"]
        - out["hr_min"]
    )

    out["hr_q25"] = (
        target - rng.uniform(1.0, 3.0)
    )

    out["hr_q75"] = (
        target + rng.uniform(1.0, 3.0)
    )

    out["hr_slope"] = rng.uniform(
        -0.20,
        -0.05
    )

    out["hr_start_end_change"] = rng.uniform(
        -15.0,
        -4.0
    )

    return out


def modify_combined(row, rng):

    out = modify_high_hr(
        row,
        rng
    )

    out = modify_spo2(
        out,
        rng
    )

    return out


def modify_ppg_amplitude(row, rng):

    out = row.copy()

    scale = rng.uniform(
        0.15,
        0.35
    )

    out["ppg_std"] *= scale
    out["ppg_range"] *= scale

    out["ppg_diff_std"] *= scale
    out["ppg_diff_abs_mean"] *= scale
    out["ppg_diff_max_abs"] *= scale

    out["ppg_energy"] *= scale ** 2

    return out


def modify_ppg_noise(row, rng):

    out = row.copy()

    factor = rng.uniform(
        3.0,
        6.0
    )

    out["ppg_std"] *= factor
    out["ppg_diff_std"] *= factor
    out["ppg_diff_abs_mean"] *= factor
    out["ppg_diff_max_abs"] *= factor
    out["ppg_range"] *= rng.uniform(
        2.0,
        4.0
    )

    return out


# ============================================================
# LOAD DATA
# ============================================================

print_header(
    "THERMORA HYBRID VITAL STRESS TEST"
)

print("\nLoading:")
print(DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

df = pd.read_csv(
    DATA_FILE
)

print(
    f"\nRows:       {len(df)}"
)

print(
    f"Recordings: {df['recording_id'].nunique()}"
)


# ============================================================
# SELECT NORMAL WINDOWS
# ============================================================

print_header(
    "SELECTING NORMAL WINDOWS"
)

normal_mask = (
    (df["spo2_min"] >= 92)
    &
    (df["spo2_mean"] >= 92)
    &
    (df["hr_max"] < 115)
    &
    (df["hr_min"] > 50)
)

normal_df = df[
    normal_mask
].copy()

print(
    f"Candidate normal windows: "
    f"{len(normal_df)}"
)

# Keep all normal windows.
# This gives us a realistic false-positive estimate.
normal_df = normal_df.reset_index(
    drop=True
)

rng = np.random.default_rng(
    42
)


# ============================================================
# CREATE TEST DATA
# ============================================================

print_header(
    "GENERATING SYNTHETIC HYBRID TEST"
)

records = []


# ------------------------------------------------------------
# NORMAL
# ------------------------------------------------------------

for _, row in normal_df.iterrows():

    records.append({
        "recording_id": row["recording_id"],
        "window_start_s": row["window_start_s"],
        "anomaly_type": "normal",
        "ground_truth": 0,
        **{
            col: safe_float(row[col])
            for col in [
                "hr_mean",
                "hr_min",
                "hr_max",
                "hr_slope",
                "hr_start_end_change",
                "spo2_mean",
                "spo2_min",
                "spo2_max",
                "spo2_slope",
                "spo2_start_end_change",
                "ppg_std",
                "ppg_range",
            ]
        }
    })


# ------------------------------------------------------------
# SYNTHETIC ANOMALIES
# ------------------------------------------------------------

ANOMALY_GENERATORS = {
    "spo2_drop": modify_spo2,
    "high_hr": modify_high_hr,
    "low_hr": modify_low_hr,
    "combined_hr_spo2": modify_combined,
    "ppg_amplitude_disturbance": modify_ppg_amplitude,
    "ppg_noise_disturbance": modify_ppg_noise,
}


for anomaly_type, generator in ANOMALY_GENERATORS.items():

    for _, row in normal_df.iterrows():

        modified = generator(
            row.copy(),
            rng
        )

        records.append({
            "recording_id": row["recording_id"],
            "window_start_s": row["window_start_s"],
            "anomaly_type": anomaly_type,
            "ground_truth": 1,
            **{
                col: safe_float(modified[col])
                for col in [
                    "hr_mean",
                    "hr_min",
                    "hr_max",
                    "hr_slope",
                    "hr_start_end_change",
                    "spo2_mean",
                    "spo2_min",
                    "spo2_max",
                    "spo2_slope",
                    "spo2_start_end_change",
                    "ppg_std",
                    "ppg_range",
                ]
            }
        })


test_df = pd.DataFrame(
    records
)

print(
    f"Normal samples:    "
    f"{len(normal_df)}"
)

print(
    f"Total test samples:"
    f" {len(test_df)}"
)


# ============================================================
# BASELINES
# ============================================================

print_header(
    "BUILDING RECORDING BASELINES"
)

test_df["hr_baseline"] = (
    test_df
    .groupby("recording_id")["hr_mean"]
    .transform("median")
)

test_df["spo2_baseline"] = (
    test_df
    .groupby("recording_id")["spo2_mean"]
    .transform("median")
)

test_df["hr_baseline"] = (
    test_df["hr_baseline"]
    .fillna(test_df["hr_mean"])
)

test_df["spo2_baseline"] = (
    test_df["spo2_baseline"]
    .fillna(test_df["spo2_mean"])
)


# ============================================================
# PHYSIOLOGICAL RULES
# ============================================================

test_df["hr_relative_deviation"] = (
    np.abs(
        test_df["hr_mean"]
        - test_df["hr_baseline"]
    )
    /
    test_df["hr_baseline"].replace(
        0,
        np.nan
    )
).fillna(0.0)


test_df["hr_absolute_abnormal"] = (
    (test_df["hr_max"] >= HR_HIGH_THRESHOLD)
    |
    (test_df["hr_min"] <= HR_LOW_THRESHOLD)
)


test_df["hr_relative_abnormal"] = (
    test_df["hr_relative_deviation"]
    >= HR_RELATIVE_CHANGE
)


test_df["hr_abnormal"] = (
    test_df["hr_absolute_abnormal"]
    |
    test_df["hr_relative_abnormal"]
)


test_df["spo2_baseline_drop"] = (
    test_df["spo2_baseline"]
    - test_df["spo2_mean"]
)


test_df["spo2_low"] = (
    test_df["spo2_min"]
    < SPO2_LOW_THRESHOLD
)


test_df["spo2_downward"] = (
    test_df["spo2_baseline_drop"]
    >= SPO2_DROP_THRESHOLD
)


test_df["spo2_abnormal"] = (
    test_df["spo2_low"]
    |
    test_df["spo2_downward"]
)


# ============================================================
# PPG QUALITY
# ============================================================

test_df["ppg_quality_poor"] = (
    (test_df["ppg_std"] < PPG_STD_MIN)
    |
    (test_df["ppg_range"] < PPG_RANGE_MIN)
)


# ============================================================
# HYBRID EVIDENCE
# ============================================================

test_df["signal_count"] = (
    test_df["hr_abnormal"].astype(int)
    +
    test_df["spo2_abnormal"].astype(int)
)


test_df["strong_evidence"] = (
    test_df["signal_count"] >= 2
)


# For this stress test we intentionally don't use the ML score.
# The goal is to test whether the physiological layer itself
# behaves correctly.
#
# This avoids giving the Isolation Forest an unfair advantage
# and lets us evaluate the actual safety logic.


test_df["raw_vital_evidence"] = (
    test_df["strong_evidence"]
)


# ============================================================
# PERSISTENCE TEST
# ============================================================

print_header(
    "APPLYING PERSISTENCE"
)

test_df = test_df.sort_values(
    [
        "recording_id",
        "window_start_s",
        "anomaly_type",
    ]
).reset_index(
    drop=True
)


# Because synthetic anomalies are generated independently,
# we evaluate single-window physiological evidence first.
#
# Then we perform a dedicated persistence experiment below.

test_df["vital_anomaly_single_window"] = (
    test_df["raw_vital_evidence"]
)


# ============================================================
# SINGLE-WINDOW RESULTS
# ============================================================

print_header(
    "SINGLE-WINDOW PERFORMANCE"
)

single_results = []

for anomaly_type in [
    "normal",
    *ANOMALY_GENERATORS.keys(),
]:

    subset = test_df[
        test_df["anomaly_type"]
        == anomaly_type
    ]

    detected = float(
        subset[
            "vital_anomaly_single_window"
        ].mean()
    )

    single_results.append({
        "anomaly_type": anomaly_type,
        "samples": len(subset),
        "detection_rate": (
            np.nan
            if anomaly_type == "normal"
            else detected
        ),
        "false_positive_rate": (
            detected
            if anomaly_type == "normal"
            else np.nan
        ),
    })


single_df = pd.DataFrame(
    single_results
)

print(
    single_df.to_string(
        index=False
    )
)


# ============================================================
# PERSISTENCE-SPECIFIC EXPERIMENT
# ============================================================

print_header(
    "PERSISTENCE EXPERIMENT"
)

# Build controlled sequences:
#
# 2 consecutive normal windows
# 2 consecutive abnormal windows
#
# This directly tests the intended persistence mechanism.

sequence_records = []

sequence_id = 0

for _, row in normal_df.head(100).iterrows():

    sequence_id += 1

    base = row.copy()

    # Normal 1
    sequence_records.append({
        "sequence_id": sequence_id,
        "position": 0,
        "anomaly_type": "normal",
        "ground_truth": 0,
        **base.to_dict()
    })

    # Normal 2
    sequence_records.append({
        "sequence_id": sequence_id,
        "position": 1,
        "anomaly_type": "normal",
        "ground_truth": 0,
        **base.to_dict()
    })

    # Pick anomaly type
    anomaly_type = list(
        ANOMALY_GENERATORS.keys()
    )[
        sequence_id
        % len(ANOMALY_GENERATORS)
    ]

    generator = ANOMALY_GENERATORS[
        anomaly_type
    ]

    abnormal_1 = generator(
        base.copy(),
        rng
    )

    abnormal_2 = generator(
        base.copy(),
        rng
    )

    sequence_records.append({
        "sequence_id": sequence_id,
        "position": 2,
        "anomaly_type": anomaly_type,
        "ground_truth": 1,
        **abnormal_1.to_dict()
    })

    sequence_records.append({
        "sequence_id": sequence_id,
        "position": 3,
        "anomaly_type": anomaly_type,
        "ground_truth": 1,
        **abnormal_2.to_dict()
    })


sequence_df = pd.DataFrame(
    sequence_records
)


# Calculate sequence physiological evidence.
sequence_df["hr_baseline"] = (
    sequence_df
    .groupby("sequence_id")["hr_mean"]
    .transform("median")
)

sequence_df["spo2_baseline"] = (
    sequence_df
    .groupby("sequence_id")["spo2_mean"]
    .transform("median")
)

sequence_df["hr_relative_deviation"] = (
    np.abs(
        sequence_df["hr_mean"]
        - sequence_df["hr_baseline"]
    )
    /
    sequence_df["hr_baseline"].replace(
        0,
        np.nan
    )
).fillna(0.0)

sequence_df["hr_abnormal"] = (
    (sequence_df["hr_max"] >= HR_HIGH_THRESHOLD)
    |
    (sequence_df["hr_min"] <= HR_LOW_THRESHOLD)
    |
    (
        sequence_df["hr_relative_deviation"]
        >= HR_RELATIVE_CHANGE
    )
)

sequence_df["spo2_abnormal"] = (
    (sequence_df["spo2_min"] < SPO2_LOW_THRESHOLD)
    |
    (
        (
            sequence_df["spo2_baseline"]
            - sequence_df["spo2_mean"]
        )
        >= SPO2_DROP_THRESHOLD
    )
)

sequence_df["raw_evidence"] = (
    sequence_df["hr_abnormal"]
    |
    sequence_df["spo2_abnormal"]
)

sequence_df["persistent_anomaly"] = False


for sequence_id, group in sequence_df.groupby(
    "sequence_id"
):

    consecutive = 0

    for idx in group.sort_values(
        "position"
    ).index:

        if bool(
            sequence_df.loc[
                idx,
                "raw_evidence"
            ]
        ):
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= PERSISTENCE_WINDOWS:

            sequence_df.loc[
                idx,
                "persistent_anomaly"
            ] = True


tp = int(
    np.sum(
        (
            sequence_df["ground_truth"]
            == 1
        )
        &
        (
            sequence_df["persistent_anomaly"]
            == True
        )
    )
)

tn = int(
    np.sum(
        (
            sequence_df["ground_truth"]
            == 0
        )
        &
        (
            sequence_df["persistent_anomaly"]
            == False
        )
    )
)

fp = int(
    np.sum(
        (
            sequence_df["ground_truth"]
            == 0
        )
        &
        (
            sequence_df["persistent_anomaly"]
            == True
        )
    )
)

fn = int(
    np.sum(
        (
            sequence_df["ground_truth"]
            == 1
        )
        &
        (
            sequence_df["persistent_anomaly"]
            == False
        )
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


print(
    f"Persistence windows: "
    f"{PERSISTENCE_WINDOWS}"
)

print(
    f"True positives:       {tp}"
)

print(
    f"True negatives:       {tn}"
)

print(
    f"False positives:      {fp}"
)

print(
    f"False negatives:      {fn}"
)

print(
    f"Precision:            {precision:.4f}"
)

print(
    f"Recall:               {recall:.4f}"
)

print(
    f"F1:                   {f1:.4f}"
)

print(
    f"False-positive rate:  {fpr:.4f}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header(
    "HYBRID VITAL STRESS TEST SUMMARY"
)

normal_fpr = float(
    single_df.loc[
        single_df["anomaly_type"]
        == "normal",
        "false_positive_rate"
    ].iloc[0]
)

print(
    f"Normal-window FPR: "
    f"{normal_fpr:.4f}"
)

print(
    f"Persistence FPR:   "
    f"{fpr:.4f}"
)

print(
    f"Persistence recall:"
    f" {recall:.4f}"
)

print(
    f"Persistence F1:    "
    f"{f1:.4f}"
)


# ============================================================
# SAVE
# ============================================================

test_df.to_csv(
    OUTPUT_FILE,
    index=False
)

summary = {
    "module": "THERMORA hybrid vital anomaly detector",
    "test_type": "synthetic_hybrid_stress_test",
    "normal_windows": int(
        len(normal_df)
    ),
    "total_test_windows": int(
        len(test_df)
    ),
    "single_window_normal_fpr": normal_fpr,
    "persistence_windows": PERSISTENCE_WINDOWS,
    "persistence_precision": precision,
    "persistence_recall": recall,
    "persistence_f1": f1,
    "persistence_false_positive_rate": fpr,
    "warning": (
        "Synthetic anomalies are controlled engineering "
        "tests and are not clinical validation."
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


print_header(
    "HYBRID VITAL STRESS TEST COMPLETED"
)

print(
    "Results:"
)

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)

print(
    "\nPaste the complete terminal output here."
)