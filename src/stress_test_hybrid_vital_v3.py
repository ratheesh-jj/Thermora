import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# THERMORA HYBRID VITAL DETECTOR V3
#
# Modality-aware persistence:
#
#   HR + SpO2 abnormal together
#       -> immediate anomaly
#
#   HR abnormal alone
#       -> 2 consecutive windows
#
#   SpO2 abnormal alone
#       -> 2 consecutive windows
#
#   PPG disturbance alone
#       -> NOT a vital anomaly
#
# Baselines are frozen from ORIGINAL normal windows.
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
    / "hybrid_vital_stress_test_v3.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "hybrid_vital_stress_summary_v3.json"
)


# ============================================================
# THRESHOLDS
# ============================================================

SPO2_LOW_THRESHOLD = 90.0
HR_HIGH_THRESHOLD = 120.0
HR_LOW_THRESHOLD = 50.0

HR_RELATIVE_CHANGE = 0.25
SPO2_DROP_THRESHOLD = 3.0

PPG_STD_MIN = 0.005
PPG_RANGE_MIN = 0.02

PERSISTENCE_WINDOWS = 2


# ============================================================
# SYNTHETIC MODIFIERS
# ============================================================

def modify_spo2(row, rng):

    out = row.copy()

    target = rng.uniform(86.0, 89.0)

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

    out["spo2_q25"] = target - rng.uniform(
        0.5, 1.5
    )

    out["spo2_q75"] = target + rng.uniform(
        0.2, 1.0
    )

    out["spo2_slope"] = rng.uniform(
        -0.15, -0.03
    )

    out["spo2_start_end_change"] = rng.uniform(
        -5.0, -1.0
    )

    return out


def modify_high_hr(row, rng):

    out = row.copy()

    target = rng.uniform(
        120.0, 135.0
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

    out["hr_q25"] = target - rng.uniform(
        2.0, 5.0
    )

    out["hr_q75"] = target + rng.uniform(
        2.0, 5.0
    )

    out["hr_slope"] = rng.uniform(
        0.05, 0.25
    )

    out["hr_start_end_change"] = rng.uniform(
        5.0, 15.0
    )

    return out


def modify_low_hr(row, rng):

    out = row.copy()

    target = rng.uniform(
        42.0, 48.0
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

    out["hr_q25"] = target - rng.uniform(
        1.0, 3.0
    )

    out["hr_q75"] = target + rng.uniform(
        1.0, 3.0
    )

    out["hr_slope"] = rng.uniform(
        -0.20, -0.05
    )

    out["hr_start_end_change"] = rng.uniform(
        -15.0, -4.0
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
        0.15, 0.35
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
        3.0, 6.0
    )

    out["ppg_std"] *= factor
    out["ppg_diff_std"] *= factor
    out["ppg_diff_abs_mean"] *= factor
    out["ppg_diff_max_abs"] *= factor

    out["ppg_range"] *= rng.uniform(
        2.0, 4.0
    )

    return out


GENERATORS = {
    "spo2_drop": modify_spo2,
    "high_hr": modify_high_hr,
    "low_hr": modify_low_hr,
    "combined_hr_spo2": modify_combined,
    "ppg_amplitude_disturbance": modify_ppg_amplitude,
    "ppg_noise_disturbance": modify_ppg_noise,
}


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("THERMORA HYBRID VITAL STRESS TEST V3")
print("=" * 70)

print("\nLoading:")
print(DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print(
    f"\nRows:       {len(df)}"
)

print(
    f"Recordings: {df['recording_id'].nunique()}"
)


# ============================================================
# ORIGINAL NORMAL WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING ORIGINAL NORMAL WINDOWS")
print("=" * 70)

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
].copy().reset_index(drop=True)

print(
    f"Original normal windows: "
    f"{len(normal_df)}"
)


# ============================================================
# FROZEN BASELINES
# ============================================================

hr_baselines = (
    normal_df
    .groupby("recording_id")["hr_mean"]
    .median()
    .to_dict()
)

spo2_baselines = (
    normal_df
    .groupby("recording_id")["spo2_mean"]
    .median()
    .to_dict()
)

print(
    f"Worker baselines: "
    f"{len(hr_baselines)}"
)


# ============================================================
# BUILD SYNTHETIC WINDOWS
# ============================================================

rng = np.random.default_rng(42)

records = []


for _, row in normal_df.iterrows():

    # --------------------------------------------------------
    # ORIGINAL NORMAL
    # --------------------------------------------------------

    records.append({
        "recording_id": row["recording_id"],
        "window_start_s": row["window_start_s"],
        "anomaly_type": "normal",
        "ground_truth": 0,

        "hr_mean": row["hr_mean"],
        "hr_min": row["hr_min"],
        "hr_max": row["hr_max"],
        "hr_slope": row["hr_slope"],
        "hr_start_end_change":
            row["hr_start_end_change"],

        "spo2_mean": row["spo2_mean"],
        "spo2_min": row["spo2_min"],
        "spo2_max": row["spo2_max"],
        "spo2_slope": row["spo2_slope"],
        "spo2_start_end_change":
            row["spo2_start_end_change"],

        "ppg_std": row["ppg_std"],
        "ppg_range": row["ppg_range"],
    })

    # --------------------------------------------------------
    # SYNTHETIC ABNORMAL COPIES
    # --------------------------------------------------------

    for anomaly_type, generator in GENERATORS.items():

        modified = generator(
            row.copy(),
            rng
        )

        records.append({
            "recording_id": row["recording_id"],
            "window_start_s":
                row["window_start_s"],
            "anomaly_type": anomaly_type,
            "ground_truth": 1,

            "hr_mean": modified["hr_mean"],
            "hr_min": modified["hr_min"],
            "hr_max": modified["hr_max"],
            "hr_slope": modified["hr_slope"],
            "hr_start_end_change":
                modified["hr_start_end_change"],

            "spo2_mean": modified["spo2_mean"],
            "spo2_min": modified["spo2_min"],
            "spo2_max": modified["spo2_max"],
            "spo2_slope": modified["spo2_slope"],
            "spo2_start_end_change":
                modified["spo2_start_end_change"],

            "ppg_std": modified["ppg_std"],
            "ppg_range": modified["ppg_range"],
        })


test_df = pd.DataFrame(records)


# ============================================================
# FROZEN BASELINES
# ============================================================

test_df["hr_baseline"] = (
    test_df["recording_id"]
    .map(hr_baselines)
)

test_df["spo2_baseline"] = (
    test_df["recording_id"]
    .map(spo2_baselines)
)


# ============================================================
# HR DETECTION
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

test_df["hr_abnormal"] = (
    (test_df["hr_max"] >= HR_HIGH_THRESHOLD)
    |
    (test_df["hr_min"] <= HR_LOW_THRESHOLD)
    |
    (
        test_df["hr_relative_deviation"]
        >= HR_RELATIVE_CHANGE
    )
)


# ============================================================
# SPO2 DETECTION
# ============================================================

test_df["spo2_baseline_drop"] = (
    test_df["spo2_baseline"]
    - test_df["spo2_mean"]
)

test_df["spo2_abnormal"] = (
    (test_df["spo2_min"] < SPO2_LOW_THRESHOLD)
    |
    (
        test_df["spo2_baseline_drop"]
        >= SPO2_DROP_THRESHOLD
    )
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
# MODALITY-AWARE LOGIC
# ============================================================

test_df["hr_only_evidence"] = (
    test_df["hr_abnormal"]
    &
    ~test_df["spo2_abnormal"]
)

test_df["spo2_only_evidence"] = (
    test_df["spo2_abnormal"]
    &
    ~test_df["hr_abnormal"]
)

test_df["combined_evidence"] = (
    test_df["hr_abnormal"]
    &
    test_df["spo2_abnormal"]
)


# ============================================================
# SINGLE-WINDOW RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SINGLE-WINDOW MODALITY DETECTION")
print("=" * 70)

single_results = []

for anomaly_type in [
    "normal",
    *GENERATORS.keys(),
]:

    subset = test_df[
        test_df["anomaly_type"]
        == anomaly_type
    ]

    hr_rate = float(
        subset["hr_abnormal"].mean()
    )

    spo2_rate = float(
        subset["spo2_abnormal"].mean()
    )

    combined_rate = float(
        subset["combined_evidence"].mean()
    )

    physiological_rate = float(
        (
            subset["hr_abnormal"]
            |
            subset["spo2_abnormal"]
        ).mean()
    )

    single_results.append({
        "anomaly_type": anomaly_type,
        "samples": len(subset),
        "hr_detection_rate":
            hr_rate,
        "spo2_detection_rate":
            spo2_rate,
        "combined_detection_rate":
            combined_rate,
        "physiological_detection_rate":
            physiological_rate,
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
# CONTROLLED PERSISTENCE
# ============================================================

print("\n" + "=" * 70)
print("MODALITY-AWARE 2-WINDOW PERSISTENCE")
print("=" * 70)

sequence_records = []

sequence_id = 0

normal_samples = normal_df.head(120)

for _, row in normal_samples.iterrows():

    sequence_id += 1

    recording_id = row["recording_id"]

    # Two normal windows
    sequence_records.append({
        "sequence_id": sequence_id,
        "position": 0,
        "recording_id": recording_id,
        "anomaly_type": "normal",
        "ground_truth": 0,
        **row.to_dict()
    })

    sequence_records.append({
        "sequence_id": sequence_id,
        "position": 1,
        "recording_id": recording_id,
        "anomaly_type": "normal",
        "ground_truth": 0,
        **row.to_dict()
    })

    anomaly_type = list(
        GENERATORS.keys()
    )[
        (sequence_id - 1)
        % len(GENERATORS)
    ]

    generator = GENERATORS[
        anomaly_type
    ]

    abnormal1 = generator(
        row.copy(),
        rng
    )

    abnormal2 = generator(
        row.copy(),
        rng
    )

    sequence_records.append({
        "sequence_id": sequence_id,
        "position": 2,
        "recording_id": recording_id,
        "anomaly_type": anomaly_type,
        "ground_truth": 1,
        **abnormal1.to_dict()
    })

    sequence_records.append({
        "sequence_id": sequence_id,
        "position": 3,
        "recording_id": recording_id,
        "anomaly_type": anomaly_type,
        "ground_truth": 1,
        **abnormal2.to_dict()
    })


sequence_df = pd.DataFrame(
    sequence_records
)


# ------------------------------------------------------------
# Frozen baselines
# ------------------------------------------------------------

sequence_df["hr_baseline"] = (
    sequence_df["recording_id"]
    .map(hr_baselines)
)

sequence_df["spo2_baseline"] = (
    sequence_df["recording_id"]
    .map(spo2_baselines)
)


# ------------------------------------------------------------
# HR
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# SpO2
# ------------------------------------------------------------

sequence_df["spo2_baseline_drop"] = (
    sequence_df["spo2_baseline"]
    - sequence_df["spo2_mean"]
)

sequence_df["spo2_abnormal"] = (
    (sequence_df["spo2_min"] < SPO2_LOW_THRESHOLD)
    |
    (
        sequence_df["spo2_baseline_drop"]
        >= SPO2_DROP_THRESHOLD
    )
)


# ------------------------------------------------------------
# Modality-aware state machine
# ------------------------------------------------------------

sequence_df["vital_anomaly"] = False
sequence_df["trigger_type"] = "NONE"

for sid, group in sequence_df.groupby(
    "sequence_id"
):

    group = group.sort_values(
        "position"
    )

    hr_count = 0
    spo2_count = 0

    for idx in group.index:

        hr_bad = bool(
            sequence_df.loc[
                idx,
                "hr_abnormal"
            ]
        )

        spo2_bad = bool(
            sequence_df.loc[
                idx,
                "spo2_abnormal"
            ]
        )

        # Strong combined evidence:
        # immediate trigger.
        if hr_bad and spo2_bad:

            sequence_df.loc[
                idx,
                "vital_anomaly"
            ] = True

            sequence_df.loc[
                idx,
                "trigger_type"
            ] = "COMBINED_IMMEDIATE"

        else:

            if hr_bad:
                hr_count += 1
            else:
                hr_count = 0

            if spo2_bad:
                spo2_count += 1
            else:
                spo2_count = 0

            # Single modality needs persistence.
            if hr_count >= PERSISTENCE_WINDOWS:

                sequence_df.loc[
                    idx,
                    "vital_anomaly"
                ] = True

                sequence_df.loc[
                    idx,
                    "trigger_type"
                ] = "HR_PERSISTENCE"

            elif spo2_count >= PERSISTENCE_WINDOWS:

                sequence_df.loc[
                    idx,
                    "vital_anomaly"
                ] = True

                sequence_df.loc[
                    idx,
                    "trigger_type"
                ] = "SPO2_PERSISTENCE"


# ============================================================
# METRICS
# ============================================================

y_true = (
    sequence_df["ground_truth"]
    .to_numpy()
)

y_pred = (
    sequence_df["vital_anomaly"]
    .to_numpy()
)


tp = int(
    np.sum(
        (y_true == 1)
        &
        (y_pred == 1)
    )
)

tn = int(
    np.sum(
        (y_true == 0)
        &
        (y_pred == 0)
    )
)

fp = int(
    np.sum(
        (y_true == 0)
        &
        (y_pred == 1)
    )
)

fn = int(
    np.sum(
        (y_true == 1)
        &
        (y_pred == 0)
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
    f"True positives:      {tp}"
)

print(
    f"True negatives:      {tn}"
)

print(
    f"False positives:     {fp}"
)

print(
    f"False negatives:     {fn}"
)

print(
    f"Precision:           {precision:.4f}"
)

print(
    f"Recall:              {recall:.4f}"
)

print(
    f"F1:                  {f1:.4f}"
)

print(
    f"False-positive rate: {fpr:.4f}"
)


# ============================================================
# TRIGGER BREAKDOWN
# ============================================================

print("\n" + "=" * 70)
print("TRIGGER BREAKDOWN")
print("=" * 70)

print(
    sequence_df[
        sequence_df["vital_anomaly"]
    ]["trigger_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

test_df.to_csv(
    OUTPUT_FILE,
    index=False
)

summary = {
    "module":
        "THERMORA hybrid vital anomaly detector",

    "test":
        "modality_aware_persistence_stress_test",

    "normal_windows":
        int(len(normal_df)),

    "total_single_window_samples":
        int(len(test_df)),

    "persistence_sequences":
        int(sequence_df["sequence_id"].nunique()),

    "persistence_precision":
        precision,

    "persistence_recall":
        recall,

    "persistence_f1":
        f1,

    "persistence_false_positive_rate":
        fpr,

    "logic": {
        "combined_hr_spo2":
            "immediate anomaly",

        "hr_only":
            "2 consecutive windows",

        "spo2_only":
            "2 consecutive windows",

        "ppg_only":
            "not a vital anomaly",
    },

    "warning":
        "Synthetic abnormalities are engineering stress tests, "
        "not clinical validation."
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


print("\n" + "=" * 70)
print("V3 STRESS TEST COMPLETED")
print("=" * 70)

print("\nResults:")
print(OUTPUT_FILE)

print(SUMMARY_FILE)