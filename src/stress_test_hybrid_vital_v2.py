import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# THERMORA HYBRID VITAL DETECTOR
# CORRECTED SYNTHETIC STRESS TEST
#
# Important:
# Worker baselines are calculated ONLY from original normal
# windows and are frozen before synthetic abnormalities are
# injected.
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
    / "hybrid_vital_stress_test_v2.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "hybrid_vital_stress_summary_v2.json"
)


# ============================================================
# PRODUCTION-STYLE THRESHOLDS
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
# HELPERS
# ============================================================

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


GENERATORS = {
    "spo2_drop": modify_spo2,
    "high_hr": modify_high_hr,
    "low_hr": modify_low_hr,
    "combined_hr_spo2": modify_combined,
    "ppg_amplitude_disturbance": modify_ppg_amplitude,
    "ppg_noise_disturbance": modify_ppg_noise,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("THERMORA HYBRID VITAL STRESS TEST V2")
print("=" * 70)

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
].copy()

normal_df = normal_df.reset_index(
    drop=True
)

print(
    f"Original normal windows: "
    f"{len(normal_df)}"
)


# ============================================================
# FREEZE BASELINES BEFORE INJECTION
# ============================================================

print("\n" + "=" * 70)
print("FREEZING WORKER BASELINES")
print("=" * 70)

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
    f"Worker baselines created: "
    f"{len(hr_baselines)}"
)


# ============================================================
# CREATE SYNTHETIC DATA
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SYNTHETIC ABNORMALITIES")
print("=" * 70)

rng = np.random.default_rng(42)

records = []


# ------------------------------------------------------------
# NORMAL
# ------------------------------------------------------------

for _, row in normal_df.iterrows():

    recording_id = row["recording_id"]

    records.append({
        "recording_id": recording_id,
        "window_start_s": row["window_start_s"],
        "anomaly_type": "normal",
        "ground_truth": 0,

        "hr_mean": safe_float(row["hr_mean"]),
        "hr_min": safe_float(row["hr_min"]),
        "hr_max": safe_float(row["hr_max"]),
        "hr_slope": safe_float(row["hr_slope"]),
        "hr_start_end_change":
            safe_float(
                row["hr_start_end_change"]
            ),

        "spo2_mean": safe_float(row["spo2_mean"]),
        "spo2_min": safe_float(row["spo2_min"]),
        "spo2_max": safe_float(row["spo2_max"]),
        "spo2_slope": safe_float(row["spo2_slope"]),
        "spo2_start_end_change":
            safe_float(
                row["spo2_start_end_change"]
            ),

        "ppg_std": safe_float(row["ppg_std"]),
        "ppg_range": safe_float(row["ppg_range"]),
    })


# ------------------------------------------------------------
# ABNORMAL COPIES
# ------------------------------------------------------------

for anomaly_type, generator in GENERATORS.items():

    for _, row in normal_df.iterrows():

        modified = generator(
            row.copy(),
            rng
        )

        recording_id = row["recording_id"]

        records.append({
            "recording_id": recording_id,
            "window_start_s": row["window_start_s"],
            "anomaly_type": anomaly_type,
            "ground_truth": 1,

            "hr_mean": safe_float(
                modified["hr_mean"]
            ),
            "hr_min": safe_float(
                modified["hr_min"]
            ),
            "hr_max": safe_float(
                modified["hr_max"]
            ),
            "hr_slope": safe_float(
                modified["hr_slope"]
            ),
            "hr_start_end_change":
                safe_float(
                    modified[
                        "hr_start_end_change"
                    ]
                ),

            "spo2_mean": safe_float(
                modified["spo2_mean"]
            ),
            "spo2_min": safe_float(
                modified["spo2_min"]
            ),
            "spo2_max": safe_float(
                modified["spo2_max"]
            ),
            "spo2_slope": safe_float(
                modified["spo2_slope"]
            ),
            "spo2_start_end_change":
                safe_float(
                    modified[
                        "spo2_start_end_change"
                    ]
                ),

            "ppg_std": safe_float(
                modified["ppg_std"]
            ),
            "ppg_range": safe_float(
                modified["ppg_range"]
            ),
        })


test_df = pd.DataFrame(
    records
)


# ============================================================
# USE FROZEN BASELINES
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
# HR RULE
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


# ============================================================
# SPO2 RULE
# ============================================================

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


# PPG quality alone must NEVER become a vital anomaly.
test_df["strong_evidence"] = (
    test_df["hr_abnormal"]
    &
    test_df["spo2_abnormal"]
)


# Individual HR/SpO2 physiological evidence.
test_df["physiological_evidence"] = (
    test_df["hr_abnormal"]
    |
    test_df["spo2_abnormal"]
)


# ============================================================
# RESULTS BY ANOMALY TYPE
# ============================================================

print("\n" + "=" * 70)
print("SINGLE-WINDOW PERFORMANCE")
print("=" * 70)

results = []

for anomaly_type in [
    "normal",
    *GENERATORS.keys(),
]:

    subset = test_df[
        test_df["anomaly_type"]
        == anomaly_type
    ]

    evidence_rate = float(
        subset[
            "physiological_evidence"
        ].mean()
    )

    strong_rate = float(
        subset[
            "strong_evidence"
        ].mean()
    )

    results.append({
        "anomaly_type": anomaly_type,
        "samples": len(subset),
        "physiological_detection_rate":
            evidence_rate
            if anomaly_type != "normal"
            else np.nan,
        "strong_hr_spo2_rate":
            strong_rate
            if anomaly_type != "normal"
            else np.nan,
        "normal_false_positive_rate":
            evidence_rate
            if anomaly_type == "normal"
            else np.nan,
    })


results_df = pd.DataFrame(
    results
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# PER-MODALITY DETECTION
# ============================================================

print("\n" + "=" * 70)
print("PER-MODALITY DETECTION")
print("=" * 70)

for anomaly_type in GENERATORS:

    subset = test_df[
        test_df["anomaly_type"]
        == anomaly_type
    ]

    hr_rate = subset[
        "hr_abnormal"
    ].mean()

    spo2_rate = subset[
        "spo2_abnormal"
    ].mean()

    print(
        f"{anomaly_type:30s}"
        f" HR={hr_rate:.3f}"
        f"  SpO2={spo2_rate:.3f}"
    )


# ============================================================
# CONTROLLED PERSISTENCE TEST
# ============================================================

print("\n" + "=" * 70)
print("CONTROLLED 2-WINDOW PERSISTENCE TEST")
print("=" * 70)

persistence_records = []

sequence_id = 0

# Use first 100 worker-normal windows.
# Each sequence contains:
#
# window 0 = normal
# window 1 = normal
# window 2 = anomaly
# window 3 = same anomaly
#
# Baseline remains the frozen worker baseline.

for _, row in normal_df.head(100).iterrows():

    sequence_id += 1

    recording_id = row["recording_id"]

    base = row.copy()

    # Normal windows
    for position in [0, 1]:

        persistence_records.append({
            "sequence_id": sequence_id,
            "position": position,
            "recording_id": recording_id,
            "anomaly_type": "normal",
            "ground_truth": 0,
            **base.to_dict()
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
        base.copy(),
        rng
    )

    abnormal2 = generator(
        base.copy(),
        rng
    )

    for position, abnormal in [
        (2, abnormal1),
        (3, abnormal2),
    ]:

        persistence_records.append({
            "sequence_id": sequence_id,
            "position": position,
            "recording_id": recording_id,
            "anomaly_type": anomaly_type,
            "ground_truth": 1,
            **abnormal.to_dict()
        })


persistence_df = pd.DataFrame(
    persistence_records
)


# Frozen baseline mapping
persistence_df["hr_baseline"] = (
    persistence_df["recording_id"]
    .map(hr_baselines)
)

persistence_df["spo2_baseline"] = (
    persistence_df["recording_id"]
    .map(spo2_baselines)
)


# HR
persistence_df["hr_relative_deviation"] = (
    np.abs(
        persistence_df["hr_mean"]
        - persistence_df["hr_baseline"]
    )
    /
    persistence_df["hr_baseline"].replace(
        0,
        np.nan
    )
).fillna(0.0)


persistence_df["hr_abnormal"] = (
    (persistence_df["hr_max"] >= HR_HIGH_THRESHOLD)
    |
    (persistence_df["hr_min"] <= HR_LOW_THRESHOLD)
    |
    (
        persistence_df[
            "hr_relative_deviation"
        ]
        >= HR_RELATIVE_CHANGE
    )
)


# SpO2
persistence_df["spo2_abnormal"] = (
    (persistence_df["spo2_min"] < SPO2_LOW_THRESHOLD)
    |
    (
        (
            persistence_df["spo2_baseline"]
            - persistence_df["spo2_mean"]
        )
        >= SPO2_DROP_THRESHOLD
    )
)


persistence_df["physiological_evidence"] = (
    persistence_df["hr_abnormal"]
    |
    persistence_df["spo2_abnormal"]
)


# ============================================================
# PERSISTENCE STATE MACHINE
# ============================================================

persistence_df["persistent_anomaly"] = False

for sequence_id, group in persistence_df.groupby(
    "sequence_id"
):

    consecutive = 0

    group = group.sort_values(
        "position"
    )

    for idx in group.index:

        if bool(
            persistence_df.loc[
                idx,
                "physiological_evidence"
            ]
        ):

            consecutive += 1

        else:

            consecutive = 0

        if consecutive >= PERSISTENCE_WINDOWS:

            persistence_df.loc[
                idx,
                "persistent_anomaly"
            ] = True


# ============================================================
# METRICS
# ============================================================

tp = int(
    (
        (persistence_df["ground_truth"] == 1)
        &
        (
            persistence_df[
                "persistent_anomaly"
            ]
            == True
        )
    ).sum()
)

tn = int(
    (
        (persistence_df["ground_truth"] == 0)
        &
        (
            persistence_df[
                "persistent_anomaly"
            ]
            == False
        )
    ).sum()
)

fp = int(
    (
        (persistence_df["ground_truth"] == 0)
        &
        (
            persistence_df[
                "persistent_anomaly"
            ]
            == True
        )
    ).sum()
)

fn = int(
    (
        (persistence_df["ground_truth"] == 1)
        &
        (
            persistence_df[
                "persistent_anomaly"
            ]
            == False
        )
    ).sum()
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
        "baseline_isolated_synthetic_stress_test",

    "normal_windows":
        int(len(normal_df)),

    "total_test_windows":
        int(len(test_df)),

    "normal_single_window_fpr":
        float(
            results_df.loc[
                results_df["anomaly_type"]
                == "normal",
                "normal_false_positive_rate"
            ].iloc[0]
        ),

    "persistence_windows":
        PERSISTENCE_WINDOWS,

    "persistence_precision":
        precision,

    "persistence_recall":
        recall,

    "persistence_f1":
        f1,

    "persistence_false_positive_rate":
        fpr,

    "warning":
        "Synthetic anomalies are engineering stress tests, "
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
print("HYBRID VITAL STRESS TEST V2 COMPLETED")
print("=" * 70)

print("\nResults:")
print(OUTPUT_FILE)

print(SUMMARY_FILE)

print(
    "\nPaste the complete terminal output here."
)