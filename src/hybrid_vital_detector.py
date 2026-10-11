import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# THERMORA HYBRID VITAL ANOMALY DETECTOR
# PRODUCTION PROTOTYPE - V3 LOGIC
#
# Independent output:
#   HR
#   SpO2
#   PPG quality
#   HR deviation
#   SpO2 deviation
#   Vital anomaly
#   Trigger reason
#
# NO unified SAFE/CAUTION/HIGH/CRITICAL score.
#
# Logic:
#
#   HR + SpO2 abnormal
#       -> immediate VITAL_ANOMALY
#
#   HR alone
#       -> 2 consecutive windows
#
#   SpO2 alone
#       -> 2 consecutive windows
#
#   PPG quality problem alone
#       -> MEASUREMENT_UNRELIABLE
#          NOT VITAL_ANOMALY
#
# Isolation Forest:
#   supporting evidence only
#   never overrides physiological logic
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "processed"
    / "vital_windows.csv"
)

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
    / "vital_hybrid_outputs_v3.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "vital_hybrid_summary_v3.json"
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

# Existing calibrated ML-support threshold.
# ML is supporting evidence only.
ML_SUPPORT_THRESHOLD = 0.875


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("THERMORA HYBRID VITAL ANOMALY DETECTOR V3")
print("=" * 70)

print("\nDataset:")
print(DATA_FILE)

print("\nModel:")
print(MODEL_FILE)


if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Vital dataset not found:\n{DATA_FILE}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Vital anomaly model not found:\n{MODEL_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
)

print(
    f"\nInput windows: "
    f"{len(df)}"
)

print(
    f"Recordings:    "
    f"{df['recording_id'].nunique()}"
)


model = joblib.load(
    MODEL_FILE
)

print(
    f"ML model:      "
    f"{type(model).__name__}"
)


# ============================================================
# LOAD FEATURE SCHEMA
# ============================================================

feature_names = None

if FEATURE_FILE.exists():

    try:

        with open(
            FEATURE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            feature_info = json.load(f)

        if isinstance(feature_info, list):

            feature_names = feature_info

        elif isinstance(feature_info, dict):

            for key in [
                "features",
                "feature_names",
                "input_features",
            ]:

                if key in feature_info:

                    feature_names = (
                        feature_info[key]
                    )

                    break

    except Exception as exc:

        print(
            f"\nFeature schema warning: "
            f"{exc}"
        )


# ============================================================
# ML FEATURE PREPARATION
# ============================================================

if feature_names is not None:

    missing_features = [
        col
        for col in feature_names
        if col not in df.columns
    ]

    if missing_features:

        print(
            "\nWARNING:"
            f" {len(missing_features)}"
            " model features are missing."
        )

        print(
            missing_features[:10]
        )

        feature_names = None


if feature_names is not None:

    X_ml = df[
        feature_names
    ].copy()

else:

    metadata_columns = {
        "recording_id",
        "window_start_s",
        "window_end_s",
        "ppg_samples",
        "numeric_samples",
    }

    feature_names = [
        col
        for col in df.columns
        if col not in metadata_columns
    ]

    X_ml = df[
        feature_names
    ].copy()


# Numeric conversion
X_ml = X_ml.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# ML ANOMALY SCORE
# ============================================================

try:

    raw_decision = (
        model.decision_function(
            X_ml
        )
    )

    # Isolation Forest:
    # lower decision function = more anomalous.
    #
    # Convert so larger = more anomalous.
    ml_raw_score = (
        -np.asarray(
            raw_decision,
            dtype=float
        )
    )

    # Normalize across current input set.
    score_low = np.nanpercentile(
        ml_raw_score,
        5
    )

    score_high = np.nanpercentile(
        ml_raw_score,
        95
    )

    if score_high > score_low:

        ml_score = (
            ml_raw_score
            - score_low
        ) / (
            score_high
            - score_low
        )

    else:

        ml_score = np.zeros(
            len(df),
            dtype=float
        )

    ml_score = np.clip(
        ml_score,
        0.0,
        1.0
    )

except Exception as exc:

    print(
        "\nML scoring failed:"
    )

    print(exc)

    ml_score = np.zeros(
        len(df),
        dtype=float
    )


df["ml_anomaly_score"] = (
    ml_score
)

df["ml_support"] = (
    df["ml_anomaly_score"]
    >= ML_SUPPORT_THRESHOLD
)


# ============================================================
# WORKER BASELINES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PERSONAL BASELINES")
print("=" * 70)


# Sort first so temporal processing is deterministic.
df = df.sort_values(
    [
        "recording_id",
        "window_start_s",
    ]
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# Baselines
#
# Use the median of the first valid windows from each
# recording/session. This prevents the abnormal current window
# from contaminating its own baseline.
# ------------------------------------------------------------

hr_baselines = {}
spo2_baselines = {}


for recording_id, group in df.groupby(
    "recording_id"
):

    group = group.sort_values(
        "window_start_s"
    )

    # Prefer physiologically reasonable windows.
    baseline_candidates = group[
        (group["spo2_mean"] >= 92)
        &
        (group["hr_min"] > 50)
        &
        (group["hr_max"] < 115)
    ]

    if len(baseline_candidates) == 0:

        baseline_candidates = group

    # Use earliest 5 valid windows.
    baseline_candidates = (
        baseline_candidates
        .head(5)
    )

    hr_value = (
        baseline_candidates[
            "hr_mean"
        ].median()
    )

    spo2_value = (
        baseline_candidates[
            "spo2_mean"
        ].median()
    )

    if np.isfinite(hr_value):

        hr_baselines[
            recording_id
        ] = float(hr_value)

    if np.isfinite(spo2_value):

        spo2_baselines[
            recording_id
        ] = float(spo2_value)


print(
    f"HR baselines:   "
    f"{len(hr_baselines)}"
)

print(
    f"SpO2 baselines: "
    f"{len(spo2_baselines)}"
)


# ============================================================
# MAP BASELINES
# ============================================================

df["hr_baseline"] = (
    df["recording_id"]
    .map(hr_baselines)
)

df["spo2_baseline"] = (
    df["recording_id"]
    .map(spo2_baselines)
)


# ============================================================
# HR ABNORMALITY
# ============================================================

df["hr_relative_deviation"] = (
    np.abs(
        df["hr_mean"]
        - df["hr_baseline"]
    )
    /
    df["hr_baseline"].replace(
        0,
        np.nan
    )
).fillna(0.0)


df["hr_high"] = (
    df["hr_max"]
    >= HR_HIGH_THRESHOLD
)


df["hr_low"] = (
    df["hr_min"]
    <= HR_LOW_THRESHOLD
)


df["hr_relative_abnormal"] = (
    df["hr_relative_deviation"]
    >= HR_RELATIVE_CHANGE
)


df["hr_abnormal"] = (
    df["hr_high"]
    |
    df["hr_low"]
    |
    df["hr_relative_abnormal"]
)


# ============================================================
# SPO2 ABNORMALITY
# ============================================================

df["spo2_baseline_drop"] = (
    df["spo2_baseline"]
    - df["spo2_mean"]
)


df["spo2_low"] = (
    df["spo2_min"]
    < SPO2_LOW_THRESHOLD
)


df["spo2_baseline_abnormal"] = (
    df["spo2_baseline_drop"]
    >= SPO2_DROP_THRESHOLD
)


df["spo2_abnormal"] = (
    df["spo2_low"]
    |
    df["spo2_baseline_abnormal"]
)


# ============================================================
# PPG QUALITY
# ============================================================

df["ppg_quality_poor"] = (
    (df["ppg_std"] < PPG_STD_MIN)
    |
    (df["ppg_range"] < PPG_RANGE_MIN)
)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

df["vital_anomaly"] = False

df["vital_status"] = "NORMAL"

df["trigger_reason"] = "NONE"

df["hr_persistence_count"] = 0

df["spo2_persistence_count"] = 0


# ============================================================
# TEMPORAL STATE MACHINE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING MODALITY-AWARE TEMPORAL DETECTOR")
print("=" * 70)


for recording_id, group in df.groupby(
    "recording_id"
):

    group = group.sort_values(
        "window_start_s"
    )

    hr_count = 0
    spo2_count = 0

    previous_time = None

    for idx in group.index:

        current_time = (
            df.loc[
                idx,
                "window_start_s"
            ]
        )

        # ----------------------------------------------------
        # Reset if temporal gap is unreasonable.
        # ----------------------------------------------------

        if previous_time is not None:

            time_gap = (
                current_time
                - previous_time
            )

            if (
                not np.isfinite(time_gap)
                or time_gap > 60
                or time_gap < 0
            ):

                hr_count = 0
                spo2_count = 0

        previous_time = current_time

        hr_bad = bool(
            df.loc[
                idx,
                "hr_abnormal"
            ]
        )

        spo2_bad = bool(
            df.loc[
                idx,
                "spo2_abnormal"
            ]
        )

        ppg_bad = bool(
            df.loc[
                idx,
                "ppg_quality_poor"
            ]
        )

        # ----------------------------------------------------
        # COMBINED PHYSIOLOGICAL EVIDENCE
        #
        # Immediate anomaly.
        # ----------------------------------------------------

        if hr_bad and spo2_bad:

            hr_count += 1
            spo2_count += 1

            df.loc[
                idx,
                "vital_anomaly"
            ] = True

            df.loc[
                idx,
                "vital_status"
            ] = "VITAL_ANOMALY"

            df.loc[
                idx,
                "trigger_reason"
            ] = "COMBINED_HR_SPO2"

        else:

            # ------------------------------------------------
            # HR persistence
            # ------------------------------------------------

            if hr_bad:

                hr_count += 1

            else:

                hr_count = 0

            # ------------------------------------------------
            # SpO2 persistence
            # ------------------------------------------------

            if spo2_bad:

                spo2_count += 1

            else:

                spo2_count = 0

            # ------------------------------------------------
            # Persistent HR
            # ------------------------------------------------

            if hr_count >= PERSISTENCE_WINDOWS:

                df.loc[
                    idx,
                    "vital_anomaly"
                ] = True

                df.loc[
                    idx,
                    "vital_status"
                ] = "VITAL_ANOMALY"

                df.loc[
                    idx,
                    "trigger_reason"
                ] = "HR_PERSISTENCE"

            # ------------------------------------------------
            # Persistent SpO2
            # ------------------------------------------------

            elif spo2_count >= PERSISTENCE_WINDOWS:

                df.loc[
                    idx,
                    "vital_anomaly"
                ] = True

                df.loc[
                    idx,
                    "vital_status"
                ] = "VITAL_ANOMALY"

                df.loc[
                    idx,
                    "trigger_reason"
                ] = "SPO2_PERSISTENCE"

            # ------------------------------------------------
            # Measurement quality
            #
            # Only report unreliable measurement when there
            # is no physiological anomaly.
            # ------------------------------------------------

            elif ppg_bad:

                df.loc[
                    idx,
                    "vital_status"
                ] = (
                    "MEASUREMENT_UNRELIABLE"
                )

                df.loc[
                    idx,
                    "trigger_reason"
                ] = "POOR_PPG_QUALITY"

            # ------------------------------------------------
            # HR deviation pending
            # ------------------------------------------------

            elif hr_bad:

                df.loc[
                    idx,
                    "vital_status"
                ] = "HR_DEVIATION"

                df.loc[
                    idx,
                    "trigger_reason"
                ] = "HR_PENDING_PERSISTENCE"

            # ------------------------------------------------
            # SpO2 deviation pending
            # ------------------------------------------------

            elif spo2_bad:

                df.loc[
                    idx,
                    "vital_status"
                ] = "SPO2_DEVIATION"

                df.loc[
                    idx,
                    "trigger_reason"
                ] = (
                    "SPO2_PENDING_PERSISTENCE"
                )

            # ------------------------------------------------
            # ML support only
            #
            # IMPORTANT:
            # ML cannot create a medical anomaly by itself.
            # ------------------------------------------------

            elif bool(
                df.loc[
                    idx,
                    "ml_support"
                ]
            ):

                df.loc[
                    idx,
                    "vital_status"
                ] = "ML_ANOMALY_SUPPORT"

                df.loc[
                    idx,
                    "trigger_reason"
                ] = "ML_SUPPORT_ONLY"


        df.loc[
            idx,
            "hr_persistence_count"
        ] = hr_count

        df.loc[
            idx,
            "spo2_persistence_count"
        ] = spo2_count


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VITAL OUTPUT SUMMARY")
print("=" * 70)


status_counts = (
    df["vital_status"]
    .value_counts()
)

print(
    status_counts.to_string()
)


print(
    f"\nMean HR: "
    f"{df['hr_mean'].mean():.2f} BPM"
)

print(
    f"Mean SpO2: "
    f"{df['spo2_mean'].mean():.2f}%"
)

print(
    f"Mean ML score: "
    f"{df['ml_anomaly_score'].mean():.4f}"
)

print(
    f"HR abnormal windows: "
    f"{int(df['hr_abnormal'].sum())}"
)

print(
    f"SpO2 abnormal windows: "
    f"{int(df['spo2_abnormal'].sum())}"
)

print(
    f"Poor PPG windows: "
    f"{int(df['ppg_quality_poor'].sum())}"
)

print(
    f"Vital anomalies: "
    f"{int(df['vital_anomaly'].sum())}"
)


# ============================================================
# TRIGGER BREAKDOWN
# ============================================================

print("\n" + "=" * 70)
print("TRIGGER BREAKDOWN")
print("=" * 70)

trigger_counts = (
    df[
        df["vital_anomaly"]
    ]["trigger_reason"]
    .value_counts()
)

if len(trigger_counts) > 0:

    print(
        trigger_counts.to_string()
    )

else:

    print(
        "No vital anomalies detected."
    )


# ============================================================
# SAVE CLEAN OUTPUT
# ============================================================

output_columns = [
    "recording_id",
    "window_start_s",
    "window_end_s",

    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_baseline",
    "hr_relative_deviation",
    "hr_high",
    "hr_low",
    "hr_relative_abnormal",
    "hr_abnormal",

    "spo2_mean",
    "spo2_min",
    "spo2_max",
    "spo2_baseline",
    "spo2_baseline_drop",
    "spo2_low",
    "spo2_baseline_abnormal",
    "spo2_abnormal",

    "ppg_std",
    "ppg_range",
    "ppg_quality_poor",

    "ml_anomaly_score",
    "ml_support",

    "hr_persistence_count",
    "spo2_persistence_count",

    "vital_anomaly",
    "vital_status",
    "trigger_reason",
]


output_columns = [
    col
    for col in output_columns
    if col in df.columns
]


df[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "module":
        "THERMORA hybrid vital anomaly detector",

    "version":
        "V3 modality-aware production prototype",

    "input_windows":
        int(len(df)),

    "recordings":
        int(df["recording_id"].nunique()),

    "vital_anomalies":
        int(df["vital_anomaly"].sum()),

    "hr_abnormal_windows":
        int(df["hr_abnormal"].sum()),

    "spo2_abnormal_windows":
        int(df["spo2_abnormal"].sum()),

    "poor_ppg_windows":
        int(df["ppg_quality_poor"].sum()),

    "ml_support_windows":
        int(df["ml_support"].sum()),

    "status_distribution":
        {
            str(k): int(v)
            for k, v in status_counts.items()
        },

    "trigger_distribution":
        {
            str(k): int(v)
            for k, v in trigger_counts.items()
        },

    "thresholds": {
        "spo2_low":
            SPO2_LOW_THRESHOLD,

        "hr_high":
            HR_HIGH_THRESHOLD,

        "hr_low":
            HR_LOW_THRESHOLD,

        "hr_relative_change":
            HR_RELATIVE_CHANGE,

        "spo2_baseline_drop":
            SPO2_DROP_THRESHOLD,

        "persistence_windows":
            PERSISTENCE_WINDOWS,

        "ml_support":
            ML_SUPPORT_THRESHOLD,
    },

    "logic": {
        "combined_hr_spo2":
            "immediate vital anomaly",

        "hr_only":
            "2 consecutive windows",

        "spo2_only":
            "2 consecutive windows",

        "poor_ppg_only":
            "measurement unreliable, not vital anomaly",

        "ml_only":
            "supporting evidence only",
    },

    "clinical_warning":
        "This prototype is not clinically validated and "
        "must not be used for diagnosis or emergency medical decisions."
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
print("PRODUCTION V3 DETECTOR COMPLETED")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)