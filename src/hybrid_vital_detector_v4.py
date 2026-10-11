import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# THERMORA HYBRID VITAL DETECTOR V4
# EPISODE-BASED PRODUCTION PROTOTYPE
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "vital_hybrid_episode_outputs_v4.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "vital_hybrid_episode_summary_v4.json"
)


# ============================================================
# THRESHOLDS
# ============================================================

SPO2_LOW_THRESHOLD = 90.0

HR_HIGH_THRESHOLD = 120.0
HR_LOW_THRESHOLD = 50.0

HR_RELATIVE_CHANGE = 0.25
SPO2_DROP_THRESHOLD = 3.0

PERSISTENCE_WINDOWS = 2

MAX_WINDOW_GAP_SECONDS = 60.0

ML_SUPPORT_THRESHOLD = 0.875

PPG_STD_MIN = 0.005
PPG_RANGE_MIN = 0.02


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("THERMORA HYBRID VITAL DETECTOR V4")
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

df = pd.read_csv(DATA_FILE)

df = df.sort_values(
    [
        "recording_id",
        "window_start_s"
    ]
).reset_index(drop=True)

print(
    f"\nInput windows: {len(df)}"
)

print(
    f"Recordings:    {df['recording_id'].nunique()}"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print(
    f"ML model: {type(model).__name__}"
)


# ============================================================
# ML FEATURES
# ============================================================

# ============================================================
# LOAD EXACT TRAINING FEATURE SCHEMA
# ============================================================

FEATURE_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "models"
    / "thermora_vital_anomaly_features.json"
)

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Vital model feature schema not found:\n"
        f"{FEATURE_FILE}"
    )

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_info = json.load(f)


# The saved schema may be either:
#   ["feature1", "feature2", ...]
# or:
#   {"features": [...]}
# or:
#   {"feature_names": [...]}

if isinstance(feature_info, list):

    feature_names = feature_info

elif isinstance(feature_info, dict):

    feature_names = None

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

    if feature_names is None:

        raise ValueError(
            "Could not find feature list in:\n"
            f"{FEATURE_FILE}"
        )

else:

    raise ValueError(
        "Unexpected feature schema format."
    )


# Check that every trained feature exists.
missing_features = [
    col
    for col in feature_names
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "The current dataset is missing "
        "trained model features:\n"
        +
        "\n".join(
            missing_features
        )
    )


# IMPORTANT:
# Use ONLY the exact features used during model training.
# Do not automatically pass every dataframe column.

X_ml = df[
    feature_names
].copy()

X_ml = X_ml.apply(
    pd.to_numeric,
    errors="coerce"
)

print(
    f"ML features expected: "
    f"{len(feature_names)}"
)

print(
    f"ML features provided: "
    f"{X_ml.shape[1]}"
)# ============================================================
# LOAD EXACT TRAINING FEATURE SCHEMA
# ============================================================

FEATURE_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "models"
    / "thermora_vital_anomaly_features.json"
)

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Vital model feature schema not found:\n"
        f"{FEATURE_FILE}"
    )

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_info = json.load(f)


# The saved schema may be either:
#   ["feature1", "feature2", ...]
# or:
#   {"features": [...]}
# or:
#   {"feature_names": [...]}

if isinstance(feature_info, list):

    feature_names = feature_info

elif isinstance(feature_info, dict):

    feature_names = None

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

    if feature_names is None:

        raise ValueError(
            "Could not find feature list in:\n"
            f"{FEATURE_FILE}"
        )

else:

    raise ValueError(
        "Unexpected feature schema format."
    )


# Check that every trained feature exists.
missing_features = [
    col
    for col in feature_names
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "The current dataset is missing "
        "trained model features:\n"
        +
        "\n".join(
            missing_features
        )
    )


# IMPORTANT:
# Use ONLY the exact features used during model training.
# Do not automatically pass every dataframe column.

X_ml = df[
    feature_names
].copy()

X_ml = X_ml.apply(
    pd.to_numeric,
    errors="coerce"
)

print(
    f"ML features expected: "
    f"{len(feature_names)}"
)

print(
    f"ML features provided: "
    f"{X_ml.shape[1]}"
)# ============================================================
# LOAD EXACT TRAINING FEATURE SCHEMA
# ============================================================

FEATURE_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "models"
    / "thermora_vital_anomaly_features.json"
)

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Vital model feature schema not found:\n"
        f"{FEATURE_FILE}"
    )

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_info = json.load(f)


# The saved schema may be either:
#   ["feature1", "feature2", ...]
# or:
#   {"features": [...]}
# or:
#   {"feature_names": [...]}

if isinstance(feature_info, list):

    feature_names = feature_info

elif isinstance(feature_info, dict):

    feature_names = None

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

    if feature_names is None:

        raise ValueError(
            "Could not find feature list in:\n"
            f"{FEATURE_FILE}"
        )

else:

    raise ValueError(
        "Unexpected feature schema format."
    )


# Check that every trained feature exists.
missing_features = [
    col
    for col in feature_names
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "The current dataset is missing "
        "trained model features:\n"
        +
        "\n".join(
            missing_features
        )
    )


# IMPORTANT:
# Use ONLY the exact features used during model training.
# Do not automatically pass every dataframe column.

X_ml = df[
    feature_names
].copy()

X_ml = X_ml.apply(
    pd.to_numeric,
    errors="coerce"
)

print(
    f"ML features expected: "
    f"{len(feature_names)}"
)

print(
    f"ML features provided: "
    f"{X_ml.shape[1]}"
)


# ============================================================
# ML ANOMALY SUPPORT
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING ML SUPPORT SCORE")
print("=" * 70)

try:

    raw_decision = model.decision_function(
        X_ml
    )

    # Isolation Forest:
    # lower decision_function = more anomalous
    ml_raw_score = -np.asarray(
        raw_decision,
        dtype=float
    )

    low = np.nanpercentile(
        ml_raw_score,
        5
    )

    high = np.nanpercentile(
        ml_raw_score,
        95
    )

    if high > low:

        ml_score = (
            ml_raw_score - low
        ) / (
            high - low
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
        f"ML scoring failed: {exc}"
    )

    ml_score = np.zeros(
        len(df),
        dtype=float
    )


df["ml_anomaly_score"] = ml_score

df["ml_support"] = (
    df["ml_anomaly_score"]
    >= ML_SUPPORT_THRESHOLD
)


# ============================================================
# PERSONAL BASELINES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PERSONAL BASELINES")
print("=" * 70)

hr_baselines = {}
spo2_baselines = {}

for recording_id, group in df.groupby(
    "recording_id"
):

    group = group.sort_values(
        "window_start_s"
    )

    candidates = group[
        (group["spo2_mean"] >= 92)
        &
        (group["hr_min"] > 50)
        &
        (group["hr_max"] < 115)
    ]

    if len(candidates) == 0:
        candidates = group

    candidates = candidates.head(5)

    hr_baseline = candidates[
        "hr_mean"
    ].median()

    spo2_baseline = candidates[
        "spo2_mean"
    ].median()

    if np.isfinite(hr_baseline):
        hr_baselines[
            recording_id
        ] = float(hr_baseline)

    if np.isfinite(spo2_baseline):
        spo2_baselines[
            recording_id
        ] = float(spo2_baseline)


df["hr_baseline"] = (
    df["recording_id"]
    .map(hr_baselines)
)

df["spo2_baseline"] = (
    df["recording_id"]
    .map(spo2_baselines)
)

print(
    f"HR baselines:   {len(hr_baselines)}"
)

print(
    f"SpO2 baselines: {len(spo2_baselines)}"
)


# ============================================================
# HR DETECTION
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
# SPO2 DETECTION
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
# OUTPUT STATE
# ============================================================

df["vital_anomaly"] = False

df["vital_status"] = "NORMAL"

df["trigger_reason"] = "NONE"

df["hr_persistence_count"] = 0

df["spo2_persistence_count"] = 0

df["episode_id"] = np.nan

df["episode_active"] = False


# ============================================================
# TEMPORAL DETECTION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING EPISODE-BASED DETECTOR")
print("=" * 70)

episode_counter = 0

episode_records = []


for recording_id, group in df.groupby(
    "recording_id"
):

    group = group.sort_values(
        "window_start_s"
    )

    hr_count = 0
    spo2_count = 0

    active_episode_id = None

    previous_time = None

    for idx in group.index:

        current_time = float(
            df.loc[
                idx,
                "window_start_s"
            ]
        )

        # ----------------------------------------------------
        # Reset on temporal gap
        # ----------------------------------------------------

        if previous_time is not None:

            gap = (
                current_time
                - previous_time
            )

            if (
                gap < 0
                or gap > MAX_WINDOW_GAP_SECONDS
            ):

                hr_count = 0
                spo2_count = 0
                active_episode_id = None

        previous_time = current_time

        # ----------------------------------------------------
        # Current physiological evidence
        # ----------------------------------------------------

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
        # Persistence counters
        # ----------------------------------------------------

        if hr_bad:
            hr_count += 1
        else:
            hr_count = 0

        if spo2_bad:
            spo2_count += 1
        else:
            spo2_count = 0

        df.loc[
            idx,
            "hr_persistence_count"
        ] = hr_count

        df.loc[
            idx,
            "spo2_persistence_count"
        ] = spo2_count

        # ----------------------------------------------------
        # Confirmation
        # ----------------------------------------------------

        confirmed = False
        reason = "NONE"

        # Combined HR + SpO2 = immediate
        if hr_bad and spo2_bad:

            confirmed = True
            reason = "COMBINED_HR_SPO2"

        # HR alone = persistence
        elif hr_count >= PERSISTENCE_WINDOWS:

            confirmed = True
            reason = "HR_PERSISTENCE"

        # SpO2 alone = persistence
        elif spo2_count >= PERSISTENCE_WINDOWS:

            confirmed = True
            reason = "SPO2_PERSISTENCE"

        # ----------------------------------------------------
        # CONFIRMED ANOMALY
        # ----------------------------------------------------

        if confirmed:

            if active_episode_id is None:

                episode_counter += 1

                active_episode_id = (
                    episode_counter
                )

                episode_records.append({
                    "episode_id":
                        active_episode_id,

                    "recording_id":
                        recording_id,

                    "start_time_s":
                        current_time,

                    "end_time_s":
                        current_time,

                    "trigger_reason":
                        reason,
                })

            else:

                episode_records[
                    -1
                ]["end_time_s"] = (
                    current_time
                )

                # Upgrade trigger if combined evidence
                # appears during the episode.
                if reason == "COMBINED_HR_SPO2":

                    episode_records[
                        -1
                    ]["trigger_reason"] = (
                        "COMBINED_HR_SPO2"
                    )

            # Window output
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
            ] = reason

            df.loc[
                idx,
                "episode_id"
            ] = active_episode_id

            df.loc[
                idx,
                "episode_active"
            ] = True

        # ----------------------------------------------------
        # NOT CONFIRMED
        # ----------------------------------------------------

        else:

            # Close an existing episode once both
            # physiological signals return to normal.
            if (
                active_episode_id is not None
                and not hr_bad
                and not spo2_bad
            ):

                active_episode_id = None

            if ppg_bad:

                df.loc[
                    idx,
                    "vital_status"
                ] = (
                    "MEASUREMENT_UNRELIABLE"
                )

                df.loc[
                    idx,
                    "trigger_reason"
                ] = (
                    "POOR_PPG_QUALITY"
                )

            elif hr_bad:

                df.loc[
                    idx,
                    "vital_status"
                ] = "HR_DEVIATION"

                df.loc[
                    idx,
                    "trigger_reason"
                ] = (
                    "HR_PENDING_PERSISTENCE"
                )

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

            elif bool(
                df.loc[
                    idx,
                    "ml_support"
                ]
            ):

                df.loc[
                    idx,
                    "vital_status"
                ] = (
                    "ML_ANOMALY_SUPPORT"
                )

                df.loc[
                    idx,
                    "trigger_reason"
                ] = (
                    "ML_SUPPORT_ONLY"
                )


# ============================================================
# BUILD EPISODE TABLE
# ============================================================

if episode_records:

    episodes_df = pd.DataFrame(
        episode_records
    )

    episodes_df["duration_seconds"] = (
        episodes_df["end_time_s"]
        - episodes_df["start_time_s"]
        + 30.0
    )

else:

    episodes_df = pd.DataFrame(
        columns=[
            "episode_id",
            "recording_id",
            "start_time_s",
            "end_time_s",
            "trigger_reason",
            "duration_seconds",
            "episode_type",
        ]
    )


# ============================================================
# DETERMINE EPISODE TYPE
# ============================================================

if len(episodes_df) > 0:

    episode_types = []

    for _, episode in episodes_df.iterrows():

        recording_id = episode[
            "recording_id"
        ]

        episode_id = episode[
            "episode_id"
        ]

        rows = df[
            (df["recording_id"] == recording_id)
            &
            (df["episode_id"] == episode_id)
        ]

        has_hr = bool(
            rows["hr_abnormal"].any()
        )

        has_spo2 = bool(
            rows["spo2_abnormal"].any()
        )

        if has_hr and has_spo2:

            episode_type = (
                "COMBINED_HR_SPO2"
            )

        elif has_hr:

            episode_type = "HR"

        elif has_spo2:

            episode_type = "SPO2"

        else:

            episode_type = "UNKNOWN"

        episode_types.append(
            episode_type
        )

    episodes_df["episode_type"] = (
        episode_types
    )


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
    f"Confirmed anomaly windows: "
    f"{int(df['vital_anomaly'].sum())}"
)

print(
    f"Vital anomaly episodes: "
    f"{len(episodes_df)}"
)


# ============================================================
# EPISODE LIST
# ============================================================

print("\n" + "=" * 70)
print("VITAL ANOMALY EPISODES")
print("=" * 70)

if len(episodes_df) > 0:

    display_columns = [
        "episode_id",
        "recording_id",
        "episode_type",
        "start_time_s",
        "end_time_s",
        "duration_seconds",
        "trigger_reason",
    ]

    print(
        episodes_df[
            display_columns
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No confirmed vital anomaly episodes."
    )


# ============================================================
# EPISODE TYPE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("EPISODE TYPE DISTRIBUTION")
print("=" * 70)

if len(episodes_df) > 0:

    type_counts = (
        episodes_df[
            "episode_type"
        ]
        .value_counts()
    )

    print(
        type_counts.to_string()
    )

else:

    type_counts = pd.Series(
        dtype=int
    )

    print(
        "No episodes."
    )


# ============================================================
# EPISODE DURATION
# ============================================================

if len(episodes_df) > 0:

    print("\n" + "=" * 70)
    print("EPISODE DURATION")
    print("=" * 70)

    print(
        f"Mean duration: "
        f"{episodes_df['duration_seconds'].mean():.1f} sec"
    )

    print(
        f"Median duration: "
        f"{episodes_df['duration_seconds'].median():.1f} sec"
    )

    print(
        f"Maximum duration: "
        f"{episodes_df['duration_seconds'].max():.1f} sec"
    )


# ============================================================
# TRIGGER DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRIGGER DISTRIBUTION")
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
        "No confirmed triggers."
    )


# ============================================================
# SAVE WINDOW OUTPUT
# ============================================================

window_columns = [
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

    "episode_id",
    "episode_active",
]

window_columns = [
    col
    for col in window_columns
    if col in df.columns
]

df[
    window_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {
    "module":
        "THERMORA hybrid vital anomaly detector",

    "version":
        "V4 episode-based prototype",

    "input_windows":
        int(len(df)),

    "recordings":
        int(df["recording_id"].nunique()),

    "confirmed_anomaly_windows":
        int(df["vital_anomaly"].sum()),

    "vital_anomaly_episodes":
        int(len(episodes_df)),

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

    "episode_type_distribution":
        {
            str(k): int(v)
            for k, v in type_counts.items()
        },

    "mean_episode_duration_seconds":
        (
            float(
                episodes_df[
                    "duration_seconds"
                ].mean()
            )
            if len(episodes_df) > 0
            else None
        ),

    "median_episode_duration_seconds":
        (
            float(
                episodes_df[
                    "duration_seconds"
                ].median()
            )
            if len(episodes_df) > 0
            else None
        ),

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
            "immediate confirmation",

        "hr_only":
            "2 consecutive windows",

        "spo2_only":
            "2 consecutive windows",

        "poor_ppg":
            "measurement unreliable",

        "ml_only":
            "supporting evidence only",

        "episode_grouping":
            "continuous confirmed abnormal windows grouped into one episode",
    },

    "clinical_warning":
        "Prototype only. Not clinically validated and not intended for diagnosis or emergency medical decisions.",
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

print("\n" + "=" * 70)
print("V4 EPISODE-BASED DETECTOR COMPLETED")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)