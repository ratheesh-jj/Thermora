import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# THERMORA HYBRID VITAL DETECTOR V4
#
# EPISODE-LEVEL STRESS TEST
#
# Instead of treating the first abnormal window as a "miss",
# this version evaluates whether the COMPLETE abnormal episode
# was eventually detected.
#
# Logic:
#
# HR + SpO2 abnormal
#       -> immediate anomaly
#
# HR alone
#       -> 2 consecutive windows
#
# SpO2 alone
#       -> 2 consecutive windows
#
# PPG-only disturbance
#       -> no vital anomaly
#
# Evaluation:
#   - Episode recall
#   - Episode precision
#   - Episode false-positive rate
#   - Detection delay
#   - Detection by anomaly type
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
    / "hybrid_vital_episode_test_v4.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "vital_anomaly"
    / "results"
    / "hybrid_vital_episode_summary_v4.json"
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

PPG_STD_MIN = 0.005
PPG_RANGE_MIN = 0.02


# ============================================================
# SYNTHETIC MODIFIERS
# ============================================================

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
print("THERMORA HYBRID VITAL EPISODE STRESS TEST V4")
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
# IDENTIFY ORIGINAL NORMAL WINDOWS
# ============================================================

normal_mask = (
    (df["spo2_min"] >= 92)
    &
    (df["spo2_mean"] >= 92)
    &
    (df["hr_max"] < 115)
    &
    (df["hr_min"] > 50)
)

normal_df = (
    df[normal_mask]
    .copy()
    .reset_index(drop=True)
)

print(
    f"\nOriginal normal windows: "
    f"{len(normal_df)}"
)


# ============================================================
# FROZEN WORKER BASELINES
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
# DETECTOR FUNCTION
# ============================================================

def add_detector_columns(data):

    data = data.copy()

    data["hr_baseline"] = (
        data["recording_id"]
        .map(hr_baselines)
    )

    data["spo2_baseline"] = (
        data["recording_id"]
        .map(spo2_baselines)
    )

    # --------------------------------------------------------
    # HR
    # --------------------------------------------------------

    data["hr_relative_deviation"] = (
        np.abs(
            data["hr_mean"]
            - data["hr_baseline"]
        )
        /
        data["hr_baseline"].replace(
            0,
            np.nan
        )
    ).fillna(0.0)

    data["hr_abnormal"] = (
        (data["hr_max"] >= HR_HIGH_THRESHOLD)
        |
        (data["hr_min"] <= HR_LOW_THRESHOLD)
        |
        (
            data["hr_relative_deviation"]
            >= HR_RELATIVE_CHANGE
        )
    )

    # --------------------------------------------------------
    # SpO2
    # --------------------------------------------------------

    data["spo2_baseline_drop"] = (
        data["spo2_baseline"]
        - data["spo2_mean"]
    )

    data["spo2_abnormal"] = (
        (data["spo2_min"] < SPO2_LOW_THRESHOLD)
        |
        (
            data["spo2_baseline_drop"]
            >= SPO2_DROP_THRESHOLD
        )
    )

    # --------------------------------------------------------
    # PPG quality
    # --------------------------------------------------------

    data["ppg_quality_poor"] = (
        (data["ppg_std"] < PPG_STD_MIN)
        |
        (data["ppg_range"] < PPG_RANGE_MIN)
    )

    return data


# ============================================================
# CREATE EPISODES
# ============================================================

print("\n" + "=" * 70)
print("CREATING SYNTHETIC ABNORMAL EPISODES")
print("=" * 70)

rng = np.random.default_rng(42)

episodes = []

episode_id = 0

# Use 120 original normal windows.
# Each produces one controlled episode.

for _, row in normal_df.head(120).iterrows():

    episode_id += 1

    recording_id = row["recording_id"]

    # --------------------------------------------------------
    # Two normal windows before event
    # --------------------------------------------------------

    normal1 = row.copy()
    normal2 = row.copy()

    normal1["episode_id"] = episode_id
    normal1["episode_position"] = 0
    normal1["anomaly_type"] = "normal"
    normal1["ground_truth"] = 0

    normal2["episode_id"] = episode_id
    normal2["episode_position"] = 1
    normal2["anomaly_type"] = "normal"
    normal2["ground_truth"] = 0

    episodes.append(normal1)
    episodes.append(normal2)

    # --------------------------------------------------------
    # Select anomaly
    # --------------------------------------------------------

    anomaly_type = list(
        GENERATORS.keys()
    )[
        (episode_id - 1)
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

    abnormal1["episode_id"] = episode_id
    abnormal1["episode_position"] = 2
    abnormal1["anomaly_type"] = anomaly_type
    abnormal1["ground_truth"] = 1

    abnormal2["episode_id"] = episode_id
    abnormal2["episode_position"] = 3
    abnormal2["anomaly_type"] = anomaly_type
    abnormal2["ground_truth"] = 1

    episodes.append(abnormal1)
    episodes.append(abnormal2)


episode_df = pd.DataFrame(
    episodes
).reset_index(drop=True)


# ============================================================
# RUN DETECTOR
# ============================================================

episode_df = add_detector_columns(
    episode_df
)


# ============================================================
# MODALITY-AWARE EPISODE STATE MACHINE
# ============================================================

episode_df["vital_anomaly"] = False
episode_df["trigger_type"] = "NONE"
episode_df["detection_delay_windows"] = np.nan


for episode_id, group in episode_df.groupby(
    "episode_id"
):

    group = group.sort_values(
        "episode_position"
    )

    hr_count = 0
    spo2_count = 0

    episode_detected = False
    detection_position = None

    for idx in group.index:

        hr_bad = bool(
            episode_df.loc[
                idx,
                "hr_abnormal"
            ]
        )

        spo2_bad = bool(
            episode_df.loc[
                idx,
                "spo2_abnormal"
            ]
        )

        # ----------------------------------------------------
        # Combined HR + SpO2:
        # immediate
        # ----------------------------------------------------

        if hr_bad and spo2_bad:

            episode_df.loc[
                idx,
                "vital_anomaly"
            ] = True

            episode_df.loc[
                idx,
                "trigger_type"
            ] = "COMBINED_IMMEDIATE"

            if not episode_detected:

                episode_detected = True
                detection_position = (
                    episode_df.loc[
                        idx,
                        "episode_position"
                    ]
                )

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

            if hr_count >= PERSISTENCE_WINDOWS:

                episode_df.loc[
                    idx,
                    "vital_anomaly"
                ] = True

                episode_df.loc[
                    idx,
                    "trigger_type"
                ] = "HR_PERSISTENCE"

                if not episode_detected:

                    episode_detected = True
                    detection_position = (
                        episode_df.loc[
                            idx,
                            "episode_position"
                        ]
                    )

            elif spo2_count >= PERSISTENCE_WINDOWS:

                episode_df.loc[
                    idx,
                    "vital_anomaly"
                ] = True

                episode_df.loc[
                    idx,
                    "trigger_type"
                ] = "SPO2_PERSISTENCE"

                if not episode_detected:

                    episode_detected = True
                    detection_position = (
                        episode_df.loc[
                            idx,
                            "episode_position"
                        ]
                    )

    # --------------------------------------------------------
    # Store episode-level delay
    # --------------------------------------------------------

    if episode_detected:

        # Abnormality begins at position 2.
        delay = (
            detection_position - 2
        )

        episode_df.loc[
            group.index,
            "detection_delay_windows"
        ] = delay


# ============================================================
# EPISODE-LEVEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EPISODE-LEVEL EVALUATION")
print("=" * 70)


episode_results = []


for episode_id, group in episode_df.groupby(
    "episode_id"
):

    group = group.sort_values(
        "episode_position"
    )

    anomaly_type = (
        group["anomaly_type"]
        .iloc[-1]
    )

    ground_truth = int(
        group["ground_truth"].max()
    )

    detected = bool(
        group["vital_anomaly"].any()
    )

    if detected:

        delay = float(
            group[
                "detection_delay_windows"
            ].dropna().iloc[0]
        )

    else:

        delay = np.nan

    episode_results.append({
        "episode_id": episode_id,
        "anomaly_type": anomaly_type,
        "ground_truth": ground_truth,
        "detected": int(detected),
        "detection_delay_windows": delay,
        "detection_delay_seconds":
            delay * 30
            if np.isfinite(delay)
            else np.nan,
    })


episode_results_df = pd.DataFrame(
    episode_results
)


# ============================================================
# EPISODE METRICS
# ============================================================

abnormal_episodes = (
    episode_results_df[
        episode_results_df["ground_truth"] == 1
    ]
)

normal_episodes = (
    episode_results_df[
        episode_results_df["ground_truth"] == 0
    ]
)

episode_tp = int(
    abnormal_episodes["detected"].sum()
)

episode_fn = int(
    len(abnormal_episodes)
    - episode_tp
)

episode_fp = int(
    normal_episodes["detected"].sum()
)

episode_tn = int(
    len(normal_episodes)
    - episode_fp
)

episode_precision = (
    episode_tp
    /
    (episode_tp + episode_fp)
    if episode_tp + episode_fp > 0
    else 0.0
)

episode_recall = (
    episode_tp
    /
    (episode_tp + episode_fn)
    if episode_tp + episode_fn > 0
    else 0.0
)

episode_f1 = (
    2
    * episode_precision
    * episode_recall
    /
    (episode_precision + episode_recall)
    if episode_precision + episode_recall > 0
    else 0.0
)

episode_fpr = (
    episode_fp
    /
    (episode_fp + episode_tn)
    if episode_fp + episode_tn > 0
    else 0.0
)


print(
    f"Abnormal episodes:  "
    f"{len(abnormal_episodes)}"
)

print(
    f"Detected episodes:   "
    f"{episode_tp}"
)

print(
    f"Missed episodes:     "
    f"{episode_fn}"
)

print(
    f"False alarm episodes:"
    f" {episode_fp}"
)

print(
    f"Episode precision:   "
    f"{episode_precision:.4f}"
)

print(
    f"Episode recall:      "
    f"{episode_recall:.4f}"
)

print(
    f"Episode F1:          "
    f"{episode_f1:.4f}"
)

print(
    f"Episode FPR:         "
    f"{episode_fpr:.4f}"
)


# ============================================================
# DETECTION BY TYPE
# ============================================================

print("\n" + "=" * 70)
print("EPISODE DETECTION BY ANOMALY TYPE")
print("=" * 70)

by_type = (
    abnormal_episodes
    .groupby("anomaly_type")
    .agg(
        episodes=("episode_id", "count"),
        detected=("detected", "sum"),
        mean_delay_windows=(
            "detection_delay_windows",
            "mean"
        ),
        mean_delay_seconds=(
            "detection_delay_seconds",
            "mean"
        ),
    )
    .reset_index()
)

by_type["recall"] = (
    by_type["detected"]
    /
    by_type["episodes"]
)

print(
    by_type.to_string(
        index=False
    )
)


# ============================================================
# TRIGGER BREAKDOWN
# ============================================================

print("\n" + "=" * 70)
print("TRIGGER BREAKDOWN")
print("=" * 70)

trigger_counts = (
    episode_df[
        episode_df["vital_anomaly"]
    ]["trigger_type"]
    .value_counts()
)

print(
    trigger_counts.to_string()
)


# ============================================================
# DETECTION DELAY
# ============================================================

detected_abnormal = abnormal_episodes[
    abnormal_episodes["detected"] == 1
]

if len(detected_abnormal) > 0:

    print("\n" + "=" * 70)
    print("DETECTION DELAY")
    print("=" * 70)

    print(
        f"Mean delay:   "
        f"{detected_abnormal['detection_delay_seconds'].mean():.1f} sec"
    )

    print(
        f"Median delay: "
        f"{detected_abnormal['detection_delay_seconds'].median():.1f} sec"
    )

    print(
        f"Maximum delay:"
        f" {detected_abnormal['detection_delay_seconds'].max():.1f} sec"
    )


# ============================================================
# SAVE
# ============================================================

episode_results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

summary = {
    "module":
        "THERMORA hybrid vital anomaly detector",

    "test":
        "episode_level_modality_aware_stress_test",

    "total_episodes":
        int(len(episode_results_df)),

    "abnormal_episodes":
        int(len(abnormal_episodes)),

    "detected_episodes":
        episode_tp,

    "missed_episodes":
        episode_fn,

    "false_alarm_episodes":
        episode_fp,

    "episode_precision":
        episode_precision,

    "episode_recall":
        episode_recall,

    "episode_f1":
        episode_f1,

    "episode_false_positive_rate":
        episode_fpr,

    "mean_detection_delay_seconds":
        (
            float(
                detected_abnormal[
                    "detection_delay_seconds"
                ].mean()
            )
            if len(detected_abnormal) > 0
            else None
        ),

    "median_detection_delay_seconds":
        (
            float(
                detected_abnormal[
                    "detection_delay_seconds"
                ].median()
            )
            if len(detected_abnormal) > 0
            else None
        ),

    "thresholds": {
        "spo2_low": SPO2_LOW_THRESHOLD,
        "hr_high": HR_HIGH_THRESHOLD,
        "hr_low": HR_LOW_THRESHOLD,
        "hr_relative_change": HR_RELATIVE_CHANGE,
        "spo2_baseline_drop":
            SPO2_DROP_THRESHOLD,
        "persistence_windows":
            PERSISTENCE_WINDOWS,
    },

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
        "Synthetic abnormalities are engineering "
        "stress tests and do not constitute clinical validation."
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
print("V4 EPISODE STRESS TEST COMPLETED")
print("=" * 70)

print("\nResults:")
print(OUTPUT_FILE)

print(SUMMARY_FILE)