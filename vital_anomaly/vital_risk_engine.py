import os
import numpy as np
import pandas as pd


# ============================================================
# THERMORA VITAL RISK ENGINE v2
# ============================================================

BASE_DIR = r"C:\Users\OMEN\Desktop\Projects\SIH26\vital_anomaly"

DATA_PATH = os.path.join(
    BASE_DIR,
    "wearable_stress",
    "outputs",
    "wearable_all_conditions_vital_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "wearable_stress",
    "outputs",
    "vital_model_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

BASELINE_WINDOWS = 5
PERSISTENCE_WINDOWS = 2


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("THERMORA VITAL RISK ENGINE v2")
print("=" * 70)

print("\nLoading:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# REQUIRED FEATURES
# ============================================================

required = [
    "subject",
    "window_start_sec",
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "temp_mean",
    "acc_mag_std"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns:\n" +
        "\n".join(missing)
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["subject", "window_start_sec"]
).reset_index(drop=True)


# ============================================================
# ROBUST Z SCORE
# ============================================================

def robust_zscore(value, median, mad):

    if pd.isna(value):
        return 0.0

    # When baseline variation is zero, use a small scale
    # instead of forcing every deviation to zero.
    if pd.isna(mad) or mad < 1e-6:
        mad = 1.0

    return 0.6745 * (value - median) / mad


# ============================================================
# PERSONAL BASELINES
# ============================================================

baseline_rows = []

for subject, subject_df in df.groupby("subject"):

    subject_df = subject_df.sort_values(
        "window_start_sec"
    )

    baseline = subject_df.head(
        BASELINE_WINDOWS
    )

    hr_med = baseline["mean_hr_bpm"].median()
    rmssd_med = baseline["rmssd_ms"].median()
    sdnn_med = baseline["sdnn_ms"].median()
    temp_med = baseline["temp_mean"].median()

    hr_mad = (
        baseline["mean_hr_bpm"] - hr_med
    ).abs().median()

    rmssd_mad = (
        baseline["rmssd_ms"] - rmssd_med
    ).abs().median()

    sdnn_mad = (
        baseline["sdnn_ms"] - sdnn_med
    ).abs().median()

    temp_mad = (
        baseline["temp_mean"] - temp_med
    ).abs().median()

    baseline_rows.append({

        "subject": subject,

        "baseline_hr": hr_med,
        "baseline_rmssd": rmssd_med,
        "baseline_sdnn": sdnn_med,
        "baseline_temp": temp_med,

        "baseline_hr_mad": hr_mad,
        "baseline_rmssd_mad": rmssd_mad,
        "baseline_sdnn_mad": sdnn_mad,
        "baseline_temp_mad": temp_mad
    })


baseline_df = pd.DataFrame(
    baseline_rows
)

df = df.merge(
    baseline_df,
    on="subject",
    how="left"
)


# ============================================================
# DEVIATIONS
# ============================================================

df["hr_z"] = df.apply(
    lambda r: robust_zscore(
        r["mean_hr_bpm"],
        r["baseline_hr"],
        r["baseline_hr_mad"]
    ),
    axis=1
)

df["rmssd_z"] = df.apply(
    lambda r: robust_zscore(
        r["rmssd_ms"],
        r["baseline_rmssd"],
        r["baseline_rmssd_mad"]
    ),
    axis=1
)

df["sdnn_z"] = df.apply(
    lambda r: robust_zscore(
        r["sdnn_ms"],
        r["baseline_sdnn"],
        r["baseline_sdnn_mad"]
    ),
    axis=1
)

df["temp_z"] = df.apply(
    lambda r: robust_zscore(
        r["temp_mean"],
        r["baseline_temp"],
        r["baseline_temp_mad"]
    ),
    axis=1
)


# ============================================================
# HR RISK COMPONENT: 0-100
# ============================================================

# Larger deviation from personal baseline = higher score.

df["hr_score"] = np.clip(
    np.abs(df["hr_z"]) * 25,
    0,
    100
)


# ============================================================
# HRV RISK COMPONENT: 0-100
# ============================================================

# Focus on HRV reduction relative to baseline.
#
# HRV is a supporting signal, not a diagnosis.

rmssd_component = np.where(
    df["rmssd_z"] < 0,
    -df["rmssd_z"] * 40,
    0
)

sdnn_component = np.where(
    df["sdnn_z"] < 0,
    -df["sdnn_z"] * 30,
    0
)

df["hrv_score"] = np.clip(
    rmssd_component + sdnn_component,
    0,
    100
)


# ============================================================
# TEMPERATURE COMPONENT: 0-100
# ============================================================

# ============================================================
# TEMPERATURE COMPONENT
# ============================================================
#
# Skin temperature is NOT core temperature.
# It is also strongly affected by environment, contact,
# clothing and sensor placement.
#
# Therefore temperature alone should be a weak supporting
# signal in the vital-risk engine.
#
# We only treat LARGE sustained deviations as meaningful.

temp_abs_change = (
    df["temp_mean"] - df["baseline_temp"]
).abs()

# Ignore small skin-temperature changes.
df["temperature_score"] = np.where(
    temp_abs_change < 1.0,
    0.0,
    np.clip(
        (temp_abs_change - 1.0) * 20,
        0,
        50
    )
)


# ============================================================
# MOVEMENT CONTEXT
# ============================================================

movement_median = df["acc_mag_std"].median()

df["movement_state"] = np.where(
    df["acc_mag_std"] > movement_median,
    "ACTIVE",
    "LOW_MOVEMENT"
)


# ============================================================
# HR CONTEXT
# ============================================================

# High HR while relatively inactive is more suspicious.
#
# High HR during activity is treated more cautiously.

df["hr_context_score"] = 0.0

df.loc[
    (df["mean_hr_bpm"] >= 120) &
    (df["movement_state"] == "LOW_MOVEMENT"),
    "hr_context_score"
] = 100

df.loc[
    (df["mean_hr_bpm"] >= 110) &
    (df["mean_hr_bpm"] < 120) &
    (df["movement_state"] == "LOW_MOVEMENT"),
    "hr_context_score"
] = 60

df.loc[
    (df["mean_hr_bpm"] <= 50) &
    (df["movement_state"] == "LOW_MOVEMENT"),
    "hr_context_score"
] = 70


# ============================================================
# SpO2
# ============================================================

# The current dataset does NOT contain SpO2.
#
# Therefore SpO2 contributes 0 for this experiment.
#
# We will activate this component in Sprint 2 / task 4
# using MAX30102 data.

if "spo2_mean" in df.columns:

    df["spo2_score"] = 0.0

    df.loc[
        df["spo2_mean"] < 94,
        "spo2_score"
    ] = 40

    df.loc[
        df["spo2_mean"] < 92,
        "spo2_score"
    ] = 70

    df.loc[
        df["spo2_mean"] < 90,
        "spo2_score"
    ] = 100

else:

    df["spo2_score"] = 0.0


# ============================================================
# COMPOSITE SCORE
# ============================================================

# Each component is normalized 0-100.
#
# Current weighting:
#
# HR          30%
# HRV         35%
# Temperature 15%
# SpO2        15%
# HR context   5%
#
# Temperature and SpO2 are deliberately not dominant here.
# Thermal risk will be handled separately in Sprint 3.

df["risk_score"] = (

    0.35 * df["hr_score"] +

    0.35 * df["hrv_score"] +

    0.05 * df["temperature_score"] +

    0.20 * df["spo2_score"] +

    0.05 * df["hr_context_score"]
)

df["risk_score"] = np.clip(
    df["risk_score"],
    0,
    100
)


# ============================================================
# RAW RISK STATE
# ============================================================

def classify_risk(score):

    if score >= 70:
        return "CRITICAL"

    elif score >= 40:
        return "WARNING"

    else:
        return "NORMAL"


df["raw_risk_state"] = (
    df["risk_score"]
    .apply(classify_risk)
)


# ============================================================
# PERSISTENCE
# ============================================================

# ============================================================
# PERSISTENCE
# ============================================================

def apply_persistence(subject_df):

    subject_df = subject_df.sort_values(
        "window_start_sec"
    ).copy()

    final_states = []

    warning_count = 0
    critical_count = 0

    for state in subject_df["raw_risk_state"]:

        if state == "CRITICAL":

            critical_count += 1
            warning_count += 1

        elif state == "WARNING":

            warning_count += 1
            critical_count = 0

        else:

            warning_count = 0
            critical_count = 0

        if critical_count >= PERSISTENCE_WINDOWS:

            final_state = "CRITICAL"

        elif warning_count >= PERSISTENCE_WINDOWS:

            final_state = "WARNING"

        else:

            final_state = "NORMAL"

        final_states.append(final_state)

    subject_df["risk_state"] = final_states

    return subject_df


# Avoid pandas groupby.apply removing the grouping column.
# Explicit concatenation keeps 'subject' guaranteed to exist.

subject_results = []

for subject, subject_df in df.groupby("subject"):

    processed = apply_persistence(
        subject_df
    )

    subject_results.append(
        processed
    )

df = pd.concat(
    subject_results,
    ignore_index=True
)

df = df.sort_values(
    ["subject", "window_start_sec"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "vital_risk_engine_output_v2.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK ENGINE RESULTS")
print("=" * 70)

print("\nRaw risk states:")

print(
    df["raw_risk_state"].value_counts()
)


print("\nAfter persistence:")

print(
    df["risk_state"].value_counts()
)


print("\nRisk score statistics:")

print(
    df["risk_score"].describe()
)


print("\nRisk score by condition:")

print(
    df.groupby("condition")["risk_score"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)


print("\nRisk state by condition (%):")

condition_table = (
    pd.crosstab(
        df["condition"],
        df["risk_state"],
        normalize="index"
    ) * 100
)

print(
    condition_table.round(2)
)


# ============================================================
# COMPONENT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("COMPONENT STATISTICS")
print("=" * 70)

component_cols = [
    "hr_score",
    "hrv_score",
    "temperature_score",
    "spo2_score",
    "hr_context_score",
    "risk_score"
]

print(
    df[component_cols]
    .describe()
    .round(2)
)


# ============================================================
# SUBJECT SUMMARY
# ============================================================

subject_summary = (
    df
    .groupby("subject")
    .agg(
        windows=("risk_score", "count"),

        mean_risk=("risk_score", "mean"),

        max_risk=("risk_score", "max"),

        warning_windows=(
            "risk_state",
            lambda x: (x == "WARNING").sum()
        ),

        critical_windows=(
            "risk_state",
            lambda x: (x == "CRITICAL").sum()
        )
    )
    .reset_index()
)

subject_summary_path = os.path.join(
    OUTPUT_DIR,
    "subject_risk_summary_v2.csv"
)

subject_summary.to_csv(
    subject_summary_path,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nMain output:")
print(output_path)

print("\nSubject summary:")
print(subject_summary_path)

print("\nNOTE:")
print(
    "SpO2 is not present in this dataset and therefore "
    "is currently disabled."
)