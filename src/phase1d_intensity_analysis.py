"""
THERMORA - Phase 1D
Movement Intensity Analysis

Purpose:
    Convert WISDM activity windows into a practical
    THERMORA movement-intensity representation.

IMPORTANT:
    This version DOES NOT train the final classifier.

    It first:
        1. Loads the validated contiguous-window dataset
        2. Calculates movement-energy indicators
        3. Creates data-driven intensity bands
        4. Shows the distribution
        5. Shows WISDM activity -> intensity relationship
        6. Checks participant balance
        7. Saves the labeled dataset

Intensity classes:
    REST
    LIGHT
    MODERATE
    HIGH

The labels are MOVEMENT INTENSITY proxies.
They are NOT clinical exertion or MET measurements.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "wisdm_activity"
    / "thermora_activity_models_v2"
    / "activity_features_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "wisdm_activity"
    / "thermora_activity_models_v2"
    / "phase1d_intensity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ACTIVITY NAMES
# ============================================================

ACTIVITY_MAP = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Sitting",
    "E": "Standing",
    "F": "Typing",
    "G": "Brushing Teeth",
    "H": "Eating Soup",
    "I": "Eating Chips",
    "J": "Eating Pasta",
    "K": "Drinking",
    "L": "Eating Sandwich",
    "M": "Kicking",
    "O": "Catching",
    "P": "Dribbling Basketball",
    "Q": "Writing",
    "R": "Folding Clothes",
    "S": "Cleaning",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING THERMORA ACTIVITY DATA")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n{INPUT_FILE}\n\n"
            "Run train_activity_model.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Windows      : {len(df):,}"
    )

    print(
        f"Participants : "
        f"{df['participant'].nunique()}"
    )

    print(
        f"Activities   : "
        f"{df['activity'].nunique()}"
    )

    return df


# ============================================================
# MOVEMENT FEATURES
# ============================================================

def calculate_movement_features(df):

    print()
    print("=" * 70)
    print("CALCULATING MOVEMENT INTENSITY INDICATORS")
    print("=" * 70)

    required = [
        "acc_mag_rms",
        "acc_mag_std",
        "acc_mag_range",
        "acc_mag_q25",
        "acc_mag_q75",
        "acc_mag_mean",
        "gyro_mag_rms",
        "gyro_mag_std",
        "gyro_mag_range",
        "gyro_mag_q25",
        "gyro_mag_q75",
        "gyro_mag_mean",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Clean values
    # --------------------------------------------------------

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df[required] = df[required].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df[required] = df[required].fillna(
        df[required].median()
    )

    # --------------------------------------------------------
    # ACCELERATION MOVEMENT ENERGY
    # --------------------------------------------------------

    # RMS represents overall acceleration magnitude.
    #
    # STD captures variation around the mean.
    #
    # RANGE captures the overall movement excursion.

    df["acc_energy"] = (
        df["acc_mag_std"]
        +
        0.5 * df["acc_mag_range"]
    )

    # --------------------------------------------------------
    # GYRO MOVEMENT ENERGY
    # --------------------------------------------------------

    df["gyro_energy"] = (
        df["gyro_mag_std"]
        +
        0.5 * df["gyro_mag_range"]
    )

    # --------------------------------------------------------
    # COMBINED MOVEMENT SCORE
    # --------------------------------------------------------

    # Normalize each component using robust percentiles.
    # This prevents one feature with a large numerical range
    # from dominating the score.

    def robust_normalize(series):

        low = np.percentile(
            series,
            5
        )

        high = np.percentile(
            series,
            95
        )

        if high <= low:

            return np.zeros(
                len(series)
            )

        result = (
            series - low
        ) / (
            high - low
        )

        return np.clip(
            result,
            0,
            1
        )

    df["acc_energy_norm"] = robust_normalize(
        df["acc_energy"]
    )

    df["gyro_energy_norm"] = robust_normalize(
        df["gyro_energy"]
    )

    # Acceleration receives slightly higher weight because
    # it is the primary movement signal.
    #
    # Gyroscope still contributes significantly.

    df["movement_score"] = (
        0.60 * df["acc_energy_norm"]
        +
        0.40 * df["gyro_energy_norm"]
    )

    # --------------------------------------------------------
    # FREQUENCY INFORMATION
    # --------------------------------------------------------

    if "acc_mag_spectral_centroid" in df.columns:

        freq_score = robust_normalize(
            df["acc_mag_spectral_centroid"]
        )

        df["movement_score"] = (
            0.80 * df["movement_score"]
            +
            0.20 * freq_score
        )

    print(
        "Movement indicators calculated."
    )

    print()
    print(
        "Movement score statistics:"
    )

    print(
        df["movement_score"].describe()
    )

    return df


# ============================================================
# ACTIVITY-BASED SANITY CHECK
# ============================================================

def activity_score_summary(df):

    print()
    print("=" * 70)
    print("MOVEMENT SCORE BY WISDM ACTIVITY")
    print("=" * 70)

    summary = (
        df
        .groupby("activity")[
            "movement_score"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
        .sort_values(
            "median"
        )
    )

    summary["activity_name"] = (
        summary.index.map(
            lambda x:
            ACTIVITY_MAP.get(
                x,
                x
            )
        )
    )

    summary = summary[
        [
            "activity_name",
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    ]

    print(
        summary.to_string()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "activity_movement_score_summary.csv"
    )

    return summary


# ============================================================
# DATA-DRIVEN INTENSITY BANDS
# ============================================================

def create_intensity_labels(df):

    print()
    print("=" * 70)
    print("CREATING MOVEMENT INTENSITY BANDS")
    print("=" * 70)

    score = df[
        "movement_score"
    ]

    # --------------------------------------------------------
    # Quantile-based initial bands
    # --------------------------------------------------------
    #
    # We deliberately use quantiles for this first experiment.
    #
    # This gives us approximately balanced classes and allows
    # us to inspect whether movement patterns naturally separate.
    #
    # THESE ARE NOT CLINICAL THRESHOLDS.
    #

    q25 = score.quantile(
        0.25
    )

    q50 = score.quantile(
        0.50
    )

    q75 = score.quantile(
        0.75
    )

    print(
        f"25th percentile : {q25:.4f}"
    )

    print(
        f"50th percentile : {q50:.4f}"
    )

    print(
        f"75th percentile : {q75:.4f}"
    )

    def assign_intensity(value):

        if value <= q25:

            return "REST"

        elif value <= q50:

            return "LIGHT"

        elif value <= q75:

            return "MODERATE"

        else:

            return "HIGH"

    df["intensity"] = (
        df["movement_score"]
        .apply(assign_intensity)
    )

    # Ordered categorical representation
    intensity_order = [
        "REST",
        "LIGHT",
        "MODERATE",
        "HIGH"
    ]

    df["intensity"] = pd.Categorical(
        df["intensity"],
        categories=intensity_order,
        ordered=True
    )

    return df, {
        "REST_upper": float(q25),
        "LIGHT_upper": float(q50),
        "MODERATE_upper": float(q75),
        "HIGH_upper": 1.0
    }


# ============================================================
# INTENSITY DISTRIBUTION
# ============================================================

def print_intensity_distribution(df):

    print()
    print("=" * 70)
    print("INTENSITY DISTRIBUTION")
    print("=" * 70)

    counts = (
        df["intensity"]
        .value_counts()
        .reindex(
            [
                "REST",
                "LIGHT",
                "MODERATE",
                "HIGH"
            ]
        )
        .fillna(0)
        .astype(int)
    )

    percentages = (
        counts
        / len(df)
        * 100
    )

    distribution = pd.DataFrame(
        {
            "windows": counts,
            "percentage": percentages
        }
    )

    print(
        distribution.to_string(
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    distribution.to_csv(
        OUTPUT_DIR
        / "intensity_distribution.csv"
    )

    return distribution


# ============================================================
# ACTIVITY -> INTENSITY CROSS TAB
# ============================================================

def activity_intensity_crosstab(df):

    print()
    print("=" * 70)
    print("WISDM ACTIVITY -> MOVEMENT INTENSITY")
    print("=" * 70)

    table = pd.crosstab(
        df["activity"],
        df["intensity"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=[
            "REST",
            "LIGHT",
            "MODERATE",
            "HIGH"
        ],
        fill_value=0
    )

    table["activity_name"] = (
        table.index.map(
            lambda x:
            ACTIVITY_MAP.get(
                x,
                x
            )
        )
    )

    table = table[
        [
            "activity_name",
            "REST",
            "LIGHT",
            "MODERATE",
            "HIGH"
        ]
    ]

    print(
        table.to_string(
            float_format=lambda x:
            f"{x:.1f}"
        )
    )

    table.to_csv(
        OUTPUT_DIR
        / "activity_to_intensity.csv"
    )

    return table


# ============================================================
# PARTICIPANT BALANCE
# ============================================================

def participant_balance(df):

    print()
    print("=" * 70)
    print("PARTICIPANT INTENSITY BALANCE")
    print("=" * 70)

    table = pd.crosstab(
        df["participant"],
        df["intensity"],
        normalize="index"
    ) * 100

    table = table.reindex(
        columns=[
            "REST",
            "LIGHT",
            "MODERATE",
            "HIGH"
        ],
        fill_value=0
    )

    print(
        table.round(1).to_string()
    )

    table.to_csv(
        OUTPUT_DIR
        / "participant_intensity_distribution.csv"
    )

    # Calculate participant-level statistics
    participant_summary = pd.DataFrame(
        {
            "participant": table.index,
            "REST_pct": table["REST"].values,
            "LIGHT_pct": table["LIGHT"].values,
            "MODERATE_pct": table["MODERATE"].values,
            "HIGH_pct": table["HIGH"].values,
        }
    )

    participant_summary.to_csv(
        OUTPUT_DIR
        / "participant_intensity_balance.csv",
        index=False
    )

    return participant_summary


# ============================================================
# PLOTS
# ============================================================

def create_plots(
    df,
    activity_summary
):

    print()
    print("=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Movement score histogram
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        df["movement_score"],
        bins=60
    )

    plt.xlabel(
        "Movement Score"
    )

    plt.ylabel(
        "Number of Windows"
    )

    plt.title(
        "THERMORA Movement Score Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "movement_score_distribution.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Intensity distribution
    # --------------------------------------------------------

    order = [
        "REST",
        "LIGHT",
        "MODERATE",
        "HIGH"
    ]

    counts = (
        df["intensity"]
        .value_counts()
        .reindex(order)
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        order,
        counts.values
    )

    plt.xlabel(
        "Movement Intensity"
    )

    plt.ylabel(
        "Number of Windows"
    )

    plt.title(
        "THERMORA Movement Intensity Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "intensity_distribution.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Activity median score
    # --------------------------------------------------------

    plot_data = activity_summary.sort_values(
        "median"
    )

    labels = [
        str(x)
        for x in plot_data[
            "activity_name"
        ]
    ]

    values = plot_data[
        "median"
    ].values

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        labels,
        values
    )

    plt.xlabel(
        "Median Movement Score"
    )

    plt.ylabel(
        "Activity"
    )

    plt.title(
        "Median Movement Score by WISDM Activity"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "activity_movement_scores.png",
        dpi=200
    )

    plt.close()

    print(
        "Plots saved."
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    df,
    thresholds
):

    print()
    print("=" * 70)
    print("SAVING INTENSITY DATASET")
    print("=" * 70)

    output_columns = [
        "participant",
        "activity",
        "movement_score",
        "acc_energy",
        "gyro_energy",
        "acc_energy_norm",
        "gyro_energy_norm",
        "intensity"
    ]

    output_columns = [
        c for c in output_columns
        if c in df.columns
    ]

    output_df = df[
        output_columns
    ].copy()

    output_file = (
        OUTPUT_DIR
        / "thermora_intensity_dataset.csv"
    )

    output_df.to_csv(
        output_file,
        index=False
    )

    with open(
        OUTPUT_DIR
        / "intensity_thresholds.json",
        "w"
    ) as f:

        json.dump(
            thresholds,
            f,
            indent=4
        )

    print(
        f"Saved:\n{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THERMORA - PHASE 1D")
    print("MOVEMENT INTENSITY ANALYSIS")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Calculate movement indicators
    # --------------------------------------------------------

    df = calculate_movement_features(
        df
    )

    # --------------------------------------------------------
    # Activity score analysis
    # --------------------------------------------------------

    activity_summary = (
        activity_score_summary(
            df
        )
    )

    # --------------------------------------------------------
    # Create intensity labels
    # --------------------------------------------------------

    df, thresholds = (
        create_intensity_labels(
            df
        )
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print_intensity_distribution(
        df
    )

    # --------------------------------------------------------
    # Activity relationship
    # --------------------------------------------------------

    activity_intensity_crosstab(
        df
    )

    # --------------------------------------------------------
    # Participant balance
    # --------------------------------------------------------

    participant_balance(
        df
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    create_plots(
        df,
        activity_summary
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        df,
        thresholds
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 1D ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "Intensity thresholds:"
    )

    print(
        json.dumps(
            thresholds,
            indent=4
        )
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These labels represent movement intensity "
        "derived from wrist motion."
    )

    print(
        "They are NOT clinical exertion measurements."
    )

    print()
    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()