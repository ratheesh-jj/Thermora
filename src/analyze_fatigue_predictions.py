"""
THERMORA FATIGUE PREDICTION ANALYSIS
=====================================

Analyzes the batch fatigue predictions and checks whether
fatigue estimates behave sensibly across:

    Initial Rest
    Rep1
    Rep2
    Rep3
    Rep4
    Rep5
    Final Rest

Also compares predictions with the available fatigue
annotations in the original MxD files.

This is a validation/analysis script only.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26"
)

PREDICTIONS_FILE = (
    BASE_DIR
    / "results"
    / "fatigue_batch_predictions.csv"
)

MXD_ROOT = (
    BASE_DIR
    / "fatigue_prediction"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "fatigue_prediction_analysis.csv"
)


# ============================================================
# IDENTIFY SEGMENT
# ============================================================

def identify_segment(filename):

    name = filename.lower()

    if "initial_rest" in name:
        return "Initial Rest"

    if "final_rest" in name:
        return "Final Rest"

    match = re.search(
        r"rep([1-5])",
        name
    )

    if match:

        return (
            "Rep "
            + match.group(1)
        )

    return "Unknown"


# ============================================================
# FIND SOURCE FILE
# ============================================================

def find_source_file(
    filename
):

    matches = list(
        MXD_ROOT.rglob(
            filename
        )
    )

    if matches:

        return matches[0]

    return None


# ============================================================
# LOAD SOURCE LABELS
# ============================================================

def get_source_fatigue_labels(
    file_path
):

    if file_path is None:

        return (
            np.nan,
            np.nan
        )

    try:

        df = pd.read_csv(
            file_path
        )

    except Exception:

        return (
            np.nan,
            np.nan
        )

    initial = np.nan
    final = np.nan

    # --------------------------------------------------------
    # Find fatigue columns
    # --------------------------------------------------------

    initial_columns = [
        c for c in df.columns
        if "Physical Fatigue - Initial"
        in str(c)
    ]

    final_columns = [
        c for c in df.columns
        if "Physical Fatigue - Final"
        in str(c)
    ]

    if initial_columns:

        values = pd.to_numeric(
            df[
                initial_columns[0]
            ],
            errors="coerce"
        )

        values = values[
            values >= 0
        ]

        if len(values):

            initial = float(
                values.iloc[0]
            )

    if final_columns:

        values = pd.to_numeric(
            df[
                final_columns[0]
            ],
            errors="coerce"
        )

        values = values[
            values >= 0
        ]

        if len(values):

            final = float(
                values.iloc[0]
            )

    return (
        initial,
        final
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    print("=" * 70)
    print("LOADING FATIGUE PREDICTIONS")
    print("=" * 70)

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    print(
        f"Prediction rows: "
        f"{len(df)}"
    )

    return df


# ============================================================
# ADD SEGMENT
# ============================================================

def add_segments(
    df
):

    df["segment"] = (
        df["file"]
        .apply(
            identify_segment
        )
    )

    return df


# ============================================================
# SEGMENT SUMMARY
# ============================================================

def segment_summary(
    df
):

    order = [

        "Initial Rest",

        "Rep 1",

        "Rep 2",

        "Rep 3",

        "Rep 4",

        "Rep 5",

        "Final Rest",
    ]

    summary = (
        df[
            df["segment"].isin(order)
        ]
        .groupby(
            "segment"
        )[
            "fatigue_score"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .reindex(order)
    )

    return summary


# ============================================================
# PARTICIPANT PROGRESSION
# ============================================================

def participant_progression(
    df
):

    order = [

        "Initial Rest",
        "Rep 1",
        "Rep 2",
        "Rep 3",
        "Rep 4",
        "Rep 5",
        "Final Rest",
    ]

    rows = []

    for participant, group in df.groupby(
        "participant_id"
    ):

        for segment in order:

            values = group.loc[
                group["segment"] == segment,
                "fatigue_score"
            ]

            if len(values):

                rows.append({

                    "participant_id":
                        participant,

                    "segment":
                        segment,

                    "mean_fatigue":
                        values.mean(),

                    "std_fatigue":
                        values.std(),

                    "windows":
                        len(values),
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# CORRELATION WITH SOURCE LABELS
# ============================================================

def source_label_analysis(
    df
):

    print()
    print("=" * 70)
    print("CHECKING ORIGINAL FATIGUE LABELS")
    print("=" * 70)

    records = []

    unique_files = (
        df["file"]
        .drop_duplicates()
        .tolist()
    )

    for filename in unique_files:

        path = find_source_file(
            filename
        )

        initial, final = (
            get_source_fatigue_labels(
                path
            )
        )

        prediction_values = df.loc[
            df["file"] == filename,
            "fatigue_score"
        ]

        if len(prediction_values) == 0:

            continue

        mean_prediction = (
            prediction_values.mean()
        )

        records.append({

            "file":
                filename,

            "segment":
                identify_segment(
                    filename
                ),

            "predicted_fatigue":
                mean_prediction,

            "source_initial_fatigue":
                initial,

            "source_final_fatigue":
                final,
        })

    labels = pd.DataFrame(
        records
    )

    print(
        f"Files with label information: "
        f"{len(labels)}"
    )

    return labels


# ============================================================
# CORRELATION
# ============================================================

def calculate_correlations(
    labels
):

    print()
    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    for column in [

        "source_initial_fatigue",

        "source_final_fatigue",
    ]:

        valid = labels[
            [
                "predicted_fatigue",
                column
            ]
        ].dropna()

        if len(valid) < 3:

            print(
                f"{column}: insufficient data"
            )

            continue

        correlation = (
            valid[
                "predicted_fatigue"
            ]
            .corr(
                valid[column]
            )
        )

        mae = np.mean(
            np.abs(
                valid[
                    "predicted_fatigue"
                ]
                - valid[column]
            )
        )

        print(
            f"{column}"
        )

        print(
            f"  Samples: "
            f"{len(valid)}"
        )

        print(
            f"  Correlation: "
            f"{correlation:.4f}"
        )

        print(
            f"  MAE: "
            f"{mae:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THERMORA FATIGUE PREDICTION ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_predictions()

    # --------------------------------------------------------
    # Segment
    # --------------------------------------------------------

    df = add_segments(
        df
    )

    # --------------------------------------------------------
    # Segment summary
    # --------------------------------------------------------

    summary = segment_summary(
        df
    )

    print()
    print("=" * 70)
    print("FATIGUE BY SEGMENT")
    print("=" * 70)

    print(
        summary.to_string()
    )

    # --------------------------------------------------------
    # Participant progression
    # --------------------------------------------------------

    progression = (
        participant_progression(
            df
        )
    )

    print()
    print("=" * 70)
    print("PARTICIPANT PROGRESSION")
    print("=" * 70)

    if len(progression):

        pivot = progression.pivot(
            index="participant_id",
            columns="segment",
            values="mean_fatigue"
        )

        order = [

            "Initial Rest",
            "Rep 1",
            "Rep 2",
            "Rep 3",
            "Rep 4",
            "Rep 5",
            "Final Rest",
        ]

        pivot = pivot.reindex(
            columns=[
                c for c in order
                if c in pivot.columns
            ]
        )

        print(
            pivot.round(2).to_string()
        )

    # --------------------------------------------------------
    # Source labels
    # --------------------------------------------------------

    labels = source_label_analysis(
        df
    )

    if len(labels):

        calculate_correlations(
            labels
        )

    # --------------------------------------------------------
    # Save combined analysis
    # --------------------------------------------------------

    summary_output = (
        summary.reset_index()
    )

    summary_output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("ANALYSIS SAVED")
    print("=" * 70)

    print(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION NOTES")
    print("=" * 70)

    print(
        "The segment trend is a sanity check, "
        "not a formal model-validation metric."
    )

    print(
        "The participant-held-out validation "
        "result remains the primary generalization "
        "evidence for the fatigue model."
    )

    print(
        "Do not use these predictions for "
        "medical diagnosis or safety-critical "
        "decisions without further validation."
    )


if __name__ == "__main__":

    main()