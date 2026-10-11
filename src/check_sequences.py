from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)


def main():

    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        [
            "participant_id",
            "condition_id",
            "time_min",
        ]
    )

    print("=" * 70)
    print("THERMOSENTINEL SEQUENCE DIAGNOSTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check every participant-condition combination
    # --------------------------------------------------------

    grouped = df.groupby(
        [
            "participant_id",
            "condition_id",
        ]
    )

    total_sequences = 0
    valid_sequences = 0

    for (participant, condition), group in grouped:

        total_sequences += 1

        group = group.sort_values(
            "time_min"
        )

        time_diff = (
            group["time_min"]
            .diff()
            .dropna()
        )

        if len(time_diff) == 0:
            continue

        median_step = time_diff.median()

        duration = (
            group["time_min"].max()
            - group["time_min"].min()
        )

        print(
            f"P{participant:02d} | "
            f"C{condition:02d} | "
            f"Rows={len(group):3d} | "
            f"Duration={duration:6.1f} min | "
            f"Median step={median_step:.2f} min"
        )

        if np.isclose(
            median_step,
            1.0,
            atol=0.05,
        ):
            valid_sequences += 1

    print("\n" + "=" * 70)

    print(
        f"Total participant-condition sequences: "
        f"{total_sequences}"
    )

    print(
        f"Approximately 1-minute sequences: "
        f"{valid_sequences}"
    )


if __name__ == "__main__":
    main()