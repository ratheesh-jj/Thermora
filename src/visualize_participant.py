from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)


def main():

    df = pd.read_csv(DATA_PATH)

    participant_id = 1

    participant = df[
        df["participant_id"] == participant_id
    ].copy()

    participant = participant.sort_values(
        ["condition_id", "time_min"]
    )

    conditions = sorted(
        participant["condition_id"].unique()
    )

    print("=" * 70)
    print(
        f"PARTICIPANT {participant_id} "
        "TRIAL ANALYSIS"
    )
    print("=" * 70)

    print(
        f"\nNumber of conditions: "
        f"{len(conditions)}"
    )

    for condition in conditions:

        trial = participant[
            participant["condition_id"] == condition
        ].copy()

        print(
            f"\nCondition {condition}"
        )

        print(
            f"  Activity: "
            f"{trial['activity_label'].iloc[0]}"
        )

        print(
            f"  Rows: "
            f"{len(trial)}"
        )

        print(
            f"  Duration: "
            f"{trial['time_min'].min():.1f} → "
            f"{trial['time_min'].max():.1f} min"
        )

        print(
            f"  Core temp: "
            f"{trial['core_temp_c'].min():.2f} → "
            f"{trial['core_temp_c'].max():.2f} °C"
        )

        print(
            f"  Work/rest states: "
            f"{trial['work_rest'].unique()}"
        )

    # --------------------------------------------------------
    # Plot each experimental condition separately
    # --------------------------------------------------------

    plt.figure(figsize=(12, 6))

    for condition in conditions:

        trial = participant[
            participant["condition_id"] == condition
        ].copy()

        plt.plot(
            trial["time_min"],
            trial["core_temp_c"],
            label=f"Condition {condition}",
        )

    plt.xlabel("Minutes within condition")
    plt.ylabel("Core temperature (°C)")

    plt.title(
        f"Participant {participant_id} — "
        "Core Temperature by Experimental Condition"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()