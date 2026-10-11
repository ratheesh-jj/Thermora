from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("THERMOSENTINEL DATA ANALYSIS")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nTarget statistics:")
    print(df["core_temp_c"].describe())

    print("\nParticipants:")
    print(df["participant_id"].unique())

    print("\nObservations per participant:")
    print(
        df.groupby("participant_id")
        .size()
        .describe()
    )

    print("\nCore temperature missing:")
    print(
        df["core_temp_c"].isna().sum()
    )

    print("\nCore temperature range:")
    print(
        df["core_temp_c"].min(),
        "to",
        df["core_temp_c"].max()
    )


if __name__ == "__main__":
    main()