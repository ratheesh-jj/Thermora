from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("THERMOSENTINEL DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset path:\n{DATA_PATH}")
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\nParticipant count:")
    print(df["participant_id"].nunique())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()