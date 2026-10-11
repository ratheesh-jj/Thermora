from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "sequences"
)

TEST_PATH = (
    SEQUENCE_DIR
    / "test_10min.npz"
)


def main():

    data = np.load(TEST_PATH)

    print("=" * 70)
    print("THERMOSENTINEL SEQUENCE METADATA CHECK")
    print("=" * 70)

    print("\nFiles contained in NPZ:")
    print(data.files)

    print("\nX shape:")
    print(data["X"].shape)

    print("\ny shape:")
    print(data["y"].shape)

    print("\nMetadata shape:")
    print(data["metadata"].shape)

    print("\nFirst 5 metadata rows:")

    for row in data["metadata"][:5]:
        print(row)

    print("\nExpected metadata columns:")
    print(
        "0 = participant_id\n"
        "1 = condition_id\n"
        "2 = input_end_time\n"
        "3 = target_time\n"
        "4 = current_core_temp"
    )

    if data["metadata"].shape[1] >= 5:
        print(
            "\n✅ current_core_temp is present."
        )
    else:
        print(
            "\n❌ current_core_temp is NOT present."
        )


if __name__ == "__main__":
    main()