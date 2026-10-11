from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "test_10min.npz"
)


def main():

    data = np.load(TEST_PATH)

    future_core = data["y"].astype(float)

    current_core = (
        data["metadata"][:, 4]
        .astype(float)
    )

    delta = (
        future_core
        - current_core
    )

    print("=" * 70)
    print("10-MINUTE CORE TEMPERATURE CHANGE")
    print("=" * 70)

    print(
        f"\nCurrent core temperature:"
    )

    print(
        f"Mean: {current_core.mean():.3f}"
    )

    print(
        f"Std : {current_core.std():.3f}"
    )

    print(
        "\nFuture core temperature:"
    )

    print(
        f"Mean: {future_core.mean():.3f}"
    )

    print(
        f"Std : {future_core.std():.3f}"
    )

    print(
        "\nDelta core temperature "
        "(future - current):"
    )

    print(
        f"Mean: {delta.mean():.4f} °C"
    )

    print(
        f"Std : {delta.std():.4f} °C"
    )

    print(
        f"Min : {delta.min():.4f} °C"
    )

    print(
        f"Max : {delta.max():.4f} °C"
    )

    print(
        "\nPercentage of sequences:"
    )

    print(
        f"Heating: "
        f"{(delta > 0.02).mean() * 100:.2f}%"
    )

    print(
        f"Stable: "
        f"{(np.abs(delta) <= 0.02).mean() * 100:.2f}%"
    )

    print(
        f"Cooling: "
        f"{(delta < -0.02).mean() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()