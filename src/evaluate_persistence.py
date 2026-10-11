from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "test_10min.npz"
)


def main():

    data = np.load(TEST_PATH)

    y_true = data["y"].astype(float)

    metadata = data["metadata"]

    # Metadata:
    # [participant_id,
    #  condition_id,
    #  input_end_time,
    #  target_time,
    #  current_core_temp]

    current_core_temp = (
        metadata[:, 4]
        .astype(float)
    )

    predictions = current_core_temp

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    print("=" * 70)
    print("THERMOSENTINEL — PERSISTENCE BASELINE")
    print("=" * 70)

    print(
        f"\nMAE :  {mae:.4f} °C"
    )

    print(
        f"RMSE:  {rmse:.4f} °C"
    )

    print(
        f"R²  :  {r2:.4f}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "This baseline simply assumes that the "
        "core temperature 10 minutes from now "
        "will be equal to the current core temperature."
    )


if __name__ == "__main__":
    main()