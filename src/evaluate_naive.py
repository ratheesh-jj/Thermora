from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "test_10min.npz"
)


def main():

    data = np.load(TEST_PATH)

    X = data["X"]
    y = data["y"]

    # The first feature columns are:
    #
    # 0 age
    # 1 body mass
    # 2 ambient temperature
    # 3 humidity
    # 4 heart rate
    # 5 insulated skin temperature
    # 6 sex
    #
    # The current core temperature is NOT an input feature.
    #
    # Therefore, for the naive baseline we need the
    # corresponding measured core temperature at time t.
    #
    # We did not save that in the sequence dataset.
    # So this script is intentionally NOT used yet.
    
    print("Current temporal dataset shape:", X.shape)
    print("Target shape:", y.shape)
    print(
        "\nNaive baseline cannot yet be calculated because "
        "the current core temperature was not saved in the "
        "sequence metadata."
    )


if __name__ == "__main__":
    main()