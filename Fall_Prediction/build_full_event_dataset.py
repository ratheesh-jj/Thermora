import os
import numpy as np
import pandas as pd

PROJECT_DIR = r"C:\Thermora Fall Prediction Model"

INPUT_X = os.path.join(PROJECT_DIR, "bits2_imu_windows.npz")
INPUT_META = os.path.join(PROJECT_DIR, "window_metadata.csv")

OUTPUT_X = os.path.join(
    PROJECT_DIR, "thermora_full_event_dataset.npz"
)

OUTPUT_META = os.path.join(
    PROJECT_DIR, "thermora_full_event_metadata.csv"
)

WINDOW_SIZE = 20
CHANNELS = 6


def main():

    print("=" * 60)
    print("BUILDING FULL THERMORA EVENT DATASET")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD EXISTING BITS-2 WINDOWS
    # ---------------------------------------------------------

    data = np.load(INPUT_X)

    X = data["X"]
    y = data["y"]

    meta = pd.read_csv(INPUT_META)

    print("\nOriginal dataset:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("Metadata:", meta.shape)

    # ---------------------------------------------------------
    # CHECK DATA
    # ---------------------------------------------------------

    required_columns = ["user", "label"]

    for column in required_columns:
        if column not in meta.columns:
            raise ValueError(
                f"Missing required metadata column: {column}"
            )

    # ---------------------------------------------------------
    # PARTICIPANT-INDEPENDENT SPLIT
    #
    # SAME SPLIT AS BEFORE
    # Train: users 1-29
    # Validation: users 30-35
    # Test: users 36-41
    # ---------------------------------------------------------

    train_users = list(range(1, 30))
    val_users = list(range(30, 36))
    test_users = list(range(36, 42))

    train_mask = meta["user"].isin(train_users).to_numpy()
    val_mask = meta["user"].isin(val_users).to_numpy()
    test_mask = meta["user"].isin(test_users).to_numpy()

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    meta_train = meta.loc[train_mask].copy()
    meta_val = meta.loc[val_mask].copy()
    meta_test = meta.loc[test_mask].copy()

    print("\nParticipant-independent split:")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    # ---------------------------------------------------------
    # EVENT-CENTERING
    #
    # We identify high-motion regions INSIDE FALL recordings.
    #
    # IMPORTANT:
    # This is a heuristic event detector, NOT ground truth.
    # ---------------------------------------------------------

    def acceleration_magnitude(X_data):
        return np.linalg.norm(X_data[:, :, :3], axis=2)

    def motion_score(window):
        acc = window[:, :3]
        gyro = window[:, 3:]

        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        acc_peak = np.max(acc_mag)
        gyro_peak = np.max(gyro_mag)

        acc_std = np.std(acc_mag)
        gyro_std = np.std(gyro_mag)

        return (
            acc_peak
            + 0.5 * gyro_peak
            + 0.5 * acc_std
            + 0.5 * gyro_std
        )

    # ---------------------------------------------------------
    # Instead of deleting the original windows, create an
    # additional "event score" for every window.
    # ---------------------------------------------------------

    print("\nCalculating event scores...")

    scores = np.zeros(len(X), dtype=np.float32)

    for i in range(len(X)):
        scores[i] = motion_score(X[i])

    # ---------------------------------------------------------
    # Calculate threshold ONLY from TRAINING FALL WINDOWS.
    # This prevents test information leaking into training.
    # ---------------------------------------------------------

    train_fall_mask = train_mask & (y == 1)

    train_fall_scores = scores[train_fall_mask]

    threshold = np.percentile(
        train_fall_scores,
        70
    )

    print(
        "\nEvent-score threshold:",
        round(float(threshold), 4)
    )

    # ---------------------------------------------------------
    # EVENT-CENTERED LABEL
    #
    # Existing label:
    #   0 = ADL
    #   1 = fall recording
    #
    # New event label:
    #   0 = non-event
    #   1 = fall-event candidate
    # ---------------------------------------------------------

    event_label = np.zeros(len(X), dtype=np.int64)

    fall_recording = y == 1

    event_label[
        fall_recording & (scores >= threshold)
    ] = 1

    # ADL windows remain 0.
    # ---------------------------------------------------------

    print("\nNew event label distribution:")

    unique, counts = np.unique(
        event_label,
        return_counts=True
    )

    for label, count in zip(unique, counts):

        name = "NON-EVENT" if label == 0 else "FALL-EVENT"

        print(
            f"{name:12s}: {count}"
        )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    np.savez_compressed(
        OUTPUT_X,
        X=X.astype(np.float32),
        y=event_label
    )

    output_meta = meta.copy()

    output_meta["original_label"] = y
    output_meta["event_label"] = event_label
    output_meta["event_score"] = scores
    output_meta["event_threshold"] = threshold

    output_meta.to_csv(
        OUTPUT_META,
        index=False
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print("\nSaved:")
    print(OUTPUT_X)
    print(OUTPUT_META)

    print("\nIMPORTANT:")
    print("The event labels are heuristic candidate labels.")
    print("They are NOT manually verified fall timestamps.")
    print("We will evaluate this carefully before calling it")
    print("a final THERMORA fall detector.")


if __name__ == "__main__":
    main()