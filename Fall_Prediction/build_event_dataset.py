import os
import glob
import numpy as np
import pandas as pd

PROJECT_DIR = r"C:\Thermora Fall Prediction Model"

# Settings
FS = 20                    # BITS-2 sampling rate
WINDOW_SIZE = 20           # 1 second
STRIDE = 10                # 0.5 second
PRE_EVENT = 10             # 0.5 sec before candidate
POST_EVENT = 10            # 0.5 sec after candidate

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "thermora_event_dataset.npz"
)

META_FILE = os.path.join(
    PROJECT_DIR,
    "thermora_event_metadata.csv"
)


def load_sensor_data(csv_file):
    df = pd.read_csv(csv_file)

    df["sensor"] = df["Unnamed: 5"].astype(str).str.lower()

    acc = df[df["sensor"] == "acc"].copy()
    gyro = df[df["sensor"] == "gyro"].copy()

    if len(acc) == 0 or len(gyro) == 0:
        return None

    acc = acc[["x", "y", "z"]].astype(float).reset_index(drop=True)
    gyro = gyro[["x", "y", "z"]].astype(float).reset_index(drop=True)

    n = min(len(acc), len(gyro))

    acc = acc.iloc[:n].to_numpy()
    gyro = gyro.iloc[:n].to_numpy()

    return np.hstack([acc, gyro])


def find_event_sample(data):
    """
    Finds the strongest acceleration event.

    IMPORTANT:
    This is a candidate-event detector, not ground-truth annotation.
    """

    acc = data[:, :3]
    gyro = data[:, 3:]

    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    # Smooth acceleration slightly
    acc_series = pd.Series(acc_mag)
    acc_smooth = acc_series.rolling(
        window=5,
        center=True,
        min_periods=1
    ).mean().to_numpy()

    # Strong acceleration peak
    peak = int(np.argmax(acc_smooth))

    return peak, acc_mag, gyro_mag


def make_window(data, center):
    """
    Creates a 20-sample window centered around the candidate event.
    """

    start = center - PRE_EVENT
    end = center + POST_EVENT

    # Pad if event is near beginning/end
    window = np.zeros((WINDOW_SIZE, 6), dtype=np.float32)

    src_start = max(0, start)
    src_end = min(len(data), end)

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    window[dst_start:dst_end] = data[src_start:src_end]

    return window


def main():

    print("=" * 60)
    print("BUILDING EVENT-CENTERED THERMORA DATASET")
    print("=" * 60)

    csv_files = sorted(
        glob.glob(
            os.path.join(PROJECT_DIR, "user5_fall*.csv")
        )
    )

    if not csv_files:
        print("ERROR: No user5_fall*.csv files found.")
        return

    print(f"\nFound {len(csv_files)} fall recordings.\n")

    X = []
    y = []
    metadata = []

    # ---------------------------------------------------------
    # 1. FALL EVENT WINDOWS
    # ---------------------------------------------------------

    print("Creating FALL event windows...\n")

    for csv_file in csv_files:

        filename = os.path.basename(csv_file)

        data = load_sensor_data(csv_file)

        if data is None:
            print(f"Skipping {filename}")
            continue

        event, acc_mag, gyro_mag = find_event_sample(data)

        print(
            f"{filename:25s} "
            f"event sample={event:5d} "
            f"AccMag={acc_mag[event]:8.2f} "
            f"GyroMag={gyro_mag[event]:8.2f}"
        )

        # Create several windows around the candidate
        offsets = [-20, -10, 0, 10, 20]

        for offset in offsets:

            center = event + offset

            window = make_window(data, center)

            X.append(window)
            y.append(1)

            metadata.append({
                "file": filename,
                "label": 1,
                "event_sample": event,
                "window_center": center,
                "window_type": "fall_candidate"
            })

    # ---------------------------------------------------------
    # 2. NON-FALL WINDOWS
    #
    # We use portions of the fall recordings FAR away from
    # the detected event only as provisional negatives.
    # These are NOT ideal negatives, but useful for testing
    # the event-centered pipeline.
    # ---------------------------------------------------------

    print("\nCreating provisional NON-FALL windows...\n")

    for csv_file in csv_files:

        filename = os.path.basename(csv_file)

        data = load_sensor_data(csv_file)

        if data is None:
            continue

        event, _, _ = find_event_sample(data)

        # Windows far away from candidate event
        candidate_centers = [
            50,
            100,
            150,
            len(data) - 50,
            len(data) - 100
        ]

        for center in candidate_centers:

            if abs(center - event) < 50:
                continue

            if center < 10 or center > len(data) - 10:
                continue

            window = make_window(data, center)

            X.append(window)
            y.append(0)

            metadata.append({
                "file": filename,
                "label": 0,
                "event_sample": event,
                "window_center": center,
                "window_type": "provisional_nonfall"
            })

    # ---------------------------------------------------------
    # 3. SAVE
    # ---------------------------------------------------------

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    metadata = pd.DataFrame(metadata)

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y
    )

    metadata.to_csv(
        META_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("DATASET CREATED")
    print("=" * 60)

    print(f"\nX shape : {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nClass distribution:")

    unique, counts = np.unique(y, return_counts=True)

    for label, count in zip(unique, counts):
        name = "NON-FALL" if label == 0 else "FALL"
        print(f"{name:10s}: {count}")

    print(f"\nSaved:")
    print(OUTPUT_FILE)
    print(META_FILE)

    print("\nIMPORTANT:")
    print("Fall windows are centered on detected candidate events.")
    print("They are NOT manually confirmed ground-truth timestamps.")
    print("This dataset is for pipeline experimentation.")


if __name__ == "__main__":
    main()