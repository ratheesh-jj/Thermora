import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================
# LOAD
# ============================================================

data = np.load("thermora_raw_event_windows_v2.npz")
X = data["X"]

metadata = pd.read_csv(
    "thermora_raw_event_metadata_v2.csv"
)

artifact = joblib.load(
    "thermora_event_model_v2.pkl"
)

model = artifact["model"]
threshold = artifact["threshold"]


# ============================================================
# EXACT FEATURE EXTRACTION
# ============================================================

def extract_features(X):

    features = []

    for sample in X:

        acc = sample[:, 0:3]
        gyro = sample[:, 3:6]

        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        acc_jerk = np.diff(acc_mag)
        gyro_change = np.diff(gyro_mag)

        f = []

        for signal in [
            acc[:, 0], acc[:, 1], acc[:, 2],
            gyro[:, 0], gyro[:, 1], gyro[:, 2],
            acc_mag, gyro_mag, acc_jerk, gyro_change
        ]:

            f.extend([
                np.mean(signal),
                np.std(signal),
                np.min(signal),
                np.max(signal),
                np.ptp(signal),
                np.sqrt(np.mean(signal ** 2)),
                np.median(signal),
                np.percentile(signal, 10),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.percentile(signal, 90),
            ])

        f.extend([
            np.max(acc_mag),
            np.max(gyro_mag),
            np.mean(acc_mag),
            np.std(acc_mag),
            np.mean(gyro_mag),
            np.std(gyro_mag),
            np.max(np.abs(acc_jerk)),
            np.mean(np.abs(acc_jerk)),
            np.std(acc_jerk),
            np.max(np.abs(gyro_change)),
            np.mean(np.abs(gyro_change)),
            np.std(gyro_change),
        ])

        mid = len(sample) // 2

        for signal in [acc_mag, gyro_mag]:

            first = signal[:mid]
            second = signal[mid:]

            f.extend([
                np.mean(first),
                np.std(first),
                np.max(first),
                np.mean(second),
                np.std(second),
                np.max(second),
                np.mean(second) - np.mean(first),
                np.max(second) - np.max(first),
            ])

        f.extend([
            np.sum(acc_mag ** 2),
            np.sum(gyro_mag ** 2),
            np.mean(acc_mag ** 2),
            np.mean(gyro_mag ** 2),
        ])

        acc_threshold = np.percentile(acc_mag, 90)
        gyro_threshold = np.percentile(gyro_mag, 90)

        f.extend([
            np.mean(acc_mag > acc_threshold),
            np.mean(gyro_mag > gyro_threshold),
            np.mean(acc_mag > 10),
            np.mean(acc_mag > 15),
            np.mean(gyro_mag > 5),
            np.mean(gyro_mag > 10),
        ])

        features.append(f)

    return np.asarray(features, dtype=np.float32)


# ============================================================
# FIND MISSED WINDOWS
# ============================================================

mask = (
    (metadata["split"] == "test") &
    (metadata["label"] == 1)
)

indices = np.where(mask)[0]

features = extract_features(X[indices])

probabilities = model.predict_proba(features)[:, 1]

predictions = probabilities >= threshold

missed = indices[~predictions]

print("Missed windows:")
print(missed)

print()
print("Creating plots...")


# ============================================================
# PLOT EACH MISSED WINDOW
# ============================================================

for number, idx in enumerate(missed, start=1):

    window = X[idx]

    time = np.arange(20) / 20.0

    acc_x = window[:, 0]
    acc_y = window[:, 1]
    acc_z = window[:, 2]

    gyro_x = window[:, 3]
    gyro_y = window[:, 4]
    gyro_z = window[:, 5]

    acc_mag = np.linalg.norm(
        window[:, 0:3],
        axis=1
    )

    gyro_mag = np.linalg.norm(
        window[:, 3:6],
        axis=1
    )

    probability = probabilities[
        np.where(indices == idx)[0][0]
    ]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)

    plt.plot(time, acc_x, label="Acc X")
    plt.plot(time, acc_y, label="Acc Y")
    plt.plot(time, acc_z, label="Acc Z")
    plt.plot(time, acc_mag, label="Acc Magnitude")

    plt.ylabel("Acceleration")
    plt.title(
        f"Missed Window | Index {idx} | "
        f"Fall Probability = {probability:.3f}"
    )

    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(time, gyro_x, label="Gyro X")
    plt.plot(time, gyro_y, label="Gyro Y")
    plt.plot(time, gyro_z, label="Gyro Z")
    plt.plot(time, gyro_mag, label="Gyro Magnitude")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Gyroscope")
    plt.legend()
    plt.grid(True)

    filename = f"missed_detail_{number}.png"

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", filename)


print()
print("DONE.")