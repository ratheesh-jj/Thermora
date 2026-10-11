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
# EXACT FEATURE EXTRACTOR
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
# FIND MISSED + DETECTED
# ============================================================

test_mask = (
    (metadata["split"] == "test") &
    (metadata["label"] == 1)
)

indices = np.where(test_mask)[0]

features = extract_features(X[indices])

probabilities = model.predict_proba(features)[:, 1]

predictions = probabilities >= threshold


missed = indices[~predictions]
detected = indices[predictions]


print("Missed:", len(missed))
print("Detected:", len(detected))

print()
print("MISSED:")
print(missed)

print()
print("DETECTED:")
print(detected)


# ============================================================
# PLOT
# ============================================================

# Use 8 missed + 8 detected
missed = missed[:8]
detected = detected[:8]


for pair_number, (miss_idx, det_idx) in enumerate(
    zip(missed, detected), start=1
):

    miss_window = X[miss_idx]
    det_window = X[det_idx]

    miss_acc_mag = np.linalg.norm(
        miss_window[:, 0:3],
        axis=1
    )

    det_acc_mag = np.linalg.norm(
        det_window[:, 0:3],
        axis=1
    )

    time = np.arange(20) / 20.0

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        miss_acc_mag,
        label=f"Missed | index {miss_idx}"
    )

    plt.plot(
        time,
        det_acc_mag,
        label=f"Detected | index {det_idx}"
    )

    plt.axhline(
        10,
        linestyle="--",
        label="10 m/s²"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration Magnitude")
    plt.title(
        f"Missed vs Detected Fall Candidate #{pair_number}"
    )

    plt.legend()
    plt.grid(True)

    filename = (
        f"comparison_{pair_number}.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", filename)


print()
print("DONE.")