import numpy as np
import pandas as pd
import joblib

# ============================================================
# LOAD
# ============================================================

data = np.load("thermora_raw_event_windows_v2.npz")
X = data["X"]
y = data["y"]

meta = pd.read_csv("thermora_raw_event_metadata_v2.csv")

model_data = joblib.load("thermora_event_model_v2.pkl")

model = model_data["model"]
threshold = model_data["threshold"]

print("Dataset:", X.shape)
print("Threshold:", threshold)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(X):

    features = []

    for w in X:

        acc = w[:, 0:3]
        gyro = w[:, 3:6]

        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        acc_jerk = np.diff(acc_mag, prepend=acc_mag[0])
        gyro_change = np.diff(gyro_mag, prepend=gyro_mag[0])

        signals = [
            acc[:, 0],
            acc[:, 1],
            acc[:, 2],
            gyro[:, 0],
            gyro[:, 1],
            gyro[:, 2],
            acc_mag,
            gyro_mag,
            acc_jerk,
            gyro_change
        ]

        f = []

        # Basic statistics
        for s in signals:
            f.extend([
                np.mean(s),
                np.std(s),
                np.min(s),
                np.max(s),
                np.ptp(s),
                np.sqrt(np.mean(s ** 2)),
                np.median(s),
                np.percentile(s, 10),
                np.percentile(s, 25),
                np.percentile(s, 75),
                np.percentile(s, 90)
            ])

        # Impact features
        f.extend([
            np.max(acc_mag),
            np.mean(acc_mag),
            np.std(acc_mag),
            np.max(np.abs(acc_jerk)),
            np.mean(np.abs(acc_jerk)),
            np.max(gyro_mag),
            np.mean(gyro_mag),
            np.std(gyro_mag),
            np.max(np.abs(gyro_change)),
            np.mean(np.abs(gyro_change))
        ])

        # First / second half
        mid = len(w) // 2

        for s in signals:
            first = s[:mid]
            second = s[mid:]

            f.extend([
                np.mean(first),
                np.mean(second),
                np.std(first),
                np.std(second),
                np.max(first),
                np.max(second),
                np.mean(second) - np.mean(first)
            ])

        # Energy
        for s in signals:
            f.append(np.sum(s ** 2))

        # High-motion fractions
        f.extend([
            np.mean(acc_mag > 12),
            np.mean(acc_mag > 15),
            np.mean(acc_mag > 20),
            np.mean(gyro_mag > 5),
            np.mean(gyro_mag > 10),
            np.mean(np.abs(acc_jerk) > 5)
        ])

        features.append(f)

    return np.asarray(features)


# ============================================================
# RUN MODEL
# ============================================================

features = extract_features(X)

print("Features:", features.shape)

prob = model.predict_proba(features)[:, 1]
pred = (prob >= threshold).astype(int)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n==============================")
print("FULL DATASET CHECK")
print("==============================")

print("Accuracy:", np.mean(pred == y))

fall_mask = y == 1

recall = np.sum(
    (pred == 1) & fall_mask
) / np.sum(fall_mask)

print("Fall recall:", recall)


# ============================================================
# SHOW MISSED FALLS
# ============================================================

missed = np.where(
    (y == 1) & (pred == 0)
)[0]

print("\nMissed falls:", len(missed))

print("\nFirst 20 missed windows:")

for i in missed[:20]:

    print(
        f"Index {i:4d} | "
        f"Probability {prob[i]:.3f}"
    )