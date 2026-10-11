import numpy as np
import pandas as pd
import joblib


# ============================================================
# LOAD
# ============================================================

data = np.load("thermora_raw_event_windows_v2.npz")

X = data["X"]
y = data["y"]

metadata = pd.read_csv(
    "thermora_raw_event_metadata_v2.csv"
)

artifact = joblib.load(
    "thermora_event_model_v2.pkl"
)

model = artifact["model"]
threshold = artifact["threshold"]

print("Model loaded")
print("Threshold:", threshold)
print("X shape:", X.shape)


# ============================================================
# EXACT SAME FEATURE EXTRACTION USED FOR TRAINING
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

        # Raw axis statistics
        for signal in [
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

        # Peak / impact
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

        # First half vs second half
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

        # Energy
        f.extend([
            np.sum(acc_mag ** 2),
            np.sum(gyro_mag ** 2),

            np.mean(acc_mag ** 2),
            np.mean(gyro_mag ** 2),
        ])

        # High-motion fractions
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
# TEST REAL FALL WINDOWS
# ============================================================

mask = (
    (metadata["split"] == "test") &
    (metadata["label"] == 1)
)

indices = np.where(mask)[0]

print()
print("Real test fall windows:", len(indices))
print("Extracting features...")


X_fall = X[indices]

features = extract_features(X_fall)

print("Feature shape:", features.shape)
print("Expected feature count:", artifact["feature_count"])

print()
print("Running model...")

probabilities = model.predict_proba(features)[:, 1]

predictions = (
    probabilities >= threshold
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("REAL BITS-2 FALL WINDOW RESULTS")
print("=" * 60)

for i, probability, prediction in zip(
    indices,
    probabilities,
    predictions
):

    print(
        f"Index: {i:4d} | "
        f"Fall probability: {probability:.3f} | "
        f"Prediction: "
        f"{'FALL' if prediction else 'NO FALL'}"
    )


detected = np.sum(predictions)

print()
print("=" * 60)
print(f"Detected: {detected}/{len(indices)}")
print(
    f"Detection rate: "
    f"{detected / len(indices) * 100:.2f}%"
)
print("=" * 60)