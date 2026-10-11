import numpy as np
import pandas as pd
import joblib


DATASET = "thermora_raw_event_windows_v2.npz"
METADATA = "thermora_raw_event_metadata_v2.csv"
MODEL = "thermora_event_model_v2.pkl"

THRESHOLD = 0.28


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
            acc_mag, gyro_mag,
            acc_jerk, gyro_change
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


# =========================
# LOAD
# =========================

data = np.load(DATASET)
X = data["X"]
y = data["y"]

meta = pd.read_csv(METADATA)

model_data = joblib.load(MODEL)
model = model_data["model"]
threshold = model_data["threshold"]

print("THERMORA REAL-TIME TEST")
print("========================")
print("Dataset:", X.shape)
print("Metadata:", meta.shape)
print("Threshold:", threshold)


# =========================
# CHECK METADATA
# =========================

print("\nMetadata columns:")
print(meta.columns.tolist())

print("\nFirst rows:")
print(meta.head())


# =========================
# FIND TEST PARTICIPANTS
# =========================

test_users = [36, 37, 38, 39, 40, 41]

# Find user column automatically
user_column = None

for col in meta.columns:
    if "user" in col.lower() or "participant" in col.lower():
        user_column = col
        break

if user_column is None:
    raise ValueError("Could not find user/participant column in metadata.")

print("\nUsing participant column:", user_column)

test_mask = meta[user_column].astype(str).str.extract(r"(\d+)")[0].astype(int).isin(test_users)

test_indices = np.where(test_mask)[0]

print("Test windows:", len(test_indices))

test_fall_indices = test_indices[y[test_indices] == 1]

print("Test fall windows:", len(test_fall_indices))


# =========================
# TEST REAL FALL WINDOWS
# =========================

print("\nTesting first 10 real fall windows...")
print("---------------------------------------")

detected = 0

for count, idx in enumerate(test_fall_indices[:10], start=1):

    window = X[idx:idx + 1]

    features = extract_features(window)

    probability = model.predict_proba(features)[0, 1]

    prediction = probability >= threshold

    if prediction:
        detected += 1

    print(
        f"{count:02d} | "
        f"Dataset index={idx} | "
        f"Probability={probability:.3f} | "
        f"{'FALL DETECTED' if prediction else 'NO FALL'}"
    )


print("\n========================")
print("RESULT")
print("========================")
print("Detected:", detected, "/ 10")