import numpy as np
import pandas as pd
import joblib


DATA = "thermora_raw_event_windows_v2.npz"
META = "thermora_raw_event_metadata_v2.csv"
MODEL = "thermora_event_model_v2.pkl"


# ============================================================
# EXACT FEATURE EXTRACTION USED FOR TRAINING
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

        # Peak / impact features
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
# LOAD
# ============================================================

print("=" * 65)
print("THERMORA REAL BITS-2 MODEL DEBUG")
print("=" * 65)

data = np.load(DATA)

X = data["X"]
y = data["y"]

metadata = pd.read_csv(META)

artifact = joblib.load(MODEL)

model = artifact["model"]
threshold = artifact["threshold"]

print()
print("Dataset:", X.shape)
print("Labels :", y.shape)
print("Model  :", type(model).__name__)
print("Threshold:", threshold)


# ============================================================
# FEATURES
# ============================================================

print()
print("Extracting features...")

X_features = extract_features(X)

print("Features:", X_features.shape)

# Safety check
expected = artifact["feature_count"]

print("Expected features:", expected)

if X_features.shape[1] != expected:
    raise ValueError(
        f"FEATURE MISMATCH: generated {X_features.shape[1]}, "
        f"model expects {expected}"
    )

print("Feature count MATCHED.")


# ============================================================
# PREDICTIONS
# ============================================================

prob = model.predict_proba(X_features)[:, 1]

pred = (
    prob >= threshold
).astype(int)


# ============================================================
# FULL DATASET RESULTS
# ============================================================

print()
print("=" * 65)
print("FULL DATASET CHECK")
print("=" * 65)

accuracy = np.mean(pred == y)

fall_mask = y == 1

fall_recall = np.sum(
    (pred == 1) & fall_mask
) / np.sum(fall_mask)

print()
print("Accuracy   :", round(accuracy, 4))
print("Fall recall:", round(fall_recall, 4))

print()
print("Actual falls :", np.sum(y == 1))
print("Detected falls:", np.sum((pred == 1) & (y == 1)))
print("Missed falls  :", np.sum((pred == 0) & (y == 1)))


# ============================================================
# TEST SET ONLY
# ============================================================

test_mask = metadata["split"].values == "test"

test_indices = np.where(test_mask)[0]

test_y = y[test_indices]
test_prob = prob[test_indices]
test_pred = pred[test_indices]

test_accuracy = np.mean(test_pred == test_y)

test_fall_mask = test_y == 1

test_recall = np.sum(
    (test_pred == 1) & test_fall_mask
) / np.sum(test_fall_mask)

print()
print("=" * 65)
print("TEST SET CHECK")
print("=" * 65)

print()
print("Test windows :", len(test_indices))
print("Test falls   :", np.sum(test_y == 1))
print("Test normal  :", np.sum(test_y == 0))

print()
print("Accuracy    :", round(test_accuracy, 4))
print("Fall recall :", round(test_recall, 4))


# ============================================================
# MISSED TEST FALL WINDOWS
# ============================================================

missed = test_indices[
    (test_y == 1) & (test_pred == 0)
]

print()
print("=" * 65)
print("MISSED TEST FALL WINDOWS")
print("=" * 65)

print()
print("Number missed:", len(missed))

for i in missed:

    row = metadata.iloc[i]

    print(
        f"Index={i:4d} | "
        f"Probability={prob[i]:.3f} | "
        f"File={row.get('file', 'N/A')}"
    )


# ============================================================
# STRONGEST MISSED FALLS
# ============================================================

print()
print("=" * 65)
print("MISSED FALLS — HIGHEST PROBABILITY FIRST")
print("=" * 65)

missed_sorted = sorted(
    missed,
    key=lambda i: prob[i],
    reverse=True
)

for i in missed_sorted[:20]:

    print(
        f"Index={i:4d} | "
        f"Probability={prob[i]:.3f}"
    )


print()
print("=" * 65)
print("DEBUG COMPLETE")
print("=" * 65)