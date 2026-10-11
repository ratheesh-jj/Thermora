import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "thermora_raw_event_windows_v2.npz"
METADATA_PATH = "thermora_raw_event_metadata_v2.csv"
MODEL_PATH = "thermora_event_model_v2.pkl"

WINDOW_SIZE = 20       # 20 samples = 1 second at 20 Hz
STRIDE = 10            # 10 samples = 0.5 second stride
TEST_USERS = [36, 37, 38, 39, 40, 41]


# ============================================================
# EXACT V2 FEATURE EXTRACTION
# ============================================================

def extract_features(X):

    features = []

    for sample in X:

        # ----------------------------------------------------
        # Separate accelerometer and gyroscope
        # ----------------------------------------------------

        acc = sample[:, 0:3]
        gyro = sample[:, 3:6]

        # Magnitudes
        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        # Changes
        acc_jerk = np.diff(acc_mag)
        gyro_change = np.diff(gyro_mag)

        f = []

        # ----------------------------------------------------
        # 10 signals × 11 statistics = 110 features
        # ----------------------------------------------------

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

        for signal in signals:

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

        # ----------------------------------------------------
        # Peak / impact features = 12
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # First half vs second half = 16
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Energy = 4
        # ----------------------------------------------------

        f.extend([
            np.sum(acc_mag ** 2),
            np.sum(gyro_mag ** 2),
            np.mean(acc_mag ** 2),
            np.mean(gyro_mag ** 2),
        ])

        # ----------------------------------------------------
        # High-motion fractions = 6
        # ----------------------------------------------------

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
# LOAD DATA
# ============================================================

print()
print("THERMORA — REAL-TIME SLIDING-WINDOW TEST")
print("=" * 50)

data = np.load(DATASET_PATH)

X = data["X"]
y = data["y"]

metadata = pd.read_csv(METADATA_PATH)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
threshold = model_data["threshold"]

print(f"Dataset shape : {X.shape}")
print(f"Labels shape  : {y.shape}")
print(f"Metadata rows : {len(metadata)}")
print(f"Threshold     : {threshold:.3f}")


# ============================================================
# VERIFY FEATURE COUNT
# ============================================================

print()
print("Checking feature extractor...")

test_features = extract_features(X[:1])

print(f"Feature count : {test_features.shape[1]}")
print(f"Model expects : {model_data['feature_count']}")

if test_features.shape[1] != model_data["feature_count"]:
    raise RuntimeError(
        "FEATURE COUNT MISMATCH — STOPPING."
    )

print("Feature count MATCHED.")


# ============================================================
# IDENTIFY HELD-OUT TEST PARTICIPANTS
# ============================================================

test_mask = metadata["user"].isin(TEST_USERS)

test_indices = np.flatnonzero(test_mask)

print()
print("HELD-OUT TEST SET")
print("-" * 50)
print(f"Test participants : {TEST_USERS}")
print(f"Test windows      : {len(test_indices)}")

test_fall_indices = test_indices[y[test_indices] == 1]

print(f"Fall windows      : {len(test_fall_indices)}")


# ============================================================
# REAL-TIME SLIDING-WINDOW SIMULATION
# ============================================================
#
# Each stored window represents:
#
#       20 IMU samples
#              ↓
#        feature extraction
#              ↓
#          V2 model
#              ↓
#       fall probability
#
# STRIDE = 10 represents a new prediction every 0.5 sec.
#
# NOTE:
# The prepared V2 dataset already contains windows.
# Therefore this section simulates the inference stage
# using those real windows rather than artificially
# generating new sensor data.
# ============================================================

predictions = []
probabilities = []
actual_labels = []
dataset_indices = []

for idx in test_indices:

    window = X[idx]

    # Safety check
    if window.shape != (WINDOW_SIZE, 6):
        raise RuntimeError(
            f"Unexpected window shape at index {idx}: "
            f"{window.shape}"
        )

    # Convert one window into batch format
    window_batch = window[np.newaxis, :, :]

    # Exact V2 features
    features = extract_features(window_batch)

    # Model prediction
    probability = model.predict_proba(features)[0, 1]

    prediction = int(probability >= threshold)

    predictions.append(prediction)
    probabilities.append(probability)
    actual_labels.append(int(y[idx]))
    dataset_indices.append(idx)


# ============================================================
# RESULTS
# ============================================================

predictions = np.asarray(predictions)
probabilities = np.asarray(probabilities)
actual_labels = np.asarray(actual_labels)
dataset_indices = np.asarray(dataset_indices)


# ------------------------------------------------------------
# Overall accuracy
# ------------------------------------------------------------

accuracy = np.mean(predictions == actual_labels)


# ------------------------------------------------------------
# Fall recall
# ------------------------------------------------------------

fall_mask = actual_labels == 1

actual_falls = np.sum(fall_mask)
detected_falls = np.sum(
    predictions[fall_mask] == 1
)

missed_falls = actual_falls - detected_falls

fall_recall = (
    detected_falls / actual_falls
    if actual_falls > 0
    else 0
)


# ------------------------------------------------------------
# Fall precision
# ------------------------------------------------------------

predicted_falls = np.sum(predictions == 1)

true_positives = np.sum(
    (predictions == 1) &
    (actual_labels == 1)
)

fall_precision = (
    true_positives / predicted_falls
    if predicted_falls > 0
    else 0
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 50)
print("SLIDING-WINDOW RESULTS")
print("=" * 50)

print(f"Window size       : {WINDOW_SIZE} samples")
print(f"Window duration   : {WINDOW_SIZE / 20:.1f} seconds")
print(f"Stride            : {STRIDE} samples")
print(f"Prediction period : {STRIDE / 20:.1f} seconds")

print()
print(f"Test windows      : {len(actual_labels)}")
print(f"Actual falls      : {actual_falls}")
print(f"Detected falls    : {detected_falls}")
print(f"Missed falls      : {missed_falls}")

print()
print(f"Accuracy           : {accuracy:.4f}")
print(f"Fall precision     : {fall_precision:.4f}")
print(f"Fall recall        : {fall_recall:.4f}")


# ============================================================
# SHOW MISSED FALL WINDOWS
# ============================================================

missed_mask = (
    (actual_labels == 1) &
    (predictions == 0)
)

missed_indices = dataset_indices[missed_mask]
missed_probs = probabilities[missed_mask]

print()
print("=" * 50)
print("MISSED FALL WINDOWS")
print("=" * 50)

if len(missed_indices) == 0:

    print("No missed fall windows.")

else:

    for idx, probability in zip(
        missed_indices,
        missed_probs
    ):

        row = metadata.iloc[idx]

        print(
            f"Index={idx} | "
            f"User={row['user']} | "
            f"Probability={probability:.3f} | "
            f"File={row['file']} | "
            f"Start={row['window_start_sec']:.2f}s"
        )


# ============================================================
# SHOW SAMPLE STREAM
# ============================================================

print()
print("=" * 50)
print("SAMPLE STREAM PREDICTIONS")
print("=" * 50)

print(
    "Index | User | Time(s) | Probability | Prediction"
)

print("-" * 50)

for idx in test_indices[:20]:

    row = metadata.iloc[idx]

    probability = probabilities[
        np.where(dataset_indices == idx)[0][0]
    ]

    prediction = int(
        probability >= threshold
    )

    result = "FALL" if prediction else "NO FALL"

    print(
        f"{idx:5d} | "
        f"{int(row['user']):4d} | "
        f"{row['window_start_sec']:7.2f} | "
        f"{probability:11.3f} | "
        f"{result}"
    )

print()
print("=" * 50)
print("TEST COMPLETE")
print("=" * 50)