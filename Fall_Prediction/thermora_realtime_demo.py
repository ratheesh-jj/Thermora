import numpy as np
import joblib
import time

# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "thermora_event_model_v2.pkl"

FS = 20
WINDOW_SIZE = 20

# Model-selected threshold
THRESHOLD = 0.28

rng = np.random.default_rng(42)


# ============================================================
# LOAD MODEL
# ============================================================

artifact = joblib.load(MODEL_FILE)

model = artifact["model"]

print("=" * 65)
print("THERMORA REAL-TIME FALL DETECTION DEMO")
print("=" * 65)

print()
print("Model loaded:", MODEL_FILE)
print("Threshold:", THRESHOLD)
print("Sampling rate:", FS, "Hz")
print()


# ============================================================
# FEATURE EXTRACTION
# MUST MATCH TRAINING
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

        # Raw signals
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
                np.percentile(signal, 90)
            ])

        # Impact features
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
            np.std(gyro_change)
        ])

        # First / second half
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
                np.max(second) - np.max(first)
            ])

        # Energy
        f.extend([
            np.sum(acc_mag ** 2),
            np.sum(gyro_mag ** 2),

            np.mean(acc_mag ** 2),
            np.mean(gyro_mag ** 2)
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
            np.mean(gyro_mag > 10)
        ])

        features.append(f)

    return np.asarray(features, dtype=np.float32)


# ============================================================
# NORMAL IMU SAMPLE
# ============================================================

def normal_sample():

    ax = rng.normal(0, 1.2)
    ay = rng.normal(0, 1.2)
    az = 9.81 + rng.normal(0, 1.0)

    gx = rng.normal(0, 2.0)
    gy = rng.normal(0, 2.0)
    gz = rng.normal(0, 2.0)

    return [
        ax, ay, az,
        gx, gy, gz
    ]


# ============================================================
# SIMULATED FALL
# ============================================================

def fall_window():

    samples = []

    for i in range(WINDOW_SIZE):

        # Before impact
        if i < 6:

            ax = rng.normal(0, 1.5)
            ay = rng.normal(0, 1.5)
            az = 9.81 + rng.normal(0, 1.2)

            gx = rng.normal(0, 3)
            gy = rng.normal(0, 3)
            gz = rng.normal(0, 3)

        # Impact
        elif i < 9:

            ax = rng.normal(0, 8)
            ay = rng.normal(0, 8)
            az = rng.normal(25, 8)

            gx = rng.normal(0, 25)
            gy = rng.normal(0, 25)
            gz = rng.normal(0, 25)

        # After impact
        else:

            ax = rng.normal(0, 0.8)
            ay = rng.normal(0, 0.8)
            az = 9.81 + rng.normal(0, 0.7)

            gx = rng.normal(0, 1)
            gy = rng.normal(0, 1)
            gz = rng.normal(0, 1)

        samples.append([
            ax, ay, az,
            gx, gy, gz
        ])

    return np.asarray(samples)


# ============================================================
# PREDICTION
# ============================================================

def predict(window):

    features = extract_features(
        window.reshape(1, WINDOW_SIZE, 6)
    )

    probability = model.predict_proba(features)[0, 1]

    prediction = probability >= THRESHOLD

    return probability, prediction


# ============================================================
# PHASE 1 — NORMAL ACTIVITY
# ============================================================

print("=" * 65)
print("PHASE 1: NORMAL ACTIVITY")
print("=" * 65)

buffer = []

for i in range(20):

    buffer.append(normal_sample())

buffer = np.asarray(buffer)

probability, prediction = predict(buffer)

print()
print(
    f"Fall probability: {probability:.3f}"
)

print(
    "Prediction:",
    "FALL DETECTED" if prediction else "NO FALL"
)


# ============================================================
# PHASE 2 — SIMULATED FALL
# ============================================================

print()
print("=" * 65)
print("PHASE 2: SIMULATED FALL")
print("=" * 65)

time.sleep(1)

fall = fall_window()

probability, prediction = predict(fall)

print()
print(
    f"Fall probability: {probability:.3f}"
)

print(
    "Prediction:",
    "FALL DETECTED" if prediction else "NO FALL"
)


# ============================================================
# PHASE 3 — RECOVERY
# ============================================================

print()
print("=" * 65)
print("PHASE 3: RECOVERY / NORMAL ACTIVITY")
print("=" * 65)

time.sleep(1)

buffer = []

for i in range(20):

    buffer.append(normal_sample())

buffer = np.asarray(buffer)

probability, prediction = predict(buffer)

print()
print(
    f"Fall probability: {probability:.3f}"
)

print(
    "Prediction:",
    "FALL DETECTED" if prediction else "NO FALL"
)


print()
print("=" * 65)
print("DEMO COMPLETE")
print("=" * 65)