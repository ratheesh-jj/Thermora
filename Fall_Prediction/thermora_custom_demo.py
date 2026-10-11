import numpy as np
import joblib
import os


MODEL_FILE = "thermora_event_model_v2.pkl"
DATA_FILE = "thermora_raw_event_windows_v2.npz"
META_FILE = "thermora_raw_event_metadata_v2.csv"


# ============================================================
# EXACT FEATURE EXTRACTOR USED DURING MODEL TRAINING
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
# LOAD MODEL + REAL BITS-2 DATA
# ============================================================

if not os.path.exists(MODEL_FILE):
    print("ERROR: Model file not found:", MODEL_FILE)
    input("\nPress Enter to exit...")
    raise SystemExit

if not os.path.exists(DATA_FILE):
    print("ERROR: Dataset file not found:", DATA_FILE)
    input("\nPress Enter to exit...")
    raise SystemExit


model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
threshold = model_data["threshold"]

data = np.load(DATA_FILE)

X = data["X"]
y = data["y"]


print("\n==============================================")
print("        THERMORA FALL DETECTION DEMO")
print("==============================================")
print("Model loaded successfully")
print("Threshold:", threshold)
print("Dataset:", X.shape)
print("==============================================\n")


# ============================================================
# MENU
# ============================================================

while True:

    print("\nChoose an option:")
    print("1. Test NORMAL sample")
    print("2. Test FALL sample")
    print("3. Enter CUSTOM 20x6 IMU values")
    print("4. Exit")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":

        # Find an actual non-fall BITS-2 window
        indices = np.where(y == 0)[0]

        idx = indices[0]

        sample = X[idx:idx + 1]

        features = extract_features(sample)
        probability = model.predict_proba(features)[0, 1]

        result = "FALL DETECTED" if probability >= threshold else "NO FALL"

        print("\n----------------------------------------------")
        print("NORMAL SAMPLE")
        print("----------------------------------------------")
        print(f"Fall probability : {probability:.3f}")
        print(f"Threshold        : {threshold:.3f}")
        print(f"Prediction       : {result}")
        print("----------------------------------------------")

    elif choice == "2":

        # Find an actual fall-event candidate from BITS-2
        indices = np.where(y == 1)[0]

        idx = indices[0]

        sample = X[idx:idx + 1]

        features = extract_features(sample)
        probability = model.predict_proba(features)[0, 1]

        result = "FALL DETECTED" if probability >= threshold else "NO FALL"

        print("\n----------------------------------------------")
        print("FALL SAMPLE")
        print("----------------------------------------------")
        print(f"Fall probability : {probability:.3f}")
        print(f"Threshold        : {threshold:.3f}")
        print(f"Prediction       : {result}")
        print("----------------------------------------------")

    elif choice == "3":

        print("\nEnter 20 rows.")
        print("Each row must contain:")
        print("ax ay az gx gy gz")
        print("\nExample:")
        print("0.2 -0.4 9.8 1.2 0.5 -0.8\n")

        rows = []

        for i in range(20):

            while True:

                try:
                    values = input(f"Row {i + 1}/20: ").split()

                    if len(values) != 6:
                        print("Please enter exactly 6 values.")
                        continue

                    values = [float(v) for v in values]

                    rows.append(values)
                    break

                except ValueError:
                    print("Invalid numbers. Try again.")

        sample = np.array(rows, dtype=np.float32).reshape(1, 20, 6)

        features = extract_features(sample)

        probability = model.predict_proba(features)[0, 1]

        result = "FALL DETECTED" if probability >= threshold else "NO FALL"

        print("\n----------------------------------------------")
        print("CUSTOM IMU INPUT")
        print("----------------------------------------------")
        print(f"Fall probability : {probability:.3f}")
        print(f"Threshold        : {threshold:.3f}")
        print(f"Prediction       : {result}")
        print("----------------------------------------------")

    elif choice == "4":

        print("\nExiting THERMORA demo.")
        break

    else:

        print("Invalid choice. Enter 1, 2, 3 or 4.")