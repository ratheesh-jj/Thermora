import os
import glob
import numpy as np
import pandas as pd

# ============================================================
# BITS-2 FALL EVENT INSPECTOR
# ============================================================
# Put this script in:
# C:\Thermora Fall Prediction Model
#
# For now it can analyze the uploaded user5_fall*.csv files.
# It does NOT retrain the model or change your existing files.
# It finds candidate high-motion events using acceleration +
# gyroscope together.
# ============================================================

project_path = os.path.dirname(__file__)

files = glob.glob(os.path.join(project_path, "user5_fall*.csv"))

# Also allow the uploaded files to be inspected if copied here.
if not files:
    print("No user5_fall*.csv files found.")
    print("Copy the CSV files into the project folder and run again.")
    raise SystemExit

print("======================================")
print("BITS-2 FALL EVENT INSPECTION")
print("======================================")

for filepath in sorted(files):

    name = os.path.basename(filepath)

    df = pd.read_csv(filepath)

    df["sensor"] = df["Unnamed: 5"].astype(str).str.lower()

    acc = df[df["sensor"] == "acc"].copy()
    gyro = df[df["sensor"] == "gyro"].copy()

    if len(acc) < 10 or len(gyro) < 10:
        print(f"\n{name}: insufficient acc/gyro data")
        continue

    acc_xyz = acc[["x", "y", "z"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().to_numpy(dtype=float)

    gyro_xyz = gyro[["x", "y", "z"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna().to_numpy(dtype=float)

    # Use the common number of samples.
    n = min(len(acc_xyz), len(gyro_xyz))

    acc_xyz = acc_xyz[:n]
    gyro_xyz = gyro_xyz[:n]

    acc_mag = np.linalg.norm(acc_xyz, axis=1)
    gyro_mag = np.linalg.norm(gyro_xyz, axis=1)

    # Robust baseline statistics.
    acc_med = np.median(acc_mag)
    acc_mad = np.median(np.abs(acc_mag - acc_med)) + 1e-8

    gyro_med = np.median(gyro_mag)
    gyro_mad = np.median(np.abs(gyro_mag - gyro_med)) + 1e-8

    # Robust scores: useful because different recordings have
    # very different absolute sensor scales.
    acc_score = np.abs(acc_mag - acc_med) / (1.4826 * acc_mad)
    gyro_score = np.abs(gyro_mag - gyro_med) / (1.4826 * gyro_mad)

    # Combined motion score.
    motion_score = acc_score + gyro_score

    # Top candidate points, separated so we don't print the same
    # impact repeatedly.
    order = np.argsort(motion_score)[::-1]

    selected = []

    for idx in order:
        if all(abs(idx - old) >= 10 for old in selected):
            selected.append(int(idx))

        if len(selected) >= 5:
            break

    selected.sort()

    print(f"\n{name}")
    print("--------------------------------------")
    print("ACC samples :", len(acc_xyz))
    print("GYRO samples:", len(gyro_xyz))
    print("ACC magnitude max :", round(float(np.max(acc_mag)), 3))
    print("GYRO magnitude max:", round(float(np.max(gyro_mag)), 3))

    print("\nTop candidate events:")
    print("Candidate | Sample | AccMag | GyroMag | MotionScore")
    print("----------------------------------------------------")

    for rank, idx in enumerate(selected, start=1):
        print(
            f"{rank:9d} | "
            f"{idx:6d} | "
            f"{acc_mag[idx]:6.2f} | "
            f"{gyro_mag[idx]:7.2f} | "
            f"{motion_score[idx]:11.2f}"
        )

print("\n======================================")
print("DONE")
print("======================================")
print("These are CANDIDATE events, not confirmed labels.")
print("We will use them to understand the raw recordings before")
print("creating the new event-centered training dataset.")
