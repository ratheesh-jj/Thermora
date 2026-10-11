import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = r"C:\Users\OMEN\Desktop\Projects\SIH26\vital_anomaly\wearable_stress"
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

WINDOW_SEC = 60
MIN_IBI = 30

def read_simple_file(path):
    """
    Read Empatica-style CSV:
    first line = start timestamp / metadata
    following lines = numeric samples
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [x.strip() for x in f if x.strip()]

    if not lines:
        return None, np.array([])

    try:
        start_time = pd.to_datetime(lines[0].split(",")[0])
    except Exception:
        start_time = None

    values = []

    for line in lines[1:]:
        parts = line.split(",")

        try:
            nums = [float(x) for x in parts]
        except ValueError:
            continue

        # Ignore sampling-frequency metadata rows
        if len(nums) == 1:
            values.append(nums[0])

    return start_time, np.asarray(values, dtype=float)


def read_acc(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [x.strip() for x in f if x.strip()]

    if not lines:
        return None, np.empty((0, 3))

    try:
        start_time = pd.to_datetime(lines[0].split(",")[0])
    except Exception:
        start_time = None

    values = []

    for line in lines[1:]:
        parts = line.split(",")

        if len(parts) < 3:
            continue

        try:
            nums = [float(x) for x in parts[:3]]
        except ValueError:
            continue

        # Skip metadata row such as 32,32,32
        if all(abs(x - nums[0]) < 1e-9 for x in nums):
            continue

        values.append(nums)

    return start_time, np.asarray(values, dtype=float)


def read_ibi(path):
    """
    IBI.csv:
    first line = start timestamp
    following rows = elapsed time, IBI seconds
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [x.strip() for x in f if x.strip()]

    if len(lines) < 2:
        return None, np.empty((0, 2))

    try:
        start_time = pd.to_datetime(lines[0].split(",")[0])
    except Exception:
        start_time = None

    values = []

    for line in lines[1:]:
        parts = line.split(",")

        if len(parts) < 2:
            continue

        try:
            t = float(parts[0])
            ibi = float(parts[1])
        except ValueError:
            continue

        if 0 < ibi < 5:
            values.append([t, ibi])

    return start_time, np.asarray(values, dtype=float)


def safe_slope(y):
    if len(y) < 2:
        return np.nan

    x = np.arange(len(y))

    try:
        return np.polyfit(x, y, 1)[0]
    except Exception:
        return np.nan


def hrv_features(ibi_values):
    ibi_ms = np.asarray(ibi_values) * 1000.0

    if len(ibi_ms) < MIN_IBI:
        return {
            "ibi_count": len(ibi_ms),
            "mean_rr_ms": np.nan,
            "mean_hr_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "pnn50_pct": np.nan,
            "hrv_range_ms": np.nan,
        }

    diff = np.diff(ibi_ms)

    return {
        "ibi_count": len(ibi_ms),
        "mean_rr_ms": np.mean(ibi_ms),
        "mean_hr_bpm": 60000.0 / np.mean(ibi_ms),
        "sdnn_ms": np.std(ibi_ms, ddof=1),
        "rmssd_ms": np.sqrt(np.mean(diff ** 2)),
        "pnn50_pct": 100.0 * np.mean(np.abs(diff) > 50),
        "hrv_range_ms": np.max(ibi_ms) - np.min(ibi_ms),
    }


def aggregate_signal(values, prefix):
    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_std": np.std(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
    }


def process_subject(folder, condition, subject):
    files = {
        name: os.path.join(folder, name + ".csv")
        for name in ["HR", "TEMP", "EDA", "ACC", "IBI"]
    }

    if not all(os.path.exists(x) for x in files.values()):
        return []

    _, hr = read_simple_file(files["HR"])
    _, temp = read_simple_file(files["TEMP"])
    _, eda = read_simple_file(files["EDA"])
    _, acc = read_acc(files["ACC"])
    _, ibi = read_ibi(files["IBI"])

    if len(ibi) == 0:
        return []

    # IBI elapsed time determines our common timeline.
    start_t = ibi[:, 0].min()
    end_t = ibi[:, 0].max()

    rows = []

    t = start_t

    while t + WINDOW_SEC <= end_t:

        end = t + WINDOW_SEC

        ibi_mask = (ibi[:, 0] >= t) & (ibi[:, 0] < end)
        current_ibi = ibi[ibi_mask, 1]

        hrv = hrv_features(current_ibi)

        # We need sufficient RR intervals for meaningful HRV.
        if hrv["ibi_count"] < MIN_IBI:
            t += WINDOW_SEC
            continue

        row = {
            "condition": condition,
            "subject": subject,
            "window_start_sec": t,
            "window_end_sec": end,
        }

        row.update(hrv)

        # HR
        if len(hr) > 0:
            # approximate sample rate from recording size/duration
            idx1 = int((t / max(end_t, 1)) * len(hr))
            idx2 = int((end / max(end_t, 1)) * len(hr))

            h = hr[idx1:min(idx2, len(hr))]

            if len(h):
                row.update(aggregate_signal(h, "hr"))

        # TEMP
        if len(temp) > 0:
            idx1 = int((t / max(end_t, 1)) * len(temp))
            idx2 = int((end / max(end_t, 1)) * len(temp))

            x = temp[idx1:min(idx2, len(temp))]

            if len(x):
                row.update(aggregate_signal(x, "temp"))

        # EDA
        if len(eda) > 0:
            idx1 = int((t / max(end_t, 1)) * len(eda))
            idx2 = int((end / max(end_t, 1)) * len(eda))

            x = eda[idx1:min(idx2, len(eda))]

            if len(x):
                row.update(aggregate_signal(x, "eda"))

        # ACC
        if len(acc) > 0:
            idx1 = int((t / max(end_t, 1)) * len(acc))
            idx2 = int((end / max(end_t, 1)) * len(acc))

            x = acc[idx1:min(idx2, len(acc))]

            if len(x):
                mag = np.sqrt(np.sum(x ** 2, axis=1))

                row.update(aggregate_signal(x[:, 0], "acc_x"))
                row.update(aggregate_signal(x[:, 1], "acc_y"))
                row.update(aggregate_signal(x[:, 2], "acc_z"))
                row.update(aggregate_signal(mag, "acc_mag"))

        rows.append(row)

        t += WINDOW_SEC

    return rows


all_rows = []

for condition in ["STRESS", "AEROBIC", "ANAEROBIC"]:

    condition_dir = os.path.join(ROOT, condition)

    if not os.path.isdir(condition_dir):
        continue

    subjects = [
        x for x in os.listdir(condition_dir)
        if os.path.isdir(os.path.join(condition_dir, x))
    ]

    print(f"\n{condition}: {len(subjects)} subjects")

    for subject in sorted(subjects):

        folder = os.path.join(condition_dir, subject)

        rows = process_subject(
            folder,
            condition,
            subject
        )

        all_rows.extend(rows)

        print(
            f"  {subject}: {len(rows)} windows"
        )


df = pd.DataFrame(all_rows)

print("\n===================================")
print("FINAL DATASET")
print("===================================")

print("Shape:", df.shape)

print("\nConditions:")
print(df["condition"].value_counts())

print("\nSubjects:")
print(df["subject"].nunique())

out_path = os.path.join(
    OUT,
    "wearable_all_conditions_vital_features.csv"
)

df.to_csv(out_path, index=False)

print("\nSaved:")
print(out_path)