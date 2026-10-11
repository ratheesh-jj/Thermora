import zipfile
import re
import numpy as np
import pandas as pd

ZIP_PATH = "full_dataset.zip"

OUT_X = "thermora_raw_event_windows_v2.npz"
OUT_META = "thermora_raw_event_metadata_v2.csv"

FS = 20
WINDOW = 20
HALF = WINDOW // 2

# Number of negative windows per ADL recording
ADL_WINDOWS = 6

TRAIN_USERS = set(range(1, 30))
VAL_USERS = set(range(30, 36))
TEST_USERS = set(range(36, 42))


def get_user_id(filename):
    m = re.search(r"user(\d+)", filename.lower())
    return int(m.group(1)) if m else None


def read_motion(zf, filename):
    df = pd.read_csv(zf.open(filename))

    sensor = df["Unnamed: 5"].astype(str).str.lower()

    acc = df.loc[sensor == "acc", ["x", "y", "z"]].to_numpy(float)
    gyro = df.loc[sensor == "gyro", ["x", "y", "z"]].to_numpy(float)

    n = min(len(acc), len(gyro))

    if n < WINDOW:
        return None

    acc = np.nan_to_num(acc[:n])
    gyro = np.nan_to_num(gyro[:n])

    return np.concatenate([acc, gyro], axis=1)


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    if mad < 1e-9:
        return np.zeros_like(x)

    return (x - med) / (1.4826 * mad)


def motion_score(signal):
    acc = signal[:, :3]
    gyro = signal[:, 3:]

    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    acc_change = np.r_[0, np.abs(np.diff(acc_mag))]

    az = np.maximum(robust_z(acc_mag), 0)
    gz = np.maximum(robust_z(gyro_mag), 0)
    dz = np.maximum(robust_z(acc_change), 0)

    return az + 0.5 * gz + 0.5 * dz


def get_split(user):
    if user in TRAIN_USERS:
        return "train"
    elif user in VAL_USERS:
        return "val"
    elif user in TEST_USERS:
        return "test"
    return "unknown"


def make_window(signal, start):
    end = start + WINDOW

    if start < 0 or end > len(signal):
        return None

    return signal[start:end]


def strongest_fall_candidate(signal):
    score = motion_score(signal)

    # Small smoothing to reduce one-sample noise.
    smooth = np.convolve(
        score,
        np.ones(3) / 3,
        mode="same"
    )

    valid_start = HALF
    valid_end = len(signal) - HALF

    if valid_end <= valid_start:
        return None

    peak = valid_start + np.argmax(
        smooth[valid_start:valid_end]
    )

    return int(peak - HALF)


def fast_adl_candidates(signal):
    """
    Efficient ADL selection.

    Instead of checking every possible 1-second window,
    sample windows at 0.5-second intervals and rank them.
    """

    score = motion_score(signal)

    # Only inspect windows every HALF samples.
    starts = np.arange(
        0,
        len(signal) - WINDOW + 1,
        HALF
    )

    if len(starts) == 0:
        return []

    # Vectorized window scoring.
    windows = np.lib.stride_tricks.sliding_window_view(
        score,
        WINDOW
    )[starts]

    scores = (
        np.max(windows, axis=1)
        + 0.5 * np.std(windows, axis=1)
    )

    # Highest-motion ADL windows first.
    order = np.argsort(scores)[::-1]

    selected = []

    for idx in order:
        start = int(starts[idx])

        # Keep selected windows separated.
        if all(abs(start - old) >= WINDOW for old in selected):
            selected.append(start)

        if len(selected) >= ADL_WINDOWS:
            break

    return selected


# ============================================================
# MAIN
# ============================================================

print("=" * 65)
print("THERMORA OPTIMIZED RAW BITS-2 EVENT WINDOW BUILDER V2")
print("=" * 65)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:

    files = [
        name
        for name in zf.namelist()
        if name.lower().endswith(".csv")
    ]

    fall_files = [
        name
        for name in files
        if "/fall/" in name.lower()
    ]

    adl_files = [
        name
        for name in files
        if "/adl/" in name.lower()
    ]

    print()
    print("Raw CSV files found:", len(files))
    print("Fall recordings    :", len(fall_files))
    print("ADL recordings     :", len(adl_files))
    print()

    X_list = []
    y_list = []
    metadata = []

    # ========================================================
    # FALL RECORDINGS
    # ========================================================

    print("Processing FALL recordings...")

    fall_count = 0

    for filename in fall_files:

        user = get_user_id(filename)

        if user is None:
            continue

        signal = read_motion(zf, filename)

        if signal is None:
            continue

        start = strongest_fall_candidate(signal)

        if start is None:
            continue

        window = make_window(signal, start)

        if window is None:
            continue

        X_list.append(window.astype(np.float32))
        y_list.append(1)

        metadata.append({
            "user": user,
            "split": get_split(user),
            "label": 1,
            "source": "fall_candidate",
            "file": filename,
            "window_start_sample": start,
            "window_start_sec": start / FS
        })

        fall_count += 1

        if fall_count % 50 == 0:
            print(
                f"Processed fall recordings: "
                f"{fall_count}/{len(fall_files)}"
            )

    # ========================================================
    # ADL RECORDINGS
    # ========================================================

    print()
    print("Processing ADL recordings...")

    adl_count = 0

    for filename in adl_files:

        user = get_user_id(filename)

        if user is None:
            continue

        signal = read_motion(zf, filename)

        if signal is None:
            continue

        starts = fast_adl_candidates(signal)

        for start in starts:

            window = make_window(signal, start)

            if window is None:
                continue

            X_list.append(window.astype(np.float32))
            y_list.append(0)

            metadata.append({
                "user": user,
                "split": get_split(user),
                "label": 0,
                "source": "adl_hard_negative",
                "file": filename,
                "window_start_sample": start,
                "window_start_sec": start / FS
            })

        adl_count += 1

        if adl_count % 100 == 0:
            print(
                f"Processed ADL recordings: "
                f"{adl_count}/{len(adl_files)}"
            )


# ============================================================
# SAVE
# ============================================================

X = np.stack(X_list)
y = np.asarray(y_list, dtype=np.int64)

metadata = pd.DataFrame(metadata)

np.savez_compressed(
    OUT_X,
    X=X,
    y=y
)

metadata.to_csv(
    OUT_META,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 65)
print("DATASET SUMMARY")
print("=" * 65)

print("X shape:", X.shape)
print("y shape:", y.shape)

print(
    "Fall-event candidates:",
    int((y == 1).sum())
)

print(
    "ADL/non-event windows :",
    int((y == 0).sum())
)

print()

for split in ["train", "val", "test"]:

    mask = metadata["split"] == split

    falls = (
        metadata.loc[mask, "label"] == 1
    ).sum()

    nonfalls = (
        metadata.loc[mask, "label"] == 0
    ).sum()

    print(
        f"{split.upper():5s}: "
        f"windows={mask.sum():5d} "
        f"fall={falls:4d} "
        f"nonfall={nonfalls:4d}"
    )

print()
print("Saved:")
print(OUT_X)
print(OUT_META)

print()
print("IMPORTANT:")
print("Fall positives are motion-based candidate events.")
print("They are NOT confirmed BITS-2 event timestamps.")
print("Participant split is preserved.")