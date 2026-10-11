import os
import re
import zipfile
import numpy as np
import pandas as pd

PROJECT_DIR = r"C:\Thermora Fall Prediction Model"
ZIP_FILE = os.path.join(PROJECT_DIR, "full_dataset.zip")

OUT_X = os.path.join(PROJECT_DIR, "thermora_raw_event_windows.npz")
OUT_META = os.path.join(PROJECT_DIR, "thermora_raw_event_metadata.csv")

FS = 20
WINDOW = 20          # 1 second
HALF = WINDOW // 2
RNG = np.random.default_rng(42)


def user_number(path):
    m = re.search(r"user(\d+)", path.lower())
    return int(m.group(1)) if m else -1


def read_recording(z, path):
    df = pd.read_csv(z.open(path))
    if "Unnamed: 5" not in df.columns:
        return None

    sensor = df["Unnamed: 5"].astype(str).str.lower()
    acc = df.loc[sensor == "acc", ["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").dropna()
    gyro = df.loc[sensor == "gyro", ["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").dropna()

    n = min(len(acc), len(gyro))
    if n < WINDOW:
        return None

    acc = acc.iloc[:n].to_numpy(dtype=np.float32)
    gyro = gyro.iloc[:n].to_numpy(dtype=np.float32)
    return np.hstack([acc, gyro])


def peak_candidates(data):
    acc = data[:, :3]
    gyro = data[:, 3:]
    am = np.linalg.norm(acc, axis=1)
    gm = np.linalg.norm(gyro, axis=1)

    # Robust normalization so users/recordings with different scales
    # do not dominate the score.
    def robust_z(v):
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        scale = 1.4826 * mad
        if scale < 1e-6:
            scale = np.std(v) + 1e-6
        return (v - med) / scale

    score = robust_z(am) + 0.5 * robust_z(gm)

    # Prefer local maxima, separated by at least 1 second.
    order = np.argsort(score)[::-1]
    chosen = []
    min_sep = FS

    for idx in order:
        idx = int(idx)
        if idx < HALF or idx >= len(data) - HALF:
            continue
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)
        if len(chosen) == 3:
            break

    return chosen


def make_window(data, center):
    start = center - HALF
    end = start + WINDOW
    if start < 0 or end > len(data):
        return None
    return data[start:end]


def main():
    print("=" * 65)
    print("THERMORA RAW BITS-2 EVENT WINDOW BUILDER")
    print("=" * 65)

    if not os.path.exists(ZIP_FILE):
        print("\nERROR: full_dataset.zip was not found.")
        print("Put full_dataset.zip directly inside:")
        print(PROJECT_DIR)
        return

    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        all_csv = [n for n in z.namelist() if n.lower().endswith(".csv")]
        fall_files = [n for n in all_csv if "/fall/" in n.lower()]
        adl_files = [n for n in all_csv if "/adl/" in n.lower()]

        print(f"\nRaw CSV files found: {len(all_csv)}")
        print(f"Fall recordings    : {len(fall_files)}")
        print(f"ADL recordings     : {len(adl_files)}")

        X = []
        y = []
        meta = []

        # Positive event candidates from every fall recording.
        for i, path in enumerate(sorted(fall_files)):
            data = read_recording(z, path)
            if data is None:
                continue

            candidates = peak_candidates(data)

            for rank, center in enumerate(candidates, start=1):
                w = make_window(data, center)
                if w is None:
                    continue

                X.append(w)
                y.append(1)
                meta.append({
                    "user": user_number(path),
                    "file": os.path.basename(path),
                    "source": "fall",
                    "label": 1,
                    "candidate_rank": rank,
                    "center_sample": center
                })

            if (i + 1) % 50 == 0:
                print(f"Processed fall recordings: {i + 1}/{len(fall_files)}")

        positive_count = len(X)

        # Negative windows from ADLs. We sample several windows per ADL
        # recording, avoiding excessive duplication.
        for i, path in enumerate(sorted(adl_files)):
            data = read_recording(z, path)
            if data is None:
                continue

            possible = np.arange(HALF, len(data) - HALF)
            if len(possible) == 0:
                continue

            # Up to 6 random windows per ADL recording.
            k = min(6, len(possible))
            centers = RNG.choice(possible, size=k, replace=False)

            for center in centers:
                w = make_window(data, int(center))
                if w is None:
                    continue

                X.append(w)
                y.append(0)
                meta.append({
                    "user": user_number(path),
                    "file": os.path.basename(path),
                    "source": "adl",
                    "label": 0,
                    "candidate_rank": 0,
                    "center_sample": int(center)
                })

            if (i + 1) % 100 == 0:
                print(f"Processed ADL recordings: {i + 1}/{len(adl_files)}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    meta = pd.DataFrame(meta)

    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Fall-event candidates:", int((y == 1).sum()))
    print("ADL/non-event windows :", int((y == 0).sum()))

    print("\nUsers:")
    print(sorted(meta["user"].unique().tolist()))

    # Keep the same participant-independent split used previously.
    train_users = set(range(1, 30))
    val_users = set(range(30, 36))
    test_users = set(range(36, 42))

    for name, users in [
        ("TRAIN", train_users),
        ("VAL", val_users),
        ("TEST", test_users)
    ]:
        mask = meta["user"].isin(users)
        print(
            f"{name:5s}: windows={int(mask.sum()):6d} "
            f"fall={int(((y == 1) & mask.to_numpy()).sum()):5d} "
            f"nonfall={int(((y == 0) & mask.to_numpy()).sum()):5d}"
        )

    np.savez_compressed(OUT_X, X=X, y=y)
    meta.to_csv(OUT_META, index=False)

    print("\nSaved:")
    print(OUT_X)
    print(OUT_META)

    print("\nIMPORTANT:")
    print("Fall positives are candidate event windows detected from raw")
    print("fall recordings. BITS-2 does not provide explicit event")
    print("timestamps in this ZIP, so these are NOT confirmed ground truth.")
    print("ADL windows are genuine non-fall activity recordings.")
    print("The participant split is preserved to prevent user leakage.")


if __name__ == "__main__":
    main()
