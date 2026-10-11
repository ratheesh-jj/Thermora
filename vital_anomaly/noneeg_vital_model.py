from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# THERMORA - NON-EEG VITAL STATE MODEL V2
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "noneeg"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WINDOW_SECONDS = 10


# ============================================================
# LOAD RECORD
# ============================================================

def load_record(subject, suffix):

    path = DATA_DIR / f"Subject{subject}_{suffix}"

    record = wfdb.rdrecord(str(path))

    df = pd.DataFrame(
        record.p_signal,
        columns=record.sig_name
    )

    df["time_sec"] = (
        np.arange(len(df)) / record.fs
    )

    return df, record.fs


# ============================================================
# READ ANNOTATIONS
# ============================================================

def read_annotations(subject):

    path = DATA_DIR / f"Subject{subject}_AccTempEDA"

    try:
        ann = wfdb.rdann(
            str(path),
            "atr"
        )
    except Exception as e:

        print(
            f"Annotation error for Subject {subject}: {e}"
        )

        return []

    labels = []

    # The actual state names are stored in aux_note.
    for sample, note in zip(
        ann.sample,
        ann.aux_note
    ):

        if note is None:
            continue

        note = str(note).strip()

        if not note:
            continue

        time_sec = sample / 8.0

        labels.append(
            (time_sec, note)
        )

    return labels


# ============================================================
# MAP ANNOTATION TEXT TO STATE
# ============================================================

def normalize_label(text):

    text = str(text).strip().lower()

    if text == "relax":
        return "RELAX"

    if text == "physicalstress":
        return "PHYSICAL_STRESS"

    if text == "emotionalstress":
        return "EMOTIONAL_STRESS"

    if text == "cognitivestress":
        return "COGNITIVE_STRESS"

    return None


def build_label_intervals(subject):

    annotations = read_annotations(subject)

    intervals = []

    current_label = None
    current_time = None

    for time_sec, symbol in annotations:

        label = normalize_label(symbol)

        if label is None:
            continue

        if current_label is not None:

            intervals.append(
                (
                    current_time,
                    time_sec,
                    current_label
                )
            )

        current_label = label
        current_time = time_sec

    if current_label is not None:

        # Use end of recording as final endpoint.
        try:
            record, _ = load_record(
                subject,
                "AccTempEDA"
            )

            end_time = record["time_sec"].iloc[-1]

            intervals.append(
                (
                    current_time,
                    end_time,
                    current_label
                )
            )

        except Exception:
            pass

    return intervals


# ============================================================
# FIND LABEL FOR WINDOW
# ============================================================

def label_window(
    center_time,
    intervals
):

    for start, end, label in intervals:

        if start <= center_time < end:

            return label

    return None


# ============================================================
# PROCESS SUBJECT
# ============================================================

def process_subject(subject):

    print(
        f"[{subject:02d}/20] Subject{subject}"
    )

    acc_df, acc_fs = load_record(
        subject,
        "AccTempEDA"
    )

    vital_df, vital_fs = load_record(
        subject,
        "SpO2HR"
    )

    intervals = build_label_intervals(
        subject
    )

    # --------------------------------------------------------
    # Synchronize HR and SpO2 onto 8-Hz timeline
    # --------------------------------------------------------

    time = acc_df["time_sec"].values

    acc_df["hr"] = np.interp(
        time,
        vital_df["time_sec"],
        vital_df["hr"]
    )

    acc_df["SpO2"] = np.interp(
        time,
        vital_df["time_sec"],
        vital_df["SpO2"]
    )

    acc_df["subject"] = subject

    # --------------------------------------------------------
    # Windowing
    # --------------------------------------------------------

    samples_per_window = int(
        acc_fs * WINDOW_SECONDS
    )

    step = samples_per_window // 2

    rows = []

    for start in range(
        0,
        len(acc_df) - samples_per_window + 1,
        step
    ):

        w = acc_df.iloc[
            start:start + samples_per_window
        ]

        center_time = float(
            w["time_sec"].iloc[len(w) // 2]
        )

        label = label_window(
            center_time,
            intervals
        )

        if label is None:
            continue

        ax = w["ax"].values
        ay = w["ay"].values
        az = w["az"].values

        acc_mag = np.sqrt(
            ax**2 +
            ay**2 +
            az**2
        )

        row = {

            "subject": subject,

            "time_sec": center_time,

            "label": label,

            # -------------------------
            # HR
            # -------------------------

            "hr_mean": w["hr"].mean(),
            "hr_std": w["hr"].std(),
            "hr_min": w["hr"].min(),
            "hr_max": w["hr"].max(),

            # -------------------------
            # SpO2
            # -------------------------

            "spo2_mean": w["SpO2"].mean(),
            "spo2_std": w["SpO2"].std(),
            "spo2_min": w["SpO2"].min(),

            # -------------------------
            # Temperature
            # -------------------------

            "temp_mean": w["temp"].mean(),
            "temp_std": w["temp"].std(),
            "temp_min": w["temp"].min(),
            "temp_max": w["temp"].max(),

            # -------------------------
            # Acceleration
            # -------------------------

            "acc_x_rms": np.sqrt(
                np.mean(ax**2)
            ),

            "acc_y_rms": np.sqrt(
                np.mean(ay**2)
            ),

            "acc_z_rms": np.sqrt(
                np.mean(az**2)
            ),

            "acc_mag_mean":
                acc_mag.mean(),

            "acc_mag_std":
                acc_mag.std(),

            # -------------------------
            # EDA
            # -------------------------

            "eda_mean":
                w["EDA"].mean(),

            "eda_std":
                w["EDA"].std(),

            "eda_min":
                w["EDA"].min(),

            "eda_max":
                w["EDA"].max()
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "THERMORA - NON-EEG VITAL STATE MODEL V2"
    )
    print("=" * 70)

    all_data = []

    for subject in range(1, 21):

        try:

            result = process_subject(
                subject
            )

            if len(result) > 0:

                all_data.append(result)

        except Exception as e:

            print(
                f"ERROR Subject {subject}: {e}"
            )

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    print("\n" + "=" * 70)
    print("LABELED DATASET")
    print("=" * 70)

    print(
        f"Windows: {len(data):,}"
    )

    print(
        f"Subjects: {data['subject'].nunique()}"
    )

    print("\nClass distribution:")

    print(
        data["label"].value_counts()
    )

    feature_columns = [
        c for c in data.columns
        if c not in [
            "subject",
            "time_sec",
            "label"
        ]
    ]

    # --------------------------------------------------------
    # Participant-held-out split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            data,
            groups=data["subject"]
        )
    )

    train = data.iloc[
        train_idx
    ].copy()

    test = data.iloc[
        test_idx
    ].copy()

    print("\nTrain subjects:")
    print(
        sorted(
            train["subject"].unique()
        )
    )

    print("\nTest subjects:")
    print(
        sorted(
            test["subject"].unique()
        )
    )

    # --------------------------------------------------------
    # Feature groups
    # --------------------------------------------------------

    groups = {

        "HR + SpO2": [
            c for c in feature_columns
            if c.startswith("hr_")
            or c.startswith("spo2_")
        ],

        "HR + SpO2 + Temperature": [
            c for c in feature_columns
            if c.startswith("hr_")
            or c.startswith("spo2_")
            or c.startswith("temp_")
        ],

        "All wearable signals": feature_columns
    }

    results = []

    for group_name, features in groups.items():

        print("\n" + "=" * 70)
        print(group_name)
        print("=" * 70)

        X_train = train[features]
        y_train = train["label"]

        X_test = test[features]
        y_test = test["label"]

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        rf.fit(
            X_train,
            y_train
        )

        pred_rf = rf.predict(
            X_test
        )

        rf_acc = accuracy_score(
            y_test,
            pred_rf
        )

        rf_f1 = f1_score(
            y_test,
            pred_rf,
            average="macro"
        )

        print(
            f"\nRandom Forest"
        )

        print(
            f"Accuracy: {rf_acc:.4f}"
        )

        print(
            f"Macro F1: {rf_f1:.4f}"
        )

        # ----------------------------------------------------
        # HistGradientBoosting
        # ----------------------------------------------------

        hgb = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42
        )

        # Encode labels
        classes = sorted(
            y_train.unique()
        )

        label_to_int = {
            label: i
            for i, label in enumerate(classes)
        }

        int_to_label = {
            i: label
            for label, i in label_to_int.items()
        }

        y_train_int = y_train.map(
            label_to_int
        )

        hgb.fit(
            X_train,
            y_train_int
        )

        pred_hgb_int = hgb.predict(
            X_test
        )

        pred_hgb = pd.Series(
            pred_hgb_int
        ).map(
            int_to_label
        )

        hgb_acc = accuracy_score(
            y_test,
            pred_hgb
        )

        hgb_f1 = f1_score(
            y_test,
            pred_hgb,
            average="macro"
        )

        print(
            f"\nHistGradientBoosting"
        )

        print(
            f"Accuracy: {hgb_acc:.4f}"
        )

        print(
            f"Macro F1: {hgb_f1:.4f}"
        )

        results.extend([
            {
                "feature_group": group_name,
                "model": "RandomForest",
                "accuracy": rf_acc,
                "macro_f1": rf_f1
            },
            {
                "feature_group": group_name,
                "model": "HistGradientBoosting",
                "accuracy": hgb_acc,
                "macro_f1": hgb_f1
            }
        ])

        # ----------------------------------------------------
        # Detailed report for all wearable signals
        # ----------------------------------------------------

        if group_name == "All wearable signals":

            print(
                "\nDetailed classification report:"
            )

            print(
                classification_report(
                    y_test,
                    pred_hgb,
                    zero_division=0
                )
            )

            predictions = test[
                [
                    "subject",
                    "time_sec",
                    "label"
                ]
            ].copy()

            predictions[
                "prediction"
            ] = pred_hgb.values

            predictions.to_csv(
                OUTPUT_DIR /
                "noneeg_vital_predictions.csv",
                index=False
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_DIR /
        "noneeg_vital_model_comparison.csv",
        index=False
    )

    data.to_csv(
        OUTPUT_DIR /
        "noneeg_vital_labeled_windows.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nFiles saved:")
    print(
        OUTPUT_DIR /
        "noneeg_vital_labeled_windows.csv"
    )

    print(
        OUTPUT_DIR /
        "noneeg_vital_model_comparison.csv"
    )

    print(
        OUTPUT_DIR /
        "noneeg_vital_predictions.csv"
    )


if __name__ == "__main__":
    main()