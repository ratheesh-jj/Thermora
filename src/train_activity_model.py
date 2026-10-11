"""
THERMORA - Phase 1B
Deployment-oriented Activity Recognition

Dataset:
    WISDM Smartphone and Smartwatch Activity and Biometrics Dataset

Sensors:
    Watch accelerometer + gyroscope

Improvements over previous version:
    1. Correctly synchronizes ACC + GYRO
    2. Splits data into contiguous activity segments
    3. Prevents windows from crossing activity changes
    4. Uses participant-held-out evaluation
    5. Trains:
        - Random Forest
        - XGBoost
        - LightGBM
    6. Saves all results and models

Run from:
    C:\\Users\\OMEN\\Desktop\\Projects\\SIH26
"""

import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import signal
from scipy.stats import skew, kurtosis

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\OMEN\Desktop\Projects\SIH26")

ACCEL_DIR = PROJECT_ROOT / "wisdm_activity" / "raw" / "watch" / "accel"
GYRO_DIR = PROJECT_ROOT / "wisdm_activity" / "raw" / "watch" / "gyro"

OUTPUT_DIR = PROJECT_ROOT / "wisdm_activity" / "thermora_activity_models_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 10
OVERLAP = 0.5

# WISDM watch data is approximately 20 Hz
SAMPLE_RATE = 20

# Maximum allowed time difference between ACC and GYRO samples
SYNC_TOLERANCE_NS = 50_000_000  # 50 ms

RANDOM_STATE = 42


# ============================================================
# WISDM ACTIVITY MAP
# ============================================================

# WISDM activity codes.
#
# We keep the codes as the primary labels.
# Names are included for readability.

ACTIVITY_MAP = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Sitting",
    "E": "Standing",
    "F": "Typing",
    "G": "Brushing Teeth",
    "H": "Eating Soup",
    "I": "Eating Chips",
    "J": "Eating Pasta",
    "K": "Drinking",
    "L": "Eating Sandwich",
    "M": "Kicking",
    "O": "Catching",
    "P": "Dribbling Basketball",
    "Q": "Writing",
    "R": "Folding Clothes",
    "S": "Cleaning",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_files(directory):
    """Find all CSV/TXT files recursively."""

    files = []

    for ext in ["*.csv", "*.txt"]:
        files.extend(directory.rglob(ext))

    return sorted(files)


def extract_participant_id(filename):
    """
    Extract participant ID from filename.

    Expected examples:
        1600.csv
        1600_accel.csv
        accel_1600.txt
    """

    name = Path(filename).stem

    matches = re.findall(r"\d+", name)

    if not matches:
        return None

    # Usually the participant ID is the only number.
    # WISDM watch participants are around 1600-1650.
    for value in matches:
        number = int(value)

        if 1000 <= number <= 2000:
            return number

    return int(matches[0])


def read_wisdm_file(filepath):
    """
    Read WISDM raw sensor file.

    Expected format:

        subject,activity,timestamp,x,y,z

    Some files may use whitespace or semicolon separators.
    """

    rows = []

    with open(filepath, "r", errors="ignore") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # Remove trailing semicolon
            line = line.rstrip(";")

            # Handle comma / semicolon / whitespace
            parts = re.split(r"[,\s;]+", line)

            if len(parts) < 6:
                continue

            try:

                subject = int(float(parts[0]))
                activity = str(parts[1]).strip()

                timestamp = int(float(parts[2]))

                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])

                rows.append(
                    [
                        subject,
                        activity,
                        timestamp,
                        x,
                        y,
                        z
                    ]
                )

            except (ValueError, TypeError):
                continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject",
                "activity",
                "timestamp",
                "x",
                "y",
                "z"
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "subject",
            "activity",
            "timestamp",
            "x",
            "y",
            "z"
        ]
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# DISCOVER SENSOR FILES
# ============================================================

def discover_sensor_files(directory):

    files = find_files(directory)

    mapping = {}

    for filepath in files:

        participant = extract_participant_id(filepath.name)

        if participant is None:
            continue

        mapping[participant] = filepath

    return mapping


# ============================================================
# SYNCHRONIZATION
# ============================================================

def synchronize_sensor_data(acc_df, gyro_df):

    if acc_df.empty or gyro_df.empty:
        return pd.DataFrame()

    # Rename sensor columns
    acc = acc_df.rename(
        columns={
            "x": "acc_x",
            "y": "acc_y",
            "z": "acc_z"
        }
    )

    gyro = gyro_df.rename(
        columns={
            "x": "gyro_x",
            "y": "gyro_y",
            "z": "gyro_z"
        }
    )

    # IMPORTANT:
    # Keep activity only from accelerometer.
    # This prevents activity_x/activity_y problems.
    acc = acc[
        [
            "subject",
            "activity",
            "timestamp",
            "acc_x",
            "acc_y",
            "acc_z"
        ]
    ]

    gyro = gyro[
        [
            "subject",
            "timestamp",
            "gyro_x",
            "gyro_y",
            "gyro_z"
        ]
    ]

    acc = acc.sort_values("timestamp")
    gyro = gyro.sort_values("timestamp")

    merged = pd.merge_asof(
        acc,
        gyro,
        on="timestamp",
        by="subject",
        direction="nearest",
        tolerance=SYNC_TOLERANCE_NS
    )

    merged = merged.dropna(
        subset=[
            "gyro_x",
            "gyro_y",
            "gyro_z"
        ]
    )

    return merged.reset_index(drop=True)


# ============================================================
# CONTIGUOUS ACTIVITY SEGMENTS
# ============================================================

def split_contiguous_segments(df):

    """
    Split data whenever:

        1. activity changes
        2. timestamp gap is too large

    This is the key fix compared with the previous script.
    """

    if df.empty:
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Timestamp difference in nanoseconds
    timestamp_diff = df["timestamp"].diff()

    # 20 Hz -> approximately 50 ms per sample.
    # Allow some tolerance.
    max_gap_ns = int(0.20 * 1e9)

    activity_change = df["activity"].ne(
        df["activity"].shift()
    )

    time_gap = timestamp_diff > max_gap_ns

    split_points = activity_change | time_gap

    segment_id = split_points.cumsum()

    segments = []

    for _, segment in df.groupby(segment_id):

        segment = segment.reset_index(drop=True)

        if len(segment) >= int(WINDOW_SECONDS * SAMPLE_RATE):

            segments.append(segment)

    return segments


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def zero_crossing_rate(x):

    x = np.asarray(x)

    if len(x) < 2:
        return 0.0

    return np.mean(
        np.diff(np.signbit(x)) != 0
    )


def dominant_frequency(x, fs):

    if len(x) < 4:
        return 0.0

    x = np.asarray(x)

    x = x - np.mean(x)

    frequencies = np.fft.rfftfreq(
        len(x),
        d=1 / fs
    )

    spectrum = np.abs(
        np.fft.rfft(x)
    )

    if len(spectrum) <= 1:
        return 0.0

    spectrum[0] = 0

    index = np.argmax(spectrum)

    return float(frequencies[index])


def spectral_centroid(x, fs):

    x = np.asarray(x)

    if len(x) < 4:
        return 0.0

    x = x - np.mean(x)

    spectrum = np.abs(
        np.fft.rfft(x)
    )

    frequencies = np.fft.rfftfreq(
        len(x),
        d=1 / fs
    )

    total = np.sum(spectrum)

    if total == 0:
        return 0.0

    return float(
        np.sum(
            frequencies * spectrum
        ) / total
    )


def extract_axis_features(x, prefix, fs):

    x = np.asarray(x)

    features = {}

    features[f"{prefix}_mean"] = np.mean(x)
    features[f"{prefix}_std"] = np.std(x)
    features[f"{prefix}_min"] = np.min(x)
    features[f"{prefix}_max"] = np.max(x)

    features[f"{prefix}_median"] = np.median(x)

    features[f"{prefix}_q25"] = np.percentile(
        x, 25
    )

    features[f"{prefix}_q75"] = np.percentile(
        x, 75
    )

    features[f"{prefix}_rms"] = np.sqrt(
        np.mean(x ** 2)
    )

    features[f"{prefix}_range"] = (
        np.max(x) - np.min(x)
    )

    features[f"{prefix}_zcr"] = zero_crossing_rate(x)

    features[f"{prefix}_skew"] = skew(
        x,
        bias=False
    )

    features[f"{prefix}_kurtosis"] = kurtosis(
        x,
        bias=False
    )

    features[
        f"{prefix}_dominant_freq"
    ] = dominant_frequency(
        x,
        fs
    )

    features[
        f"{prefix}_spectral_centroid"
    ] = spectral_centroid(
        x,
        fs
    )

    return features


def extract_window_features(window):

    features = {}

    sensor_columns = {
        "acc_x": "acc_x",
        "acc_y": "acc_y",
        "acc_z": "acc_z",
        "gyro_x": "gyro_x",
        "gyro_y": "gyro_y",
        "gyro_z": "gyro_z"
    }

    # --------------------------------------------------------
    # Individual axes
    # --------------------------------------------------------

    for column, prefix in sensor_columns.items():

        values = window[column].values

        features.update(
            extract_axis_features(
                values,
                prefix,
                SAMPLE_RATE
            )
        )

    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    acc_mag = np.sqrt(
        window["acc_x"] ** 2 +
        window["acc_y"] ** 2 +
        window["acc_z"] ** 2
    ).values

    gyro_mag = np.sqrt(
        window["gyro_x"] ** 2 +
        window["gyro_y"] ** 2 +
        window["gyro_z"] ** 2
    ).values

    features.update(
        extract_axis_features(
            acc_mag,
            "acc_mag",
            SAMPLE_RATE
        )
    )

    features.update(
        extract_axis_features(
            gyro_mag,
            "gyro_mag",
            SAMPLE_RATE
        )
    )

    return features


# ============================================================
# WINDOW GENERATION
# ============================================================

def create_windows_for_segment(
    segment,
    participant_id
):

    window_size = int(
        WINDOW_SECONDS * SAMPLE_RATE
    )

    step_size = int(
        window_size * (1 - OVERLAP)
    )

    windows = []

    if len(segment) < window_size:
        return windows

    for start in range(
        0,
        len(segment) - window_size + 1,
        step_size
    ):

        end = start + window_size

        window = segment.iloc[
            start:end
        ]

        # Extra timestamp validation
        duration_ns = (
            window["timestamp"].iloc[-1]
            -
            window["timestamp"].iloc[0]
        )

        # Expected duration ~9.95 seconds
        # Allow broad tolerance.
        if duration_ns > 15_000_000_000:
            continue

        features = extract_window_features(
            window
        )

        features["participant"] = participant_id

        # Since segment is guaranteed to contain
        # only one activity:
        activity = window["activity"].mode()[0]

        features["activity"] = activity

        windows.append(features)

    return windows


# ============================================================
# BUILD DATASET
# ============================================================

def build_feature_dataset(
    accel_files,
    gyro_files
):

    all_features = []

    paired_participants = sorted(
        set(accel_files.keys())
        &
        set(gyro_files.keys())
    )

    print()
    print("=" * 70)
    print("PAIRED PARTICIPANTS")
    print("=" * 70)
    print(
        f"Accelerometer participants : "
        f"{len(accel_files)}"
    )

    print(
        f"Gyroscope participants     : "
        f"{len(gyro_files)}"
    )

    print(
        f"Paired participants        : "
        f"{len(paired_participants)}"
    )

    print()

    for index, participant in enumerate(
        paired_participants,
        start=1
    ):

        print(
            f"[{index:02d}/{len(paired_participants)}] "
            f"Participant {participant}",
            end=" ... "
        )

        try:

            acc_df = read_wisdm_file(
                accel_files[participant]
            )

            gyro_df = read_wisdm_file(
                gyro_files[participant]
            )

            if acc_df.empty or gyro_df.empty:
                print("SKIPPED - empty file")
                continue

            synced = synchronize_sensor_data(
                acc_df,
                gyro_df
            )

            if synced.empty:
                print("SKIPPED - no synchronized data")
                continue

            segments = split_contiguous_segments(
                synced
            )

            participant_windows = []

            for segment in segments:

                windows = create_windows_for_segment(
                    segment,
                    participant
                )

                participant_windows.extend(
                    windows
                )

            all_features.extend(
                participant_windows
            )

            print(
                f"{len(participant_windows)} windows"
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

    if not all_features:
        raise RuntimeError(
            "No windows were generated."
        )

    features_df = pd.DataFrame(
        all_features
    )

    return features_df


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(features_df):

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Windows      : {len(features_df):,}"
    )

    print(
        f"Participants : "
        f"{features_df['participant'].nunique()}"
    )

    print(
        f"Activities   : "
        f"{features_df['activity'].nunique()}"
    )

    print()

    # --------------------------------------------------------
    # Save raw features
    # --------------------------------------------------------

    features_path = (
        OUTPUT_DIR /
        "activity_features_v2.csv"
    )

    features_df.to_csv(
        features_path,
        index=False
    )

    # --------------------------------------------------------
    # Prepare X/y
    # --------------------------------------------------------

    groups = features_df["participant"].values

    y_raw = features_df["activity"].values

    # Encode labels
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_raw
    )

    label_names = list(
        label_encoder.classes_
    )

    # Feature columns
    excluded = [
        "participant",
        "activity"
    ]

    feature_columns = [
        col
        for col in features_df.columns
        if col not in excluded
    ]

    X = features_df[
        feature_columns
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # Participant-held-out split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]

    train_groups = groups[
        train_idx
    ]

    test_groups = groups[
        test_idx
    ]

    print("=" * 70)
    print("PARTICIPANT-HELD-OUT SPLIT")
    print("=" * 70)

    print(
        f"Training windows : "
        f"{len(X_train):,}"
    )

    print(
        f"Testing windows  : "
        f"{len(X_test):,}"
    )

    print(
        f"Training people  : "
        f"{len(np.unique(train_groups))}"
    )

    print(
        f"Testing people   : "
        f"{len(np.unique(test_groups))}"
    )

    print()

    print(
        "Test participants:"
    )

    print(
        sorted(
            np.unique(test_groups)
        )
    )

    # Save configuration
    config = {
        "window_seconds": WINDOW_SECONDS,
        "overlap": OVERLAP,
        "sample_rate": SAMPLE_RATE,
        "sync_tolerance_ms": 50,
        "num_features": len(feature_columns),
        "num_classes": len(label_names),
        "label_mapping": {
            str(code): ACTIVITY_MAP.get(
                code,
                code
            )
            for code in label_names
        },
        "train_participants": sorted(
            [int(x) for x in np.unique(train_groups)]
        ),
        "test_participants": sorted(
            [int(x) for x in np.unique(test_groups)]
        )
    }

    with open(
        OUTPUT_DIR /
        "training_config_v2.json",
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    # Save label mapping
    with open(
        OUTPUT_DIR /
        "activity_labels_v2.json",
        "w"
    ) as f:

        json.dump(
            {
                str(i): {
                    "code": code,
                    "name": ACTIVITY_MAP.get(
                        code,
                        code
                    )
                }
                for i, code in enumerate(
                    label_names
                )
            },
            f,
            indent=4
        )

    results = []

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_pred = rf.predict(
        X_test
    )

    rf_accuracy = accuracy_score(
        y_test,
        rf_pred
    )

    rf_f1 = f1_score(
        y_test,
        rf_pred,
        average="macro"
    )

    print(
        f"Accuracy : {rf_accuracy:.4f}"
    )

    print(
        f"Macro F1 : {rf_f1:.4f}"
    )

    results.append(
        {
            "model": "RandomForest",
            "accuracy": rf_accuracy,
            "macro_f1": rf_f1
        }
    )

    joblib.dump(
        rf,
        OUTPUT_DIR /
        "thermora_activity_rf_v2.pkl"
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    try:

        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        xgb.fit(
            X_train,
            y_train
        )

        xgb_pred = xgb.predict(
            X_test
        )

        xgb_accuracy = accuracy_score(
            y_test,
            xgb_pred
        )

        xgb_f1 = f1_score(
            y_test,
            xgb_pred,
            average="macro"
        )

        print(
            f"Accuracy : {xgb_accuracy:.4f}"
        )

        print(
            f"Macro F1 : {xgb_f1:.4f}"
        )

        results.append(
            {
                "model": "XGBoost",
                "accuracy": xgb_accuracy,
                "macro_f1": xgb_f1
            }
        )

        joblib.dump(
            xgb,
            OUTPUT_DIR /
            "thermora_activity_xgb_v2.pkl"
        )

    except ImportError:

        print(
            "XGBoost not installed. Skipping."
        )

        xgb_pred = None

    # ========================================================
    # LIGHTGBM
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING LIGHTGBM")
    print("=" * 70)

    try:

        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1
        )

        lgbm.fit(
            X_train,
            y_train
        )

        lgbm_pred = lgbm.predict(
            X_test
        )

        lgbm_accuracy = accuracy_score(
            y_test,
            lgbm_pred
        )

        lgbm_f1 = f1_score(
            y_test,
            lgbm_pred,
            average="macro"
        )

        print(
            f"Accuracy : {lgbm_accuracy:.4f}"
        )

        print(
            f"Macro F1 : {lgbm_f1:.4f}"
        )

        results.append(
            {
                "model": "LightGBM",
                "accuracy": lgbm_accuracy,
                "macro_f1": lgbm_f1
            }
        )

        joblib.dump(
            lgbm,
            OUTPUT_DIR /
            "thermora_activity_lgbm_v2.pkl"
        )

    except ImportError:

        print(
            "LightGBM not installed. Skipping."
        )

        lgbm_pred = None

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        OUTPUT_DIR /
        "model_comparison_v2.csv",
        index=False
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model_name = (
        results_df.iloc[0]["model"]
    )

    print()
    print("=" * 70)
    print(
        f"BEST MODEL: {best_model_name}"
    )
    print("=" * 70)

    # Select predictions
    if best_model_name == "RandomForest":

        best_model = rf
        best_pred = rf_pred

    elif best_model_name == "XGBoost":

        best_model = xgb
        best_pred = xgb_pred

    else:

        best_model = lgbm
        best_pred = lgbm_pred

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        best_pred,
        target_names=[
            ACTIVITY_MAP.get(
                code,
                code
            )
            for code in label_names
        ],
        zero_division=0
    )

    print()
    print(report)

    with open(
        OUTPUT_DIR /
        "classification_report_v2.txt",
        "w"
    ) as f:

        f.write(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        best_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            ACTIVITY_MAP.get(
                code,
                code
            )
            for code in label_names
        ],
        columns=[
            ACTIVITY_MAP.get(
                code,
                code
            )
            for code in label_names
        ]
    )

    cm_df.to_csv(
        OUTPUT_DIR /
        "confusion_matrix_v2.csv"
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    predictions_df = features_df.iloc[
        test_idx
    ][
        [
            "participant",
            "activity"
        ]
    ].copy()

    predictions_df[
        "predicted_activity"
    ] = [
        label_names[i]
        for i in best_pred
    ]

    predictions_df[
        "actual_activity_name"
    ] = predictions_df[
        "activity"
    ].map(ACTIVITY_MAP)

    predictions_df[
        "predicted_activity_name"
    ] = predictions_df[
        "predicted_activity"
    ].map(ACTIVITY_MAP)

    predictions_df.to_csv(
        OUTPUT_DIR /
        "test_predictions_v2.csv",
        index=False
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": best_model.feature_importances_
            }
        )

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        )

        importance_df.to_csv(
            OUTPUT_DIR /
            "feature_importance_v2.csv",
            index=False
        )

        print()
        print("=" * 70)
        print("TOP 20 FEATURES")
        print("=" * 70)

        print(
            importance_df.head(20).to_string(
                index=False
            )
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    best_accuracy = accuracy_score(
        y_test,
        best_pred
    )

    best_f1 = f1_score(
        y_test,
        best_pred,
        average="macro"
    )

    summary = {
        "best_model": best_model_name,
        "accuracy": float(best_accuracy),
        "macro_f1": float(best_f1),
        "num_windows": int(len(features_df)),
        "num_features": int(len(feature_columns)),
        "num_participants": int(
            features_df["participant"].nunique()
        ),
        "num_classes": int(
            features_df["activity"].nunique()
        ),
        "window_seconds": WINDOW_SECONDS,
        "overlap": OVERLAP,
        "sample_rate": SAMPLE_RATE
    }

    with open(
        OUTPUT_DIR /
        "final_summary_v2.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print("=" * 70)
    print("THERMORA PHASE 1B COMPLETE")
    print("=" * 70)

    print(
        f"Best model : {best_model_name}"
    )

    print(
        f"Accuracy   : {best_accuracy:.4f}"
    )

    print(
        f"Macro F1   : {best_f1:.4f}"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THERMORA - ACTIVITY RECOGNITION v2")
    print("=" * 70)

    print()
    print(
        "Accelerometer directory:"
    )

    print(
        ACCEL_DIR
    )

    print()
    print(
        "Gyroscope directory:"
    )

    print(
        GYRO_DIR
    )

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not ACCEL_DIR.exists():

        raise FileNotFoundError(
            f"Accelerometer directory not found:\n"
            f"{ACCEL_DIR}"
        )

    if not GYRO_DIR.exists():

        raise FileNotFoundError(
            f"Gyroscope directory not found:\n"
            f"{GYRO_DIR}"
        )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    accel_files = discover_sensor_files(
        ACCEL_DIR
    )

    gyro_files = discover_sensor_files(
        GYRO_DIR
    )

    print()
    print(
        f"Found {len(accel_files)} "
        f"accelerometer files."
    )

    print(
        f"Found {len(gyro_files)} "
        f"gyroscope files."
    )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    features_df = build_feature_dataset(
        accel_files,
        gyro_files
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_models(
        features_df
    )


if __name__ == "__main__":
    main()