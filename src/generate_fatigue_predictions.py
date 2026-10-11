"""
THERMORA FATIGUE BATCH INFERENCE
=================================

Runs the validated THERMORA fatigue model across multiple
real MxD 3-minute IMU4 windows.

Pipeline:

MxD CSV
   ↓
3-minute window
   ↓
156 features
   ↓
SimpleImputer
   ↓
RandomForestRegressor
   ↓
Fatigue score 0-10
   ↓
SAFE / CAUTION / HIGH

This script is for model/inference validation.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26"
)

MODEL_DIR = (
    BASE_DIR
    / "fatigue_prediction"
    / "fatigue_deployable_model"
)

MODEL_FILE = (
    MODEL_DIR
    / "thermora_fatigue_rf.pkl"
)

IMPUTER_FILE = (
    MODEL_DIR
    / "thermora_fatigue_imputer.pkl"
)

FEATURE_FILE = (
    MODEL_DIR
    / "thermora_fatigue_features.json"
)

MXD_ROOT = (
    BASE_DIR
    / "fatigue_prediction"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "fatigue_batch_predictions.csv"
)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE_HZ = 2

WINDOW_SECONDS = 180

WINDOW_SAMPLES = (
    SAMPLE_RATE_HZ
    * WINDOW_SECONDS
)

# Use a 30-second stride, matching model_info.json.
STRIDE_SECONDS = 30

STRIDE_SAMPLES = (
    SAMPLE_RATE_HZ
    * STRIDE_SECONDS
)

# Maximum number of CSV files to process in this test.
# Set to None later if you want the complete dataset.
MAX_FILES = 30


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING FATIGUE MODEL")
    print("=" * 70)

    model = joblib.load(
        MODEL_FILE
    )

    imputer = joblib.load(
        IMPUTER_FILE
    )

    with open(
        FEATURE_FILE,
        "r"
    ) as f:

        feature_names = json.load(f)

    print(
        f"Model: {type(model).__name__}"
    )

    print(
        f"Trees: {model.n_estimators}"
    )

    print(
        f"Expected features: "
        f"{len(feature_names)}"
    )

    if model.n_features_in_ != len(
        feature_names
    ):

        raise RuntimeError(
            "Model feature count does not "
            "match feature JSON."
        )

    if imputer.n_features_in_ != len(
        feature_names
    ):

        raise RuntimeError(
            "Imputer feature count does not "
            "match feature JSON."
        )

    return (
        model,
        imputer,
        feature_names
    )


# ============================================================
# SENSOR CLEANING
# ============================================================

def clean_sensor_column(
    df,
    column
):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # MxD invalid sentinel
    values = values.replace(
        [-2, -2.0],
        np.nan
    )

    values = values.interpolate(
        limit_direction="both"
    )

    return values.to_numpy(
        dtype=float
    )


# ============================================================
# STATISTICS
# ============================================================

def safe_stats(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "rms": np.nan,
            "abs_mean": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "first": np.nan,
            "last": np.nan,
            "change": np.nan,
            "slope": np.nan,
        }

    mean = np.mean(values)

    std = np.std(values)

    if len(values) > 1:

        diff = np.diff(values)

        change = np.mean(
            np.abs(diff)
        )

        slope = np.polyfit(
            np.arange(len(values)),
            values,
            1
        )[0]

    else:

        change = 0.0
        slope = 0.0

    if std > 1e-12:

        z = (
            values - mean
        ) / std

        skew = np.mean(
            z ** 3
        )

        kurtosis = (
            np.mean(
                z ** 4
            ) - 3.0
        )

    else:

        skew = 0.0
        kurtosis = 0.0

    return {

        "mean": mean,
        "std": std,
        "median": np.median(values),
        "min": np.min(values),
        "max": np.max(values),

        "range":
            np.max(values)
            - np.min(values),

        "rms":
            np.sqrt(
                np.mean(
                    values ** 2
                )
            ),

        "abs_mean":
            np.mean(
                np.abs(values)
            ),

        "q25":
            np.percentile(
                values,
                25
            ),

        "q75":
            np.percentile(
                values,
                75
            ),

        "skew":
            skew,

        "kurtosis":
            kurtosis,

        "first":
            values[0],

        "last":
            values[-1],

        "change":
            change,

        "slope":
            slope,
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(df):

    columns = {

        "acc_x":
            "IMU_4_ax_g_",

        "acc_y":
            "IMU_4_ay_g_",

        "acc_z":
            "IMU_4_az_g_",

        "gyro_x":
            "IMU_4_gx_dps_",

        "gyro_y":
            "IMU_4_gy_dps_",

        "gyro_z":
            "IMU_4_gz_dps_",
    }

    for column in columns.values():

        if column not in df.columns:

            raise ValueError(
                f"Missing IMU column: "
                f"{column}"
            )

    acc_x = clean_sensor_column(
        df,
        columns["acc_x"]
    )

    acc_y = clean_sensor_column(
        df,
        columns["acc_y"]
    )

    acc_z = clean_sensor_column(
        df,
        columns["acc_z"]
    )

    gyro_x = clean_sensor_column(
        df,
        columns["gyro_x"]
    )

    gyro_y = clean_sensor_column(
        df,
        columns["gyro_y"]
    )

    gyro_z = clean_sensor_column(
        df,
        columns["gyro_z"]
    )

    def fill_missing(values):

        if not np.isfinite(
            values
        ).any():

            return np.zeros(
                len(values)
            )

        return values

    acc_x = fill_missing(acc_x)
    acc_y = fill_missing(acc_y)
    acc_z = fill_missing(acc_z)

    gyro_x = fill_missing(gyro_x)
    gyro_y = fill_missing(gyro_y)
    gyro_z = fill_missing(gyro_z)

    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    acc_mag = np.sqrt(
        acc_x ** 2
        + acc_y ** 2
        + acc_z ** 2
    )

    gyro_mag = np.sqrt(
        gyro_x ** 2
        + gyro_y ** 2
        + gyro_z ** 2
    )

    # --------------------------------------------------------
    # Dynamic acceleration
    # --------------------------------------------------------

    dynamic_acc = (
        acc_mag
        - np.mean(acc_mag)
    )

    # --------------------------------------------------------
    # Jerk
    # --------------------------------------------------------

    jerk = np.diff(
        acc_mag,
        prepend=acc_mag[0]
    )

    features = {}

    # ========================================================
    # BASIC AXES
    # ========================================================

    channels = {

        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,

        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
    }

    for channel_name, values in channels.items():

        stats = safe_stats(
            values
        )

        if channel_name.startswith(
            "acc_"
        ):

            prefix = "IMU4_acc_"

        else:

            prefix = "IMU4_gyro_"

        axis = channel_name[-1]

        for stat_name, value in stats.items():

            features[
                f"{prefix}{axis}_{stat_name}"
            ] = value

    # ========================================================
    # ACCELERATION MAGNITUDE
    # ========================================================

    stats = safe_stats(
        acc_mag
    )

    for stat_name, value in stats.items():

        features[
            f"IMU4_acc_mag_{stat_name}"
        ] = value

    # ========================================================
    # DYNAMIC ACCELERATION
    # ========================================================

    stats = safe_stats(
        dynamic_acc
    )

    dynamic_stats = [

        "mean",
        "std",
        "median",
        "min",
        "max",
        "range",
        "rms",
        "abs_mean",
        "q25",
        "q75",
        "skew",
        "kurtosis",
    ]

    for stat_name in dynamic_stats:

        features[
            f"IMU4_dynamic_acc_{stat_name}"
        ] = stats[
            stat_name
        ]

    # ========================================================
    # ACCELERATION ENERGY
    # ========================================================

    features[
        "IMU4_acc_energy"
    ] = np.sum(
        acc_mag ** 2
    )

    # ========================================================
    # ACTIVITY VARIANCE
    # ========================================================

    features[
        "IMU4_activity_variance"
    ] = np.var(
        dynamic_acc
    )

    # ========================================================
    # JERK
    # ========================================================

    jerk_stats = safe_stats(
        jerk
    )

    jerk_stat_names = [

        "mean",
        "std",
        "median",
        "min",
        "max",
        "range",
        "rms",
        "abs_mean",
        "q25",
        "q75",
        "skew",
        "kurtosis",
    ]

    for stat_name in jerk_stat_names:

        features[
            f"IMU4_jerk_{stat_name}"
        ] = jerk_stats[
            stat_name
        ]

    features[
        "IMU4_jerk_energy"
    ] = np.sum(
        jerk ** 2
    )

    # ========================================================
    # GYROSCOPE MAGNITUDE
    # ========================================================

    stats = safe_stats(
        gyro_mag
    )

    for stat_name, value in stats.items():

        features[
            f"IMU4_gyro_mag_{stat_name}"
        ] = value

    # ========================================================
    # GYROSCOPE ENERGY
    # ========================================================

    features[
        "IMU4_gyro_energy"
    ] = np.sum(
        gyro_mag ** 2
    )

    return features


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_input(
    feature_dict,
    feature_names
):

    actual = set(
        feature_dict.keys()
    )

    expected = set(
        feature_names
    )

    extra = sorted(
        actual - expected
    )

    missing = sorted(
        expected - actual
    )

    if extra or missing:

        raise RuntimeError(
            "Feature schema mismatch.\n"
            f"Extra: {extra}\n"
            f"Missing: {missing}"
        )

    values = [
        feature_dict[name]
        for name in feature_names
    ]

    return pd.DataFrame(
        [values],
        columns=feature_names
    )


# ============================================================
# FATIGUE RISK
# ============================================================

def get_fatigue_risk(
    score
):

    if score >= 7:

        return "HIGH"

    elif score >= 4:

        return "CAUTION"

    else:

        return "SAFE"


# ============================================================
# FIND DATASET FILES
# ============================================================

def find_csv_files():

    print()
    print("=" * 70)
    print("SEARCHING FOR MxD CSV FILES")
    print("=" * 70)

    files = []

    for file_path in MXD_ROOT.rglob(
        "*.csv"
    ):

        try:

            header = pd.read_csv(
                file_path,
                nrows=2
            )

            required = [

                "IMU_4_ax_g_",
                "IMU_4_ay_g_",
                "IMU_4_az_g_",
                "IMU_4_gx_dps_",
                "IMU_4_gy_dps_",
                "IMU_4_gz_dps_",
            ]

            if all(
                c in header.columns
                for c in required
            ):

                files.append(
                    file_path
                )

        except Exception:

            continue

    files.sort()

    print(
        f"Compatible CSV files found: "
        f"{len(files)}"
    )

    if not files:

        raise RuntimeError(
            "No compatible IMU4 CSV files found."
        )

    if MAX_FILES is not None:

        files = files[
            :MAX_FILES
        ]

        print(
            f"Testing first "
            f"{len(files)} files."
        )

    return files


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    file_path,
    model,
    imputer,
    feature_names
):

    df = pd.read_csv(
        file_path
    )

    if len(df) < WINDOW_SAMPLES:

        return []

    rows = []

    # --------------------------------------------------------
    # Sliding 3-minute windows
    # Every 30 seconds
    # --------------------------------------------------------

    start_positions = range(
        0,
        len(df) - WINDOW_SAMPLES + 1,
        STRIDE_SAMPLES
    )

    for start in start_positions:

        end = (
            start
            + WINDOW_SAMPLES
        )

        window = df.iloc[
            start:end
        ].copy()

        feature_dict = extract_features(
            window
        )

        X = build_input(
            feature_dict,
            feature_names
        )

        X_imputed = imputer.transform(
            X
        )

        prediction = model.predict(
            X_imputed
        )[0]

        prediction = float(
            np.clip(
                prediction,
                0,
                10
            )
        )

        risk = get_fatigue_risk(
            prediction
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        participant = ""

        condition = ""

        session = ""

        repetition = ""

        file_name = file_path.name

        parent_name = (
            file_path.parent.name
        )

        # Example:
        # P001C001S002
        #
        # Extract identifiers if possible.

        import re

        match = re.search(
            r"(P\d+C\d+S\d+)",
            file_name
            + " "
            + parent_name
        )

        if match:

            identifier = (
                match.group(1)
            )

            match_p = re.search(
                r"P(\d+)",
                identifier
            )

            match_c = re.search(
                r"C(\d+)",
                identifier
            )

            match_s = re.search(
                r"S(\d+)",
                identifier
            )

            if match_p:
                participant = (
                    "P"
                    + match_p.group(1)
                )

            if match_c:
                condition = (
                    "C"
                    + match_c.group(1)
                )

            if match_s:
                session = (
                    "S"
                    + match_s.group(1)
                )

        # ----------------------------------------------------
        # Time information
        # ----------------------------------------------------

        start_time = None
        end_time = None

        if "Time" in window.columns:

            time_values = pd.to_numeric(
                window["Time"],
                errors="coerce"
            )

            if time_values.notna().any():

                start_time = (
                    time_values.iloc[0]
                )

                end_time = (
                    time_values.iloc[-1]
                )

        rows.append({

            "participant_id":
                participant,

            "condition_id":
                condition,

            "session_id":
                session,

            "file":
                file_name,

            "window_start_sample":
                start,

            "window_end_sample":
                end - 1,

            "window_start_time":
                start_time,

            "window_end_time":
                end_time,

            "fatigue_score":
                round(
                    prediction,
                    4
                ),

            "fatigue_risk":
                risk,
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THERMORA FATIGUE BATCH INFERENCE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        imputer,
        feature_names
    ) = load_model()

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = find_csv_files()

    all_rows = []

    print()
    print("=" * 70)
    print("PROCESSING DATA")
    print("=" * 70)

    for index, file_path in enumerate(
        files,
        start=1
    ):

        print(
            f"[{index}/{len(files)}] "
            f"{file_path.name}"
        )

        try:

            rows = process_file(
                file_path,
                model,
                imputer,
                feature_names
            )

            all_rows.extend(
                rows
            )

            print(
                f"    windows: "
                f"{len(rows)}"
            )

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    if not all_rows:

        raise RuntimeError(
            "No predictions were generated."
        )

    results = pd.DataFrame(
        all_rows
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FATIGUE PREDICTION SUMMARY")
    print("=" * 70)

    print(
        f"Files processed: "
        f"{results['file'].nunique()}"
    )

    print(
        f"Windows generated: "
        f"{len(results)}"
    )

    print(
        f"Participants: "
        f"{results['participant_id'].nunique()}"
    )

    print()

    distribution = (
        results["fatigue_risk"]
        .value_counts()
        .reindex(
            [
                "SAFE",
                "CAUTION",
                "HIGH"
            ],
            fill_value=0
        )
    )

    for risk, count in distribution.items():

        percentage = (
            count
            / len(results)
            * 100
        )

        print(
            f"{risk:<10}"
            f"{count:>6}"
            f" ({percentage:6.2f}%)"
        )

    print()

    print(
        f"Mean fatigue score: "
        f"{results['fatigue_score'].mean():.2f}"
    )

    print(
        f"Minimum fatigue score: "
        f"{results['fatigue_score'].min():.2f}"
    )

    print(
        f"Maximum fatigue score: "
        f"{results['fatigue_score'].max():.2f}"
    )

    print()
    print(
        f"Saved:\n{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    print(
        results[
            [
                "participant_id",
                "condition_id",
                "file",
                "window_start_time",
                "fatigue_score",
                "fatigue_risk",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("BATCH INFERENCE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()