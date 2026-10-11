"""
THERMORA FATIGUE INFERENCE TEST
================================

Tests the saved deployable fatigue model against
real MxD IMU4 data.

Pipeline:

Raw BMI270-equivalent 6-axis IMU
        ↓
3-minute window
        ↓
156 feature extraction
        ↓
Feature-schema validation
        ↓
SimpleImputer
        ↓
RandomForestRegressor
        ↓
Fatigue score 0-10
        ↓
SAFE / CAUTION / HIGH

Prototype / research use only.
Not a medical diagnostic system.
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


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SECONDS = 180

# MxD data is sampled at approximately 2 Hz.
SAMPLE_RATE_HZ = 2

WINDOW_SAMPLES = (
    WINDOW_SECONDS
    * SAMPLE_RATE_HZ
)


# ============================================================
# FIND A REAL MxD CSV
# ============================================================

def find_mxd_csv():

    candidates = list(
        MXD_ROOT.rglob("*.csv")
    )

    if not candidates:

        raise FileNotFoundError(
            "No CSV files found under:\n"
            f"{MXD_ROOT}"
        )

    imu4_files = []

    for file_path in candidates:

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
                col in header.columns
                for col in required
            ):

                imu4_files.append(
                    file_path
                )

        except Exception:
            continue

    if not imu4_files:

        raise RuntimeError(
            "Could not find an MxD CSV containing "
            "all six required IMU4 channels."
        )

    # Sort for reproducibility
    imu4_files.sort()

    return imu4_files[0]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING FATIGUE MODEL")
    print("=" * 70)

    print(
        f"Model:   {MODEL_FILE}"
    )

    print(
        f"Imputer: {IMPUTER_FILE}"
    )

    print(
        f"Features: {FEATURE_FILE}"
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Missing model:\n{MODEL_FILE}"
        )

    if not IMPUTER_FILE.exists():

        raise FileNotFoundError(
            f"Missing imputer:\n{IMPUTER_FILE}"
        )

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Missing feature file:\n{FEATURE_FILE}"
        )

    # IMPORTANT:
    # These files were saved with joblib.
    model = joblib.load(
        MODEL_FILE
    )

    imputer = joblib.load(
        IMPUTER_FILE
    )

    with open(
        FEATURE_FILE,
        "r"
    ) as file:

        feature_names = json.load(
            file
        )

    print()
    print(
        f"Model type: "
        f"{type(model)}"
    )

    print(
        f"Expected features: "
        f"{len(feature_names)}"
    )

    if hasattr(
        model,
        "n_estimators"
    ):

        print(
            f"Trees: "
            f"{model.n_estimators}"
        )

    if hasattr(
        model,
        "n_features_in_"
    ):

        print(
            f"Model input features: "
            f"{model.n_features_in_}"
        )

    if hasattr(
        imputer,
        "n_features_in_"
    ):

        print(
            f"Imputer input features: "
            f"{imputer.n_features_in_}"
        )

    return (
        model,
        imputer,
        feature_names
    )


# ============================================================
# SAFE STATISTICS
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

    std = np.std(
        values
    )

    if len(values) > 1:

        differences = np.diff(
            values
        )

        change = np.mean(
            np.abs(
                differences
            )
        )

        slope = np.polyfit(
            np.arange(
                len(values)
            ),
            values,
            1
        )[0]

    else:

        change = 0.0
        slope = 0.0

    # Prevent unstable skew/kurtosis for constant signals.
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

        "median": np.median(
            values
        ),

        "min": np.min(
            values
        ),

        "max": np.max(
            values
        ),

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

        "skew": skew,

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
# CONVERT SENSOR COLUMN
# ============================================================

def clean_sensor_column(
    df,
    column
):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # MxD uses -2 as an invalid/unavailable sentinel.
    values = values.replace(
        [-2, -2.0],
        np.nan
    )

    # Interpolate inside the window.
    values = values.interpolate(
        limit_direction="both"
    )

    return values.to_numpy(
        dtype=float
    )


# ============================================================
# EXTRACT EXACT 156 FEATURES
# ============================================================

def extract_features(df):

    print()
    print("=" * 70)
    print("EXTRACTING 156 FATIGUE FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Exact six IMU4 channels
    # --------------------------------------------------------

    column_map = {

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

    for name, column in column_map.items():

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: "
                f"{column}"
            )

    # --------------------------------------------------------
    # Load channels
    # --------------------------------------------------------

    acc_x = clean_sensor_column(
        df,
        column_map["acc_x"]
    )

    acc_y = clean_sensor_column(
        df,
        column_map["acc_y"]
    )

    acc_z = clean_sensor_column(
        df,
        column_map["acc_z"]
    )

    gyro_x = clean_sensor_column(
        df,
        column_map["gyro_x"]
    )

    gyro_y = clean_sensor_column(
        df,
        column_map["gyro_y"]
    )

    gyro_z = clean_sensor_column(
        df,
        column_map["gyro_z"]
    )

    # --------------------------------------------------------
    # If a whole channel is missing, use zeros temporarily.
    #
    # The trained imputer handles missing FEATURES later,
    # but a completely missing raw channel needs numerical
    # values for derived magnitude calculations.
    # --------------------------------------------------------

    def fill_whole_channel(values):

        if not np.isfinite(
            values
        ).any():

            return np.zeros(
                len(values)
            )

        return values

    acc_x = fill_whole_channel(
        acc_x
    )

    acc_y = fill_whole_channel(
        acc_y
    )

    acc_z = fill_whole_channel(
        acc_z
    )

    gyro_x = fill_whole_channel(
        gyro_x
    )

    gyro_y = fill_whole_channel(
        gyro_y
    )

    gyro_z = fill_whole_channel(
        gyro_z
    )

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
    #
    # Remove the mean acceleration magnitude, representing
    # the static component approximately.
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

    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    features = {}

    # ========================================================
    # 1-6. BASIC ACCELERATION / GYRO CHANNELS
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
    # 7. ACCELERATION MAGNITUDE
    # ========================================================

    stats = safe_stats(
        acc_mag
    )

    for stat_name, value in stats.items():

        features[
            f"IMU4_acc_mag_{stat_name}"
        ] = value

    # ========================================================
    # 8. DYNAMIC ACCELERATION
    #
    # EXACTLY 12 statistics expected by the trained model.
    #
    # The trained schema does NOT contain:
    #   first
    #   last
    #   change
    #   slope
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
    # 9. ACCELERATION ENERGY
    # ========================================================

    features[
        "IMU4_acc_energy"
    ] = np.sum(
        acc_mag ** 2
    )

    # ========================================================
    # 10. ACTIVITY VARIANCE
    # ========================================================

    features[
        "IMU4_activity_variance"
    ] = np.var(
        dynamic_acc
    )

    # ========================================================
    # 11. JERK
    #
    # EXACTLY 13 FEATURES
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
    # 12. GYROSCOPE MAGNITUDE
    # ========================================================

    stats = safe_stats(
        gyro_mag
    )

    for stat_name, value in stats.items():

        features[
            f"IMU4_gyro_mag_{stat_name}"
        ] = value

    # ========================================================
    # 13. GYROSCOPE ENERGY
    # ========================================================

    features[
        "IMU4_gyro_energy"
    ] = np.sum(
        gyro_mag ** 2
    )

    return features


# ============================================================
# VALIDATE FEATURE SCHEMA
# ============================================================

def validate_features(
    feature_dict,
    feature_names
):

    print()
    print("=" * 70)
    print("VALIDATING FEATURE SCHEMA")
    print("=" * 70)

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

    print(
        f"Expected features: "
        f"{len(feature_names)}"
    )

    print(
        f"Extracted features: "
        f"{len(feature_dict)}"
    )

    print(
        f"Extra features: "
        f"{len(extra)}"
    )

    print(
        f"Missing features: "
        f"{len(missing)}"
    )

    if extra:

        print()
        print(
            "EXTRA FEATURES:"
        )

        for name in extra:

            print(
                f"  {name}"
            )

    if missing:

        print()
        print(
            "MISSING FEATURES:"
        )

        for name in missing:

            print(
                f"  {name}"
            )

    if extra or missing:

        raise RuntimeError(
            "\nFeature schema mismatch.\n"
            "Prediction stopped intentionally."
        )

    print()
    print(
        "FEATURE SCHEMA: EXACT MATCH"
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(
    feature_dict,
    feature_names
):

    # IMPORTANT:
    # DataFrame preserves the exact feature names and order.

    X = pd.DataFrame(
        [
            [
                feature_dict[name]
                for name in feature_names
            ]
        ],
        columns=feature_names
    )

    print()
    print(
        f"Model input shape: "
        f"{X.shape}"
    )

    return X


# ============================================================
# PREDICT FATIGUE
# ============================================================

def predict_fatigue(
    model,
    imputer,
    X
):

    # --------------------------------------------------------
    # Impute
    # --------------------------------------------------------

    X_imputed = imputer.transform(
        X
    )

    print(
        f"After imputation: "
        f"{X_imputed.shape}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

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

    return prediction


# ============================================================
# FATIGUE RISK
# ============================================================

def fatigue_risk(
    score
):

    if score >= 7:

        return "HIGH"

    elif score >= 4:

        return "CAUTION"

    else:

        return "SAFE"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THERMORA FATIGUE INFERENCE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        imputer,
        feature_names
    ) = load_model()

    # --------------------------------------------------------
    # Find real dataset
    # --------------------------------------------------------

    csv_file = find_mxd_csv()

    print()
    print(
        "Using dataset:"
    )

    print(
        csv_file
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_file
    )

    print()
    print(
        f"Rows available: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Create 3-minute window
    # --------------------------------------------------------

    WINDOW = min(
        WINDOW_SAMPLES,
        len(df)
    )

    window = df.iloc[
        :WINDOW
    ].copy()

    print(
        f"Window size: "
        f"{len(window)} samples"
    )

    print(
        f"Window duration: "
        f"approximately "
        f"{len(window) / SAMPLE_RATE_HZ:.1f} seconds"
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    feature_dict = extract_features(
        window
    )

    # --------------------------------------------------------
    # EXACT SCHEMA CHECK
    # --------------------------------------------------------

    validate_features(
        feature_dict,
        feature_names
    )

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    X = create_model_input(
        feature_dict,
        feature_names
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = predict_fatigue(
        model,
        imputer,
        X
    )

    risk = fatigue_risk(
        prediction
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FATIGUE PREDICTION")
    print("=" * 70)

    print(
        f"Fatigue score: "
        f"{prediction:.2f} / 10"
    )

    print(
        f"Fatigue risk:  "
        f"{risk}"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "INFERENCE PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()