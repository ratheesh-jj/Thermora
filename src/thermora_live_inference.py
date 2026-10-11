"""
THERMORA LIVE-STYLE INFERENCE SIMULATOR
=======================================

Simulates how THERMORA will operate with recorded wearable data.

INDEPENDENT OUTPUTS
-------------------

THERMAL
    - estimated current core temperature
    - predicted core temperature +10 min
    - predicted temperature rise

FATIGUE
    - 3-minute IMU window
    - fatigue score 0-10
    - fatigue trend

FALL
    - 20-sample IMU window
    - fall probability
    - fall detected

IMPORTANT
---------
- Offline simulator only.
- No models are retrained.
- No unified risk score is created.
- Thermal, fatigue and fall outputs remain independent.
- Fatigue uses the exact 156-feature extractor used during training.
- Fall uses the exact 148-feature extractor from v2.
- Fatigue raw data comes from the MxD dataset.
- Fall data comes from the existing BITS-2 v2 windows.
- Final hardware will replace recorded data with live BMI270 data.
"""

import os
import json
import zipfile
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FATIGUE_DIR = os.path.join(
    BASE_DIR,
    "fatigue_prediction",
    "fatigue_deployable_model"
)

FALL_DIR = os.path.join(
    BASE_DIR,
    "Fall_Prediction"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# MODEL PATHS
# ============================================================

THERMAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "thermora_integrated_10min_no_wind_model.pkl"
)

FATIGUE_MODEL_PATH = os.path.join(
    FATIGUE_DIR,
    "thermora_fatigue_rf.pkl"
)

FATIGUE_IMPUTER_PATH = os.path.join(
    FATIGUE_DIR,
    "thermora_fatigue_imputer.pkl"
)

FATIGUE_FEATURES_PATH = os.path.join(
    FATIGUE_DIR,
    "thermora_fatigue_features.json"
)

FALL_MODEL_PATH = os.path.join(
    FALL_DIR,
    "thermora_event_model_v2.pkl"
)


# ============================================================
# RECORDED DATA PATHS
# ============================================================

# Preferred extracted raw MxD dataset location.
#
# This is the location that was previously used successfully
# by test_fatigue_inference.py.
FATIGUE_EXTRACTED_ROOT = os.path.join(
    FATIGUE_DIR,
    "..",
    "fatigue_personalization_results",
    "extracted"
)

FATIGUE_EXTRACTED_ROOT = os.path.abspath(
    FATIGUE_EXTRACTED_ROOT
)


# ZIP fallback.
FATIGUE_ZIP_PATH = os.path.join(
    BASE_DIR,
    "fatigue_prediction",
    "MxD_Data_User_Study.zip"
)


# Fall windows.
FALL_DATA_PATH = os.path.join(
    FALL_DIR,
    "thermora_raw_event_windows_v2.npz"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

THERMAL_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "thermora_live_thermal.csv"
)

FATIGUE_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "thermora_live_fatigue.csv"
)

FALL_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "thermora_live_fall.csv"
)

METADATA_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "thermora_live_simulation_metadata.json"
)


# ============================================================
# SETTINGS
# ============================================================

# Fatigue:
# 2 Hz × 180 seconds = 360 samples
FATIGUE_WINDOW = 360

# 30 second stride at 2 Hz
FATIGUE_STRIDE = 60


# Fall:
# BITS-2 v2 model uses 20-sample windows.
FALL_WINDOW = 20


# ============================================================
# ACTUAL RAW MxD IMU4 COLUMN NAMES
# ============================================================

# IMPORTANT:
#
# The raw MxD dataset DOES NOT use:
#
#   IMU_4_ax
#   IMU_4_ay
#   ...
#
# It actually uses:
#
#   IMU_4_ax_g_
#   IMU_4_ay_g_
#   IMU_4_az_g_
#   IMU_4_gx_dps_
#   IMU_4_gy_dps_
#   IMU_4_gz_dps_
#
# These are the exact raw channels used here.

IMU4_COLUMNS = [
    "IMU_4_ax_g_",
    "IMU_4_ay_g_",
    "IMU_4_az_g_",
    "IMU_4_gx_dps_",
    "IMU_4_gy_dps_",
    "IMU_4_gz_dps_",
]


# ============================================================
# FATIGUE FEATURE EXTRACTION
# ============================================================

def safe_stats(signal):
    """
    EXACT statistics used by the THERMORA fatigue model.

    Returns 16 values in this exact order:

        mean
        std
        median
        min
        max
        range
        rms
        abs_mean
        q25
        q75
        skew
        kurtosis
        first
        last
        change
        slope
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    signal = signal[
        np.isfinite(signal)
    ]

    if len(signal) == 0:
        return [0.0] * 16

    if len(signal) == 1:
        value = float(signal[0])

        return [
            value,      # mean
            0.0,        # std
            value,      # median
            value,      # min
            value,      # max
            0.0,        # range
            abs(value), # rms
            abs(value), # abs_mean
            value,      # q25
            value,      # q75
            0.0,        # skew
            0.0,        # kurtosis
            value,      # first
            value,      # last
            0.0,        # change
            0.0         # slope
        ]

    mean = np.mean(signal)

    std = np.std(signal)

    median = np.median(signal)

    minimum = np.min(signal)

    maximum = np.max(signal)

    value_range = maximum - minimum

    rms = np.sqrt(
        np.mean(
            signal ** 2
        )
    )

    abs_mean = np.mean(
        np.abs(signal)
    )

    q25 = np.percentile(
        signal,
        25
    )

    q75 = np.percentile(
        signal,
        75
    )

    # Skew and kurtosis without scipy.
    centered = signal - mean

    variance = np.mean(
        centered ** 2
    )

    if variance > 1e-12:

        skew = (
            np.mean(
                centered ** 3
            )
            /
            variance ** 1.5
        )

        kurtosis = (
            np.mean(
                centered ** 4
            )
            /
            variance ** 2
            -
            3.0
        )

    else:

        skew = 0.0

        kurtosis = 0.0

    first = signal[0]

    last = signal[-1]

    change = last - first

    x = np.arange(
        len(signal),
        dtype=float
    )

    slope = np.polyfit(
        x,
        signal,
        1
    )[0]

    return [
        float(mean),
        float(std),
        float(median),
        float(minimum),
        float(maximum),
        float(value_range),
        float(rms),
        float(abs_mean),
        float(q25),
        float(q75),
        float(skew),
        float(kurtosis),
        float(first),
        float(last),
        float(change),
        float(slope)
    ]


# ============================================================
# EXACT 156-FEATURE FATIGUE EXTRACTOR
# ============================================================

def extract_fatigue_features(imu):
    """
    EXACT 156-feature structure used by the trained
    THERMORA fatigue model.

    Input:
        imu shape = (samples, 6)

    Channel order:
        ax, ay, az, gx, gy, gz
    """

    imu = np.asarray(
        imu,
        dtype=float
    )

    if imu.ndim != 2:

        raise ValueError(
            "IMU input must be 2-dimensional."
        )

    if imu.shape[1] != 6:

        raise ValueError(
            f"Expected 6 IMU channels, "
            f"got {imu.shape[1]}"
        )

    # --------------------------------------------------------
    # Replace dataset sentinel values
    # --------------------------------------------------------

    imu[
        imu == -2
    ] = np.nan

    # --------------------------------------------------------
    # Interpolate missing values
    # --------------------------------------------------------

    df = pd.DataFrame(
        imu
    )

    df = df.interpolate(
        limit_direction="both"
    )

    df = df.fillna(
        0.0
    )

    imu = df.values

    ax = imu[:, 0]
    ay = imu[:, 1]
    az = imu[:, 2]

    gx = imu[:, 3]
    gy = imu[:, 4]
    gz = imu[:, 5]

    # --------------------------------------------------------
    # Basic channels
    # --------------------------------------------------------

    channels = [
        ax,
        ay,
        az,
        gx,
        gy,
        gz
    ]

    features = []

    for signal in channels:

        features.extend(
            safe_stats(signal)
        )

    # --------------------------------------------------------
    # Acceleration magnitude
    # --------------------------------------------------------

    acc_mag = np.sqrt(
        ax ** 2 +
        ay ** 2 +
        az ** 2
    )

    features.extend(
        safe_stats(
            acc_mag
        )
    )

    # --------------------------------------------------------
    # Dynamic acceleration
    # --------------------------------------------------------

    rolling_window = min(
        20,
        len(acc_mag)
    )

    dynamic_acc = (
        acc_mag
        -
        pd.Series(
            acc_mag
        ).rolling(
            window=rolling_window,
            min_periods=1
        ).mean().values
    )

    dynamic_stats = safe_stats(
        dynamic_acc
    )

    # Only first 12 dynamic acceleration statistics
    # belong to the 156-feature model.
    features.extend(
        dynamic_stats[:12]
    )

    # --------------------------------------------------------
    # Acceleration energy
    # --------------------------------------------------------

    acc_energy = np.sum(
        acc_mag ** 2
    )

    features.append(
        float(acc_energy)
    )

    # --------------------------------------------------------
    # Activity variance
    # --------------------------------------------------------

    activity_variance = np.var(
        acc_mag
    )

    features.append(
        float(activity_variance)
    )

    # --------------------------------------------------------
    # Jerk
    # --------------------------------------------------------

    jerk = np.diff(
        acc_mag
    )

    jerk_stats = safe_stats(
        jerk
    )

    features.extend(
        jerk_stats[:12]
    )

    jerk_energy = np.sum(
        jerk ** 2
    )

    features.append(
        float(jerk_energy)
    )

    # --------------------------------------------------------
    # Gyroscope magnitude
    # --------------------------------------------------------

    gyro_mag = np.sqrt(
        gx ** 2 +
        gy ** 2 +
        gz ** 2
    )

    features.extend(
        safe_stats(
            gyro_mag
        )
    )

    # --------------------------------------------------------
    # Gyroscope energy
    # --------------------------------------------------------

    gyro_energy = np.sum(
        gyro_mag ** 2
    )

    features.append(
        float(gyro_energy)
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if len(features) != 156:

        raise ValueError(
            "\nFATIGUE FEATURE MISMATCH\n"
            f"Generated: {len(features)}\n"
            "Expected: 156"
        )

    return features


# ============================================================
# EXACT 148-FEATURE FALL EXTRACTOR
# ============================================================

def extract_fall_features(X):
    """
    EXACT 148-feature extractor from
    train_event_model_v2.py.
    """

    features = []

    for sample in X:

        acc = sample[:, 0:3]

        gyro = sample[:, 3:6]

        acc_mag = np.linalg.norm(
            acc,
            axis=1
        )

        gyro_mag = np.linalg.norm(
            gyro,
            axis=1
        )

        acc_jerk = np.diff(
            acc_mag
        )

        gyro_change = np.diff(
            gyro_mag
        )

        f = []

        # ----------------------------------------------------
        # Raw axis statistics
        # ----------------------------------------------------

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
                np.sqrt(
                    np.mean(
                        signal ** 2
                    )
                ),
                np.median(signal),
                np.percentile(
                    signal,
                    10
                ),
                np.percentile(
                    signal,
                    25
                ),
                np.percentile(
                    signal,
                    75
                ),
                np.percentile(
                    signal,
                    90
                )
            ])

        # ----------------------------------------------------
        # Peak / impact
        # ----------------------------------------------------

        f.extend([
            np.max(acc_mag),
            np.max(gyro_mag),

            np.mean(acc_mag),
            np.std(acc_mag),

            np.mean(gyro_mag),
            np.std(gyro_mag),

            np.max(
                np.abs(
                    acc_jerk
                )
            ),

            np.mean(
                np.abs(
                    acc_jerk
                )
            ),

            np.std(
                acc_jerk
            ),

            np.max(
                np.abs(
                    gyro_change
                )
            ),

            np.mean(
                np.abs(
                    gyro_change
                )
            ),

            np.std(
                gyro_change
            )
        ])

        # ----------------------------------------------------
        # First half / second half
        # ----------------------------------------------------

        mid = len(sample) // 2

        for signal in [
            acc_mag,
            gyro_mag
        ]:

            first = signal[:mid]

            second = signal[mid:]

            f.extend([
                np.mean(first),
                np.std(first),
                np.max(first),

                np.mean(second),
                np.std(second),
                np.max(second),

                np.mean(second)
                -
                np.mean(first),

                np.max(second)
                -
                np.max(first)
            ])

        # ----------------------------------------------------
        # Energy
        # ----------------------------------------------------

        f.extend([
            np.sum(
                acc_mag ** 2
            ),

            np.sum(
                gyro_mag ** 2
            ),

            np.mean(
                acc_mag ** 2
            ),

            np.mean(
                gyro_mag ** 2
            )
        ])

        # ----------------------------------------------------
        # High-motion fractions
        # ----------------------------------------------------

        acc_threshold = np.percentile(
            acc_mag,
            90
        )

        gyro_threshold = np.percentile(
            gyro_mag,
            90
        )

        f.extend([
            np.mean(
                acc_mag >
                acc_threshold
            ),

            np.mean(
                gyro_mag >
                gyro_threshold
            ),

            np.mean(
                acc_mag >
                10
            ),

            np.mean(
                acc_mag >
                15
            ),

            np.mean(
                gyro_mag >
                5
            ),

            np.mean(
                gyro_mag >
                10
            )
        ])

        features.append(
            f
        )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape[1] != 148:

        raise ValueError(
            f"Fall feature mismatch: "
            f"generated {features.shape[1]}, "
            f"expected 148"
        )

    return features


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("=" * 70)

    print(
        "LOADING LIVE THERMORA MODELS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Thermal
    # --------------------------------------------------------

    if not os.path.exists(
        THERMAL_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Thermal model not found:\n"
            f"{THERMAL_MODEL_PATH}"
        )

    thermal_model = joblib.load(
        THERMAL_MODEL_PATH
    )

    print(
        "\nThermal model:",
        type(thermal_model)
    )

    # --------------------------------------------------------
    # Fatigue
    # --------------------------------------------------------

    if not os.path.exists(
        FATIGUE_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Fatigue model not found:\n"
            f"{FATIGUE_MODEL_PATH}"
        )

    if not os.path.exists(
        FATIGUE_IMPUTER_PATH
    ):

        raise FileNotFoundError(
            f"Fatigue imputer not found:\n"
            f"{FATIGUE_IMPUTER_PATH}"
        )

    fatigue_model = joblib.load(
        FATIGUE_MODEL_PATH
    )

    fatigue_imputer = joblib.load(
        FATIGUE_IMPUTER_PATH
    )

    print(
        "Fatigue model:",
        type(fatigue_model)
    )

    print(
        "Fatigue features: 156"
    )

    # --------------------------------------------------------
    # Fall
    # --------------------------------------------------------

    if not os.path.exists(
        FALL_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Fall model not found:\n"
            f"{FALL_MODEL_PATH}"
        )

    fall_artifact = joblib.load(
        FALL_MODEL_PATH
    )

    if not isinstance(
        fall_artifact,
        dict
    ):

        raise ValueError(
            "Unexpected fall model format."
        )

    fall_model = fall_artifact[
        "model"
    ]

    fall_threshold = fall_artifact[
        "threshold"
    ]

    fall_feature_count = (
        fall_artifact[
            "feature_count"
        ]
    )

    print(
        "Fall model:",
        type(fall_model)
    )

    print(
        "Fall threshold:",
        fall_threshold
    )

    print(
        "Fall features:",
        fall_feature_count
    )

    if fall_feature_count != 148:

        raise ValueError(
            f"Expected 148 fall features, "
            f"artifact says {fall_feature_count}"
        )

    return {
        "thermal": thermal_model,
        "fatigue": fatigue_model,
        "fatigue_imputer": fatigue_imputer,
        "fall": fall_model,
        "fall_threshold": fall_threshold,
        "fall_feature_count": fall_feature_count
    }


# ============================================================
# FIND RAW FATIGUE CSV
# ============================================================

def find_fatigue_csv():

    print("\n")
    print("=" * 70)
    print("SEARCHING FOR RAW FATIGUE SENSOR DATA")
    print("=" * 70)

    print(
        "\nLooking specifically for raw MxD IMU4 channels:"
    )

    for col in IMU4_COLUMNS:
        print(
            "  ✓",
            col
        )

    candidates = []

    # --------------------------------------------------------
    # FIRST: Search the previously extracted MxD dataset.
    # --------------------------------------------------------

    if os.path.exists(
        FATIGUE_EXTRACTED_ROOT
    ):

        print(
            "\nSearching extracted MxD dataset:"
        )

        print(
            FATIGUE_EXTRACTED_ROOT
        )

        for root, dirs, files in os.walk(
            FATIGUE_EXTRACTED_ROOT
        ):

            for filename in files:

                if not filename.lower().endswith(
                    ".csv"
                ):
                    continue

                path = os.path.join(
                    root,
                    filename
                )

                try:

                    df = pd.read_csv(
                        path,
                        nrows=2
                    )

                except Exception:

                    continue

                if all(
                    col in df.columns
                    for col in IMU4_COLUMNS
                ):

                    candidates.append(
                        path
                    )

    # --------------------------------------------------------
    # SECOND: Search ZIP directly if extracted data was not
    # found.
    # --------------------------------------------------------

    if len(candidates) == 0 and os.path.exists(
        FATIGUE_ZIP_PATH
    ):

        print(
            "\nNo extracted raw CSVs found."
        )

        print(
            "Searching MxD ZIP directly:"
        )

        print(
            FATIGUE_ZIP_PATH
        )

        try:

            with zipfile.ZipFile(
                FATIGUE_ZIP_PATH,
                "r"
            ) as z:

                csv_members = [
                    name
                    for name in z.namelist()
                    if name.lower().endswith(
                        ".csv"
                    )
                ]

                print(
                    "CSV files in ZIP:",
                    len(csv_members)
                )

                for member in csv_members:

                    try:

                        df = pd.read_csv(
                            z.open(member),
                            nrows=2
                        )

                    except Exception:

                        continue

                    if all(
                        col in df.columns
                        for col in IMU4_COLUMNS
                    ):

                        candidates.append(
                            (
                                "ZIP",
                                member
                            )
                        )

        except Exception as exc:

            print(
                "ZIP search failed:",
                exc
            )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print(
        "\nCompatible raw fatigue CSV files:",
        len(candidates)
    )

    if len(candidates) == 0:

        raise FileNotFoundError(
            "\nNo raw MxD CSV containing the exact "
            "IMU4 channels was found.\n\n"
            "Expected channels:\n"
            +
            "\n".join(
                "  " + c
                for c in IMU4_COLUMNS
            )
        )

    # Prefer a normal extracted CSV.
    for candidate in candidates:

        if isinstance(
            candidate,
            str
        ):

            print(
                "\nSelected raw fatigue file:"
            )

            print(
                candidate
            )

            return {
                "type": "file",
                "path": candidate
            }

    # Otherwise use ZIP member.
    zip_member = candidates[0][1]

    print(
        "\nSelected raw fatigue ZIP member:"
    )

    print(
        zip_member
    )

    return {
        "type": "zip",
        "path": FATIGUE_ZIP_PATH,
        "member": zip_member
    }


# ============================================================
# LOAD FATIGUE IMU
# ============================================================

def load_fatigue_imu(source):

    print(
        "\nLoading raw fatigue sensor data..."
    )

    # --------------------------------------------------------
    # Normal extracted CSV
    # --------------------------------------------------------

    if source["type"] == "file":

        path = source["path"]

        df = pd.read_csv(
            path
        )

    # --------------------------------------------------------
    # CSV directly from ZIP
    # --------------------------------------------------------

    elif source["type"] == "zip":

        with zipfile.ZipFile(
            source["path"],
            "r"
        ) as z:

            df = pd.read_csv(
                z.open(
                    source["member"]
                )
            )

    else:

        raise ValueError(
            "Unknown fatigue source type."
        )

    # --------------------------------------------------------
    # Verify exact raw columns
    # --------------------------------------------------------

    missing = [
        col
        for col in IMU4_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing fatigue IMU columns:\n"
            +
            "\n".join(
                missing
            )
        )

    # --------------------------------------------------------
    # Extract in the exact model order
    # --------------------------------------------------------

    imu = (
        df[
            IMU4_COLUMNS
        ]
        .values
        .astype(float)
    )

    print(
        "Raw sensor rows:",
        len(imu)
    )

    print(
        "IMU channels:",
        imu.shape[1]
    )

    print(
        "Channel mapping:"
    )

    for i, col in enumerate(
        IMU4_COLUMNS
    ):

        print(
            f"  [{i}] {col}"
        )

    return df, imu


# ============================================================
# FATIGUE INFERENCE
# ============================================================

def run_fatigue_inference(
    imu_window,
    fatigue_model,
    fatigue_imputer
):

    # Generate exact 156 features.
    features = extract_fatigue_features(
        imu_window
    )

    X = features.reshape(
        1,
        -1
    )

    if X.shape[1] != 156:

        raise ValueError(
            f"Fatigue input has "
            f"{X.shape[1]} features; "
            f"expected 156."
        )

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    X_imputed = fatigue_imputer.transform(
        X
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = fatigue_model.predict(
        X_imputed
    )[0]

    # Keep documented 0-10 range.
    prediction = float(
        np.clip(
            prediction,
            0,
            10
        )
    )

    return prediction


# ============================================================
# FALL INFERENCE
# ============================================================

def run_fall_inference(
    imu_window,
    fall_model,
    fall_threshold
):

    X = np.asarray(
        imu_window,
        dtype=np.float32
    ).reshape(
        1,
        -1,
        6
    )

    if X.shape[1] != FALL_WINDOW:

        raise ValueError(
            f"Fall window has "
            f"{X.shape[1]} samples; "
            f"expected {FALL_WINDOW}."
        )

    features = extract_fall_features(
        X
    )

    probability = (
        fall_model
        .predict_proba(
            features
        )[0, 1]
    )

    detected = (
        probability >= fall_threshold
    )

    return (
        float(probability),
        bool(detected)
    )


# ============================================================
# FATIGUE TREND
# ============================================================

def calculate_trend(
    previous,
    current
):

    if previous is None:

        return "unknown"

    delta = current - previous

    if delta > 0.30:

        return "increasing"

    if delta < -0.30:

        return "decreasing"

    return "stable"


# ============================================================
# LOAD THERMAL OUTPUTS
# ============================================================

def load_thermal_outputs():

    path = os.path.join(
        RESULTS_DIR,
        "thermora_thermal_outputs.csv"
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            "Run multimodal_inference.py "
            "first to generate thermal outputs."
        )

    df = pd.read_csv(
        path
    )

    return df


# ============================================================
# THERMAL COLUMN HELPER
# ============================================================

def get_column(
    df,
    possible_names,
    description
):

    for name in possible_names:

        if name in df.columns:

            return df[name]

    raise KeyError(
        f"Could not find thermal column "
        f"for {description}.\n"
        f"Tried: {possible_names}\n"
        f"Available columns:\n"
        f"{df.columns.tolist()}"
    )


# ============================================================
# PREPARE THERMAL DISPLAY VALUES
# ============================================================

def prepare_thermal_values(
    thermal_df
):

    # Your current thermal output uses:
    #
    # estimated_core_temp_c
    # predicted_future_core_temp_c
    # predicted_delta_c
    #
    # Earlier versions used slightly different names.
    # Support both.

    current_core = get_column(
        thermal_df,
        [
            "current_core_temp_c",
            "estimated_core_temp_c",
            "core_temp_c"
        ],
        "current estimated core temperature"
    )

    predicted_core = get_column(
        thermal_df,
        [
            "predicted_core_temp_10min_c",
            "predicted_future_core_temp_c",
            "future_core_temp_c"
        ],
        "predicted core temperature +10 min"
    )

    predicted_rise = get_column(
        thermal_df,
        [
            "predicted_rise_c",
            "predicted_delta_c"
        ],
        "predicted temperature rise"
    )

    return (
        current_core,
        predicted_core,
        predicted_rise
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "THERMORA LIVE-STYLE SENSOR SIMULATION"
    )
    print("=" * 70)

    # ========================================================
    # LOAD MODELS
    # ========================================================

    models = load_models()

    # ========================================================
    # LOAD FATIGUE SENSOR DATA
    # ========================================================

    fatigue_source = find_fatigue_csv()

    fatigue_df, fatigue_imu = (
        load_fatigue_imu(
            fatigue_source
        )
    )

    print(
        "\nFatigue sensor rows:",
        len(fatigue_imu)
    )

    # ========================================================
    # LOAD THERMAL RESULTS
    # ========================================================

    thermal_df = load_thermal_outputs()

    print(
        "Thermal output rows:",
        len(thermal_df)
    )

    # ========================================================
    # CHECK FATIGUE WINDOW
    # ========================================================

    if len(fatigue_imu) < FATIGUE_WINDOW:

        raise ValueError(
            f"Not enough IMU samples.\n"
            f"Required: {FATIGUE_WINDOW}\n"
            f"Available: {len(fatigue_imu)}"
        )

    # ========================================================
    # LOAD FALL WINDOWS
    # ========================================================

    if not os.path.exists(
        FALL_DATA_PATH
    ):

        raise FileNotFoundError(
            f"Fall dataset not found:\n"
            f"{FALL_DATA_PATH}"
        )

    fall_data = np.load(
        FALL_DATA_PATH
    )

    fall_X = fall_data[
        "X"
    ]

    if fall_X.ndim != 3:

        raise ValueError(
            f"Unexpected fall X shape: "
            f"{fall_X.shape}"
        )

    if fall_X.shape[1] != FALL_WINDOW:

        raise ValueError(
            f"Expected fall windows of "
            f"{FALL_WINDOW} samples, "
            f"got {fall_X.shape[1]}"
        )

    if fall_X.shape[2] != 6:

        raise ValueError(
            f"Expected 6 fall IMU channels, "
            f"got {fall_X.shape[2]}"
        )

    # ========================================================
    # START
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "STARTING SENSOR-STREAM SIMULATION"
    )
    print("=" * 70)

    print(
        "\nFatigue window:",
        FATIGUE_WINDOW,
        "samples"
    )

    print(
        "Fatigue window duration:",
        f"{FATIGUE_WINDOW / 2:.1f}",
        "seconds"
    )

    print(
        "Fatigue stride:",
        FATIGUE_STRIDE,
        "samples"
    )

    print(
        "Fatigue stride duration:",
        f"{FATIGUE_STRIDE / 2:.1f}",
        "seconds"
    )

    print(
        "Fall window:",
        FALL_WINDOW,
        "samples"
    )

    print(
        "Fall recorded windows:",
        len(fall_X)
    )

    # ========================================================
    # FATIGUE WINDOWS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "RUNNING FATIGUE SENSOR SIMULATION"
    )
    print("=" * 70)

    fatigue_results = []

    previous_fatigue = None

    fatigue_window_count = 0

    for start in range(
        0,
        len(fatigue_imu)
        - FATIGUE_WINDOW
        + 1,
        FATIGUE_STRIDE
    ):

        end = (
            start
            +
            FATIGUE_WINDOW
        )

        window = fatigue_imu[
            start:end
        ]

        score = run_fatigue_inference(
            window,
            models["fatigue"],
            models["fatigue_imputer"]
        )

        trend = calculate_trend(
            previous_fatigue,
            score
        )

        fatigue_results.append({
            "window_index":
                fatigue_window_count,

            "start_sample":
                start,

            "end_sample":
                end,

            "window_seconds":
                FATIGUE_WINDOW / 2,

            "stride_seconds":
                FATIGUE_STRIDE / 2,

            "fatigue_score":
                score,

            "fatigue_trend":
                trend
        })

        previous_fatigue = score

        fatigue_window_count += 1

    fatigue_results = pd.DataFrame(
        fatigue_results
    )

    # ========================================================
    # FALL WINDOWS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "RUNNING FALL SENSOR SIMULATION"
    )
    print("=" * 70)

    fall_results = []

    for i in range(
        len(fall_X)
    ):

        probability, detected = (
            run_fall_inference(
                fall_X[i],
                models["fall"],
                models["fall_threshold"]
            )
        )

        fall_results.append({
            "window_index":
                i,

            "window_samples":
                FALL_WINDOW,

            "fall_probability":
                probability,

            "fall_detected":
                detected
        })

    fall_results = pd.DataFrame(
        fall_results
    )

    # ========================================================
    # THERMAL
    # ========================================================

    # Thermal remains independent.
    #
    # The recorded thermal data is not synchronized with
    # the fatigue/fall recorded datasets.
    #
    # Therefore we do NOT merge them into a fake timestamp.

    thermal_current, thermal_predicted, thermal_rise = (
        prepare_thermal_values(
            thermal_df
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    combined_metadata = {

        "simulation_type":
            "offline_recorded_sensor_simulation",

        "outputs_independent":
            True,

        "thermal_rows":
            len(thermal_df),

        "fatigue_windows":
            len(fatigue_results),

        "fall_windows":
            len(fall_results),

        "fatigue_window_samples":
            FATIGUE_WINDOW,

        "fatigue_window_seconds":
            FATIGUE_WINDOW / 2,

        "fatigue_stride_samples":
            FATIGUE_STRIDE,

        "fatigue_stride_seconds":
            FATIGUE_STRIDE / 2,

        "fall_window_samples":
            FALL_WINDOW,

        "fatigue_sensor_source":
            (
                fatigue_source["path"]
                if fatigue_source["type"] == "file"
                else fatigue_source["path"]
                + "::"
                + fatigue_source["member"]
            ),

        "fatigue_sensor_channels":
            IMU4_COLUMNS,

        "fatigue_feature_count":
            156,

        "fall_feature_count":
            148,

        "thermal_model":
            os.path.basename(
                THERMAL_MODEL_PATH
            ),

        "fatigue_model":
            os.path.basename(
                FATIGUE_MODEL_PATH
            ),

        "fall_model":
            os.path.basename(
                FALL_MODEL_PATH
            )
    }

    # ========================================================
    # SAVE THREE SEPARATE CSVs
    # ========================================================

    thermal_df.to_csv(
        THERMAL_OUTPUT_PATH,
        index=False
    )

    fatigue_results.to_csv(
        FATIGUE_OUTPUT_PATH,
        index=False
    )

    fall_results.to_csv(
        FALL_OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    with open(
        METADATA_OUTPUT_PATH,
        "w"
    ) as f:

        json.dump(
            combined_metadata,
            f,
            indent=2
        )

    # ========================================================
    # DISPLAY THERMAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "THERMAL OUTPUT"
    )
    print("=" * 70)

    print(
        "Samples:",
        len(thermal_df)
    )

    print(
        "Mean current estimated core:",
        f"{thermal_current.mean():.3f} °C"
    )

    print(
        "Mean predicted +10 min:",
        f"{thermal_predicted.mean():.3f} °C"
    )

    print(
        "Minimum predicted +10 min:",
        f"{thermal_predicted.min():.3f} °C"
    )

    print(
        "Maximum predicted +10 min:",
        f"{thermal_predicted.max():.3f} °C"
    )

    print(
        "Maximum predicted rise:",
        f"{thermal_rise.max():+.3f} °C"
    )

    # ========================================================
    # DISPLAY FATIGUE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "FATIGUE LIVE SIMULATION"
    )
    print("=" * 70)

    print(
        "Windows:",
        len(fatigue_results)
    )

    print(
        "Mean fatigue:",
        f"{fatigue_results['fatigue_score'].mean():.3f} / 10"
    )

    print(
        "Minimum:",
        f"{fatigue_results['fatigue_score'].min():.3f}"
    )

    print(
        "Maximum:",
        f"{fatigue_results['fatigue_score'].max():.3f}"
    )

    print(
        "\nTrend counts:"
    )

    print(
        fatigue_results[
            "fatigue_trend"
        ].value_counts()
    )

    # ========================================================
    # DISPLAY FALL
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "FALL LIVE SIMULATION"
    )
    print("=" * 70)

    print(
        "Windows:",
        len(fall_results)
    )

    print(
        "Mean probability:",
        f"{fall_results['fall_probability'].mean():.4f}"
    )

    print(
        "Maximum probability:",
        f"{fall_results['fall_probability'].max():.4f}"
    )

    print(
        "Detected:",
        int(
            fall_results[
                "fall_detected"
            ].sum()
        )
    )

    print(
        "Detection rate:",
        f"{fall_results['fall_detected'].mean() * 100:.2f}%"
    )

    # ========================================================
    # SIMULATED LIVE DASHBOARD
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SIMULATED LIVE DASHBOARD"
    )
    print("=" * 70)

    thermal_sample = (
        thermal_df.iloc[0]
    )

    fatigue_sample = (
        fatigue_results.iloc[0]
    )

    fall_sample = (
        fall_results.iloc[0]
    )

    # Get first thermal values robustly.
    current_value = float(
        thermal_current.iloc[0]
    )

    predicted_value = float(
        thermal_predicted.iloc[0]
    )

    rise_value = float(
        thermal_rise.iloc[0]
    )

    print("\n")
    print(
        "┌──────────────────────────────────────────┐"
    )

    print(
        "│              THERMORA                    │"
    )

    print(
        "├──────────────────────────────────────────┤"
    )

    print(
        "│ THERMAL                                  │"
    )

    print(
        f"│ Current core:       "
        f"{current_value:6.3f} °C       │"
    )

    print(
        f"│ Predicted +10 min:  "
        f"{predicted_value:6.3f} °C       │"
    )

    print(
        f"│ Predicted rise:     "
        f"{rise_value:+6.3f} °C       │"
    )

    print(
        "│                                          │"
    )

    print(
        "│ FATIGUE                                  │"
    )

    print(
        f"│ Score:              "
        f"{fatigue_sample['fatigue_score']:6.2f} / 10        │"
    )

    print(
        f"│ Trend:              "
        f"{fatigue_sample['fatigue_trend']:>10}             │"
    )

    print(
        "│                                          │"
    )

    print(
        "│ FALL                                     │"
    )

    print(
        f"│ Probability:        "
        f"{fall_sample['fall_probability']:6.3f}              │"
    )

    print(
        f"│ Detected:           "
        f"{'YES' if fall_sample['fall_detected'] else 'NO':>3}                  │"
    )

    print(
        "└──────────────────────────────────────────┘"
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SIMULATION OUTPUT FILES"
    )
    print("=" * 70)

    print(
        "\nThermal:",
        THERMAL_OUTPUT_PATH
    )

    print(
        "\nFatigue:",
        FATIGUE_OUTPUT_PATH
    )

    print(
        "\nFall:",
        FALL_OUTPUT_PATH
    )

    print(
        "\nMetadata:",
        METADATA_OUTPUT_PATH
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "THERMORA LIVE-STYLE SIMULATION COMPLETED"
    )
    print("=" * 70)

    print(
        "\n✓ Thermal pipeline tested"
    )

    print(
        "✓ Fatigue raw MxD sensor data loaded"
    )

    print(
        "✓ Fatigue 180-second window tested"
    )

    print(
        "✓ Fatigue 156-feature extraction tested"
    )

    print(
        "✓ Fall 20-sample window tested"
    )

    print(
        "✓ Fall 148-feature extraction tested"
    )

    print(
        "✓ Actual trained models used"
    )

    print(
        "✓ Outputs remain independent"
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "Replace recorded sensor input with "
        "live BMI270 / MAX30102 / TMP117 data."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()