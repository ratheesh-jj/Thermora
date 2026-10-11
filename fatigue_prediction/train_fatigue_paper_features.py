from pathlib import Path

import warnings
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GroupKFold

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction"
)

WINDOW_FILE = (
    BASE
    / "mxd_imu4_windows"
    / "imu4_3min_windows.csv"
)

RAW_ROOT = (
    BASE
    / "fatigue_personalization_results"
    / "extracted"
    / "MxD_Data_User_Study"
    / "MxD_Data_User_Study"
    / "composite"
)

OUT = (
    BASE
    / "fatigue_paper_features_v2"
)

OUT.mkdir(
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# WARNINGS
# ============================================================

# We handle constant signals explicitly below.
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation"
)


# ============================================================
# CHECK FILES
# ============================================================

if not WINDOW_FILE.exists():

    raise FileNotFoundError(
        f"\nWindow dataset not found:\n{WINDOW_FILE}"
    )

if not RAW_ROOT.exists():

    raise FileNotFoundError(
        f"\nRaw Composite dataset not found:\n{RAW_ROOT}"
    )


# ============================================================
# LOAD ORIGINAL WINDOW DATA
# ============================================================

windows = pd.read_csv(
    WINDOW_FILE
)

print("=" * 75)
print("THERMORA PAPER-STYLE FATIGUE FEATURES V2")
print("=" * 75)

print(
    f"Original windows : {len(windows)}"
)

print(
    f"Participants      : "
    f"{windows['participant_id'].nunique()}"
)

print(
    f"Repetitions       : "
    f"{windows[['participant_id', 'repetition']].drop_duplicates().shape[0]}"
)

print()

print(
    f"Raw dataset:\n{RAW_ROOT}"
)

print()


# ============================================================
# EXPECTED DATASET
# ============================================================

expected_participants = sorted(
    windows[
        "participant_id"
    ]
    .astype(str)
    .unique()
)

expected_repetitions = (
    windows[
        [
            "participant_id",
            "repetition"
        ]
    ]
    .drop_duplicates()
)


# ============================================================
# BUILD EXACT PARTICIPANT FOLDER MAP
# ============================================================

print(
    "Indexing participant folders..."
)

participant_folders = {}

for folder in RAW_ROOT.iterdir():

    if not folder.is_dir():
        continue

    name = folder.name.upper()

    # Example:
    # P001C001S002

    if name.startswith("P"):

        pid = name[
            :4
        ]

        if (
            len(pid) == 4
            and pid[1:].isdigit()
        ):

            participant_folders[
                pid
            ] = folder


print(
    f"Indexed participant folders: "
    f"{len(participant_folders)}"
)

print()


# ============================================================
# VERIFY PARTICIPANTS BEFORE PROCESSING
# ============================================================

missing_participants = [
    p
    for p in expected_participants
    if p not in participant_folders
]

if missing_participants:

    print(
        "WARNING: Missing participant folders:"
    )

    print(
        missing_participants
    )

else:

    print(
        "All expected participant folders found."
    )

print()


# ============================================================
# FILE CACHE
# ============================================================

raw_cache = {}


# ============================================================
# FIND REPETITION FILE
# ============================================================

def find_rep_file(
    participant_folder,
    pid,
    rep
):

    patterns = [

        f"Rep{rep}_{pid}*.csv",

        f"rep{rep}_{pid}*.csv",

        f"*Rep{rep}*{pid}*.csv",

        f"*rep{rep}*{pid}*.csv"
    ]

    for pattern in patterns:

        matches = list(
            participant_folder.glob(
                pattern
            )
        )

        if matches:

            return matches[0]

    return None


# ============================================================
# LOAD REPETITION
# ============================================================

def load_repetition(
    pid,
    rep
):

    key = (
        pid,
        rep
    )

    if key in raw_cache:

        return raw_cache[key]

    if pid not in participant_folders:

        raw_cache[key] = None

        return None

    folder = participant_folders[
        pid
    ]

    file = find_rep_file(
        folder,
        pid,
        rep
    )

    if file is None:

        raw_cache[key] = None

        return None

    try:

        segment = pd.read_csv(
            file
        )

        raw_cache[key] = segment

        return segment

    except Exception as e:

        print(
            f"ERROR reading {file}: {e}"
        )

        raw_cache[key] = None

        return None


# ============================================================
# NUMERIC CLEANING
# ============================================================

def clean_numeric(series):

    x = pd.to_numeric(
        series,
        errors="coerce"
    )

    return x.replace(
        [
            -2,
            -999,
            -9999
        ],
        np.nan
    )


# ============================================================
# ADD STATISTICS
# ============================================================

def add_statistics(
    features,
    prefix,
    values
):

    x = np.asarray(
        values,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return

    mean = np.mean(x)

    std = (
        np.std(
            x,
            ddof=1
        )
        if len(x) > 1
        else 0.0
    )

    features[
        f"{prefix}_mean"
    ] = mean

    features[
        f"{prefix}_std"
    ] = std

    features[
        f"{prefix}_median"
    ] = np.median(x)

    features[
        f"{prefix}_min"
    ] = np.min(x)

    features[
        f"{prefix}_max"
    ] = np.max(x)

    features[
        f"{prefix}_range"
    ] = np.max(x) - np.min(x)

    features[
        f"{prefix}_rms"
    ] = np.sqrt(
        np.mean(
            x ** 2
        )
    )

    features[
        f"{prefix}_abs_mean"
    ] = np.mean(
        np.abs(x)
    )

    features[
        f"{prefix}_q25"
    ] = np.percentile(
        x,
        25
    )

    features[
        f"{prefix}_q75"
    ] = np.percentile(
        x,
        75
    )

    # --------------------------------------------------------
    # Avoid meaningless skew/kurtosis on constant signals
    # --------------------------------------------------------

    if (
        len(x) >= 3
        and std > 1e-12
    ):

        s = skew(
            x,
            bias=False
        )

        if np.isfinite(s):

            features[
                f"{prefix}_skew"
            ] = s

    if (
        len(x) >= 4
        and std > 1e-12
    ):

        k = kurtosis(
            x,
            bias=False
        )

        if np.isfinite(k):

            features[
                f"{prefix}_kurtosis"
            ] = k


# ============================================================
# TREND
# ============================================================

def add_trend(
    features,
    prefix,
    values
):

    x = np.asarray(
        values,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) < 2:

        return

    features[
        f"{prefix}_first"
    ] = x[0]

    features[
        f"{prefix}_last"
    ] = x[-1]

    features[
        f"{prefix}_change"
    ] = x[-1] - x[0]

    t = np.arange(
        len(x),
        dtype=float
    )

    if np.std(x) > 1e-12:

        slope = np.polyfit(
            t,
            x,
            1
        )[0]

    else:

        slope = 0.0

    features[
        f"{prefix}_slope"
    ] = slope


# ============================================================
# IMU COLUMNS
# ============================================================

IMU_COLUMNS = [

    "IMU_4_ax_g_",
    "IMU_4_ay_g_",
    "IMU_4_az_g_",

    "IMU_4_gx_dps_",
    "IMU_4_gy_dps_",
    "IMU_4_gz_dps_"
]


# ============================================================
# BUILD ONE WINDOW
# ============================================================

def build_features(
    segment,
    start_ms,
    end_ms
):

    if "Time" not in segment.columns:

        return None

    time = clean_numeric(
        segment["Time"]
    )

    mask = (
        (time >= start_ms)
        &
        (time <= end_ms)
    )

    w = segment.loc[
        mask
    ].copy()

    if len(w) < 20:

        return None

    features = {}

    imu_data = {}


    # ========================================================
    # IMU AXES
    # ========================================================

    for col in IMU_COLUMNS:

        if col not in w.columns:

            continue

        x = clean_numeric(
            w[col]
        ).to_numpy()

        imu_data[
            col
        ] = x

        if "ax" in col:

            prefix = "IMU4_acc_x"

        elif "ay" in col:

            prefix = "IMU4_acc_y"

        elif "az" in col:

            prefix = "IMU4_acc_z"

        elif "gx" in col:

            prefix = "IMU4_gyro_x"

        elif "gy" in col:

            prefix = "IMU4_gyro_y"

        else:

            prefix = "IMU4_gyro_z"


        add_statistics(
            features,
            prefix,
            x
        )

        add_trend(
            features,
            prefix,
            x
        )


    # ========================================================
    # ACCELERATION MAGNITUDE
    # ========================================================

    required_acc = [

        "IMU_4_ax_g_",
        "IMU_4_ay_g_",
        "IMU_4_az_g_"
    ]

    if all(
        c in imu_data
        for c in required_acc
    ):

        ax = imu_data[
            "IMU_4_ax_g_"
        ]

        ay = imu_data[
            "IMU_4_ay_g_"
        ]

        az = imu_data[
            "IMU_4_az_g_"
        ]

        valid = (
            np.isfinite(ax)
            &
            np.isfinite(ay)
            &
            np.isfinite(az)
        )

        ax = ax[valid]
        ay = ay[valid]
        az = az[valid]

        if len(ax):

            acc_mag = np.sqrt(
                ax ** 2
                +
                ay ** 2
                +
                az ** 2
            )

            add_statistics(
                features,
                "IMU4_acc_mag",
                acc_mag
            )

            add_trend(
                features,
                "IMU4_acc_mag",
                acc_mag
            )

            dynamic = (
                acc_mag
                -
                np.mean(
                    acc_mag
                )
            )

            add_statistics(
                features,
                "IMU4_dynamic_acc",
                dynamic
            )

            features[
                "IMU4_acc_energy"
            ] = np.mean(
                acc_mag ** 2
            )

            features[
                "IMU4_activity_variance"
            ] = np.var(
                acc_mag
            )

            # ------------------------------------------------
            # JERK
            # ------------------------------------------------

            if len(acc_mag) >= 2:

                jerk = np.diff(
                    acc_mag
                )

                add_statistics(
                    features,
                    "IMU4_jerk",
                    jerk
                )

                features[
                    "IMU4_jerk_energy"
                ] = np.mean(
                    jerk ** 2
                )


    # ========================================================
    # GYRO MAGNITUDE
    # ========================================================

    required_gyro = [

        "IMU_4_gx_dps_",
        "IMU_4_gy_dps_",
        "IMU_4_gz_dps_"
    ]

    if all(
        c in imu_data
        for c in required_gyro
    ):

        gx = imu_data[
            "IMU_4_gx_dps_"
        ]

        gy = imu_data[
            "IMU_4_gy_dps_"
        ]

        gz = imu_data[
            "IMU_4_gz_dps_"
        ]

        valid = (
            np.isfinite(gx)
            &
            np.isfinite(gy)
            &
            np.isfinite(gz)
        )

        gx = gx[valid]
        gy = gy[valid]
        gz = gz[valid]

        if len(gx):

            gyro_mag = np.sqrt(
                gx ** 2
                +
                gy ** 2
                +
                gz ** 2
            )

            add_statistics(
                features,
                "IMU4_gyro_mag",
                gyro_mag
            )

            add_trend(
                features,
                "IMU4_gyro_mag",
                gyro_mag
            )

            features[
                "IMU4_gyro_energy"
            ] = np.mean(
                gyro_mag ** 2
            )


    # ========================================================
    # HR
    # ========================================================

    if "HR_Processed" in w.columns:

        hr = clean_numeric(
            w[
                "HR_Processed"
            ]
        ).dropna().to_numpy()

        add_statistics(
            features,
            "HR",
            hr
        )

        add_trend(
            features,
            "HR",
            hr
        )


    # ========================================================
    # RAW HR
    # ========================================================

    if "HR" in w.columns:

        hr_raw = clean_numeric(
            w["HR"]
        ).dropna().to_numpy()

        add_statistics(
            features,
            "HR_raw",
            hr_raw
        )


    # ========================================================
    # HRV
    # ========================================================

    if "HRV" in w.columns:

        hrv = clean_numeric(
            w["HRV"]
        ).dropna().to_numpy()

        add_statistics(
            features,
            "HRV",
            hrv
        )

        add_trend(
            features,
            "HRV",
            hrv
        )


    # ========================================================
    # RR
    # ========================================================

    if "RR" in w.columns:

        rr = clean_numeric(
            w["RR"]
        ).dropna().to_numpy()

        add_statistics(
            features,
            "RR",
            rr
        )

        add_trend(
            features,
            "RR",
            rr
        )


    # ========================================================
    # SKIN TEMPERATURE
    # ========================================================

    if "Temperature" in w.columns:

        temp = clean_numeric(
            w["Temperature"]
        ).dropna().to_numpy()

        add_statistics(
            features,
            "SkinTemp",
            temp
        )

        add_trend(
            features,
            "SkinTemp",
            temp
        )


    # ========================================================
    # PERSONAL FEATURES
    # ========================================================

    for col in [
        "Age",
        "Weight",
        "Height",
        "Weights added"
    ]:

        if col not in w.columns:

            continue

        values = clean_numeric(
            w[col]
        ).dropna()

        if len(values):

            name = col.replace(
                " ",
                "_"
            )

            features[
                name
            ] = values.iloc[0]


    # ========================================================
    # GENDER
    # ========================================================

    if "Gender" in w.columns:

        gender = (
            w["Gender"]
            .dropna()
        )

        if len(gender):

            g = str(
                gender.iloc[0]
            ).strip().lower()

            if g in [
                "male",
                "m",
                "1"
            ]:

                features[
                    "Gender_male"
                ] = 1

            elif g in [
                "female",
                "f",
                "0"
            ]:

                features[
                    "Gender_male"
                ] = 0


    return features


# ============================================================
# BUILD ALL WINDOWS
# ============================================================

rows = []

missing_raw_files = []

failed_windows = []

print(
    "Building paper-style features..."
)

print()


for i, (_, window) in enumerate(
    windows.iterrows(),
    start=1
):

    pid = str(
        window[
            "participant_id"
        ]
    )

    rep = int(
        window[
            "repetition"
        ]
    )

    start_ms = float(
        window[
            "window_start"
        ]
    )

    end_ms = float(
        window[
            "window_end"
        ]
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        i == 1
        or pid != str(
            windows.iloc[
                i - 2
            ]["participant_id"]
        )
    ):

        print(
            f"Processing {pid}"
        )


    # --------------------------------------------------------
    # Load repetition
    # --------------------------------------------------------

    segment = load_repetition(
        pid,
        rep
    )

    if segment is None:

        missing_raw_files.append(
            (
                pid,
                rep
            )
        )

        continue


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = build_features(
        segment,
        start_ms,
        end_ms
    )

    if features is None:

        failed_windows.append(
            (
                pid,
                rep,
                start_ms,
                end_ms
            )
        )

        continue


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    features[
        "participant_id"
    ] = pid

    features[
        "repetition"
    ] = rep

    features[
        "window_start"
    ] = start_ms

    features[
        "window_end"
    ] = end_ms

    features[
        "fatigue"
    ] = float(
        window[
            "fatigue"
        ]
    )


    rows.append(
        features
    )


# ============================================================
# BUILD DATAFRAME
# ============================================================

data = pd.DataFrame(
    rows
)

print()

print("=" * 75)
print("FEATURE EXTRACTION CHECK")
print("=" * 75)

print(
    f"Original windows : {len(windows)}"
)

print(
    f"Processed windows: {len(data)}"
)

print(
    f"Missing windows  : "
    f"{len(windows) - len(data)}"
)

print(
    f"Participants     : "
    f"{data['participant_id'].nunique()}"
)

print(
    f"Repetitions      : "
    f"{data[['participant_id', 'repetition']].drop_duplicates().shape[0]}"
)

print(
    f"Columns          : "
    f"{len(data.columns)}"
)

print()


# ============================================================
# DETAILED MISSING DATA REPORT
# ============================================================

if missing_raw_files:

    print(
        "Missing raw repetition files:"
    )

    for item in sorted(
        set(missing_raw_files)
    ):

        print(
            f"  {item}"
        )

    print()

if failed_windows:

    print(
        "Windows that could not be processed:"
    )

    print(
        f"  Count: {len(failed_windows)}"
    )

    print()


# ============================================================
# CRITICAL VALIDATION
# ============================================================

original_keys = set(
    zip(
        windows[
            "participant_id"
        ].astype(str),

        windows[
            "repetition"
        ].astype(int),

        windows[
            "window_start"
        ].astype(float),

        windows[
            "window_end"
        ].astype(float)
    )
)

processed_keys = set(
    zip(
        data[
            "participant_id"
        ].astype(str),

        data[
            "repetition"
        ].astype(int),

        data[
            "window_start"
        ].astype(float),

        data[
            "window_end"
        ].astype(float)
    )
)

missing_keys = (
    original_keys
    -
    processed_keys
)

extra_keys = (
    processed_keys
    -
    original_keys
)

print(
    f"Missing original keys: "
    f"{len(missing_keys)}"
)

print(
    f"Unexpected extra keys: "
    f"{len(extra_keys)}"
)

print()


# ============================================================
# DO NOT TRAIN IF DATA WAS LOST
# ============================================================

if (
    len(data)
    != len(windows)
):

    print("=" * 75)

    print(
        "STOPPING BEFORE TRAINING"
    )

    print("=" * 75)

    print()

    print(
        "Feature extraction did not preserve "
        "all original windows."
    )

    print(
        "Fix the missing raw files/windows before "
        "using this as a benchmark."
    )

    print()

    raise RuntimeError(
        f"Expected {len(windows)} windows, "
        f"but only {len(data)} were processed."
    )


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

feature_file = (
    OUT /
    "paper_style_features.csv"
)

data.to_csv(
    feature_file,
    index=False
)

print(
    f"Feature dataset saved:\n"
    f"{feature_file}"
)

print()


# ============================================================
# FEATURE GROUPS
# ============================================================

META = {
    "participant_id",
    "repetition",
    "window_start",
    "window_end",
    "fatigue"
}


ALL_FEATURES = [
    c
    for c in data.columns
    if c not in META
]


IMU_FEATURES = [
    c
    for c in ALL_FEATURES
    if c.startswith(
        "IMU4_"
    )
]


PHYSIO_FEATURES = [
    c
    for c in ALL_FEATURES
    if c.startswith(
        (
            "HR_",
            "HRV_",
            "RR_",
            "SkinTemp_"
        )
    )
]


PERSONAL_FEATURES = [
    c
    for c in ALL_FEATURES
    if c in [
        "Age",
        "Weight",
        "Height",
        "Weights_added",
        "Gender_male"
    ]
]


print(
    f"IMU features      : "
    f"{len(IMU_FEATURES)}"
)

print(
    f"Physio features   : "
    f"{len(PHYSIO_FEATURES)}"
)

print(
    f"Personal features : "
    f"{len(PERSONAL_FEATURES)}"
)

print()


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = {

    "IMU4_ONLY":
        IMU_FEATURES,

    "PHYSIO_ONLY":
        PHYSIO_FEATURES
        +
        PERSONAL_FEATURES,

    "IMU4_PLUS_PHYSIO":
        IMU_FEATURES
        +
        PHYSIO_FEATURES
        +
        PERSONAL_FEATURES
}


# ============================================================
# MODELS
# ============================================================

def make_rf():

    return RandomForestRegressor(

        n_estimators=500,

        min_samples_leaf=3,

        max_features="sqrt",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


def make_xgb():

    if not XGB_AVAILABLE:

        return None

    return XGBRegressor(

        n_estimators=500,

        max_depth=5,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=3,

        reg_alpha=0.1,

        reg_lambda=1.0,

        objective="reg:squarederror",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


models = {
    "RF": make_rf
}

if XGB_AVAILABLE:

    models[
        "XGB"
    ] = make_xgb

else:

    print(
        "WARNING: XGBoost not installed."
    )

    print(
        "Only RF will be tested."
    )

    print()


# ============================================================
# TARGET / GROUPS
# ============================================================

y = data[
    "fatigue"
]

groups = data[
    "participant_id"
]


# ============================================================
# 5-FOLD GROUP CV
# ============================================================

gkf = GroupKFold(
    n_splits=5
)

results = []

predictions = []

feature_importance_rows = []


# ============================================================
# TRAIN
# ============================================================

for (
    experiment_name,
    feature_list
) in experiments.items():

    print()

    print("=" * 75)

    print(
        experiment_name
    )

    print(
        f"Features: "
        f"{len(feature_list)}"
    )

    print("=" * 75)


    X = data[
        feature_list
    ]


    for (
        model_name,
        model_factory
    ) in models.items():

        print()

        print(
            f"--- {model_name} ---"
        )


        for fold, (
            train_idx,
            test_idx
        ) in enumerate(

            gkf.split(
                X,
                y,
                groups
            ),

            start=1
        ):

            X_train = X.iloc[
                train_idx
            ]

            X_test = X.iloc[
                test_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            y_test = y.iloc[
                test_idx
            ]


            # ------------------------------------------------
            # IMPUTATION
            # ------------------------------------------------

            imputer = SimpleImputer(
                strategy="median"
            )

            X_train_imp = (
                imputer
                .fit_transform(
                    X_train
                )
            )

            X_test_imp = (
                imputer
                .transform(
                    X_test
                )
            )


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            model = model_factory()

            if model is None:

                continue


            model.fit(
                X_train_imp,
                y_train
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                X_test_imp
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            mae = mean_absolute_error(
                y_test,
                prediction
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    prediction
                )
            )

            r2 = r2_score(
                y_test,
                prediction
            )


            print(
                f"Fold {fold}: "
                f"MAE={mae:.4f} | "
                f"RMSE={rmse:.4f} | "
                f"R²={r2:.4f}"
            )


            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            results.append({

                "experiment":
                    experiment_name,

                "model":
                    model_name,

                "fold":
                    fold,

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "R2":
                    r2,

                "train_samples":
                    len(train_idx),

                "test_samples":
                    len(test_idx),

                "train_participants":
                    groups.iloc[
                        train_idx
                    ].nunique(),

                "test_participants":
                    groups.iloc[
                        test_idx
                    ].nunique()
            })


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            pred = data.iloc[
                test_idx
            ][
                [
                    "participant_id",
                    "repetition",
                    "window_start",
                    "window_end",
                    "fatigue"
                ]
            ].copy()

            pred[
                "prediction"
            ] = prediction

            pred[
                "absolute_error"
            ] = np.abs(
                prediction
                -
                y_test.to_numpy()
            )

            pred[
                "experiment"
            ] = experiment_name

            pred[
                "model"
            ] = model_name

            pred[
                "fold"
            ] = fold

            predictions.append(
                pred
            )


            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------

            if hasattr(
                model,
                "feature_importances_"
            ):

                importances = (
                    model.feature_importances_
                )

                for (
                    feature,
                    importance
                ) in zip(
                    feature_list,
                    importances
                ):

                    feature_importance_rows.append({

                        "experiment":
                            experiment_name,

                        "model":
                            model_name,

                        "fold":
                            fold,

                        "feature":
                            feature,

                        "importance":
                            importance
                    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.concat(
    predictions,
    ignore_index=True
)


results_df.to_csv(
    OUT /
    "fold_results.csv",
    index=False
)

predictions_df.to_csv(
    OUT /
    "predictions.csv",
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if feature_importance_rows:

    importance_df = pd.DataFrame(
        feature_importance_rows
    )

    importance_summary = (
        importance_df
        .groupby(
            [
                "experiment",
                "model",
                "feature"
            ]
        )[
            "importance"
        ]
        .mean()
        .reset_index()
        .sort_values(
            "importance",
            ascending=False
        )
    )

    importance_summary.to_csv(
        OUT /
        "feature_importance.csv",
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

summary = (

    results_df

    .groupby(
        [
            "experiment",
            "model"
        ]
    )

    .agg(

        MAE_mean=(
            "MAE",
            "mean"
        ),

        MAE_std=(
            "MAE",
            "std"
        ),

        RMSE_mean=(
            "RMSE",
            "mean"
        ),

        RMSE_std=(
            "RMSE",
            "std"
        ),

        R2_mean=(
            "R2",
            "mean"
        ),

        R2_std=(
            "R2",
            "std"
        )
    )

    .reset_index()
)


summary.to_csv(
    OUT /
    "summary.csv",
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()

print("=" * 75)
print("FINAL COMPARISON")
print("=" * 75)

print()


for _, row in (
    summary.iterrows()
):

    print(
        f"{row['experiment']} | "
        f"{row['model']}"
    )

    print(
        f"  MAE  = "
        f"{row['MAE_mean']:.4f} "
        f"± "
        f"{row['MAE_std']:.4f}"
    )

    print(
        f"  RMSE = "
        f"{row['RMSE_mean']:.4f} "
        f"± "
        f"{row['RMSE_std']:.4f}"
    )

    print(
        f"  R²   = "
        f"{row['R2_mean']:.4f} "
        f"± "
        f"{row['R2_std']:.4f}"
    )

    print()


# ============================================================
# COMPARISON WITH PREVIOUS BASELINE
# ============================================================

PREVIOUS_BASELINE = 1.6715

best = (
    summary
    .sort_values(
        "MAE_mean"
    )
    .iloc[0]
)

improvement = (
    (
        PREVIOUS_BASELINE
        -
        best["MAE_mean"]
    )
    /
    PREVIOUS_BASELINE
) * 100


print("=" * 75)
print("BEST RESULT")
print("=" * 75)

print(
    f"Experiment : "
    f"{best['experiment']}"
)

print(
    f"Model      : "
    f"{best['model']}"
)

print(
    f"MAE        : "
    f"{best['MAE_mean']:.4f} "
    f"± "
    f"{best['MAE_std']:.4f}"
)

print(
    f"RMSE       : "
    f"{best['RMSE_mean']:.4f} "
    f"± "
    f"{best['RMSE_std']:.4f}"
)

print(
    f"R²         : "
    f"{best['R2_mean']:.4f} "
    f"± "
    f"{best['R2_std']:.4f}"
)

print()

print(
    f"Change vs previous "
    f"IMU4 RF baseline: "
    f"{improvement:.2f}%"
)

print()

print(
    f"Results saved to:\n{OUT}"
)

print()

print("=" * 75)
print("DONE")
print("=" * 75)