from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import GroupKFold


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction"
)

# Existing 3-minute IMU4 window dataset
DATA_FILE = (
    BASE
    / "mxd_imu4_windows"
    / "imu4_3min_windows.csv"
)

# EXACT location of your extracted Composite MxD dataset
RAW_ROOT = (
    BASE
    / "fatigue_personalization_results"
    / "extracted"
    / "MxD_Data_User_Study"
    / "MxD_Data_User_Study"
    / "composite"
)

# Output directory
OUT = (
    BASE
    / "fatigue_multimodal_models"
)

OUT.mkdir(
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# CHECK PATHS
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nIMU4 window dataset not found:\n{DATA_FILE}"
    )


if not RAW_ROOT.exists():

    raise FileNotFoundError(
        f"\nMxD Composite dataset not found:\n{RAW_ROOT}"
    )


# ============================================================
# LOAD WINDOW DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

print("=" * 75)
print("THERMORA MULTIMODAL FATIGUE EXPERIMENT")
print("=" * 75)

print(
    f"Rows: {len(df)}"
)

print(
    f"Participants: "
    f"{df['participant_id'].nunique()}"
)

print(
    f"Repetitions: "
    f"{df[['participant_id', 'repetition']].drop_duplicates().shape[0]}"
)

print()

print(
    f"Raw MxD dataset:\n{RAW_ROOT}"
)

print()


# ============================================================
# PARTICIPANTS
# ============================================================

participants = sorted(
    df["participant_id"]
    .astype(str)
    .unique()
)


# ============================================================
# FIND PARTICIPANT FOLDER
# ============================================================

def find_participant_folder(pid):

    # Exact expected folder format:
    # P001C001S002
    matches = list(
        RAW_ROOT.glob(
            f"{pid}C*"
        )
    )

    # If exact pattern doesn't work, fall back to recursive search
    if not matches:

        matches = list(
            RAW_ROOT.rglob(
                f"{pid}*"
            )
        )

    dirs = [
        x for x in matches
        if x.is_dir()
    ]

    if not dirs:
        return None

    # Prefer directory containing CSV files
    for d in dirs:

        if list(d.glob("*.csv")):

            return d

    return dirs[0]


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
# SAFE NUMERIC CONVERSION
# ============================================================

def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ============================================================
# PHYSIOLOGICAL FEATURE EXTRACTION
# ============================================================

def extract_physio_features(
    segment,
    start_ms,
    end_ms
):

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if "Time" not in segment.columns:

        return None

    t = safe_numeric(
        segment["Time"]
    )

    mask = (
        (t >= start_ms)
        &
        (t <= end_ms)
    )

    w = segment.loc[
        mask
    ].copy()

    # Need enough samples for a meaningful window
    if len(w) < 20:

        return None

    features = {}


    # ========================================================
    # GENERIC STATISTICS
    # ========================================================

    def add_stats(
        prefix,
        col
    ):

        if col not in w.columns:

            return

        x = safe_numeric(
            w[col]
        )

        # MxD invalid-value sentinels
        x = x.replace(
            [
                -2,
                -999,
                -9999
            ],
            np.nan
        )

        x = x.dropna()

        if len(x) == 0:

            return

        features[
            f"{prefix}_mean"
        ] = x.mean()

        features[
            f"{prefix}_std"
        ] = x.std()

        features[
            f"{prefix}_min"
        ] = x.min()

        features[
            f"{prefix}_max"
        ] = x.max()

        features[
            f"{prefix}_median"
        ] = x.median()

        features[
            f"{prefix}_range"
        ] = (
            x.max()
            -
            x.min()
        )


    # ========================================================
    # HEART RATE
    # ========================================================

    add_stats(
        "HR",
        "HR_Processed"
    )

    # Raw HR
    add_stats(
        "HR_raw",
        "HR"
    )


    # ========================================================
    # HRV
    # ========================================================

    add_stats(
        "HRV",
        "HRV"
    )

    add_stats(
        "RR",
        "RR"
    )


    # ========================================================
    # SKIN TEMPERATURE
    # ========================================================

    add_stats(
        "SkinTemp",
        "Temperature"
    )


    # ========================================================
    # HEART RATE TREND
    # ========================================================

    if "HR_Processed" in w.columns:

        hr = safe_numeric(
            w["HR_Processed"]
        )

        hr = hr.replace(
            [
                -2,
                -999,
                -9999
            ],
            np.nan
        ).dropna()

        if len(hr) >= 2:

            features[
                "HR_first"
            ] = hr.iloc[0]

            features[
                "HR_last"
            ] = hr.iloc[-1]

            features[
                "HR_change"
            ] = (
                hr.iloc[-1]
                -
                hr.iloc[0]
            )


    # ========================================================
    # SKIN TEMPERATURE TREND
    # ========================================================

    if "Temperature" in w.columns:

        temp = safe_numeric(
            w["Temperature"]
        )

        temp = temp.replace(
            [
                -2,
                -999,
                -9999
            ],
            np.nan
        ).dropna()

        if len(temp) >= 2:

            features[
                "SkinTemp_first"
            ] = temp.iloc[0]

            features[
                "SkinTemp_last"
            ] = temp.iloc[-1]

            features[
                "SkinTemp_change"
            ] = (
                temp.iloc[-1]
                -
                temp.iloc[0]
            )


    # ========================================================
    # PERSONAL CHARACTERISTICS
    # ========================================================

    for col in [
        "Age",
        "Weight",
        "Height",
        "Weights added"
    ]:

        if col not in w.columns:

            continue

        values = safe_numeric(
            w[col]
        )

        values = values.replace(
            [
                -2,
                -999,
                -9999
            ],
            np.nan
        ).dropna()

        if len(values):

            feature_name = (
                col
                .replace(
                    " ",
                    "_"
                )
            )

            features[
                feature_name
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
# BUILD PHYSIOLOGICAL WINDOW DATA
# ============================================================

physio_rows = []

print(
    "Building physiological features..."
)

print()


for pid in participants:

    print(
        f"Processing {pid}"
    )

    participant_folder = (
        find_participant_folder(
            pid
        )
    )

    if participant_folder is None:

        print(
            f"  WARNING: participant folder not found"
        )

        continue

    participant_windows = df[
        df[
            "participant_id"
        ].astype(str)
        ==
        pid
    ]

    for _, row in (
        participant_windows
        .iterrows()
    ):

        rep = int(
            row["repetition"]
        )

        start_ms = float(
            row["window_start"]
        )

        end_ms = float(
            row["window_end"]
        )

        rep_file = find_rep_file(
            participant_folder,
            pid,
            rep
        )

        if rep_file is None:

            print(
                f"  WARNING: "
                f"Rep{rep} not found for {pid}"
            )

            continue

        try:

            segment = pd.read_csv(
                rep_file
            )

        except Exception as e:

            print(
                f"  ERROR reading "
                f"{rep_file}: {e}"
            )

            continue

        features = extract_physio_features(
            segment,
            start_ms,
            end_ms
        )

        if features is None:

            continue

        # Window identifiers
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

        physio_rows.append(
            features
        )


# ============================================================
# PHYSIO DATAFRAME
# ============================================================

physio = pd.DataFrame(
    physio_rows
)

print()

print("=" * 75)
print("PHYSIO WINDOW DATASET")
print("=" * 75)

print(
    f"Rows: {len(physio)}"
)

print(
    f"Participants: "
    f"{physio['participant_id'].nunique()}"
)

print(
    f"Features: "
    f"{len(physio.columns)}"
)

print()


if len(physio) == 0:

    raise RuntimeError(
        "No physiological windows were created."
    )


# ============================================================
# MERGE IMU4 + PHYSIO
# ============================================================

KEYS = [
    "participant_id",
    "repetition",
    "window_start",
    "window_end"
]

merged = df.merge(
    physio,
    on=KEYS,
    how="inner"
)

print(
    f"Merged windows: {len(merged)}"
)

print()


# ============================================================
# CHECK MERGE
# ============================================================

if len(merged) == 0:

    raise RuntimeError(
        "No windows survived the IMU4 + physiological merge."
    )


# ============================================================
# SAVE MERGED DATASET
# ============================================================

merged.to_csv(
    OUT /
    "imu4_physio_merged.csv",
    index=False
)


# ============================================================
# DEFINE IMU FEATURES
# ============================================================

DROP = {
    "participant_id",
    "repetition",
    "window_start",
    "window_end",
    "fatigue"
}

imu_features = [
    c
    for c in df.columns
    if c not in DROP
]


# ============================================================
# DEFINE PHYSIO FEATURES
# ============================================================

physio_features = [
    c
    for c in physio.columns
    if c not in {
        "participant_id",
        "repetition",
        "window_start",
        "window_end"
    }
]


# Remove target if accidentally present
physio_features = [
    c
    for c in physio_features
    if c != "fatigue"
]


print(
    f"IMU features: "
    f"{len(imu_features)}"
)

print(
    f"Physio features: "
    f"{len(physio_features)}"
)

print()


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = {

    # --------------------------------------------------------
    # A: Current baseline
    # --------------------------------------------------------

    "IMU4_ONLY":
        imu_features,

    # --------------------------------------------------------
    # B: Physiological sensors only
    # --------------------------------------------------------

    "PHYSIO_ONLY":
        physio_features,

    # --------------------------------------------------------
    # C: Multimodal
    # --------------------------------------------------------

    "IMU4_PLUS_PHYSIO":
        imu_features
        +
        physio_features
}


# Remove duplicates
for name in experiments:

    experiments[name] = list(
        dict.fromkeys(
            experiments[name]
        )
    )


# ============================================================
# MODELS
# ============================================================

def make_rf():

    return RandomForestRegressor(

        n_estimators=400,

        min_samples_leaf=3,

        max_features="sqrt",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


def make_hgb():

    return HistGradientBoostingRegressor(

        max_iter=300,

        learning_rate=0.05,

        max_leaf_nodes=31,

        min_samples_leaf=10,

        l2_regularization=1.0,

        random_state=RANDOM_STATE
    )


models = {

    "RF":
        make_rf,

    "HGB":
        make_hgb
}


# ============================================================
# TARGET + GROUPS
# ============================================================

y = merged[
    "fatigue"
]

groups = merged[
    "participant_id"
]


# ============================================================
# 5-FOLD PARTICIPANT-HELD-OUT CV
# ============================================================

gkf = GroupKFold(
    n_splits=5
)

results = []

predictions = []


# ============================================================
# RUN EXPERIMENTS
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


    X = merged[
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

            # ------------------------------------------------
            # Train/test split
            # ------------------------------------------------

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
            # Median imputation
            #
            # FIT ONLY ON TRAINING PARTICIPANTS
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


            # ------------------------------------------------
            # PRINT
            # ------------------------------------------------

            print(
                f"Fold {fold}: "
                f"MAE={mae:.4f} | "
                f"RMSE={rmse:.4f} | "
                f"R²={r2:.4f}"
            )


            # ------------------------------------------------
            # SAVE METRICS
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
            # SAVE PREDICTIONS
            # ------------------------------------------------

            pred_df = merged.iloc[
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

            pred_df[
                "prediction"
            ] = prediction

            pred_df[
                "absolute_error"
            ] = np.abs(
                prediction
                -
                y_test.to_numpy()
            )

            pred_df[
                "experiment"
            ] = experiment_name

            pred_df[
                "model"
            ] = model_name

            pred_df[
                "fold"
            ] = fold

            predictions.append(
                pred_df
            )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.concat(
    predictions,
    ignore_index=True
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

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

print(
    "FINAL COMPARISON"
)

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
# BEST MODEL
# ============================================================

best = (
    summary
    .sort_values(
        "MAE_mean"
    )
    .iloc[0]
)


print(
    "=" * 75
)

print(
    "BEST RESULT"
)

print(
    "=" * 75
)

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
    f"Results saved to:\n{OUT}"
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)