import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# THERMORA
# DEPLOYABLE 10-MINUTE CORE TEMPERATURE PREDICTOR
# NO-WIND VERSION
# ============================================================
#
# TWO-STAGE PIPELINE
#
# Stage 1:
#   Estimate current core temperature using only deployable
#   THERMORA signals.
#
# Stage 2:
#   Predict core temperature 10 minutes into the future.
#
# WIND IS NOT USED.
#
# This is a research/prototype model and is NOT a clinical
# or medical-grade core-temperature measurement system.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)

CORE_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "thermora_core_temp_estimator_no_wind.pkl"
)

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Accept target observations approximately 10 minutes away.
TARGET_TOLERANCE_MIN = 0.75


# ============================================================
# DEPLOYABLE FEATURES
# ============================================================
#
# These are features that THERMORA can realistically obtain
# from its wristband + environmental node + user profile.
#
# IMPORTANT:
# Wind is deliberately excluded.
#
# Direct core temperature is also excluded because the wristband
# cannot directly measure core temperature.
# ============================================================

BASE_FEATURES = [

    # Activity
    "activity_code",

    # Work/rest state
    "work_rest",

    # Personal factors
    "sex",
    "age_years",
    "body_mass_kg",
    "height_cm",

    # Environment
    "humidity_pct",
    "ambient_temp_c",

    # Physiological signals
    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",

    # Skin temperature
    "skin_temp_mean_c"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("THERMORA 10-MINUTE CORE TEMPERATURE PREDICTOR")
print("DEPLOYABLE VERSION - NO WIND")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading environmental/physiological dataset...")

df = pd.read_csv(DATA_FILE)

print(
    f"Original rows: {len(df)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    BASE_FEATURES
    + [
        "participant_id",
        "condition_id",
        "time_min",
        "core_temp_c"
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# CLEAN TIME AND TARGET
# ============================================================

df["time_min"] = pd.to_numeric(
    df["time_min"],
    errors="coerce"
)

df["core_temp_c"] = pd.to_numeric(
    df["core_temp_c"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "participant_id",
        "condition_id",
        "time_min",
        "core_temp_c"
    ]
).copy()


df = df.sort_values(
    [
        "participant_id",
        "condition_id",
        "time_min"
    ]
).reset_index(drop=True)


print(
    f"Valid rows: {len(df)}"
)

print(
    f"Participants: "
    f"{df['participant_id'].nunique()}"
)


# ============================================================
# BUILD +10 MINUTE TARGET
# ============================================================
#
# For each observation at time t:
#
#       target = core temperature at approximately t + 10 min
#
# Same participant + same condition only.
#
# This avoids mixing different experiments/participants.
# ============================================================

print()
print("-" * 70)
print("BUILDING +10 MINUTE TARGETS")
print("-" * 70)


future_targets = []

actual_target_offsets = []


for (
    participant_id,
    condition_id
), group in df.groupby(
    [
        "participant_id",
        "condition_id"
    ],
    sort=False
):

    group = group.sort_values(
        "time_min"
    ).reset_index(drop=True)


    times = group[
        "time_min"
    ].to_numpy(
        dtype=float
    )

    core_values = group[
        "core_temp_c"
    ].to_numpy(
        dtype=float
    )


    for i in range(
        len(group)
    ):

        current_time = times[i]

        desired_time = (
            current_time + 10.0
        )


        # ----------------------------------------------------
        # Find observations at or after desired target time
        # ----------------------------------------------------

        future_mask = (
            times >= desired_time
        )


        if not future_mask.any():

            future_targets.append(
                np.nan
            )

            actual_target_offsets.append(
                np.nan
            )

            continue


        future_indices = np.where(
            future_mask
        )[0]


        # Closest future observation
        closest_idx = future_indices[
            np.argmin(
                np.abs(
                    times[
                        future_indices
                    ]
                    - desired_time
                )
            )
        ]


        actual_offset = (
            times[closest_idx]
            - current_time
        )


        # ----------------------------------------------------
        # Accept only approximately +10 minute targets
        # ----------------------------------------------------

        if (
            abs(
                actual_offset - 10.0
            )
            <= TARGET_TOLERANCE_MIN
        ):

            future_targets.append(
                core_values[
                    closest_idx
                ]
            )

            actual_target_offsets.append(
                actual_offset
            )

        else:

            future_targets.append(
                np.nan
            )

            actual_target_offsets.append(
                np.nan
            )


# Safety check
if len(future_targets) != len(df):

    raise RuntimeError(
        "Target construction failed: "
        "target count does not match dataframe rows."
    )


df["future_core_temp_c"] = (
    future_targets
)

df["target_offset_min"] = (
    actual_target_offsets
)


# ============================================================
# KEEP VALID TARGETS
# ============================================================

valid = df[
    pd.notna(
        df["future_core_temp_c"]
    )
].copy()


print(
    f"+10 minute target rows: "
    f"{len(valid)}"
)

print(
    f"Participants with targets: "
    f"{valid['participant_id'].nunique()}"
)


if len(valid) == 0:

    raise RuntimeError(
        "No valid +10 minute targets were created."
    )


# ============================================================
# TARGET OFFSET CHECK
# ============================================================

offset_mean = (
    valid["target_offset_min"]
    .mean()
)

offset_std = (
    valid["target_offset_min"]
    .std()
)

offset_min = (
    valid["target_offset_min"]
    .min()
)

offset_max = (
    valid["target_offset_min"]
    .max()
)


print()
print(
    "Target offset statistics:"
)

print(
    f"  Mean: "
    f"{offset_mean:.3f} min"
)

print(
    f"  Std : "
    f"{offset_std:.3f} min"
)

print(
    f"  Min : "
    f"{offset_min:.3f} min"
)

print(
    f"  Max : "
    f"{offset_max:.3f} min"
)


# ============================================================
# PARTICIPANT-HELD-OUT SPLIT
# ============================================================
#
# IMPORTANT:
# The same participant must NEVER appear in both training
# and testing.
# ============================================================

print()
print("-" * 70)
print("PARTICIPANT-HELD-OUT SPLIT")
print("-" * 70)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)


train_idx, test_idx = next(
    splitter.split(
        valid,
        groups=valid[
            "participant_id"
        ]
    )
)


train_df = valid.iloc[
    train_idx
].copy()

test_df = valid.iloc[
    test_idx
].copy()


train_participants = sorted(
    train_df[
        "participant_id"
    ].unique()
)

test_participants = sorted(
    test_df[
        "participant_id"
    ].unique()
)


print(
    f"Train participants: "
    f"{len(train_participants)}"
)

print(
    f"Test participants: "
    f"{len(test_participants)}"
)


print(
    "Test participants:",
    [
        int(x)
        for x in test_participants
    ]
)


print(
    f"Training rows: "
    f"{len(train_df)}"
)

print(
    f"Testing rows: "
    f"{len(test_df)}"
)


# ============================================================
# LOAD STAGE-1 CORE TEMPERATURE MODEL
# ============================================================

print()
print("-" * 70)
print("LOADING NO-WIND CORE-TEMPERATURE ESTIMATOR")
print("-" * 70)


if not CORE_MODEL_FILE.exists():

    raise FileNotFoundError(

        "\nCould not find:\n"
        f"{CORE_MODEL_FILE}\n\n"

        "Train the Stage-1 model first using:\n"

        "python src\\"
        "train_core_temp_estimator_no_wind.py"
    )


core_model = joblib.load(
    CORE_MODEL_FILE
)


print(
    "Core-temperature estimator loaded."
)


# ============================================================
# STAGE 1
# ESTIMATE CURRENT CORE TEMPERATURE
# ============================================================
#
# IMPORTANT:
# The actual current core temperature is NOT supplied to the
# deployed predictor.
#
# Instead:
#
#   HR
#   HRV
#   skin temperature
#   ambient temperature
#   humidity
#   activity
#   personal factors
#
# are used to estimate current core temperature.
# ============================================================

print()
print("-" * 70)
print("STAGE 1 - ESTIMATING CURRENT CORE TEMPERATURE")
print("-" * 70)


X_train_core = train_df[
    BASE_FEATURES
].copy()

X_test_core = test_df[
    BASE_FEATURES
].copy()


train_df[
    "estimated_core_temp_c"
] = core_model.predict(
    X_train_core
)

test_df[
    "estimated_core_temp_c"
] = core_model.predict(
    X_test_core
)


# ============================================================
# STAGE 1 EVALUATION
# ============================================================

stage1_mae = mean_absolute_error(
    test_df["core_temp_c"],
    test_df[
        "estimated_core_temp_c"
    ]
)


stage1_rmse = np.sqrt(
    mean_squared_error(
        test_df["core_temp_c"],
        test_df[
            "estimated_core_temp_c"
        ]
    )
)


stage1_r2 = r2_score(
    test_df["core_temp_c"],
    test_df[
        "estimated_core_temp_c"
    ]
)


print(
    f"Stage 1 MAE  : "
    f"{stage1_mae:.6f} °C"
)

print(
    f"Stage 1 RMSE : "
    f"{stage1_rmse:.6f} °C"
)

print(
    f"Stage 1 R²   : "
    f"{stage1_r2:.6f}"
)


# ============================================================
# STAGE 2 FEATURES
# ============================================================
#
# These are ONLY present-time/deployable features.
#
# There is:
#
#   NO wind
#   NO future core temperature
#   NO future HR
#   NO future skin temperature
#
# ============================================================

STAGE2_FEATURES = [

    # Estimated current physiological state
    "estimated_core_temp_c",

    # Physiological signals
    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",

    # Skin temperature
    "skin_temp_mean_c",

    # Environment
    "ambient_temp_c",
    "humidity_pct",

    # Personal factors
    "age_years",
    "body_mass_kg",
    "height_cm",
    "sex",

    # Activity / work state
    "activity_code",
    "work_rest"
]


TARGET = (
    "future_core_temp_c"
)


# ============================================================
# CATEGORICAL / NUMERIC FEATURES
# ============================================================

categorical_features = [

    "sex",
    "activity_code",
    "work_rest"
]


numeric_features = [

    feature
    for feature in STAGE2_FEATURES
    if feature not in categorical_features
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# TRAINING DATA
# ============================================================

X_train = train_df[
    STAGE2_FEATURES
].copy()

X_test = test_df[
    STAGE2_FEATURES
].copy()


y_train = train_df[
    TARGET
].copy()

y_test = test_df[
    TARGET
].copy()


print()
print("-" * 70)
print("STAGE 2 DATA")
print("-" * 70)

print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples : "
    f"{len(X_test)}"
)

print(
    f"Features before preprocessing: "
    f"{len(STAGE2_FEATURES)}"
)


# ============================================================
# MODELS
# ============================================================

models = {

    "XGBoost": XGBRegressor(

        n_estimators=500,

        max_depth=6,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=RANDOM_STATE,

        n_jobs=-1
    ),


    "LightGBM": LGBMRegressor(

        n_estimators=500,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=-1,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=RANDOM_STATE,

        verbosity=-1,

        n_jobs=-1
    ),


    "HistGradientBoosting":
        HistGradientBoostingRegressor(

            max_iter=400,

            learning_rate=0.05,

            max_leaf_nodes=31,

            random_state=RANDOM_STATE
        )
}


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

best_model_name = None
best_pipeline = None
best_mae = float("inf")

predictions_by_model = {}


for name, model in models.items():

    print()
    print("-" * 70)
    print(
        f"TRAINING STAGE 2: {name}"
    )
    print("-" * 70)


    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )


    predictions_by_model[
        name
    ] = predictions


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    print(
        f"MAE  : "
        f"{mae:.6f} °C"
    )

    print(
        f"RMSE : "
        f"{rmse:.6f} °C"
    )

    print(
        f"R²   : "
        f"{r2:.6f}"
    )


    results.append({

        "model":
            name,

        "MAE_C":
            float(mae),

        "RMSE_C":
            float(rmse),

        "R2":
            float(r2)
    })


    # --------------------------------------------------------
    # Select best by MAE
    # --------------------------------------------------------

    if mae < best_mae:

        best_mae = mae

        best_model_name = name

        best_pipeline = pipeline


# ============================================================
# PERSISTENCE BASELINE
# ============================================================
#
# Simplest possible prediction:
#
#   future core temperature
#   =
#   estimated current core temperature
#
# This tells us whether the ML model actually improves over
# simply assuming temperature stays constant.
# ============================================================

print()
print("-" * 70)
print("PERSISTENCE BASELINE")
print("-" * 70)


persistence_predictions = (
    test_df[
        "estimated_core_temp_c"
    ].to_numpy()
)


persistence_mae = (
    mean_absolute_error(
        y_test,
        persistence_predictions
    )
)


persistence_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        persistence_predictions
    )
)


persistence_r2 = r2_score(
    y_test,
    persistence_predictions
)


print(
    f"MAE  : "
    f"{persistence_mae:.6f} °C"
)

print(
    f"RMSE : "
    f"{persistence_rmse:.6f} °C"
)

print(
    f"R²   : "
    f"{persistence_r2:.6f}"
)


results.append({

    "model":
        "Estimated-current persistence",

    "MAE_C":
        float(persistence_mae),

    "RMSE_C":
        float(persistence_rmse),

    "R2":
        float(persistence_r2)
})


# ============================================================
# IMPROVEMENT OVER PERSISTENCE
# ============================================================

improvement = (

    (
        persistence_mae
        - best_mae
    )
    / persistence_mae

) * 100


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    results
)


comparison_file = (
    RESULT_DIR
    / "integrated_10min_no_wind_model_comparison.csv"
)


comparison_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# SAVE BEST PIPELINE
# ============================================================

model_file = (
    MODEL_DIR
    / "thermora_integrated_10min_no_wind_model.pkl"
)


joblib.dump(
    best_pipeline,
    model_file
)


# ============================================================
# GENERATE FINAL TEST PREDICTIONS
# ============================================================

best_predictions = (
    predictions_by_model[
        best_model_name
    ]
)


test_output = test_df[
    [
        "participant_id",
        "condition_id",
        "time_min",
        "target_offset_min",
        "core_temp_c",
        "future_core_temp_c",
        "estimated_core_temp_c"
    ]
].copy()


# Actual future core temperature
test_output[
    "actual_future_core_temp_c"
] = test_output[
    "future_core_temp_c"
]


# ------------------------------------------------------------
# The Stage-2 model directly predicts future core temperature.
# ------------------------------------------------------------

test_output[
    "predicted_future_core_temp_c"
] = (
    best_predictions
)


# ------------------------------------------------------------
# Predicted change in core temperature
# ------------------------------------------------------------

test_output[
    "predicted_delta_c"
] = (

    test_output[
        "predicted_future_core_temp_c"
    ]

    -

    test_output[
        "estimated_core_temp_c"
    ]
)


# ------------------------------------------------------------
# Actual change relative to estimated current temperature
# ------------------------------------------------------------

test_output[
    "actual_delta_from_estimated_c"
] = (

    test_output[
        "actual_future_core_temp_c"
    ]

    -

    test_output[
        "estimated_core_temp_c"
    ]
)


# ------------------------------------------------------------
# Absolute prediction error
# ------------------------------------------------------------

test_output[
    "absolute_error_c"
] = abs(

    test_output[
        "predicted_future_core_temp_c"
    ]

    -

    test_output[
        "actual_future_core_temp_c"
    ]
)


prediction_file = (
    RESULT_DIR
    / "integrated_10min_no_wind_test_predictions.csv"
)


test_output.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "model_type":
        "Two-stage deployable thermal prediction pipeline",

    "wind_required":
        False,

    "target_horizon_minutes":
        10.0,

    "target_tolerance_minutes":
        TARGET_TOLERANCE_MIN,

    "target_rows":
        int(len(valid)),

    "stage_1": {

        "model":
            "No-wind core temperature estimator",

        "MAE_C":
            float(stage1_mae),

        "RMSE_C":
            float(stage1_rmse),

        "R2":
            float(stage1_r2)
    },


    "stage_2": {

        "best_model":
            str(best_model_name),

        "MAE_C":
            float(best_mae),

        "RMSE_C":
            float(
                comparison_df[
                    comparison_df["model"]
                    == best_model_name
                ]["RMSE_C"].iloc[0]
            ),

        "R2":
            float(
                comparison_df[
                    comparison_df["model"]
                    == best_model_name
                ]["R2"].iloc[0]
            ),

        "persistence_MAE_C":
            float(persistence_mae),

        "persistence_RMSE_C":
            float(persistence_rmse),

        "persistence_R2":
            float(persistence_r2),

        "improvement_over_persistence_percent":
            float(improvement)
    },


    "participants": {

        "total":
            int(
                valid[
                    "participant_id"
                ].nunique()
            ),

        "train":
            int(
                len(train_participants)
            ),

        "test":
            int(
                len(test_participants)
            ),

        "test_participants":
            [
                int(x)
                for x in test_participants
            ]
    },


    "features": [
        str(x)
        for x in STAGE2_FEATURES
    ]
}


summary_file = (
    RESULT_DIR
    / "integrated_10min_no_wind_summary.json"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL NO-WIND 10-MINUTE MODEL")
print("=" * 70)


print(
    f"Best model: "
    f"{best_model_name}"
)


best_result = comparison_df[
    comparison_df["model"]
    == best_model_name
].iloc[0]


print(
    f"MAE  : "
    f"{best_result['MAE_C']:.6f} °C"
)

print(
    f"RMSE : "
    f"{best_result['RMSE_C']:.6f} °C"
)

print(
    f"R²   : "
    f"{best_result['R2']:.6f}"
)


print()
print(
    f"Persistence MAE: "
    f"{persistence_mae:.6f} °C"
)


print(
    f"Improvement over persistence: "
    f"{improvement:.2f}%"
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)


print(
    f"Model:\n"
    f"{model_file}"
)

print()

print(
    f"Comparison:\n"
    f"{comparison_file}"
)

print()

print(
    f"Predictions:\n"
    f"{prediction_file}"
)

print()

print(
    f"Summary:\n"
    f"{summary_file}"
)


print()
print("=" * 70)
print("NO-WIND 10-MINUTE PREDICTOR COMPLETE")
print("=" * 70)