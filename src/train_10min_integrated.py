import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_FOLDS = 5

HORIZON_MINUTES = 10.0

# Maximum allowed difference between requested
# +10 minute target and the actual future observation.
#
# Example:
# requested target = 30.0 min
# acceptable target = 29.5–30.5 min
#
TARGET_TOLERANCE_MINUTES = 0.75


# ============================================================
# FEATURES AVAILABLE TO THERMORA
# ============================================================

CORE_FEATURES = [
    "activity_code",
    "work_rest",
    "sex",
    "age_years",
    "body_mass_kg",
    "height_cm",
    "humidity_pct",
    "ambient_temp_c",
    "wind_mps",
    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",
    "skin_temp_mean_c",
]

CORE_TARGET = "core_temp_c"

GROUP = "participant_id"

PREDICTOR_FEATURES = [
    "age_years",
    "body_mass_kg",
    "ambient_temp_c",
    "humidity_pct",
    "heart_rate_bpm",
    "skin_temp_mean_c",
    "sex_male",
    "estimated_core_temp_c",
]


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def encode_categories(df):

    df = df.copy()

    mappings = {}

    for col in [
        "activity_code",
        "work_rest",
        "sex"
    ]:

        df[col] = df[col].astype("string")
        df[col] = df[col].fillna("UNKNOWN")

        values = sorted(
            df[col].unique().tolist()
        )

        mapping = {
            value: i
            for i, value in enumerate(values)
        }

        mappings[col] = mapping

        df[col] = df[col].map(mapping)

    # Male indicator
    male_code = mappings["sex"].get(
        "M",
        mappings["sex"].get("Male", -999)
    )

    df["sex_male"] = (
        df["sex"] == male_code
    ).astype(float)

    return df, mappings


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("THERMORA — INTEGRATED 10-MINUTE CORE-TEMPERATURE PREDICTOR V2")
print("=" * 75)

print("\nLoading:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nOriginal shape:", df.shape)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = list(
    set(
        CORE_FEATURES
        + [
            CORE_TARGET,
            GROUP,
            "time_min",
            "condition_id"
        ]
    )
)

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "time_min",
    "age_years",
    "body_mass_kg",
    "height_cm",
    "humidity_pct",
    "ambient_temp_c",
    "wind_mps",
    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",
    "skin_temp_mean_c",
    CORE_TARGET
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID TIME / TARGET
# ============================================================

df = df.dropna(
    subset=[
        GROUP,
        "condition_id",
        "time_min",
        CORE_TARGET
    ]
).copy()

print(
    "\nRows with valid time + target:",
    len(df)
)

print(
    "Participants:",
    df[GROUP].nunique()
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        GROUP,
        "condition_id",
        "time_min"
    ]
).reset_index(
    drop=True
)


# ============================================================
# CREATE 10-MINUTE FUTURE TARGET
# ============================================================

print("\n" + "=" * 75)
print("CREATING 10-MINUTE FUTURE TARGET")
print("=" * 75)

print(
    "\nTarget horizon:",
    HORIZON_MINUTES,
    "minutes"
)

print(
    "Allowed timing error:",
    TARGET_TOLERANCE_MINUTES,
    "minutes"
)


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# For every current observation:
#
# current time = t
# requested target = t + 10
#
# Search ONLY within the same:
#
# participant_id + condition_id
#
# and choose the closest future observation.
#
# ------------------------------------------------------------

target_rows = []

grouped = df.groupby(
    [
        GROUP,
        "condition_id"
    ],
    sort=False
)

for (
    participant,
    condition
), group in grouped:

    group = group.sort_values(
        "time_min"
    ).reset_index(
        drop=True
    )

    times = group[
        "time_min"
    ].values

    core = group[
        CORE_TARGET
    ].values

    for i in range(
        len(group)
    ):

        current_time = times[i]

        desired_time = (
            current_time
            + HORIZON_MINUTES
        )

        # Find first observation at or after
        # desired target time.
        future_position = np.searchsorted(
            times,
            desired_time,
            side="left"
        )

        if future_position >= len(times):
            continue

        # Candidate immediately after target
        candidate_indices = [
            future_position
        ]

        # Also check observation immediately before
        # desired target.
        if future_position > i + 1:
            candidate_indices.append(
                future_position - 1
            )

        best_index = min(
            candidate_indices,
            key=lambda j: abs(
                times[j]
                - desired_time
            )
        )

        # Must actually be in the future.
        if best_index <= i:
            continue

        actual_difference = abs(
            times[best_index]
            - desired_time
        )

        if (
            actual_difference
            > TARGET_TOLERANCE_MINUTES
        ):
            continue

        target_rows.append({
            "participant_id": participant,
            "condition_id": condition,
            "current_time_min": current_time,
            "target_time_min": times[best_index],
            "future_core_temp_c": core[best_index],
            "target_offset_min": (
                times[best_index]
                - current_time
            )
        })


targets = pd.DataFrame(
    target_rows
)

print(
    "\nRows with valid approximately +10 minute target:",
    len(targets)
)

if len(targets) == 0:

    raise RuntimeError(
        "No valid future targets were created. "
        "Inspect the time_min spacing."
    )

print(
    "Participants with targets:",
    targets[
        "participant_id"
    ].nunique()
)

print(
    "Actual target offset:",
    round(
        targets[
            "target_offset_min"
        ].mean(),
        4
    ),
    "±",
    round(
        targets[
            "target_offset_min"
        ].std(),
        4
    ),
    "minutes"
)


# ============================================================
# MERGE CURRENT SENSOR STATE
# ============================================================

data = df.merge(
    targets,
    left_on=[
        GROUP,
        "condition_id",
        "time_min"
    ],
    right_on=[
        "participant_id",
        "condition_id",
        "current_time_min"
    ],
    how="inner",
    suffixes=("", "_target")
)

print(
    "\nIntegrated rows:",
    len(data)
)

print(
    "Integrated participants:",
    data[GROUP].nunique()
)


# ============================================================
# ENCODE CATEGORIES
# ============================================================

data, mappings = encode_categories(
    data
)


# ============================================================
# PARTICIPANT HOLDOUT
# ============================================================

print("\n" + "=" * 75)
print("PARTICIPANT-HOLDOUT SPLIT")
print("=" * 75)

participants = np.array(
    sorted(
        data[GROUP].unique()
    )
)

rng = np.random.RandomState(
    RANDOM_STATE
)

shuffled = participants.copy()

rng.shuffle(
    shuffled
)

n_test = max(
    1,
    int(
        round(
            len(shuffled) * 0.20
        )
    )
)

test_participants = sorted(
    shuffled[:n_test]
)

train_participants = sorted(
    shuffled[n_test:]
)

train_mask = data[
    GROUP
].isin(
    train_participants
)

test_mask = data[
    GROUP
].isin(
    test_participants
)

train_df = data[
    train_mask
].copy()

test_df = data[
    test_mask
].copy()

print(
    "\nTraining participants:",
    len(train_participants)
)

print(
    "Testing participants :",
    len(test_participants)
)

print(
    "\nTest participants:"
)

for p in test_participants:
    print(
        " ",
        p
    )

print(
    "\nTraining rows:",
    len(train_df)
)

print(
    "Testing rows :",
    len(test_df)
)


# ============================================================
# CORE TEMP FEATURES
# ============================================================

core_numeric = [
    "age_years",
    "body_mass_kg",
    "height_cm",
    "humidity_pct",
    "ambient_temp_c",
    "wind_mps",
    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",
    "skin_temp_mean_c",
]

core_categorical = [
    "activity_code",
    "work_rest",
    "sex",
]


# ============================================================
# STAGE 1 — OOF CORE TEMP
# ============================================================

print("\n" + "=" * 75)
print("STAGE 1 — CORE TEMPERATURE ESTIMATOR")
print("=" * 75)

print(
    "\nGenerating participant-held-out OOF estimates..."
)

oof_estimates = np.full(
    len(train_df),
    np.nan
)

train_groups = train_df[
    GROUP
].values

actual_folds = min(
    N_FOLDS,
    train_df[
        GROUP
    ].nunique()
)

if actual_folds < 2:

    raise RuntimeError(
        "Not enough training participants "
        "for OOF estimation."
    )

gkf = GroupKFold(
    n_splits=actual_folds
)

for fold, (
    tr_idx,
    val_idx
) in enumerate(
    gkf.split(
        train_df,
        train_df[CORE_TARGET],
        train_groups
    ),
    start=1
):

    print(
        f"  Fold {fold}/{actual_folds}"
    )

    fold_train = train_df.iloc[
        tr_idx
    ]

    fold_val = train_df.iloc[
        val_idx
    ]

    imputer = SimpleImputer(
        strategy="median"
    )

    Xtr_num = imputer.fit_transform(
        fold_train[core_numeric]
    )

    Xva_num = imputer.transform(
        fold_val[core_numeric]
    )

    Xtr = np.hstack([
        fold_train[
            core_categorical
        ].values,
        Xtr_num
    ]).astype(
        np.float32
    )

    Xva = np.hstack([
        fold_val[
            core_categorical
        ].values,
        Xva_num
    ]).astype(
        np.float32
    )

    ytr = fold_train[
        CORE_TARGET
    ].values.astype(
        np.float32
    )

    try:

        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    except ImportError:

        from sklearn.ensemble import (
            HistGradientBoostingRegressor
        )

        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )

    model.fit(
        Xtr,
        ytr
    )

    oof_estimates[
        val_idx
    ] = model.predict(
        Xva
    )


if np.isnan(
    oof_estimates
).any():

    raise RuntimeError(
        "OOF core-temperature predictions "
        "contain NaN values."
    )

train_df[
    "estimated_core_temp_c"
] = oof_estimates


# ============================================================
# FINAL CORE MODEL
# ============================================================

print(
    "\nTraining final core-temperature model..."
)

final_core_imputer = SimpleImputer(
    strategy="median"
)

X_train_core_num = (
    final_core_imputer.fit_transform(
        train_df[core_numeric]
    )
)

X_test_core_num = (
    final_core_imputer.transform(
        test_df[core_numeric]
    )
)

X_train_core = np.hstack([
    train_df[
        core_categorical
    ].values,
    X_train_core_num
]).astype(
    np.float32
)

X_test_core = np.hstack([
    test_df[
        core_categorical
    ].values,
    X_test_core_num
]).astype(
    np.float32
)

y_train_core = train_df[
    CORE_TARGET
].values.astype(
    np.float32
)

try:

    from xgboost import XGBRegressor

    final_core_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

except ImportError:

    from sklearn.ensemble import (
        HistGradientBoostingRegressor
    )

    final_core_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )

final_core_model.fit(
    X_train_core,
    y_train_core
)

test_df[
    "estimated_core_temp_c"
] = final_core_model.predict(
    X_test_core
)


# ============================================================
# CORE TEMP EVALUATION
# ============================================================

core_mae = mean_absolute_error(
    test_df[CORE_TARGET],
    test_df["estimated_core_temp_c"]
)

core_rmse = np.sqrt(
    mean_squared_error(
        test_df[CORE_TARGET],
        test_df["estimated_core_temp_c"]
    )
)

core_r2 = r2_score(
    test_df[CORE_TARGET],
    test_df["estimated_core_temp_c"]
)

print(
    "\nCore estimator test performance:"
)

print(
    "MAE :",
    round(core_mae, 6),
    "°C"
)

print(
    "RMSE:",
    round(core_rmse, 6),
    "°C"
)

print(
    "R²  :",
    round(core_r2, 6)
)


# ============================================================
# STAGE 2 — FUTURE DELTA
# ============================================================

train_df[
    "delta_core_temp_c"
] = (
    train_df[
        "future_core_temp_c"
    ]
    -
    train_df[
        "estimated_core_temp_c"
    ]
)

test_df[
    "delta_core_temp_c"
] = (
    test_df[
        "future_core_temp_c"
    ]
    -
    test_df[
        "estimated_core_temp_c"
    ]
)


# ============================================================
# VALID PREDICTOR DATA
# ============================================================

train_valid = (
    train_df[
        PREDICTOR_FEATURES
    ]
    .notna()
    .all(axis=1)
)

test_valid = (
    test_df[
        PREDICTOR_FEATURES
    ]
    .notna()
    .all(axis=1)
)

train_pred_df = train_df[
    train_valid
].copy()

test_pred_df = test_df[
    test_valid
].copy()


print("\n" + "=" * 75)
print("STAGE 2 — 10-MINUTE DELTA PREDICTOR")
print("=" * 75)

print(
    "\nTraining rows:",
    len(train_pred_df)
)

print(
    "Testing rows :",
    len(test_pred_df)
)


# ============================================================
# X / Y
# ============================================================

X_train_delta = train_pred_df[
    PREDICTOR_FEATURES
].astype(
    np.float32
)

y_train_delta = train_pred_df[
    "delta_core_temp_c"
].astype(
    np.float32
)

X_test_delta = test_pred_df[
    PREDICTOR_FEATURES
].astype(
    np.float32
)

y_test_delta = test_pred_df[
    "delta_core_temp_c"
].astype(
    np.float32
)


# ============================================================
# DELTA MODELS
# ============================================================

delta_models = {}

try:

    from lightgbm import LGBMRegressor

    delta_models[
        "LightGBM"
    ] = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=-1
    )

except ImportError:

    print(
        "\nLightGBM unavailable."
    )


try:

    from xgboost import XGBRegressor

    delta_models[
        "XGBoost"
    ] = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

except ImportError:

    print(
        "\nXGBoost unavailable."
    )


from sklearn.ensemble import (
    HistGradientBoostingRegressor
)

delta_models[
    "HistGradientBoosting"
] = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
)


# ============================================================
# TRAIN DELTA MODELS
# ============================================================

results = []

trained_models = {}

actual_future = test_pred_df[
    "future_core_temp_c"
].values

estimated_current = test_pred_df[
    "estimated_core_temp_c"
].values

for name, model in delta_models.items():

    print(
        "\n" + "-" * 75
    )

    print(
        "Training:",
        name
    )

    model.fit(
        X_train_delta,
        y_train_delta
    )

    predicted_delta = model.predict(
        X_test_delta
    )

    predicted_future = (
        estimated_current
        + predicted_delta
    )

    mae = mean_absolute_error(
        actual_future,
        predicted_future
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_future,
            predicted_future
        )
    )

    r2 = r2_score(
        actual_future,
        predicted_future
    )

    delta_mae = mean_absolute_error(
        y_test_delta,
        predicted_delta
    )

    print(
        "Future core MAE :",
        round(mae, 6),
        "°C"
    )

    print(
        "Future core RMSE:",
        round(rmse, 6),
        "°C"
    )

    print(
        "Future core R²  :",
        round(r2, 6)
    )

    print(
        "Delta MAE       :",
        round(delta_mae, 6),
        "°C"
    )

    results.append({
        "model": name,
        "future_core_MAE_C": mae,
        "future_core_RMSE_C": rmse,
        "future_core_R2": r2,
        "delta_MAE_C": delta_mae
    })

    trained_models[
        name
    ] = model


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "PERSISTENCE BASELINE"
)

print(
    "=" * 75
)

persistence_mae = mean_absolute_error(
    actual_future,
    estimated_current
)

persistence_rmse = np.sqrt(
    mean_squared_error(
        actual_future,
        estimated_current
    )
)

persistence_r2 = r2_score(
    actual_future,
    estimated_current
)

print(
    "\nMAE :",
    round(persistence_mae, 6),
    "°C"
)

print(
    "RMSE:",
    round(persistence_rmse, 6),
    "°C"
)

print(
    "R²  :",
    round(persistence_r2, 6)
)


# ============================================================
# COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    "future_core_MAE_C"
).reset_index(
    drop=True
)

print(
    "\n" + "=" * 75
)

print(
    "10-MINUTE MODEL COMPARISON"
)

print(
    "=" * 75
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_name = results_df.iloc[
    0
]["model"]

best_model = trained_models[
    best_name
]

best_delta = best_model.predict(
    X_test_delta
)

best_future = (
    estimated_current
    + best_delta
)

best_mae = mean_absolute_error(
    actual_future,
    best_future
)

improvement = (
    (
        persistence_mae
        - best_mae
    )
    /
    persistence_mae
) * 100.0


print(
    "\n" + "=" * 75
)

print(
    "BEST 10-MINUTE MODEL"
)

print(
    "=" * 75
)

print(
    "\nModel:",
    best_name
)

print(
    "MAE:",
    round(best_mae, 6),
    "°C"
)

print(
    "Persistence:",
    round(persistence_mae, 6),
    "°C"
)

print(
    "Improvement:",
    round(improvement, 2),
    "%"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_output = test_pred_df[
    [
        GROUP,
        "condition_id",
        "time_min",
        "target_time_min",
        "target_offset_min",
        "estimated_core_temp_c",
        "future_core_temp_c"
    ]
].copy()

prediction_output[
    "predicted_delta_c"
] = best_delta

prediction_output[
    "predicted_future_core_temp_c"
] = best_future

prediction_output[
    "absolute_error_c"
] = np.abs(
    prediction_output[
        "future_core_temp_c"
    ]
    -
    prediction_output[
        "predicted_future_core_temp_c"
    ]
)

prediction_path = os.path.join(
    RESULT_DIR,
    "integrated_10min_test_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "thermora_integrated_10min_model.pkl"
)

artifact = {
    "core_temp_model": final_core_model,
    "core_temp_imputer": final_core_imputer,
    "delta_model": best_model,
    "categorical_mappings": mappings,
    "core_features": CORE_FEATURES,
    "predictor_features": PREDICTOR_FEATURES,
    "horizon_minutes": HORIZON_MINUTES,
    "target_tolerance_minutes": TARGET_TOLERANCE_MINUTES,
    "best_delta_model": best_name,
    "core_test_mae_c": float(core_mae),
    "future_test_mae_c": float(best_mae),
    "test_participants": [
        str(x)
        for x in test_participants
    ]
}

joblib.dump(
    artifact,
    model_path
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    RESULT_DIR,
    "integrated_10min_model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "dataset_rows": int(
        len(data)
    ),

    "participants": int(
        data[GROUP].nunique()
    ),

    "train_participants": int(
        len(train_participants)
    ),

    "test_participants": int(
        len(test_participants)
    ),

    "test_participant_ids": [
        str(x)
        for x in test_participants
    ],

    "horizon_minutes": HORIZON_MINUTES,

    "target_tolerance_minutes":
        TARGET_TOLERANCE_MINUTES,

    "mean_actual_target_offset_minutes":
        float(
            targets[
                "target_offset_min"
            ].mean()
        ),

    "core_estimator_test_mae_c":
        float(core_mae),

    "core_estimator_test_rmse_c":
        float(core_rmse),

    "core_estimator_test_r2":
        float(core_r2),

    "persistence_mae_c":
        float(persistence_mae),

    "best_model":
        best_name,

    "best_future_core_mae_c":
        float(best_mae),

    "improvement_over_persistence_pct":
        float(improvement)
}

summary_path = os.path.join(
    RESULT_DIR,
    "integrated_10min_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "INTEGRATED 10-MINUTE PIPELINE COMPLETE"
)

print(
    "=" * 75
)

print(
    "\nBest model:",
    best_name
)

print(
    "Future core MAE:",
    round(best_mae, 4),
    "°C"
)

print(
    "\nSaved model:"
)

print(
    model_path
)

print(
    "\nSaved predictions:"
)

print(
    prediction_path
)

print(
    "\nSaved results:"
)

print(
    results_path
)

print(
    "\nSaved summary:"
)

print(
    summary_path
)

print(
    "\n" + "=" * 75
)