import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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

TARGET = "core_temp_c"
GROUP = "participant_id"

CATEGORICAL = [
    "activity_code",
    "work_rest",
    "sex",
]

NUMERIC = [
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


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("THERMORA — CORE TEMPERATURE ESTIMATOR V2")
print("=" * 70)

print("\nLoading:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)

required = FEATURES + [TARGET, GROUP]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns:\n" +
        "\n".join(missing)
    )


# ============================================================
# SELECT DATA
# ============================================================

data = df[required].copy()

# Target must exist
data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="coerce"
)

data = data.dropna(
    subset=[TARGET, GROUP]
)

print(
    "Rows with valid target:",
    len(data)
)

print(
    "Participants:",
    data[GROUP].nunique()
)


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

print("\nCategorical values:")

mappings = {}

for col in CATEGORICAL:

    # Treat everything as a string category
    data[col] = data[col].astype("string")

    # Missing category becomes UNKNOWN
    data[col] = data[col].fillna("UNKNOWN")

    values = sorted(
        data[col].unique().tolist()
    )

    mapping = {
        value: i
        for i, value in enumerate(values)
    }

    mappings[col] = mapping

    data[col] = data[col].map(mapping)

    print(
        f"{col}:",
        len(values),
        "categories"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in NUMERIC:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


# ============================================================
# PARTICIPANT SPLIT
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANT-HOLDOUT SPLIT")
print("=" * 70)

groups = data[GROUP].values

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        data[FEATURES],
        data[TARGET],
        groups=groups
    )
)

train = data.iloc[train_idx].copy()
test = data.iloc[test_idx].copy()

train_participants = sorted(
    train[GROUP].unique()
)

test_participants = sorted(
    test[GROUP].unique()
)

print(
    "\nTraining rows:",
    len(train)
)

print(
    "Testing rows :",
    len(test)
)

print(
    "\nTraining participants:",
    len(train_participants)
)

print(
    "Testing participants :",
    len(test_participants)
)

print("\nTest participants:")

for p in test_participants:
    print(" ", p)


# ============================================================
# IMPUTATION
# ============================================================
#
# IMPORTANT:
# Imputation is fitted ONLY on training data.
# This prevents test-set information leaking into training.
#

print("\n" + "=" * 70)
print("MISSING-VALUE IMPUTATION")
print("=" * 70)

numeric_imputer = SimpleImputer(
    strategy="median"
)

X_train_numeric = numeric_imputer.fit_transform(
    train[NUMERIC]
)

X_test_numeric = numeric_imputer.transform(
    test[NUMERIC]
)

# Categorical columns are already encoded.
# UNKNOWN = 0 or another valid category.
X_train_cat = train[CATEGORICAL].values.astype(
    np.float32
)

X_test_cat = test[CATEGORICAL].values.astype(
    np.float32
)

X_train = np.hstack([
    X_train_cat,
    X_train_numeric
]).astype(np.float32)

X_test = np.hstack([
    X_test_cat,
    X_test_numeric
]).astype(np.float32)

y_train = train[TARGET].values.astype(
    np.float32
)

y_test = test[TARGET].values.astype(
    np.float32
)

print(
    "\nFinal training matrix:",
    X_train.shape
)

print(
    "Final testing matrix :",
    X_test.shape
)


# ============================================================
# BASELINES
# ============================================================

print("\n" + "=" * 70)
print("BASELINES")
print("=" * 70)

# Training mean
mean_pred = np.full(
    len(y_test),
    y_train.mean()
)

mean_mae = mean_absolute_error(
    y_test,
    mean_pred
)

mean_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        mean_pred
    )
)

mean_r2 = r2_score(
    y_test,
    mean_pred
)

print("\nTraining-mean baseline")
print("MAE :", round(mean_mae, 6))
print("RMSE:", round(mean_rmse, 6))
print("R²  :", round(mean_r2, 6))


# 37°C baseline
fixed_pred = np.full(
    len(y_test),
    37.0
)

fixed_mae = mean_absolute_error(
    y_test,
    fixed_pred
)

fixed_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        fixed_pred
    )
)

fixed_r2 = r2_score(
    y_test,
    fixed_pred
)

print("\n37°C baseline")
print("MAE :", round(fixed_mae, 6))
print("RMSE:", round(fixed_rmse, 6))
print("R²  :", round(fixed_r2, 6))


# ============================================================
# MODELS
# ============================================================

models = {}

# LightGBM
try:
    from lightgbm import LGBMRegressor

    models["LightGBM"] = LGBMRegressor(
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
    print("\nLightGBM not installed — skipping.")


# XGBoost
try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
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
    print("\nXGBoost not installed — skipping.")


# HistGradientBoosting
models["HistGradientBoosting"] = (
    HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )
)


# Random Forest
models["RandomForest"] = (
    RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
)


# ============================================================
# TRAIN
# ============================================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    print(
        "\nMAE :",
        round(mae, 6),
        "°C"
    )

    print(
        "RMSE:",
        round(rmse, 6),
        "°C"
    )

    print(
        "R²  :",
        round(r2, 6)
    )

    results.append({
        "model": name,
        "MAE_C": mae,
        "RMSE_C": rmse,
        "R2": r2
    })

    trained_models[name] = model


# ============================================================
# COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    "MAE_C"
).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_name
]

best_mae = float(
    results_df.iloc[0]["MAE_C"]
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "\nModel:",
    best_name
)

print(
    "Test MAE:",
    round(best_mae, 6),
    "°C"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "thermora_core_temp_estimator.pkl"
)

artifact = {
    "model": best_model,
    "features": FEATURES,
    "categorical_features": CATEGORICAL,
    "numeric_features": NUMERIC,
    "target": TARGET,
    "group_column": GROUP,
    "categorical_mappings": mappings,
    "numeric_imputer": numeric_imputer,
    "best_model": best_name,
    "test_mae_c": best_mae,
    "random_state": RANDOM_STATE,
}

joblib.dump(
    artifact,
    model_path
)

print(
    "\nSaved model:",
    model_path
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    RESULT_DIR,
    "core_temp_estimator_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    "Saved results:",
    results_path
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

best_predictions = best_model.predict(
    X_test
)

test_output = test[
    [GROUP, TARGET]
].copy()

test_output[
    "predicted_core_temp_c"
] = best_predictions

test_output[
    "absolute_error_c"
] = np.abs(
    test_output[TARGET]
    - test_output["predicted_core_temp_c"]
)

prediction_path = os.path.join(
    RESULT_DIR,
    "core_temp_estimator_test_predictions.csv"
)

test_output.to_csv(
    prediction_path,
    index=False
)

print(
    "Saved test predictions:",
    prediction_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CORE TEMPERATURE ESTIMATOR V2 COMPLETE")
print("=" * 70)

print(
    "\nBest model:",
    best_name
)

print(
    "Test MAE:",
    round(best_mae, 4),
    "°C"
)

print(
    "\nParticipants:",
    data[GROUP].nunique()
)

print(
    "Training participants:",
    len(train_participants)
)

print(
    "Test participants:",
    len(test_participants)
)

print("=" * 70)