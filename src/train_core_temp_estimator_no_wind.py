import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor


# ============================================================
# THERMORA
# DEPLOYABLE CORE TEMPERATURE ESTIMATOR - NO WIND
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("THERMORA CORE TEMPERATURE ESTIMATOR")
print("DEPLOYABLE VERSION - NO WIND")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"Original rows: {len(df)}")


# ============================================================
# DEPLOYABLE FEATURES
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

    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",

    "skin_temp_mean_c"
]


TARGET = "core_temp_c"


missing = [
    c for c in FEATURES + [TARGET]
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# REMOVE INVALID TARGETS
# ============================================================

df = df[
    pd.notna(df[TARGET])
].copy()

print(
    f"Rows with valid core-temp target: {len(df)}"
)


# ============================================================
# PARTICIPANT-HELD-OUT SPLIT
# ============================================================

participants = (
    df["participant_id"]
    .dropna()
    .unique()
)

print(
    f"Participants: {len(participants)}"
)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["participant_id"]
    )
)


train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()


print(
    f"Train participants: "
    f"{train_df['participant_id'].nunique()}"
)

print(
    f"Test participants: "
    f"{test_df['participant_id'].nunique()}"
)

print(
    "Test participants:",
    sorted(
        test_df["participant_id"]
        .unique()
        .tolist()
    )
)


# ============================================================
# X / y
# ============================================================

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df[TARGET].copy()
y_test = test_df[TARGET].copy()


# ============================================================
# IDENTIFY CATEGORICAL / NUMERICAL
# ============================================================

categorical_features = [
    "activity_code",
    "work_rest",
    "sex"
]

numeric_features = [
    c
    for c in FEATURES
    if c not in categorical_features
]


# ============================================================
# PREPROCESSOR
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
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
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
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    )
}


# ============================================================
# TRAIN / EVALUATE
# ============================================================

results = []

best_model = None
best_pipeline = None
best_mae = float("inf")


for name, model in models.items():

    print()
    print("-" * 70)
    print(f"Training {name}")
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


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


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
        f"MAE  : {mae:.6f} °C"
    )

    print(
        f"RMSE : {rmse:.6f} °C"
    )

    print(
        f"R²   : {r2:.6f}"
    )


    results.append({

        "model": name,
        "MAE_C": mae,
        "RMSE_C": rmse,
        "R2": r2
    })


    if mae < best_mae:

        best_mae = mae
        best_model = name
        best_pipeline = pipeline


# ============================================================
# SAVE COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

comparison_file = (
    RESULT_DIR
    / "core_temp_estimator_no_wind_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_file = (
    MODEL_DIR
    / "thermora_core_temp_estimator_no_wind.pkl"
)

joblib.dump(
    best_pipeline,
    model_file
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = test_df[
    [
        "participant_id",
        "condition_id",
        "time_min",
        TARGET
    ]
].copy()

test_predictions[
    "predicted_core_temp_c"
] = best_pipeline.predict(
    X_test
)

test_predictions[
    "error_c"
] = (
    test_predictions[
        "predicted_core_temp_c"
    ]
    - test_predictions[
        TARGET
    ]
)

test_predictions[
    "absolute_error_c"
] = abs(
    test_predictions[
        "error_c"
    ]
)

prediction_file = (
    RESULT_DIR
    / "core_temp_estimator_no_wind_predictions.csv"
)

test_predictions.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    f"Best model: {best_model}"
)

best_row = results_df[
    results_df["model"] == best_model
].iloc[0]

print(
    f"MAE  : {best_row['MAE_C']:.6f} °C"
)

print(
    f"RMSE : {best_row['RMSE_C']:.6f} °C"
)

print(
    f"R²   : {best_row['R2']:.6f}"
)


print()
print("FILES SAVED")
print("-" * 70)

print(model_file)
print(comparison_file)
print(prediction_file)


print()
print("=" * 70)
print("NO-WIND CORE TEMPERATURE ESTIMATOR COMPLETE")
print("=" * 70)