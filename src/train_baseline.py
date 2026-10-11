from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "core_temp_c"


# ------------------------------------------------------------
# MODEL A
# Environment + workload
# ------------------------------------------------------------

MODEL_A_NUMERIC = [
    "humidity_pct",
    "ambient_temp_c",
    "wind_mps",
    "solar_radiation_w_m2",
    "oxygen_consumption_ml_kg_min",
    "treadmill_speed_kmh",
    "treadmill_gradient_pct",
]

MODEL_A_CATEGORICAL = [
    "work_rest",
    "solar_present",
]


# ------------------------------------------------------------
# MODEL B
# Environment + workload + worker characteristics
# ------------------------------------------------------------

MODEL_B_NUMERIC = [
    "age_years",
    "body_mass_kg",
    "height_cm",
    "body_fat_pct",
    "vo2peak_ml_kg_min",

    "humidity_pct",
    "ambient_temp_c",
    "wind_mps",
    "solar_radiation_w_m2",

    "oxygen_consumption_ml_kg_min",
    "treadmill_speed_kmh",
    "treadmill_gradient_pct",
]

MODEL_B_CATEGORICAL = [
    "sex",
    "acclimation_status",
    "training_status",
    "work_rest",
    "solar_present",
]


# ------------------------------------------------------------
# MODEL C
# Environment + workload + worker + physiology
# ------------------------------------------------------------

MODEL_C_NUMERIC = [
    "age_years",
    "body_mass_kg",
    "height_cm",
    "body_fat_pct",
    "vo2peak_ml_kg_min",

    "humidity_pct",
    "ambient_temp_c",
    "wind_mps",
    "solar_radiation_w_m2",

    "oxygen_consumption_ml_kg_min",
    "treadmill_speed_kmh",
    "treadmill_gradient_pct",

    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",

    "insulated_skin_temp_c",
    "microclimate_temp_c",
]

MODEL_C_CATEGORICAL = [
    "sex",
    "acclimation_status",
    "training_status",
    "work_rest",
    "solar_present",
]


# ------------------------------------------------------------
# MODEL D
# Deployment-realistic:
# Environment + worker + physiological measurements
# ------------------------------------------------------------

MODEL_D_NUMERIC = [
    "age_years",
    "body_mass_kg",
    "height_cm",
    "body_fat_pct",
    "vo2peak_ml_kg_min",

    "humidity_pct",
    "ambient_temp_c",
    "wind_mps",
    "solar_radiation_w_m2",

    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",

    "insulated_skin_temp_c",
    "microclimate_temp_c",
]

MODEL_D_CATEGORICAL = [
    "sex",
    "acclimation_status",
    "training_status",
]


# ------------------------------------------------------------
# MODEL E
# Minimal practical ThermoSentinel
# ------------------------------------------------------------

MODEL_E_NUMERIC = [
    "age_years",
    "body_mass_kg",

    "ambient_temp_c",
    "humidity_pct",

    "heart_rate_bpm",
    "insulated_skin_temp_c",
]

MODEL_E_CATEGORICAL = [
    "sex",
]


# ------------------------------------------------------------
# MODEL F
# Minimal practical ThermoSentinel + HRV
# ------------------------------------------------------------

MODEL_F_NUMERIC = [
    "age_years",
    "body_mass_kg",

    "ambient_temp_c",
    "humidity_pct",

    "heart_rate_bpm",
    "rr_interval_ms",
    "sdrr_ms",
    "rmssd_ms",

    "insulated_skin_temp_c",
]

MODEL_F_CATEGORICAL = [
    "sex",
]


MODELS = {
    "Model A - Environment + Workload": {
        "numeric": MODEL_A_NUMERIC,
        "categorical": MODEL_A_CATEGORICAL,
    },

    "Model B - + Worker Characteristics": {
        "numeric": MODEL_B_NUMERIC,
        "categorical": MODEL_B_CATEGORICAL,
    },

    "Model C - + Physiology + Laboratory": {
        "numeric": MODEL_C_NUMERIC,
        "categorical": MODEL_C_CATEGORICAL,
    },

    "Model D - Deployment Realistic": {
        "numeric": MODEL_D_NUMERIC,
        "categorical": MODEL_D_CATEGORICAL,
    },

    "Model E - Minimal Wearable": {
        "numeric": MODEL_E_NUMERIC,
        "categorical": MODEL_E_CATEGORICAL,
    },

    "Model F - Minimal Wearable + HRV": {
        "numeric": MODEL_F_NUMERIC,
        "categorical": MODEL_F_CATEGORICAL,
    },
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("THERMOSENTINEL BASELINE COMPARISON")
    print("=" * 80)

    print(f"\nDataset shape: {df.shape}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:

    # Collect all columns required by all three models.
    all_numeric = set()
    all_categorical = set()

    for config in MODELS.values():
        all_numeric.update(config["numeric"])
        all_categorical.update(config["categorical"])

    required_columns = (
        ["participant_id", "condition_id", "time_min", TARGET]
        + list(all_numeric)
        + list(all_categorical)
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    # Target is the only column we MUST have.
    df = df.dropna(
        subset=[TARGET]
    ).copy()

    # Sort chronologically within participant/trial.
    df = df.sort_values(
        [
            "participant_id",
            "condition_id",
            "time_min",
        ]
    ).reset_index(drop=True)

    print(
        f"\nRows after removing missing target: "
        f"{len(df):,}"
    )

    print(
        f"Participants: "
        f"{df['participant_id'].nunique()}"
    )

    return df


# ============================================================
# CREATE ONE FIXED PARTICIPANT SPLIT
# ============================================================

def split_data(df: pd.DataFrame):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df[TARGET],
            groups=df["participant_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_participants = sorted(
        train_df["participant_id"]
        .unique()
        .tolist()
    )

    test_participants = sorted(
        test_df["participant_id"]
        .unique()
        .tolist()
    )

    print("\n" + "=" * 80)
    print("FIXED PARTICIPANT SPLIT")
    print("=" * 80)

    print(
        "\nTraining participants:"
    )
    print(train_participants)

    print(
        "\nTest participants:"
    )
    print(test_participants)

    print(
        f"\nTraining rows: {len(train_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    # Safety check:
    overlap = set(train_participants).intersection(
        set(test_participants)
    )

    if overlap:
        raise RuntimeError(
            f"Participant leakage detected: {overlap}"
        )

    return train_df, test_df


# ============================================================
# BUILD MODEL PIPELINE
# ============================================================

def build_pipeline(
    numeric_features,
    categorical_features,
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return predictions, mae, rmse, r2


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    pipeline,
    output_path,
):

    preprocessor = (
        pipeline
        .named_steps["preprocessor"]
    )

    model = (
        pipeline
        .named_steps["model"]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    return importance_df


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load + prepare
    # --------------------------------------------------------

    df = load_data()

    df = prepare_data(df)

    # --------------------------------------------------------
    # Create ONE fixed participant split.
    # All three models use this exact split.
    # --------------------------------------------------------

    train_df, test_df = split_data(df)

    results = []

    trained_models = {}

    # --------------------------------------------------------
    # Train A, B, C
    # --------------------------------------------------------

    for model_name, config in MODELS.items():

        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)

        numeric_features = config["numeric"]
        categorical_features = config["categorical"]

        features = (
            numeric_features
            + categorical_features
        )

        X_train = train_df[features]
        y_train = train_df[TARGET]

        X_test = test_df[features]
        y_test = test_df[TARGET]

        print(
            f"\nNumber of features: "
            f"{len(features)}"
        )

        print("Training...")

        pipeline = build_pipeline(
            numeric_features,
            categorical_features,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        print("Training complete.")

        predictions, mae, rmse, r2 = (
            evaluate_model(
                pipeline,
                X_test,
                y_test,
            )
        )

        print("\nResults:")
        print(
            f"MAE :  {mae:.4f} °C"
        )
        print(
            f"RMSE:  {rmse:.4f} °C"
        )
        print(
            f"R²  :  {r2:.4f}"
        )

        results.append(
            {
                "model": model_name,
                "feature_count": len(features),
                "MAE_C": mae,
                "RMSE_C": rmse,
                "R2": r2,
            }
        )

        trained_models[model_name] = (
            pipeline,
            predictions,
        )

        # ----------------------------------------------------
        # Save predictions for this model
        # ----------------------------------------------------

        prediction_df = pd.DataFrame(
            {
                "participant_id": test_df[
                    "participant_id"
                ].values,
                "condition_id": test_df[
                    "condition_id"
                ].values,
                "time_min": test_df[
                    "time_min"
                ].values,
                "actual_core_temp_c": (
                    y_test.values
                ),
                "predicted_core_temp_c": (
                    predictions
                ),
            }
        )

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("+", "plus")
        )

        prediction_path = (
            RESULTS_DIR
            / f"{safe_name}_predictions.csv"
        )

        prediction_df.to_csv(
            prediction_path,
            index=False,
        )

        print(
            f"Predictions saved to:\n"
            f"{prediction_path}"
        )

    # --------------------------------------------------------
    # Save comparison results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    comparison_path = (
        RESULTS_DIR
        / "baseline_model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nComparison saved to:\n"
        f"{comparison_path}"
    )

    # --------------------------------------------------------
    # Model C feature importance
    # --------------------------------------------------------

    model_c_name = (
    "Model C - + Physiology + Laboratory"
)

    model_c_pipeline = (
        trained_models[
            model_c_name
        ][0]
    )

    feature_importance_path = (
        RESULTS_DIR
        / "model_c_feature_importance.csv"
    )

    importance_df = save_feature_importance(
        model_c_pipeline,
        feature_importance_path,
    )

    print("\n" + "=" * 80)
    print("MODEL C — TOP 20 FEATURE IMPORTANCES")
    print("=" * 80)

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    print(
        f"\nFeature importance saved to:\n"
        f"{feature_importance_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()