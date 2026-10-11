import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "data/sequences/train_10min.npz"
VAL_PATH   = "data/sequences/validation_10min.npz"
TEST_PATH  = "data/sequences/test_10min.npz"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    metadata = data["metadata"]

    return X, y, metadata


X_train, y_train, meta_train = load_dataset(TRAIN_PATH)
X_val, y_val, meta_val = load_dataset(VAL_PATH)
X_test, y_test, meta_test = load_dataset(TEST_PATH)


print("=" * 80)
print("THERMORA — DELTA TEMPERATURE MODEL COMPARISON")
print("=" * 80)

print("\nDataset shapes:")
print(f"Train: {X_train.shape}")
print(f"Val  : {X_val.shape}")
print(f"Test : {X_test.shape}")


# ============================================================
# CURRENT CORE TEMPERATURE
# metadata:
# 0 = participant_id
# 1 = condition_id
# 2 = input_end_time
# 3 = target_time
# 4 = current_core_temp
# ============================================================

current_train = meta_train[:, 4].astype(float)
current_val   = meta_val[:, 4].astype(float)
current_test  = meta_test[:, 4].astype(float)


# ============================================================
# TARGET: DELTA TEMPERATURE
# ============================================================

delta_train = y_train - current_train
delta_val   = y_val - current_val
delta_test  = y_test - current_test


print("\nDELTA TEMPERATURE DISTRIBUTION")
print("-" * 60)

print(f"Train mean ΔT: {delta_train.mean():.4f} °C")
print(f"Train std  ΔT: {delta_train.std():.4f} °C")
print(f"Train min  ΔT: {delta_train.min():.4f} °C")
print(f"Train max  ΔT: {delta_train.max():.4f} °C")

print(f"\nHeating: {(delta_train > 0.05).mean() * 100:.2f}%")
print(f"Stable : {(np.abs(delta_train) <= 0.05).mean() * 100:.2f}%")
print(f"Cooling: {(delta_train < -0.05).mean() * 100:.2f}%")


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def engineer_features(X):
    """
    Convert:
        (samples, timesteps, features)

    into:
        (samples, engineered_features)

    Original features:
    0 = age_years
    1 = body_mass_kg
    2 = ambient_temp_c
    3 = humidity_pct
    4 = heart_rate_bpm
    5 = insulated_skin_temp_c
    6 = sex_male
    """

    n_samples, timesteps, n_features = X.shape

    feature_blocks = []

    # --------------------------------------------------------
    # Last timestep
    # --------------------------------------------------------

    last = X[:, -1, :]
    feature_blocks.append(last)

    # --------------------------------------------------------
    # Statistics over complete 20-minute window
    # --------------------------------------------------------

    feature_blocks.append(np.nanmean(X, axis=1))
    feature_blocks.append(np.nanstd(X, axis=1))
    feature_blocks.append(np.nanmin(X, axis=1))
    feature_blocks.append(np.nanmax(X, axis=1))

    # --------------------------------------------------------
    # Recent temporal windows
    # --------------------------------------------------------

    windows = {
        "last5": X[:, -5:, :],
        "last10": X[:, -10:, :],
        "last15": X[:, -15:, :]
    }

    for window in windows.values():
        feature_blocks.append(np.nanmean(window, axis=1))
        feature_blocks.append(np.nanstd(window, axis=1))

    # --------------------------------------------------------
    # Slopes / changes
    # --------------------------------------------------------

    # Feature indices that have meaningful temporal behavior
    temporal_features = [2, 3, 4, 5]

    for idx in temporal_features:

        series = X[:, :, idx]

        # Current value
        feature_blocks.append(series[:, -1][:, None])

        # Change over 5 minutes
        feature_blocks.append(
            (series[:, -1] - series[:, -6])[:, None]
        )

        # Change over 10 minutes
        feature_blocks.append(
            (series[:, -1] - series[:, -11])[:, None]
        )

        # Change over 15 minutes
        feature_blocks.append(
            (series[:, -1] - series[:, -16])[:, None]
        )

        # Linear slope over full window
        t = np.arange(timesteps)

        slopes = np.zeros(n_samples)

        for i in range(n_samples):
            valid = ~np.isnan(series[i])

            if valid.sum() >= 2:
                slopes[i] = np.polyfit(
                    t[valid],
                    series[i, valid],
                    1
                )[0]
            else:
                slopes[i] = np.nan

        feature_blocks.append(slopes[:, None])

    # --------------------------------------------------------
    # Physiological/environmental interactions
    # --------------------------------------------------------

    heart_rate = X[:, -1, 4]
    skin_temp = X[:, -1, 5]
    ambient = X[:, -1, 2]
    humidity = X[:, -1, 3]

    feature_blocks.append((heart_rate * skin_temp)[:, None])
    feature_blocks.append((heart_rate * ambient)[:, None])
    feature_blocks.append((skin_temp - ambient)[:, None])
    feature_blocks.append((skin_temp - ambient)[:, None] ** 2)
    feature_blocks.append((humidity * ambient)[:, None])

    features = np.concatenate(feature_blocks, axis=1)

    return features


print("\nEngineering temporal features...")

X_train_eng = engineer_features(X_train)
X_val_eng   = engineer_features(X_val)
X_test_eng  = engineer_features(X_test)

print(f"Engineered feature shape: {X_train_eng.shape}")


# ============================================================
# IMPUTATION
# ============================================================

print("\nFitting imputer on training data only...")

imputer = SimpleImputer(strategy="median")

X_train_eng = imputer.fit_transform(X_train_eng)
X_val_eng   = imputer.transform(X_val_eng)
X_test_eng  = imputer.transform(X_test_eng)

print("Missing values after preprocessing:")
print(f"Train: {np.isnan(X_train_eng).sum()}")
print(f"Val  : {np.isnan(X_val_eng).sum()}")
print(f"Test : {np.isnan(X_test_eng).sum()}")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

results = []
prediction_frames = []


def evaluate_model(
    name,
    model,
    X_train,
    y_train_delta,
    X_val,
    y_val_delta,
    X_test,
    y_test_delta,
    current_test,
    actual_future
):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("Training...")

    model.fit(
        X_train,
        y_train_delta
    )

    print("Training complete.")

    predicted_delta = model.predict(X_test)

    # Reconstruct future temperature
    predicted_future = current_test + predicted_delta

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

    print("\nFuture Temperature Results:")
    print(f"MAE :  {mae:.4f} °C")
    print(f"RMSE:  {rmse:.4f} °C")
    print(f"R²  :  {r2:.4f}")

    print("\nDelta Temperature Results:")
    print(f"ΔT MAE: {delta_mae:.4f} °C")

    results.append({
        "model": name,
        "MAE_C": mae,
        "RMSE_C": rmse,
        "R2": r2,
        "Delta_MAE_C": delta_mae
    })

    prediction_frames.append(
        pd.DataFrame({
            "current_core_temp": current_test,
            "actual_future_temp": actual_future,
            "actual_delta_temp": y_test_delta,
            "predicted_delta_temp": predicted_delta,
            "predicted_future_temp": predicted_future
        }).assign(model=name)
    )

    return model, predicted_delta, predicted_future


# ============================================================
# MODELS
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


lgb_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)


# ============================================================
# PERSISTENCE
# ============================================================

print("\n" + "=" * 80)
print("PERSISTENCE BASELINE")
print("=" * 80)

persistence_pred = current_test

persistence_mae = mean_absolute_error(
    y_test,
    persistence_pred
)

persistence_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        persistence_pred
    )
)

persistence_r2 = r2_score(
    y_test,
    persistence_pred
)

print(f"MAE :  {persistence_mae:.4f} °C")
print(f"RMSE:  {persistence_rmse:.4f} °C")
print(f"R²  :  {persistence_r2:.4f}")

results.append({
    "model": "Persistence",
    "MAE_C": persistence_mae,
    "RMSE_C": persistence_rmse,
    "R2": persistence_r2,
    "Delta_MAE_C": 0.0
})


# ============================================================
# XGBOOST DELTA
# ============================================================

xgb_model, xgb_delta, xgb_future = evaluate_model(
    "XGBoost ΔT",
    xgb_model,
    X_train_eng,
    delta_train,
    X_val_eng,
    delta_val,
    X_test_eng,
    delta_test,
    current_test,
    y_test
)


# ============================================================
# LIGHTGBM DELTA
# ============================================================

lgb_model, lgb_delta, lgb_future = evaluate_model(
    "LightGBM ΔT",
    lgb_model,
    X_train_eng,
    delta_train,
    X_val_eng,
    delta_val,
    X_test_eng,
    delta_test,
    current_test,
    y_test
)


# ============================================================
# ENSEMBLE
# ============================================================

print("\n" + "=" * 80)
print("ENSEMBLE — XGBoost + LightGBM")
print("=" * 80)

ensemble_delta = (
    0.5 * xgb_delta +
    0.5 * lgb_delta
)

ensemble_future = (
    current_test +
    ensemble_delta
)

ensemble_mae = mean_absolute_error(
    y_test,
    ensemble_future
)

ensemble_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ensemble_future
    )
)

ensemble_r2 = r2_score(
    y_test,
    ensemble_future
)

ensemble_delta_mae = mean_absolute_error(
    delta_test,
    ensemble_delta
)

print(f"MAE :  {ensemble_mae:.4f} °C")
print(f"RMSE:  {ensemble_rmse:.4f} °C")
print(f"R²  :  {ensemble_r2:.4f}")
print(f"ΔT MAE: {ensemble_delta_mae:.4f} °C")

results.append({
    "model": "Ensemble ΔT",
    "MAE_C": ensemble_mae,
    "RMSE_C": ensemble_rmse,
    "R2": ensemble_r2,
    "Delta_MAE_C": ensemble_delta_mae
})

prediction_frames.append(
    pd.DataFrame({
        "current_core_temp": current_test,
        "actual_future_temp": y_test,
        "actual_delta_temp": delta_test,
        "predicted_delta_temp": ensemble_delta,
        "predicted_future_temp": ensemble_future
    }).assign(model="Ensemble ΔT")
)


# ============================================================
# COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("FINAL DELTA MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# IMPROVEMENT OVER PERSISTENCE
# ============================================================

print("\n" + "=" * 80)
print("IMPROVEMENT OVER PERSISTENCE")
print("=" * 80)

for _, row in results_df.iterrows():

    if row["model"] == "Persistence":
        continue

    improvement = (
        (persistence_mae - row["MAE_C"])
        / persistence_mae
    ) * 100

    print(
        f'{row["model"]:<20}: '
        f'{improvement:+.2f}%'
    )


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_path = os.path.join(
    RESULTS_DIR,
    "delta_model_comparison.csv"
)

predictions_path = os.path.join(
    RESULTS_DIR,
    "delta_model_predictions.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE MODELS
# ============================================================

import joblib

joblib.dump(
    xgb_model,
    "models/thermora_xgboost_delta.pkl"
)

joblib.dump(
    lgb_model,
    "models/thermora_lightgbm_delta.pkl"
)

joblib.dump(
    imputer,
    "models/thermora_delta_imputer.pkl"
)


print("\nResults saved to:")
print(comparison_path)
print(predictions_path)

print("\nModels saved to:")
print("models/thermora_xgboost_delta.pkl")
print("models/thermora_lightgbm_delta.pkl")
print("models/thermora_delta_imputer.pkl")

print("\n" + "=" * 80)
print("DELTA MODEL TRAINING COMPLETE")
print("=" * 80)