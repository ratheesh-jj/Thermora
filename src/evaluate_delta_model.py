import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIG
# ============================================================

TEST_PATH = "data/sequences/test_10min.npz"

MODEL_PATH = "models/thermora_lightgbm_delta.pkl"
IMPUTER_PATH = "models/thermora_delta_imputer.pkl"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "delta_model_evaluation_predictions.csv"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_dataset(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    metadata = data["metadata"]

    return X, y, metadata


print("=" * 75)
print("THERMORA — DELTA TEMPERATURE MODEL EVALUATION")
print("=" * 75)

print("\nLoading test dataset...")

X_test, y_test, metadata = load_dataset(TEST_PATH)

print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")
print(f"Metadata     : {metadata.shape}")


# ============================================================
# CURRENT CORE TEMPERATURE
#
# metadata:
# 0 = participant_id
# 1 = condition_id
# 2 = input_end_time
# 3 = target_time
# 4 = current_core_temp
# ============================================================

current_test = metadata[:, 4].astype(float)


# ============================================================
# TARGET DELTA
# ============================================================

delta_test = y_test - current_test


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
    "age_years",
    "body_mass_kg",
    "ambient_temp_c",
    "humidity_pct",
    "heart_rate_bpm",
    "insulated_skin_temp_c",
    "sex_male"
]

print("\nSequence features:")

for i, name in enumerate(feature_names):
    print(f" {i}: {name}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 75)
print("LOADING TRAINED MODEL")
print("=" * 75)

print("\nLoading trained LightGBM model...")

model = joblib.load(MODEL_PATH)

print("Model loaded from:")
print(os.path.abspath(MODEL_PATH))


# ============================================================
# LOAD TRAINING-FITTED IMPUTER
# ============================================================

print("\nLoading training-fitted imputer...")

imputer = joblib.load(IMPUTER_PATH)

print("Imputer loaded from:")
print(os.path.abspath(IMPUTER_PATH))


# ============================================================
# TEMPORAL FEATURE ENGINEERING
#
# THIS MUST MATCH train_delta_models.py EXACTLY
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

    feature_blocks.append(
        np.nanmean(X, axis=1)
    )

    feature_blocks.append(
        np.nanstd(X, axis=1)
    )

    feature_blocks.append(
        np.nanmin(X, axis=1)
    )

    feature_blocks.append(
        np.nanmax(X, axis=1)
    )

    # --------------------------------------------------------
    # Recent temporal windows
    # --------------------------------------------------------

    windows = {
        "last5": X[:, -5:, :],
        "last10": X[:, -10:, :],
        "last15": X[:, -15:, :]
    }

    for window in windows.values():

        feature_blocks.append(
            np.nanmean(window, axis=1)
        )

        feature_blocks.append(
            np.nanstd(window, axis=1)
        )

    # --------------------------------------------------------
    # Slopes / changes
    # --------------------------------------------------------

    temporal_features = [2, 3, 4, 5]

    for idx in temporal_features:

        series = X[:, :, idx]

        # Current value
        feature_blocks.append(
            series[:, -1][:, None]
        )

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

        feature_blocks.append(
            slopes[:, None]
        )

    # --------------------------------------------------------
    # Physiological / environmental interactions
    # --------------------------------------------------------

    heart_rate = X[:, -1, 4]
    skin_temp = X[:, -1, 5]
    ambient = X[:, -1, 2]
    humidity = X[:, -1, 3]

    feature_blocks.append(
        (heart_rate * skin_temp)[:, None]
    )

    feature_blocks.append(
        (heart_rate * ambient)[:, None]
    )

    feature_blocks.append(
        (skin_temp - ambient)[:, None]
    )

    feature_blocks.append(
        (skin_temp - ambient)[:, None] ** 2
    )

    feature_blocks.append(
        (humidity * ambient)[:, None]
    )

    # --------------------------------------------------------
    # Combine everything
    # --------------------------------------------------------

    features = np.concatenate(
        feature_blocks,
        axis=1
    )

    return features


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 75)
print("FEATURE ENGINEERING")
print("=" * 75)

print("\nEngineering temporal features...")

X_test_eng = engineer_features(X_test)

print(
    f"Engineered feature shape: {X_test_eng.shape}"
)


# ============================================================
# VERIFY FEATURE COUNT
# ============================================================

expected_features = 102

if X_test_eng.shape[1] != expected_features:

    raise ValueError(
        f"\nFeature mismatch!\n"
        f"Expected: {expected_features}\n"
        f"Got:      {X_test_eng.shape[1]}\n\n"
        f"The evaluation feature engineering does not match "
        f"train_delta_models.py."
    )

print(
    f"\n✓ Feature count matches training pipeline: "
    f"{X_test_eng.shape[1]} features"
)


# ============================================================
# APPLY TRAINING-FITTED IMPUTER
# ============================================================

print("\nApplying training-fitted imputer...")

X_test_eng = imputer.transform(X_test_eng)

print(
    f"After imputation: {X_test_eng.shape}"
)

print(
    f"Remaining NaNs: {np.isnan(X_test_eng).sum()}"
)


# ============================================================
# MODEL FEATURE COUNT CHECK
# ============================================================

if hasattr(model, "n_features_in_"):

    print(
        f"Model expects: {model.n_features_in_} features"
    )

    if model.n_features_in_ != X_test_eng.shape[1]:

        raise ValueError(
            f"\nModel feature mismatch!\n"
            f"Model expects: {model.n_features_in_}\n"
            f"Input contains: {X_test_eng.shape[1]}"
        )

    print("✓ Model feature count matches.")


# ============================================================
# PREDICT DELTA
# ============================================================

print("\n" + "=" * 75)
print("PREDICTION")
print("=" * 75)

print("\nPredicting future temperature change...")

predicted_delta = model.predict(X_test_eng)


# ============================================================
# RECONSTRUCT FUTURE TEMPERATURE
# ============================================================

predicted_future = (
    current_test + predicted_delta
)


# ============================================================
# EVALUATE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predicted_future
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predicted_future
    )
)

r2 = r2_score(
    y_test,
    predicted_future
)

delta_mae = mean_absolute_error(
    delta_test,
    predicted_delta
)


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

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


# ============================================================
# IMPROVEMENT
# ============================================================

improvement = (
    (persistence_mae - mae)
    / persistence_mae
) * 100


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("THERMORA — FINAL TEST RESULTS")
print("=" * 75)

print("\nLIGHTGBM ΔT MODEL")
print("-" * 60)

print(
    f"MAE :  {mae:.4f} °C"
)

print(
    f"RMSE:  {rmse:.4f} °C"
)

print(
    f"R²  :  {r2:.4f}"
)

print(
    f"ΔT MAE: {delta_mae:.4f} °C"
)


print("\nPERSISTENCE BASELINE")
print("-" * 60)

print(
    f"MAE :  {persistence_mae:.4f} °C"
)

print(
    f"RMSE:  {persistence_rmse:.4f} °C"
)

print(
    f"R²  :  {persistence_r2:.4f}"
)


print("\nIMPROVEMENT OVER PERSISTENCE")
print("-" * 60)

print(
    f"MAE improvement: {improvement:+.2f}%"
)


# ============================================================
# TEMPERATURE DIRECTION ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("TEMPERATURE DIRECTION ANALYSIS")
print("=" * 75)

actual_direction = np.where(
    delta_test > 0.05,
    "Heating",
    np.where(
        delta_test < -0.05,
        "Cooling",
        "Stable"
    )
)

predicted_direction = np.where(
    predicted_delta > 0.05,
    "Heating",
    np.where(
        predicted_delta < -0.05,
        "Cooling",
        "Stable"
    )
)


direction_accuracy = (
    actual_direction == predicted_direction
).mean() * 100


print(
    f"\nDirection accuracy: "
    f"{direction_accuracy:.2f}%"
)


print("\nActual direction distribution:")

for direction in ["Heating", "Stable", "Cooling"]:

    percentage = (
        actual_direction == direction
    ).mean() * 100

    print(
        f"{direction:<10}: "
        f"{percentage:.2f}%"
    )


print("\nPredicted direction distribution:")

for direction in ["Heating", "Stable", "Cooling"]:

    percentage = (
        predicted_direction == direction
    ).mean() * 100

    print(
        f"{direction:<10}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "participant_id":
        metadata[:, 0],

    "condition_id":
        metadata[:, 1],

    "input_end_time":
        metadata[:, 2],

    "target_time":
        metadata[:, 3],

    "current_core_temp":
        current_test,

    "actual_future_temp":
        y_test,

    "actual_delta_temp":
        delta_test,

    "predicted_delta_temp":
        predicted_delta,

    "predicted_future_temp":
        predicted_future,

    "actual_direction":
        actual_direction,

    "predicted_direction":
        predicted_direction

})


predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)

print("\nResults:")

print(
    f"LightGBM MAE       : {mae:.4f} °C"
)

print(
    f"LightGBM RMSE      : {rmse:.4f} °C"
)

print(
    f"LightGBM R²        : {r2:.4f}"
)

print(
    f"ΔT MAE             : {delta_mae:.4f} °C"
)

print(
    f"Persistence MAE    : {persistence_mae:.4f} °C"
)

print(
    f"Improvement        : {improvement:+.2f}%"
)

print(
    f"Direction accuracy : {direction_accuracy:.2f}%"
)

print("\nPredictions saved to:")

print(
    os.path.abspath(PREDICTIONS_PATH)
)

print("\n" + "=" * 75)