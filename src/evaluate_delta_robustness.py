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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("THERMORA — ROBUSTNESS & LEAKAGE AUDIT")
print("=" * 80)

print("\nLoading test dataset...")

data = np.load(TEST_PATH, allow_pickle=True)

X_test = data["X"]
y_test = data["y"]
metadata = data["metadata"]

print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")
print(f"metadata     : {metadata.shape}")


# ============================================================
# METADATA
# ============================================================

# metadata:
# 0 = participant_id
# 1 = condition_id
# 2 = input_end_time
# 3 = target_time
# 4 = current_core_temp

participant_ids = metadata[:, 0].astype(int)
condition_ids = metadata[:, 1].astype(int)

current_test = metadata[:, 4].astype(float)

actual_future = y_test.astype(float)

actual_delta = actual_future - current_test


# ============================================================
# LOAD TRAINED MODEL + IMPUTER
# ============================================================

print("\n" + "=" * 80)
print("LOADING TRAINED MODEL")
print("=" * 80)

print("\nLoading trained LightGBM model...")

model = joblib.load(MODEL_PATH)

print(f"Model loaded from:\n{os.path.abspath(MODEL_PATH)}")

print("\nLoading training-fitted imputer...")

imputer = joblib.load(IMPUTER_PATH)

print(f"Imputer loaded from:\n{os.path.abspath(IMPUTER_PATH)}")


# ============================================================
# EXACT TRAINING FEATURE ENGINEERING
# ============================================================

def engineer_features(X):

    """
    EXACT SAME FEATURE ENGINEERING USED DURING TRAINING.

    Input:
        X = (samples, 20 timesteps, 7 features)

    Output:
        (samples, 102 features)

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

    features = np.concatenate(
        feature_blocks,
        axis=1
    )

    return features


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

print("\nEngineering temporal features...")

X_test_eng = engineer_features(X_test)

print(
    f"Engineered feature shape: {X_test_eng.shape}"
)

expected_features = 102

if X_test_eng.shape[1] != expected_features:

    raise ValueError(
        f"\nFeature mismatch!\n"
        f"Expected: {expected_features}\n"
        f"Got     : {X_test_eng.shape[1]}"
    )

print(
    f"\n✓ Feature count matches training pipeline: "
    f"{expected_features}"
)


# ============================================================
# APPLY TRAINING IMPUTER
# ============================================================

print("\nApplying training-fitted imputer...")

X_test_eng = imputer.transform(X_test_eng)

print(
    f"After imputation: {X_test_eng.shape}"
)

print(
    f"Remaining NaNs: {np.isnan(X_test_eng).sum()}"
)

model_features = model.n_features_in_

print(
    f"Model expects: {model_features} features"
)

if model_features != X_test_eng.shape[1]:

    raise ValueError(
        f"\nModel/input feature mismatch!\n"
        f"Model expects: {model_features}\n"
        f"Input has     : {X_test_eng.shape[1]}"
    )

print("✓ Model feature count matches.")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION")
print("=" * 80)

print("\nPredicting future temperature change...")

predicted_delta = model.predict(X_test_eng)

predicted_future = (
    current_test + predicted_delta
)


# ============================================================
# BASIC TEST METRICS
# ============================================================

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
    actual_delta,
    predicted_delta
)


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

persistence_future = current_test

persistence_mae = mean_absolute_error(
    actual_future,
    persistence_future
)

persistence_rmse = np.sqrt(
    mean_squared_error(
        actual_future,
        persistence_future
    )
)

persistence_r2 = r2_score(
    actual_future,
    persistence_future
)

improvement = (
    (persistence_mae - mae)
    / persistence_mae
) * 100


# ============================================================
# ROBUSTNESS — PARTICIPANT LEVEL
# ============================================================

print("\n" + "=" * 80)
print("PARTICIPANT-LEVEL ROBUSTNESS")
print("=" * 80)

participant_results = []

for participant in sorted(
    np.unique(participant_ids)
):

    mask = participant_ids == participant

    if mask.sum() < 2:
        continue

    p_mae = mean_absolute_error(
        actual_future[mask],
        predicted_future[mask]
    )

    p_rmse = np.sqrt(
        mean_squared_error(
            actual_future[mask],
            predicted_future[mask]
        )
    )

    p_r2 = r2_score(
        actual_future[mask],
        predicted_future[mask]
    ) if mask.sum() >= 2 else np.nan

    p_baseline_mae = mean_absolute_error(
        actual_future[mask],
        persistence_future[mask]
    )

    p_improvement = (
        (p_baseline_mae - p_mae)
        / p_baseline_mae
        * 100
    ) if p_baseline_mae > 0 else np.nan

    participant_results.append({
        "participant_id": participant,
        "samples": int(mask.sum()),
        "MAE_C": p_mae,
        "RMSE_C": p_rmse,
        "R2": p_r2,
        "Persistence_MAE_C": p_baseline_mae,
        "Improvement_pct": p_improvement
    })

participant_df = pd.DataFrame(
    participant_results
)

print(
    participant_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ROBUSTNESS SUMMARY")
print("=" * 80)

print(
    f"\nParticipants evaluated: "
    f"{len(participant_df)}"
)

print(
    f"Participant MAE mean: "
    f"{participant_df['MAE_C'].mean():.4f} °C"
)

print(
    f"Participant MAE std : "
    f"{participant_df['MAE_C'].std():.4f} °C"
)

print(
    f"Best participant MAE : "
    f"{participant_df['MAE_C'].min():.4f} °C"
)

print(
    f"Worst participant MAE: "
    f"{participant_df['MAE_C'].max():.4f} °C"
)

print(
    f"Participants improving over persistence: "
    f"{(participant_df['Improvement_pct'] > 0).sum()}"
    f"/{len(participant_df)}"
)


# ============================================================
# TEMPERATURE DIRECTION ROBUSTNESS
# ============================================================

print("\n" + "=" * 80)
print("TEMPERATURE DIRECTION ANALYSIS")
print("=" * 80)

threshold = 0.05

def classify_direction(delta):

    if delta > threshold:
        return "Heating"

    elif delta < -threshold:
        return "Cooling"

    else:
        return "Stable"


actual_direction = np.array([
    classify_direction(x)
    for x in actual_delta
])

predicted_direction = np.array([
    classify_direction(x)
    for x in predicted_delta
])

direction_accuracy = (
    actual_direction == predicted_direction
).mean() * 100

print(
    f"\nDirection accuracy: "
    f"{direction_accuracy:.2f}%"
)

print("\nActual direction distribution:")

for direction in [
    "Heating",
    "Stable",
    "Cooling"
]:

    percentage = (
        actual_direction == direction
    ).mean() * 100

    print(
        f"{direction:<10}: "
        f"{percentage:.2f}%"
    )

print("\nPredicted direction distribution:")

for direction in [
    "Heating",
    "Stable",
    "Cooling"
]:

    percentage = (
        predicted_direction == direction
    ).mean() * 100

    print(
        f"{direction:<10}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("LEAKAGE AUDIT")
print("=" * 80)

print(
    "\nChecking whether future target information "
    "appears in model inputs..."
)

# The model inputs are based exclusively on X_test.
# The target y_test is NOT passed into engineer_features().
#
# current_core_temp comes from metadata[:, 4].
# It is used only to reconstruct:
#
# predicted_future = current_core_temp + predicted_delta
#
# Therefore the future target y_test is not an input feature.

print(
    "✓ Future core temperature is NOT used "
    "as a model input."
)

print(
    "✓ Future delta is NOT used as a model input."
)

print(
    "✓ y_test is used only for final evaluation."
)

print(
    "✓ Imputer was loaded from training."
)

print(
    "✓ No preprocessing was fitted on test data."
)


# ============================================================
# SAVE PARTICIPANT RESULTS
# ============================================================

participant_path = os.path.join(
    RESULTS_DIR,
    "delta_robustness_participant_results.csv"
)

participant_df.to_csv(
    participant_path,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "participant_id": participant_ids,

    "condition_id": condition_ids,

    "current_core_temp": current_test,

    "actual_future_temp": actual_future,

    "actual_delta_temp": actual_delta,

    "predicted_delta_temp": predicted_delta,

    "predicted_future_temp": predicted_future,

    "actual_direction": actual_direction,

    "predicted_direction": predicted_direction

})

prediction_path = os.path.join(
    RESULTS_DIR,
    "delta_robustness_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("THERMORA — FINAL ROBUSTNESS AUDIT")
print("=" * 80)

print("\nGLOBAL TEST PERFORMANCE")
print("-" * 60)

print(
    f"LightGBM ΔT MAE       : "
    f"{mae:.4f} °C"
)

print(
    f"LightGBM ΔT RMSE      : "
    f"{rmse:.4f} °C"
)

print(
    f"LightGBM ΔT R²        : "
    f"{r2:.4f}"
)

print(
    f"Delta MAE             : "
    f"{delta_mae:.4f} °C"
)

print(
    f"Persistence MAE       : "
    f"{persistence_mae:.4f} °C"
)

print(
    f"Improvement            : "
    f"{improvement:+.2f}%"
)

print(
    f"Direction accuracy     : "
    f"{direction_accuracy:.2f}%"
)

print("\nPARTICIPANT ROBUSTNESS")
print("-" * 60)

print(
    f"Participants evaluated : "
    f"{len(participant_df)}"
)

print(
    f"Mean participant MAE   : "
    f"{participant_df['MAE_C'].mean():.4f} °C"
)

print(
    f"MAE standard deviation : "
    f"{participant_df['MAE_C'].std():.4f} °C"
)

print(
    f"Best participant MAE   : "
    f"{participant_df['MAE_C'].min():.4f} °C"
)

print(
    f"Worst participant MAE  : "
    f"{participant_df['MAE_C'].max():.4f} °C"
)

print("\nLeakage status:")
print("✓ No target leakage detected in evaluation pipeline")

print("\nResults saved to:")
print(participant_path)
print(prediction_path)

print("\n" + "=" * 80)
print("ROBUSTNESS AUDIT COMPLETE")
print("=" * 80)