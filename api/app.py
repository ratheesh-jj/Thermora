"""
ThermoSentinel / Thermora — Prediction API

Loads the trained LightGBM delta-temperature model and serves
predictions via a simple REST endpoint.

Model: thermora_lightgbm_delta.pkl  (MAE ≈ 0.111 °C)
"""

import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "thermora_lightgbm_delta.pkl"
)

IMPUTER_PATH = os.path.join(
    PROJECT_ROOT, "models", "thermora_delta_imputer.pkl"
)


# ============================================================
# LOAD MODEL + IMPUTER
# ============================================================

print("Loading LightGBM delta model...")
model = joblib.load(MODEL_PATH)
print(f"  Model loaded from: {MODEL_PATH}")

print("Loading training-fitted imputer...")
imputer = joblib.load(IMPUTER_PATH)
print(f"  Imputer loaded from: {IMPUTER_PATH}")


# ============================================================
# FEATURE ENGINEERING
#
# This MUST match train_delta_models.py / evaluate_delta_model.py
#
# Input sequence shape: (1, 20, 7)
# Output engineered features: (1, 102)
#
# Feature indices:
#   0 = age_years
#   1 = body_mass_kg
#   2 = ambient_temp_c
#   3 = humidity_pct
#   4 = heart_rate_bpm
#   5 = insulated_skin_temp_c
#   6 = sex_male
# ============================================================

def engineer_features(X):
    """
    Convert (samples, timesteps, features) → (samples, 102 engineered features).
    Mirrors the exact pipeline from train_delta_models.py.
    """

    n_samples, timesteps, n_features = X.shape

    feature_blocks = []

    # --------------------------------------------------------
    # Last timestep  →  7 features
    # --------------------------------------------------------

    last = X[:, -1, :]
    feature_blocks.append(last)

    # --------------------------------------------------------
    # Statistics over complete 20-minute window  →  4 × 7 = 28
    # --------------------------------------------------------

    feature_blocks.append(np.nanmean(X, axis=1))
    feature_blocks.append(np.nanstd(X, axis=1))
    feature_blocks.append(np.nanmin(X, axis=1))
    feature_blocks.append(np.nanmax(X, axis=1))

    # --------------------------------------------------------
    # Recent temporal windows  →  3 windows × 2 stats × 7 = 42
    # --------------------------------------------------------

    windows = {
        "last5": X[:, -5:, :],
        "last10": X[:, -10:, :],
        "last15": X[:, -15:, :],
    }

    for window in windows.values():
        feature_blocks.append(np.nanmean(window, axis=1))
        feature_blocks.append(np.nanstd(window, axis=1))

    # --------------------------------------------------------
    # Slopes / changes for temporal features  →  4 feats × 5 = 20
    # --------------------------------------------------------

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
    # Physiological / environmental interactions  →  5
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

    # --------------------------------------------------------
    # Combine  →  7 + 28 + 42 + 20 + 5 = 102
    # --------------------------------------------------------

    features = np.concatenate(feature_blocks, axis=1)

    return features


# ============================================================
# BUILD SYNTHETIC SEQUENCE
#
# The user provides a single snapshot of readings.
# We construct a 20-timestep "steady-state" sequence
# (all timesteps identical) so the feature engineering
# pipeline can run correctly.
# ============================================================

def build_sequence(
    age_years,
    body_mass_kg,
    ambient_temp_c,
    humidity_pct,
    heart_rate_bpm,
    insulated_skin_temp_c,
    sex_male,
):
    """Build a (1, 20, 7) sequence from a single snapshot."""

    snapshot = np.array(
        [
            age_years,
            body_mass_kg,
            ambient_temp_c,
            humidity_pct,
            heart_rate_bpm,
            insulated_skin_temp_c,
            sex_male,
        ],
        dtype=np.float32,
    )

    # Repeat the snapshot across 20 timesteps
    sequence = np.tile(snapshot, (20, 1))  # (20, 7)

    return sequence[np.newaxis, :, :]  # (1, 20, 7)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with fields:
        age_years, body_mass_kg, ambient_temp_c, humidity_pct,
        heart_rate_bpm, insulated_skin_temp_c, sex,
        current_core_temp_c

    Returns:
        predicted_delta, predicted_future_temp, risk_level
    """

    try:
        data = request.get_json()

        # Parse inputs
        age_years = float(data["age_years"])
        body_mass_kg = float(data["body_mass_kg"])
        ambient_temp_c = float(data["ambient_temp_c"])
        humidity_pct = float(data["humidity_pct"])
        heart_rate_bpm = float(data["heart_rate_bpm"])
        insulated_skin_temp_c = float(data["insulated_skin_temp_c"])
        sex_male = 1.0 if data["sex"] == "Male" else 0.0
        current_core_temp = float(data["current_core_temp_c"])

        # Build synthetic sequence
        X = build_sequence(
            age_years,
            body_mass_kg,
            ambient_temp_c,
            humidity_pct,
            heart_rate_bpm,
            insulated_skin_temp_c,
            sex_male,
        )

        # Engineer features (102)
        X_eng = engineer_features(X)

        # Impute missing values
        X_eng = imputer.transform(X_eng)

        # Predict delta temperature
        predicted_delta = float(model.predict(X_eng)[0])

        # Reconstruct future temperature
        predicted_future = current_core_temp + predicted_delta

        # Determine risk level
        if predicted_future < 38.0:
            risk_level = "Normal"
            risk_color = "#22c55e"
        elif predicted_future < 38.5:
            risk_level = "Elevated"
            risk_color = "#f59e0b"
        elif predicted_future < 39.0:
            risk_level = "High"
            risk_color = "#f97316"
        else:
            risk_level = "Critical"
            risk_color = "#ef4444"

        # Direction
        if predicted_delta > 0.05:
            direction = "Heating"
        elif predicted_delta < -0.05:
            direction = "Cooling"
        else:
            direction = "Stable"

        return jsonify(
            {
                "success": True,
                "current_core_temp": round(current_core_temp, 2),
                "predicted_delta": round(predicted_delta, 4),
                "predicted_future_temp": round(predicted_future, 2),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "direction": direction,
            }
        )

    except KeyError as e:
        return jsonify(
            {"success": False, "error": f"Missing field: {e}"}
        ), 400

    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e)}
        ), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "LightGBM ΔT"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
