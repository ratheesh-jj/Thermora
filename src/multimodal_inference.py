"""
THERMORA MULTIMODAL INFERENCE
=============================

Independent inference pipelines:

1. THERMAL
   - Estimated current core temperature
   - Predicted core temperature +10 minutes
   - Predicted temperature rise

2. FATIGUE
   - Fatigue score 0-10
   - Fatigue trend

3. FALL
   - Fall probability
   - Fall detected

IMPORTANT:
- Outputs are NOT merged into one risk score.
- Thermal and fatigue datasets remain independent.
- Fall uses the EXACT 148-feature extraction logic used
  to train thermora_event_model_v2.pkl.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FATIGUE_DIR = os.path.join(
    BASE_DIR,
    "fatigue_prediction",
    "fatigue_deployable_model"
)

FALL_DIR = os.path.join(
    BASE_DIR,
    "Fall_Prediction"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# MODEL PATHS
# ============================================================

THERMAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "thermora_integrated_10min_no_wind_model.pkl"
)

FATIGUE_MODEL_PATH = os.path.join(
    FATIGUE_DIR,
    "thermora_fatigue_rf.pkl"
)

FATIGUE_IMPUTER_PATH = os.path.join(
    FATIGUE_DIR,
    "thermora_fatigue_imputer.pkl"
)

FATIGUE_FEATURES_PATH = os.path.join(
    FATIGUE_DIR,
    "thermora_fatigue_features.json"
)

FALL_MODEL_PATH = os.path.join(
    FALL_DIR,
    "thermora_event_model_v2.pkl"
)


# ============================================================
# OFFLINE DATA
# ============================================================

THERMAL_DATA_PATH = os.path.join(
    RESULTS_DIR,
    "thermal_risk_temporal_hysteresis_results.csv"
)

FATIGUE_DATA_PATH = os.path.join(
    RESULTS_DIR,
    "fatigue_batch_predictions.csv"
)

# The actual raw BITS-2 windows used for fall testing
FALL_DATA_PATH = os.path.join(
    FALL_DIR,
    "thermora_raw_event_windows_v2.npz"
)

FALL_META_PATH = os.path.join(
    FALL_DIR,
    "thermora_raw_event_metadata_v2.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

THERMAL_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "thermora_thermal_outputs.csv"
)

FATIGUE_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "thermora_fatigue_outputs.csv"
)

FALL_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "thermora_fall_outputs.csv"
)


# ============================================================
# EXACT FALL FEATURE EXTRACTION
# ============================================================

def extract_fall_features(X):
    """
    EXACT feature extraction used by train_event_model_v2.py.

    Input:
        X shape = (n_windows, samples, 6)

    Channel order:
        0 = acc_x
        1 = acc_y
        2 = acc_z
        3 = gyro_x
        4 = gyro_y
        5 = gyro_z

    Output:
        148 features per window.
    """

    features = []

    for sample in X:

        acc = sample[:, 0:3]

        gyro = sample[:, 3:6]

        acc_mag = np.linalg.norm(
            acc,
            axis=1
        )

        gyro_mag = np.linalg.norm(
            gyro,
            axis=1
        )

        acc_jerk = np.diff(
            acc_mag
        )

        gyro_change = np.diff(
            gyro_mag
        )

        f = []

        # ----------------------------------------------------
        # Raw axis statistics
        # ----------------------------------------------------

        for signal in [

            acc[:, 0],
            acc[:, 1],
            acc[:, 2],

            gyro[:, 0],
            gyro[:, 1],
            gyro[:, 2],

            acc_mag,
            gyro_mag,

            acc_jerk,
            gyro_change

        ]:

            f.extend([

                np.mean(signal),

                np.std(signal),

                np.min(signal),

                np.max(signal),

                np.ptp(signal),

                np.sqrt(
                    np.mean(
                        signal ** 2
                    )
                ),

                np.median(signal),

                np.percentile(
                    signal,
                    10
                ),

                np.percentile(
                    signal,
                    25
                ),

                np.percentile(
                    signal,
                    75
                ),

                np.percentile(
                    signal,
                    90
                )
            ])

        # ----------------------------------------------------
        # Peak / impact features
        # ----------------------------------------------------

        f.extend([

            np.max(acc_mag),

            np.max(gyro_mag),

            np.mean(acc_mag),

            np.std(acc_mag),

            np.mean(gyro_mag),

            np.std(gyro_mag),

            np.max(
                np.abs(acc_jerk)
            ),

            np.mean(
                np.abs(acc_jerk)
            ),

            np.std(
                acc_jerk
            ),

            np.max(
                np.abs(gyro_change)
            ),

            np.mean(
                np.abs(gyro_change)
            ),

            np.std(
                gyro_change
            )
        ])

        # ----------------------------------------------------
        # First half vs second half
        # ----------------------------------------------------

        mid = len(sample) // 2

        for signal in [

            acc_mag,
            gyro_mag

        ]:

            first = signal[:mid]

            second = signal[mid:]

            f.extend([

                np.mean(first),

                np.std(first),

                np.max(first),

                np.mean(second),

                np.std(second),

                np.max(second),

                np.mean(second)
                -
                np.mean(first),

                np.max(second)
                -
                np.max(first)
            ])

        # ----------------------------------------------------
        # Energy
        # ----------------------------------------------------

        f.extend([

            np.sum(
                acc_mag ** 2
            ),

            np.sum(
                gyro_mag ** 2
            ),

            np.mean(
                acc_mag ** 2
            ),

            np.mean(
                gyro_mag ** 2
            )
        ])

        # ----------------------------------------------------
        # High-motion fractions
        # ----------------------------------------------------

        acc_threshold = np.percentile(
            acc_mag,
            90
        )

        gyro_threshold = np.percentile(
            gyro_mag,
            90
        )

        f.extend([

            np.mean(
                acc_mag >
                acc_threshold
            ),

            np.mean(
                gyro_mag >
                gyro_threshold
            ),

            np.mean(
                acc_mag >
                10
            ),

            np.mean(
                acc_mag >
                15
            ),

            np.mean(
                gyro_mag >
                5
            ),

            np.mean(
                gyro_mag >
                10
            )
        ])

        features.append(f)

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_models():

    print("=" * 70)
    print("LOADING THERMORA MODELS")
    print("=" * 70)

    # --------------------------------------------------------
    # THERMAL
    # --------------------------------------------------------

    if not os.path.exists(
        THERMAL_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nThermal model not found:\n"
            f"{THERMAL_MODEL_PATH}"
        )

    thermal_model = joblib.load(
        THERMAL_MODEL_PATH
    )

    print("\nTHERMAL MODEL")
    print("-" * 40)

    print(
        "Path:",
        THERMAL_MODEL_PATH
    )

    print(
        "Type:",
        type(thermal_model)
    )

    # --------------------------------------------------------
    # FATIGUE
    # --------------------------------------------------------

    if not os.path.exists(
        FATIGUE_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nFatigue model not found:\n"
            f"{FATIGUE_MODEL_PATH}"
        )

    if not os.path.exists(
        FATIGUE_IMPUTER_PATH
    ):

        raise FileNotFoundError(
            f"\nFatigue imputer not found:\n"
            f"{FATIGUE_IMPUTER_PATH}"
        )

    fatigue_model = joblib.load(
        FATIGUE_MODEL_PATH
    )

    fatigue_imputer = joblib.load(
        FATIGUE_IMPUTER_PATH
    )

    print("\nFATIGUE MODEL")
    print("-" * 40)

    print(
        "Path:",
        FATIGUE_MODEL_PATH
    )

    print(
        "Type:",
        type(fatigue_model)
    )

    print(
        "Trees:",
        getattr(
            fatigue_model,
            "n_estimators",
            "N/A"
        )
    )

    print(
        "Imputer:",
        type(fatigue_imputer)
    )

    # --------------------------------------------------------
    # FATIGUE FEATURE SCHEMA
    # --------------------------------------------------------

    fatigue_features = None

    if os.path.exists(
        FATIGUE_FEATURES_PATH
    ):

        with open(
            FATIGUE_FEATURES_PATH,
            "r"
        ) as f:

            fatigue_features = json.load(f)

        print(
            "\nFATIGUE FEATURE SCHEMA"
        )

        print(
            "Features:",
            len(fatigue_features)
        )

    # --------------------------------------------------------
    # FALL
    # --------------------------------------------------------

    if not os.path.exists(
        FALL_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nFall model not found:\n"
            f"{FALL_MODEL_PATH}"
        )

    fall_artifact = joblib.load(
        FALL_MODEL_PATH
    )

    print("\nFALL MODEL")
    print("-" * 40)

    print(
        "Path:",
        FALL_MODEL_PATH
    )

    print(
        "Artifact type:",
        type(fall_artifact)
    )

    # The v2 model was saved as a dictionary.
    if isinstance(
        fall_artifact,
        dict
    ):

        fall_model = fall_artifact[
            "model"
        ]

        fall_threshold = fall_artifact[
            "threshold"
        ]

        fall_feature_count = (
            fall_artifact[
                "feature_count"
            ]
        )

    else:

        raise ValueError(
            "Unexpected fall model format. "
            "Expected dictionary containing "
            "'model', 'threshold' and "
            "'feature_count'."
        )

    print(
        "Model:",
        type(fall_model)
    )

    print(
        "Threshold:",
        fall_threshold
    )

    print(
        "Expected features:",
        fall_feature_count
    )

    return {

        "thermal":
            thermal_model,

        "fatigue":
            fatigue_model,

        "fatigue_imputer":
            fatigue_imputer,

        "fatigue_features":
            fatigue_features,

        "fall":
            fall_model,

        "fall_threshold":
            fall_threshold,

        "fall_feature_count":
            fall_feature_count
    }


# ============================================================
# THERMAL OUTPUT
# ============================================================

def build_thermal_outputs(
    thermal_df
):

    current_column = (
        "estimated_core_temp_c"
    )

    predicted_column = (
        "predicted_future_core_temp_c"
    )

    if current_column not in thermal_df.columns:

        raise ValueError(
            f"Missing thermal column: "
            f"{current_column}"
        )

    if predicted_column not in thermal_df.columns:

        raise ValueError(
            f"Missing thermal column: "
            f"{predicted_column}"
        )

    print(
        "\nTHERMAL COLUMN MAPPING"
    )

    print(
        "Current core:",
        current_column
    )

    print(
        "Future core:",
        predicted_column
    )

    thermal_outputs = pd.DataFrame()

    for column in [

        "participant_id",
        "condition_id",
        "time_min"

    ]:

        if column in thermal_df.columns:

            thermal_outputs[
                column
            ] = thermal_df[
                column
            ]

    thermal_outputs[
        "current_core_temp_c"
    ] = thermal_df[
        current_column
    ]

    thermal_outputs[
        "predicted_core_temp_10min_c"
    ] = thermal_df[
        predicted_column
    ]

    if (
        "predicted_rise_c"
        in thermal_df.columns
    ):

        thermal_outputs[
            "predicted_rise_c"
        ] = thermal_df[
            "predicted_rise_c"
        ]

    else:

        thermal_outputs[
            "predicted_rise_c"
        ] = (

            thermal_outputs[
                "predicted_core_temp_10min_c"
            ]

            -

            thermal_outputs[
                "current_core_temp_c"
            ]
        )

    return thermal_outputs


# ============================================================
# FATIGUE OUTPUT
# ============================================================

def build_fatigue_outputs(
    fatigue_df
):

    fatigue_column = "fatigue_score"

    if fatigue_column not in fatigue_df.columns:

        raise ValueError(
            "fatigue_score column not found."
        )

    print(
        "\nFATIGUE COLUMN MAPPING"
    )

    print(
        "Fatigue score:",
        fatigue_column
    )

    fatigue_outputs = pd.DataFrame()

    for column in [

        "participant_id",
        "condition_id",
        "session_id",
        "file",
        "window_start_sample",
        "window_end_sample",
        "window_start_time",
        "window_end_time"

    ]:

        if column in fatigue_df.columns:

            fatigue_outputs[
                column
            ] = fatigue_df[
                column
            ]

    fatigue_outputs[
        "fatigue_score"
    ] = fatigue_df[
        fatigue_column
    ]

    # --------------------------------------------------------
    # Fatigue trend per participant
    # --------------------------------------------------------

    fatigue_outputs[
        "fatigue_trend"
    ] = "unknown"

    if "participant_id" in fatigue_outputs.columns:

        for participant in (
            fatigue_outputs[
                "participant_id"
            ].unique()
        ):

            mask = (
                fatigue_outputs[
                    "participant_id"
                ]
                == participant
            )

            indices = (
                fatigue_outputs[
                    mask
                ].index
            )

            previous = None

            for index in indices:

                current = (
                    fatigue_outputs.loc[
                        index,
                        "fatigue_score"
                    ]
                )

                if pd.isna(current):

                    trend = "unknown"

                elif previous is None:

                    trend = "unknown"

                elif (
                    current - previous
                    > 0.30
                ):

                    trend = "increasing"

                elif (
                    current - previous
                    < -0.30
                ):

                    trend = "decreasing"

                else:

                    trend = "stable"

                fatigue_outputs.loc[
                    index,
                    "fatigue_trend"
                ] = trend

                if pd.notna(current):

                    previous = current

    return fatigue_outputs


# ============================================================
# FALL OUTPUT
# ============================================================

def build_fall_outputs(
    fall_model,
    fall_threshold,
    expected_feature_count
):

    print("\n")
    print("=" * 70)
    print("LOADING FALL TEST WINDOWS")
    print("=" * 70)

    if not os.path.exists(
        FALL_DATA_PATH
    ):

        raise FileNotFoundError(
            f"\nFall windows not found:\n"
            f"{FALL_DATA_PATH}"
        )

    if not os.path.exists(
        FALL_META_PATH
    ):

        raise FileNotFoundError(
            f"\nFall metadata not found:\n"
            f"{FALL_META_PATH}"
        )

    # --------------------------------------------------------
    # Load raw windows
    # --------------------------------------------------------

    data = np.load(
        FALL_DATA_PATH
    )

    X = data["X"]

    print(
        "Raw fall windows:",
        X.shape
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        FALL_META_PATH
    )

    print(
        "Metadata rows:",
        len(metadata)
    )

    if len(X) != len(metadata):

        raise ValueError(
            "Fall window count does not match "
            "metadata row count."
        )

    # --------------------------------------------------------
    # Extract EXACT 148 features
    # --------------------------------------------------------

    print(
        "\nExtracting EXACT v2 fall features..."
    )

    X_features = extract_fall_features(
        X
    )

    print(
        "Generated features:",
        X_features.shape
    )

    print(
        "Expected features:",
        expected_feature_count
    )

    if (
        X_features.shape[1]
        != expected_feature_count
    ):

        raise ValueError(
            "\nFALL FEATURE MISMATCH!\n"
            f"Generated: "
            f"{X_features.shape[1]}\n"
            f"Expected: "
            f"{expected_feature_count}"
        )

    print(
        "FALL FEATURE COUNT: EXACT MATCH"
    )

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    print(
        "\nRunning fall inference..."
    )

    probabilities = (
        fall_model.predict_proba(
            X_features
        )[:, 1]
    )

    predictions = (
        probabilities
        >= fall_threshold
    )

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    fall_outputs = pd.DataFrame()

    # Copy useful metadata
    for column in [

        "participant_id",
        "file",
        "split",
        "label"

    ]:

        if column in metadata.columns:

            fall_outputs[
                column
            ] = metadata[
                column
            ]

    fall_outputs[
        "fall_probability"
    ] = probabilities

    fall_outputs[
        "fall_detected"
    ] = predictions

    return fall_outputs


# ============================================================
# THERMAL SUMMARY
# ============================================================

def display_thermal_summary(
    thermal_outputs
):

    print("\n")
    print("=" * 70)
    print("THERMAL OUTPUT")
    print("=" * 70)

    print(
        "Rows:",
        len(thermal_outputs)
    )

    print(
        "\nMean estimated current core:",
        f"{thermal_outputs['current_core_temp_c'].mean():.3f} °C"
    )

    print(
        "Mean predicted core +10 min:",
        f"{thermal_outputs['predicted_core_temp_10min_c'].mean():.3f} °C"
    )

    print(
        "Minimum predicted core +10 min:",
        f"{thermal_outputs['predicted_core_temp_10min_c'].min():.3f} °C"
    )

    print(
        "Maximum predicted core +10 min:",
        f"{thermal_outputs['predicted_core_temp_10min_c'].max():.3f} °C"
    )

    print(
        "Maximum predicted rise:",
        f"{thermal_outputs['predicted_rise_c'].max():+.3f} °C"
    )


# ============================================================
# FATIGUE SUMMARY
# ============================================================

def display_fatigue_summary(
    fatigue_outputs
):

    print("\n")
    print("=" * 70)
    print("FATIGUE OUTPUT")
    print("=" * 70)

    print(
        "Rows:",
        len(fatigue_outputs)
    )

    print(
        "\nMean fatigue:",
        f"{fatigue_outputs['fatigue_score'].mean():.3f} / 10"
    )

    print(
        "Minimum fatigue:",
        f"{fatigue_outputs['fatigue_score'].min():.3f} / 10"
    )

    print(
        "Maximum fatigue:",
        f"{fatigue_outputs['fatigue_score'].max():.3f} / 10"
    )

    print(
        "\nFatigue trend counts:"
    )

    print(
        fatigue_outputs[
            "fatigue_trend"
        ].value_counts()
    )


# ============================================================
# FALL SUMMARY
# ============================================================

def display_fall_summary(
    fall_outputs
):

    print("\n")
    print("=" * 70)
    print("FALL OUTPUT")
    print("=" * 70)

    print(
        "Rows:",
        len(fall_outputs)
    )

    print(
        "\nMean fall probability:",
        f"{fall_outputs['fall_probability'].mean():.4f}"
    )

    print(
        "Maximum fall probability:",
        f"{fall_outputs['fall_probability'].max():.4f}"
    )

    print(
        "Detected fall windows:",
        int(
            fall_outputs[
                "fall_detected"
            ].sum()
        )
    )

    print(
        "Total windows:",
        len(fall_outputs)
    )

    print(
        "\nFall detection rate:",
        f"{fall_outputs['fall_detected'].mean() * 100:.2f}%"
    )


# ============================================================
# SAMPLE THERMAL OUTPUT
# ============================================================

def display_thermal_samples(
    thermal_outputs
):

    print("\n")
    print("=" * 70)
    print("THERMAL SAMPLES")
    print("=" * 70)

    for _, row in (
        thermal_outputs
        .head(5)
        .iterrows()
    ):

        print(
            f"\nWorker: "
            f"{row.get('participant_id', 'N/A')} "
            f"| Time: "
            f"{row.get('time_min', 'N/A')} min"
        )

        print(
            f"  Current estimated core: "
            f"{row['current_core_temp_c']:.3f} °C"
        )

        print(
            f"  Predicted core +10 min: "
            f"{row['predicted_core_temp_10min_c']:.3f} °C"
        )

        print(
            f"  Predicted rise: "
            f"{row['predicted_rise_c']:+.3f} °C"
        )


# ============================================================
# SAMPLE FATIGUE OUTPUT
# ============================================================

def display_fatigue_samples(
    fatigue_outputs
):

    print("\n")
    print("=" * 70)
    print("FATIGUE SAMPLES")
    print("=" * 70)

    for _, row in (
        fatigue_outputs
        .head(5)
        .iterrows()
    ):

        print(
            f"\nWorker: "
            f"{row.get('participant_id', 'N/A')}"
        )

        print(
            f"  File: "
            f"{row.get('file', 'N/A')}"
        )

        print(
            f"  Fatigue score: "
            f"{row['fatigue_score']:.2f} / 10"
        )

        print(
            f"  Trend: "
            f"{row['fatigue_trend']}"
        )


# ============================================================
# SAMPLE FALL OUTPUT
# ============================================================

def display_fall_samples(
    fall_outputs
):

    print("\n")
    print("=" * 70)
    print("FALL SAMPLES")
    print("=" * 70)

    for _, row in (
        fall_outputs
        .head(10)
        .iterrows()
    ):

        print(
            f"\nWorker: "
            f"{row.get('participant_id', 'N/A')}"
        )

        print(
            f"  Fall probability: "
            f"{row['fall_probability']:.4f}"
        )

        print(
            f"  Fall detected: "
            f"{'YES' if row['fall_detected'] else 'NO'}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("THERMORA MULTIMODAL INFERENCE")
    print("=" * 70)

    # ========================================================
    # LOAD MODELS
    # ========================================================

    models = load_models()

    # ========================================================
    # LOAD THERMAL RESULTS
    # ========================================================

    if not os.path.exists(
        THERMAL_DATA_PATH
    ):

        raise FileNotFoundError(
            f"\nThermal results not found:\n"
            f"{THERMAL_DATA_PATH}"
        )

    thermal_df = pd.read_csv(
        THERMAL_DATA_PATH
    )

    print("\n")
    print("=" * 70)
    print("LOADING THERMAL RESULTS")
    print("=" * 70)

    print(
        "Rows:",
        len(thermal_df)
    )

    print(
        "Columns:",
        len(thermal_df.columns)
    )

    # ========================================================
    # LOAD FATIGUE RESULTS
    # ========================================================

    if not os.path.exists(
        FATIGUE_DATA_PATH
    ):

        raise FileNotFoundError(
            f"\nFatigue predictions not found:\n"
            f"{FATIGUE_DATA_PATH}"
        )

    fatigue_df = pd.read_csv(
        FATIGUE_DATA_PATH
    )

    print("\n")
    print("=" * 70)
    print("LOADING FATIGUE RESULTS")
    print("=" * 70)

    print(
        "Rows:",
        len(fatigue_df)
    )

    print(
        "Columns:",
        len(fatigue_df.columns)
    )

    # ========================================================
    # BUILD INDEPENDENT OUTPUTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PROCESSING THREE INDEPENDENT MODELS")
    print("=" * 70)

    # --------------------------------------------------------
    # Thermal
    # --------------------------------------------------------

    thermal_outputs = (
        build_thermal_outputs(
            thermal_df
        )
    )

    # --------------------------------------------------------
    # Fatigue
    # --------------------------------------------------------

    fatigue_outputs = (
        build_fatigue_outputs(
            fatigue_df
        )
    )

    # --------------------------------------------------------
    # Fall
    # --------------------------------------------------------

    fall_outputs = (
        build_fall_outputs(

            models["fall"],

            models["fall_threshold"],

            models["fall_feature_count"]
        )
    )

    # ========================================================
    # DISPLAY SUMMARIES
    # ========================================================

    display_thermal_summary(
        thermal_outputs
    )

    display_fatigue_summary(
        fatigue_outputs
    )

    display_fall_summary(
        fall_outputs
    )

    # ========================================================
    # DISPLAY SAMPLES
    # ========================================================

    display_thermal_samples(
        thermal_outputs
    )

    display_fatigue_samples(
        fatigue_outputs
    )

    display_fall_samples(
        fall_outputs
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    thermal_outputs.to_csv(
        THERMAL_OUTPUT_PATH,
        index=False
    )

    fatigue_outputs.to_csv(
        FATIGUE_OUTPUT_PATH,
        index=False
    )

    fall_outputs.to_csv(
        FALL_OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print("\n")
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        "\nTHERMAL:"
    )

    print(
        THERMAL_OUTPUT_PATH
    )

    print(
        "\nFATIGUE:"
    )

    print(
        FATIGUE_OUTPUT_PATH
    )

    print(
        "\nFALL:"
    )

    print(
        FALL_OUTPUT_PATH
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("THERMORA MULTIMODAL INFERENCE COMPLETED")
    print("=" * 70)

    print(
        "\nSTATUS:"
    )

    print(
        "✓ Thermal model connected"
    )

    print(
        "✓ Fatigue model connected"
    )

    print(
        "✓ Fall v2 model connected"
    )

    print(
        "✓ Exact 148-feature fall extractor used"
    )

    print(
        "✓ Thermal output independent"
    )

    print(
        "✓ Fatigue output independent"
    )

    print(
        "✓ Fall output independent"
    )

    print(
        "✓ No unified risk classification"
    )

    print(
        "✓ No artificial thermal/fatigue row matching"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()