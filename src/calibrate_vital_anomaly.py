import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

DATA_FILE = r"vital_anomaly\processed\vital_windows.csv"

MODEL_FILE = (
    r"vital_anomaly\models"
    r"\thermora_vital_anomaly_model.pkl"
)

FEATURE_FILE = (
    r"vital_anomaly\models"
    r"\thermora_vital_anomaly_features.json"
)

RESULT_DIR = r"vital_anomaly\results"

CALIBRATION_FILE = os.path.join(
    RESULT_DIR,
    "vital_anomaly_threshold_calibration.csv"
)

SELECTED_FILE = os.path.join(
    RESULT_DIR,
    "vital_anomaly_selected_threshold.json"
)

PREDICTION_FILE = os.path.join(
    RESULT_DIR,
    "vital_anomaly_calibrated_test_predictions.csv"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_RECORDING_FRACTION = 0.20

WINDOW_SECONDS = 30

# Test a wide range of operating thresholds.
THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.025
)


# ============================================================
# PROXY ANOMALY LABELS
# ============================================================

def make_proxy_anomaly_labels(df):
    """
    Create conservative physiological stress-test labels.

    IMPORTANT:
    These are NOT clinical ground-truth labels.

    They are only used to test whether the unsupervised
    anomaly detector responds to clearly unusual vital
    observations in BIDMC.
    """

    labels = np.zeros(
        len(df),
        dtype=int
    )

    # Low oxygen saturation
    labels |= (
        df["spo2_min"] < 90
    ).astype(int)

    labels |= (
        df["spo2_mean"] < 90
    ).astype(int)

    # Very high heart rate
    labels |= (
        df["hr_max"] >= 120
    ).astype(int)

    # Very low heart rate
    labels |= (
        df["hr_min"] <= 50
    ).astype(int)

    return labels


# ============================================================
# REPRODUCE TRAIN / TEST RECORDING SPLIT
# ============================================================

def split_recordings(df):

    recordings = sorted(
        df["recording_id"].unique()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    shuffled = recordings.copy()

    rng.shuffle(
        shuffled
    )

    n_test = max(
        1,
        int(
            round(
                len(recordings)
                * TEST_RECORDING_FRACTION
            )
        )
    )

    test_recordings = sorted(
        shuffled[:n_test]
    )

    train_recordings = sorted(
        shuffled[n_test:]
    )

    train_df = df[
        df["recording_id"].isin(
            train_recordings
        )
    ].copy()

    test_df = df[
        df["recording_id"].isin(
            test_recordings
        )
    ].copy()

    return (
        train_df,
        test_df,
        train_recordings,
        test_recordings,
    )


# ============================================================
# NORMALIZE ANOMALY SCORES
# ============================================================

def normalize_scores(
    train_scores,
    test_scores
):
    """
    Normalize scores using ONLY the training-score
    distribution.

    This prevents the test set from determining
    its own normalization.
    """

    train_scores = np.asarray(
        train_scores,
        dtype=float
    )

    test_scores = np.asarray(
        test_scores,
        dtype=float
    )

    low = np.percentile(
        train_scores,
        5
    )

    high = np.percentile(
        train_scores,
        95
    )

    if high <= low:

        return (
            np.zeros_like(test_scores),
            float(low),
            float(high)
        )

    normalized = (
        (test_scores - low)
        / (high - low)
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    return (
        normalized,
        float(low),
        float(high)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("THERMORA VITAL ANOMALY THRESHOLD CALIBRATION")
    print("=" * 70)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print()
    print("Dataset")
    print(
        f"Rows:       {len(df):,}"
    )

    print(
        f"Columns:    {len(df.columns)}"
    )

    print(
        f"Recordings: "
        f"{df['recording_id'].nunique()}"
    )

    # ========================================================
    # REPRODUCE SAME SPLIT
    # ========================================================

    (
        train_df,
        test_df,
        train_recordings,
        test_recordings,
    ) = split_recordings(
        df
    )

    print()
    print("RECORDING-LEVEL SPLIT")
    print("---------------------")

    print(
        f"Training recordings: "
        f"{len(train_recordings)}"
    )

    print(
        f"Testing recordings:  "
        f"{len(test_recordings)}"
    )

    print(
        f"Training windows:    "
        f"{len(train_df):,}"
    )

    print(
        f"Testing windows:     "
        f"{len(test_df):,}"
    )

    print()
    print("Test recordings:")
    print(
        ", ".join(test_recordings)
    )

    # ========================================================
    # PROXY LABELS
    # ========================================================

    train_df["proxy_anomaly"] = (
        make_proxy_anomaly_labels(
            train_df
        )
    )

    test_df["proxy_anomaly"] = (
        make_proxy_anomaly_labels(
            test_df
        )
    )

    print()
    print("PROXY LABELS")
    print("------------")

    print(
        f"Training proxy abnormal: "
        f"{train_df['proxy_anomaly'].sum():,}"
    )

    print(
        f"Testing proxy abnormal:  "
        f"{test_df['proxy_anomaly'].sum():,}"
    )

    print()
    print(
        "WARNING: Proxy labels are NOT "
        "clinical ground truth."
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    print()
    print("Loading trained model:")
    print(MODEL_FILE)

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model type: {type(model)}"
    )

    # ========================================================
    # LOAD FEATURE SCHEMA
    # ========================================================

    if not os.path.exists(FEATURE_FILE):

        raise FileNotFoundError(
            f"Feature schema not found:\n"
            f"{FEATURE_FILE}"
        )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        feature_info = json.load(
            f
        )

    FEATURES = feature_info[
        "features"
    ]

    print(
        f"Features: {len(FEATURES)}"
    )

    # ========================================================
    # PREPARE TRAINING DATA
    # ========================================================

    X_train = train_df[
        FEATURES
    ].copy()

    X_test = test_df[
        FEATURES
    ].copy()

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # Use training medians only.
    # --------------------------------------------------------

    train_medians = X_train.median()

    X_train = X_train.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    print()
    print(
        f"Training matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Testing matrix:  "
        f"{X_test.shape}"
    )

    # ========================================================
    # GET TRAINING SCORES
    # ========================================================

    if not hasattr(
        model,
        "decision_function"
    ):

        raise RuntimeError(
            "Selected model does not provide "
            "decision_function()."
        )

    train_raw_scores = (
        -model.decision_function(
            X_train
        )
    )

    test_raw_scores = (
        -model.decision_function(
            X_test
        )
    )

    # Normalize test scores using training distribution
    test_scores, score_low, score_high = (
        normalize_scores(
            train_raw_scores,
            test_raw_scores
        )
    )

    print()
    print("ANOMALY SCORE CALIBRATION")
    print("-------------------------")

    print(
        f"Training raw score min: "
        f"{train_raw_scores.min():.6f}"
    )

    print(
        f"Training raw score max: "
        f"{train_raw_scores.max():.6f}"
    )

    print(
        f"Normalization low:      "
        f"{score_low:.6f}"
    )

    print(
        f"Normalization high:     "
        f"{score_high:.6f}"
    )

    print()
    print(
        f"Test score min: "
        f"{test_scores.min():.4f}"
    )

    print(
        f"Test score max: "
        f"{test_scores.max():.4f}"
    )

    print(
        f"Test score mean: "
        f"{test_scores.mean():.4f}"
    )

    # ========================================================
    # THRESHOLD SWEEP
    # ========================================================

    y_true = (
        test_df[
            "proxy_anomaly"
        ].to_numpy()
    )

    results = []

    print()
    print("=" * 70)
    print("THRESHOLD CALIBRATION")
    print("=" * 70)

    for threshold in THRESHOLDS:

        y_pred = (
            test_scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        normal_mask = (
            y_true == 0
        )

        if normal_mask.sum() > 0:

            false_positive_rate = float(
                np.mean(
                    y_pred[
                        normal_mask
                    ] == 1
                )
            )

        else:

            false_positive_rate = np.nan

        anomaly_rate = float(
            np.mean(
                y_pred
            )
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        results.append({

            "threshold": float(
                threshold
            ),

            "anomaly_rate": anomaly_rate,

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "false_positive_rate": (
                false_positive_rate
            ),

            "true_negative": int(tn),

            "false_positive": int(fp),

            "false_negative": int(fn),

            "true_positive": int(tp),
        })

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # SAVE CALIBRATION TABLE
    # ========================================================

    results_df.to_csv(
        CALIBRATION_FILE,
        index=False
    )

    # ========================================================
    # SELECT OPERATING POINT
    # ========================================================

    # Preferred operating region:
    #
    #   False-positive rate <= 5%
    #
    # Within that region, maximize:
    #
    #   recall → precision → F1
    #
    # If no threshold satisfies <=5% FPR,
    # choose the lowest-FPR threshold while
    # retaining some anomaly recall.

    acceptable = results_df[
        results_df[
            "false_positive_rate"
        ] <= 0.05
    ].copy()

    if len(acceptable) > 0:

        selected = acceptable.sort_values(
            [
                "recall",
                "precision",
                "f1",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        ).iloc[0]

        selection_reason = (
            "Highest proxy recall among thresholds "
            "with proxy false-positive rate <= 5%."
        )

    else:

        candidates = results_df[
            results_df["recall"] > 0
        ].copy()

        if len(candidates) == 0:

            candidates = results_df.copy()

        selected = candidates.sort_values(
            [
                "false_positive_rate",
                "recall",
            ],
            ascending=[
                True,
                False,
            ]
        ).iloc[0]

        selection_reason = (
            "No threshold achieved <=5% proxy "
            "false-positive rate; selected the "
            "lowest-FPR operating point with "
            "non-zero recall."
        )

    selected_threshold = float(
        selected["threshold"]
    )

    # ========================================================
    # SELECTED OPERATING POINT
    # ========================================================

    print()
    print("=" * 70)
    print("SELECTED OPERATING POINT")
    print("=" * 70)

    print(
        f"Threshold:             "
        f"{selected_threshold:.4f}"
    )

    print(
        f"Anomaly rate:          "
        f"{selected['anomaly_rate'] * 100:.2f}%"
    )

    print(
        f"Proxy recall:          "
        f"{selected['recall']:.4f}"
    )

    print(
        f"Proxy precision:       "
        f"{selected['precision']:.4f}"
    )

    print(
        f"Proxy F1:              "
        f"{selected['f1']:.4f}"
    )

    print(
        f"Proxy false-positive:  "
        f"{selected['false_positive_rate']:.4f}"
    )

    print()
    print("Selection reason:")
    print(selection_reason)

    # ========================================================
    # CALIBRATED TEST PREDICTIONS
    # ========================================================

    calibrated_prediction = (
        test_scores >= selected_threshold
    ).astype(int)

    calibrated_output = test_df[
        [
            "recording_id",
            "window_start_s",
            "window_end_s",
            "hr_mean",
            "hr_min",
            "hr_max",
            "spo2_mean",
            "spo2_min",
            "spo2_max",
            "proxy_anomaly",
        ]
    ].copy()

    calibrated_output[
        "anomaly_score"
    ] = test_scores

    calibrated_output[
        "anomaly_prediction"
    ] = calibrated_prediction

    calibrated_output.to_csv(
        PREDICTION_FILE,
        index=False
    )

    # ========================================================
    # SAVE SELECTED THRESHOLD
    # ========================================================

    threshold_info = {

        "model": "IsolationForest",

        "selected_threshold": (
            selected_threshold
        ),

        "score_definition": (
            "Normalized negative Isolation "
            "Forest decision function."
        ),

        "window_seconds": WINDOW_SECONDS,

        "feature_count": len(
            FEATURES
        ),

        "training_recordings": (
            train_recordings
        ),

        "test_recordings": (
            test_recordings
        ),

        "training_windows": int(
            len(train_df)
        ),

        "test_windows": int(
            len(test_df)
        ),

        "proxy_anomaly_windows_test": int(
            y_true.sum()
        ),

        "proxy_recall": float(
            selected["recall"]
        ),

        "proxy_precision": float(
            selected["precision"]
        ),

        "proxy_f1": float(
            selected["f1"]
        ),

        "proxy_false_positive_rate": float(
            selected[
                "false_positive_rate"
            ]
        ),

        "selection_reason": (
            selection_reason
        ),

        "warning": (
            "BIDMC does not provide ground-truth "
            "vital anomaly labels. Proxy labels are "
            "only a stress-test and must not be "
            "reported as clinical diagnostic accuracy."
        ),
    }

    with open(
        SELECTED_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            threshold_info,
            f,
            indent=2
        )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 70)
    print("CALIBRATION FILES")
    print("=" * 70)

    print(
        CALIBRATION_FILE
    )

    print(
        PREDICTION_FILE
    )

    print(
        SELECTED_FILE
    )

    print()
    print("=" * 70)
    print("VITAL ANOMALY CALIBRATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()