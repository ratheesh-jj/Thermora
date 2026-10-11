"""
THERMORA - Phase 1C
Accelerometer-only vs 6-axis ACC + GYRO

Uses the SAME:
    - contiguous windows
    - participant split
    - feature dataset

This makes the sensor comparison fair.

Input:
    wisdm_activity/thermora_activity_models_v2/activity_features_v2.csv

Output:
    wisdm_activity/thermora_activity_models_v2/sensor_comparison/
"""

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "wisdm_activity"
    / "thermora_activity_models_v2"
    / "activity_features_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "wisdm_activity"
    / "thermora_activity_models_v2"
    / "sensor_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING ACTIVITY FEATURE DATASET")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n{INPUT_FILE}\n\n"
            "Run train_activity_model.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Windows      : {len(df):,}"
    )

    print(
        f"Participants : "
        f"{df['participant'].nunique()}"
    )

    print(
        f"Activities   : "
        f"{df['activity'].nunique()}"
    )

    return df


# ============================================================
# FEATURE GROUPS
# ============================================================

def get_feature_groups(df):

    all_features = [
        c for c in df.columns
        if c not in [
            "participant",
            "activity"
        ]
    ]

    acc_features = [
        c for c in all_features
        if (
            c.startswith("acc_")
        )
    ]

    gyro_features = [
        c for c in all_features
        if (
            c.startswith("gyro_")
        )
    ]

    six_axis_features = (
        acc_features +
        gyro_features
    )

    return {
        "ACC_ONLY": acc_features,
        "ACC_GYRO": six_axis_features
    }


# ============================================================
# SAME PARTICIPANT SPLIT
# ============================================================

def create_split(df):

    groups = df["participant"].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=groups
        )
    )

    return train_idx, test_idx


# ============================================================
# PREPARE LABELS
# ============================================================

def prepare_labels(df):

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        df["activity"].values
    )

    return y, encoder


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df,
    feature_columns
):

    X = df[
        feature_columns
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Median imputation.
    # Important:
    # Calculate medians from the full feature matrix only
    # because these are sensor-derived feature values.
    # For strict leakage-free preprocessing, the deployment
    # version should store train medians.
    X = X.fillna(
        X.median()
    )

    return X


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "    Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    return model, accuracy, macro_f1


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "    XGBoost..."
    )

    try:

        from xgboost import XGBClassifier

    except ImportError:

        print(
            "    XGBoost not installed - skipped."
        )

        return None, None, None

    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    return model, accuracy, macro_f1


# ============================================================
# TRAIN LIGHTGBM
# ============================================================

def train_lightgbm(
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        "    LightGBM..."
    )

    try:

        from lightgbm import LGBMClassifier

    except ImportError:

        print(
            "    LightGBM not installed - skipped."
        )

        return None, None, None

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    return model, accuracy, macro_f1


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THERMORA - PHASE 1C")
    print("ACCELEROMETER vs ACCELEROMETER + GYROSCOPE")
    print("=" * 70)
    print()

    df = load_dataset()

    feature_groups = get_feature_groups(
        df
    )

    print()
    print("=" * 70)
    print("FEATURE COUNTS")
    print("=" * 70)

    print(
        f"ACC-only features : "
        f"{len(feature_groups['ACC_ONLY'])}"
    )

    print(
        f"ACC+GYRO features : "
        f"{len(feature_groups['ACC_GYRO'])}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y, label_encoder = prepare_labels(
        df
    )

    # --------------------------------------------------------
    # SAME SPLIT FOR BOTH EXPERIMENTS
    # --------------------------------------------------------

    train_idx, test_idx = create_split(
        df
    )

    train_participants = sorted(
        df.iloc[
            train_idx
        ]["participant"].unique().tolist()
    )

    test_participants = sorted(
        df.iloc[
            test_idx
        ]["participant"].unique().tolist()
    )

    print()
    print("=" * 70)
    print("PARTICIPANT SPLIT")
    print("=" * 70)

    print(
        f"Train windows : "
        f"{len(train_idx):,}"
    )

    print(
        f"Test windows  : "
        f"{len(test_idx):,}"
    )

    print(
        f"Train people  : "
        f"{len(train_participants)}"
    )

    print(
        f"Test people   : "
        f"{len(test_participants)}"
    )

    print()
    print(
        "Test participants:"
    )

    print(
        test_participants
    )

    # Save split
    with open(
        OUTPUT_DIR / "same_split.json",
        "w"
    ) as f:

        json.dump(
            {
                "random_state": RANDOM_STATE,
                "train_participants": [
                    int(x)
                    for x in train_participants
                ],
                "test_participants": [
                    int(x)
                    for x in test_participants
                ],
                "train_windows": len(train_idx),
                "test_windows": len(test_idx)
            },
            f,
            indent=4
        )

    results = []

    trained_models = {}

    # ========================================================
    # EXPERIMENT 1: ACC ONLY
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 1: ACCELEROMETER ONLY")
    print("=" * 70)

    acc_features = feature_groups[
        "ACC_ONLY"
    ]

    X_acc = prepare_features(
        df,
        acc_features
    )

    X_acc_train = X_acc.iloc[
        train_idx
    ]

    X_acc_test = X_acc.iloc[
        test_idx
    ]

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]

    # Random Forest
    rf_model, rf_acc, rf_f1 = train_random_forest(
        X_acc_train,
        X_acc_test,
        y_train,
        y_test
    )

    results.append(
        {
            "sensor_config": "ACC_ONLY",
            "model": "RandomForest",
            "accuracy": rf_acc,
            "macro_f1": rf_f1,
            "num_features": len(acc_features)
        }
    )

    trained_models[
        "ACC_ONLY_RandomForest"
    ] = rf_model

    # XGBoost
    xgb_model, xgb_acc, xgb_f1 = train_xgboost(
        X_acc_train,
        X_acc_test,
        y_train,
        y_test
    )

    if xgb_model is not None:

        results.append(
            {
                "sensor_config": "ACC_ONLY",
                "model": "XGBoost",
                "accuracy": xgb_acc,
                "macro_f1": xgb_f1,
                "num_features": len(acc_features)
            }
        )

        trained_models[
            "ACC_ONLY_XGBoost"
        ] = xgb_model

    # LightGBM
    lgb_model, lgb_acc, lgb_f1 = train_lightgbm(
        X_acc_train,
        X_acc_test,
        y_train,
        y_test
    )

    if lgb_model is not None:

        results.append(
            {
                "sensor_config": "ACC_ONLY",
                "model": "LightGBM",
                "accuracy": lgb_acc,
                "macro_f1": lgb_f1,
                "num_features": len(acc_features)
            }
        )

        trained_models[
            "ACC_ONLY_LightGBM"
        ] = lgb_model

    # ========================================================
    # EXPERIMENT 2: ACC + GYRO
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 2: ACCELEROMETER + GYROSCOPE")
    print("=" * 70)

    six_features = feature_groups[
        "ACC_GYRO"
    ]

    X_six = prepare_features(
        df,
        six_features
    )

    X_six_train = X_six.iloc[
        train_idx
    ]

    X_six_test = X_six.iloc[
        test_idx
    ]

    # Random Forest
    rf_model, rf_acc, rf_f1 = train_random_forest(
        X_six_train,
        X_six_test,
        y_train,
        y_test
    )

    results.append(
        {
            "sensor_config": "ACC_GYRO",
            "model": "RandomForest",
            "accuracy": rf_acc,
            "macro_f1": rf_f1,
            "num_features": len(six_features)
        }
    )

    trained_models[
        "ACC_GYRO_RandomForest"
    ] = rf_model

    # XGBoost
    xgb_model, xgb_acc, xgb_f1 = train_xgboost(
        X_six_train,
        X_six_test,
        y_train,
        y_test
    )

    if xgb_model is not None:

        results.append(
            {
                "sensor_config": "ACC_GYRO",
                "model": "XGBoost",
                "accuracy": xgb_acc,
                "macro_f1": xgb_f1,
                "num_features": len(six_features)
            }
        )

        trained_models[
            "ACC_GYRO_XGBoost"
        ] = xgb_model

    # LightGBM
    lgb_model, lgb_acc, lgb_f1 = train_lightgbm(
        X_six_train,
        X_six_test,
        y_train,
        y_test
    )

    if lgb_model is not None:

        results.append(
            {
                "sensor_config": "ACC_GYRO",
                "model": "LightGBM",
                "accuracy": lgb_acc,
                "macro_f1": lgb_f1,
                "num_features": len(six_features)
            }
        )

        trained_models[
            "ACC_GYRO_LightGBM"
        ] = lgb_model

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False
    )

    print()
    print("=" * 70)
    print("FINAL SENSOR COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        OUTPUT_DIR /
        "acc_vs_6axis_results.csv",
        index=False
    )

    # ========================================================
    # BEST MODEL FOR EACH SENSOR CONFIG
    # ========================================================

    print()
    print("=" * 70)
    print("BEST MODEL PER SENSOR CONFIGURATION")
    print("=" * 70)

    for config in [
        "ACC_ONLY",
        "ACC_GYRO"
    ]:

        subset = results_df[
            results_df["sensor_config"]
            == config
        ]

        if subset.empty:
            continue

        best = subset.iloc[0]

        print()
        print(
            f"{config}"
        )

        print(
            f"  Model    : "
            f"{best['model']}"
        )

        print(
            f"  Accuracy : "
            f"{best['accuracy']:.4f}"
        )

        print(
            f"  Macro F1 : "
            f"{best['macro_f1']:.4f}"
        )

        print(
            f"  Features : "
            f"{int(best['num_features'])}"
        )

    # ========================================================
    # GYRO IMPROVEMENT
    # ========================================================

    best_acc_only = results_df[
        results_df["sensor_config"]
        == "ACC_ONLY"
    ].sort_values(
        "macro_f1",
        ascending=False
    ).iloc[0]

    best_acc_gyro = results_df[
        results_df["sensor_config"]
        == "ACC_GYRO"
    ].sort_values(
        "macro_f1",
        ascending=False
    ).iloc[0]

    f1_change = (
        best_acc_gyro["macro_f1"]
        -
        best_acc_only["macro_f1"]
    )

    accuracy_change = (
        best_acc_gyro["accuracy"]
        -
        best_acc_only["accuracy"]
    )

    print()
    print("=" * 70)
    print("GYROSCOPE CONTRIBUTION")
    print("=" * 70)

    print(
        f"Best ACC-only model:"
    )

    print(
        f"  {best_acc_only['model']}"
    )

    print(
        f"  Accuracy = "
        f"{best_acc_only['accuracy']:.4f}"
    )

    print(
        f"  Macro F1 = "
        f"{best_acc_only['macro_f1']:.4f}"
    )

    print()

    print(
        f"Best ACC+GYRO model:"
    )

    print(
        f"  {best_acc_gyro['model']}"
    )

    print(
        f"  Accuracy = "
        f"{best_acc_gyro['accuracy']:.4f}"
    )

    print(
        f"  Macro F1 = "
        f"{best_acc_gyro['macro_f1']:.4f}"
    )

    print()

    print(
        f"Accuracy change : "
        f"{accuracy_change:+.4f}"
    )

    print(
        f"Macro F1 change : "
        f"{f1_change:+.4f}"
    )

    # ========================================================
    # SAVE BEST MODELS
    # ========================================================

    for name, model in trained_models.items():

        if model is None:
            continue

        joblib.dump(
            model,
            OUTPUT_DIR /
            f"{name.lower()}.pkl"
        )

    # ========================================================
    # FINAL RECOMMENDATION
    # ========================================================

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if f1_change >= 0.02:

        print(
            "KEEP THE GYROSCOPE."
        )

        print(
            "The 6-axis configuration provides "
            "a meaningful improvement."
        )

    elif f1_change >= 0.005:

        print(
            "GYROSCOPE PROVIDES A SMALL IMPROVEMENT."
        )

        print(
            "Keep the BMI270 gyro because it can "
            "also contribute to fall detection."
        )

    else:

        print(
            "ACCELEROMETER MAY BE SUFFICIENT "
            "FOR ACTIVITY RECOGNITION."
        )

        print(
            "However, the gyro may still be "
            "valuable for fall detection."
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "best_acc_only_model": str(
            best_acc_only["model"]
        ),
        "best_acc_only_accuracy": float(
            best_acc_only["accuracy"]
        ),
        "best_acc_only_macro_f1": float(
            best_acc_only["macro_f1"]
        ),
        "best_acc_gyro_model": str(
            best_acc_gyro["model"]
        ),
        "best_acc_gyro_accuracy": float(
            best_acc_gyro["accuracy"]
        ),
        "best_acc_gyro_macro_f1": float(
            best_acc_gyro["macro_f1"]
        ),
        "accuracy_change": float(
            accuracy_change
        ),
        "macro_f1_change": float(
            f1_change
        )
    }

    with open(
        OUTPUT_DIR /
        "sensor_comparison_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print("=" * 70)
    print("PHASE 1C COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()