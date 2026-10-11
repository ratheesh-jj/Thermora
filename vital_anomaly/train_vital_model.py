# train_vital_model.py
#
# THERMORA - Vital Physiological State Model
#
# Evaluates the contribution of:
#   1. HR
#   2. HR + SpO2
#   3. HR + HRV
#   4. HR + HRV + Temperature
#   5. HR + HRV + Temperature + EDA + ACC
#
# IMPORTANT:
# Participant IDs ending in _a / _b are normalized to the
# same base participant to prevent participant leakage.
#
# Target:
#   STRESS / AEROBIC / ANAEROBIC
#
# This is a physiological-state classification experiment,
# NOT a clinical diagnosis model.

import os
import re
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\OMEN\Desktop\Projects\SIH26\vital_anomaly"

DATA_PATH = os.path.join(
    BASE_DIR,
    "wearable_stress",
    "outputs",
    "wearable_all_conditions_vital_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "wearable_stress",
    "outputs",
    "vital_model_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("THERMORA VITAL MODEL")
print("=" * 70)

print("\nLoading:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

print("\nColumns:")
for col in df.columns:
    print(" ", col)


# ============================================================
# NORMALIZE PARTICIPANT IDs
# ============================================================

def normalize_participant(pid):
    """
    Convert:
        f14_a -> f14
        f14_b -> f14
        S11_a -> S11
        S11_b -> S11

    This prevents the same participant from appearing
    in both training and testing sets.
    """

    pid = str(pid)

    return re.sub(r"_a$|_b$", "", pid, flags=re.IGNORECASE)


# Find participant column
possible_subject_cols = [
    "subject",
    "subject_id",
    "participant",
    "participant_id",
    "Subject",
    "Participant"
]

subject_col = None

for col in possible_subject_cols:
    if col in df.columns:
        subject_col = col
        break

if subject_col is None:
    raise ValueError(
        "Could not find participant column. "
        "Available columns:\n" + "\n".join(df.columns)
    )

print("\nParticipant column:", subject_col)

df["participant_group"] = df[subject_col].apply(normalize_participant)

print(
    "Unique participants after normalization:",
    df["participant_group"].nunique()
)


# ============================================================
# TARGET
# ============================================================

if "condition" not in df.columns:
    raise ValueError(
        "Expected target column 'condition' was not found."
    )

df = df.dropna(subset=["condition"]).copy()

print("\nCondition distribution:")
print(df["condition"].value_counts())


# ============================================================
# REMOVE NON-FEATURE COLUMNS
# ============================================================

metadata_cols = [
    "condition",
    subject_col,
    "participant_group",
    "window_start",
    "window_end",
    "timestamp",
    "start_time",
    "end_time"
]

metadata_cols = [
    c for c in metadata_cols
    if c in df.columns
]


# ============================================================
# IDENTIFY FEATURE GROUPS
# ============================================================

# We search by column names rather than hard-coding
# every generated feature.

hr_features = []

hrv_features = []

spo2_features = []

temp_features = []

eda_features = []

acc_features = []


for col in df.columns:

    c = col.lower()

    if col in metadata_cols:
        continue

    # -------------------------
    # HR
    # -------------------------

    if (
        "hr_" in c
        or c.startswith("hr")
        or "heart_rate" in c
    ):
        hr_features.append(col)

    # -------------------------
    # HRV
    # -------------------------

    if any(
        x in c
        for x in [
            "ibi",
            "rr_",
            "sdnn",
            "rmssd",
            "pnn",
            "hrv",
            "mean_rr"
        ]
    ):
        hrv_features.append(col)

    # -------------------------
    # SpO2
    # -------------------------

    if (
        "spo2" in c
        or "oxygen" in c
    ):
        spo2_features.append(col)

    # -------------------------
    # Temperature
    # -------------------------

    if (
        "temp" in c
        or "temperature" in c
    ):
        temp_features.append(col)

    # -------------------------
    # EDA
    # -------------------------

    if "eda" in c:
        eda_features.append(col)

    # -------------------------
    # ACC
    # -------------------------

    if (
        "acc_" in c
        or "acc" == c
        or "accel" in c
        or "movement" in c
    ):
        acc_features.append(col)


# Remove accidental duplicates
hr_features = sorted(list(set(hr_features)))
hrv_features = sorted(list(set(hrv_features)))
spo2_features = sorted(list(set(spo2_features)))
temp_features = sorted(list(set(temp_features)))
eda_features = sorted(list(set(eda_features)))
acc_features = sorted(list(set(acc_features)))


# ============================================================
# PRINT FEATURE GROUPS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE GROUPS")
print("=" * 70)

print("\nHR:")
print(hr_features)

print("\nHRV:")
print(hrv_features)

print("\nSpO2:")
print(spo2_features)

print("\nTemperature:")
print(temp_features)

print("\nEDA:")
print(eda_features)

print("\nAcceleration:")
print(acc_features)


# ============================================================
# BUILD FEATURE CONFIGURATIONS
# ============================================================

feature_groups = {

    "HR_only": (
        hr_features
    ),

    "HR_SpO2": (
        hr_features
        + spo2_features
    ),

    "HR_HRV": (
        hr_features
        + hrv_features
    ),

    "HR_HRV_Temperature": (
        hr_features
        + hrv_features
        + temp_features
    ),

    "All_Wearable": (
        hr_features
        + hrv_features
        + temp_features
        + eda_features
        + acc_features
        + spo2_features
    )
}


# Remove duplicates
for name in feature_groups:

    feature_groups[name] = sorted(
        list(set(feature_groups[name]))
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE COUNTS")
print("=" * 70)

for name, features in feature_groups.items():

    print(
        f"{name:25s}: {len(features):3d} features"
    )

    if len(features) == 0:
        print("  WARNING: No features found!")


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["condition"].astype(str)
)

print("\nClasses:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i}: {label}")


# ============================================================
# GROUPED TRAIN / TEST SPLIT
# ============================================================

groups = df["participant_group"].values

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        y,
        groups=groups
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

y_train = y[train_idx]
y_test = y[test_idx]

print("\n" + "=" * 70)
print("PARTICIPANT-HELD-OUT SPLIT")
print("=" * 70)

print(
    "\nTrain windows:",
    len(train_df)
)

print(
    "Test windows:",
    len(test_df)
)

print(
    "Train participants:",
    train_df["participant_group"].nunique()
)

print(
    "Test participants:",
    test_df["participant_group"].nunique()
)

print("\nTest participants:")

for p in sorted(
    test_df["participant_group"].unique()
):
    print(" ", p)


# ============================================================
# SAVE SPLIT INFO
# ============================================================

split_info = {

    "total_windows": int(len(df)),

    "train_windows": int(len(train_df)),

    "test_windows": int(len(test_df)),

    "train_participants": sorted(
        train_df["participant_group"].unique().tolist()
    ),

    "test_participants": sorted(
        test_df["participant_group"].unique().tolist()
    ),

    "classes": label_encoder.classes_.tolist(),

    "random_state": 42
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "split_info.json"
    ),
    "w"
) as f:

    json.dump(
        split_info,
        f,
        indent=2
    )


# ============================================================
# MODELS
# ============================================================

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    )
}


# ============================================================
# STORAGE
# ============================================================

results = []


# ============================================================
# TRAIN EACH FEATURE GROUP
# ============================================================

for group_name, features in feature_groups.items():

    print("\n")
    print("=" * 70)
    print("FEATURE GROUP:", group_name)
    print("=" * 70)

    if len(features) == 0:

        print(
            "Skipping because no features were found."
        )

        continue


    X_train = train_df[features].copy()
    X_test = test_df[features].copy()


    # Convert everything to numeric
    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # Replace inf
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # Imputer
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )


    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_test_imp = imputer.transform(
        X_test
    )


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train_imp,
            y_train
        )

        pred = model.predict(
            X_test_imp
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            pred
        )

        macro_f1 = f1_score(
            y_test,
            pred,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_test,
            pred,
            average="weighted",
            zero_division=0
        )


        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Macro F1  : {macro_f1:.4f}"
        )

        print(
            f"Weighted F1: {weighted_f1:.4f}"
        )


        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )

        print("\nClassification report:")
        print(report)


        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=label_encoder.classes_,
            columns=label_encoder.classes_
        )

        cm_path = os.path.join(
            OUTPUT_DIR,
            f"{group_name}_{model_name}_confusion_matrix.csv"
        )

        cm_df.to_csv(
            cm_path
        )


        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        pred_df = test_df[
            [
                subject_col,
                "participant_group",
                "condition"
            ]
        ].copy()

        pred_df["true_label"] = (
            label_encoder.inverse_transform(
                y_test
            )
        )

        pred_df["predicted_label"] = (
            label_encoder.inverse_transform(
                pred
            )
        )

        pred_path = os.path.join(
            OUTPUT_DIR,
            f"{group_name}_{model_name}_predictions.csv"
        )

        pred_df.to_csv(
            pred_path,
            index=False
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({

            "feature_group": group_name,

            "model": model_name,

            "num_features": len(features),

            "accuracy": accuracy,

            "macro_f1": macro_f1,

            "weighted_f1": weighted_f1

        })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False
)

results_path = os.path.join(
    OUTPUT_DIR,
    "vital_model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    "\nFeature group:",
    best["feature_group"]
)

print(
    "Model:",
    best["model"]
)

print(
    f"Accuracy: {best['accuracy']:.4f}"
)

print(
    f"Macro F1: {best['macro_f1']:.4f}"
)

print(
    f"Weighted F1: {best['weighted_f1']:.4f}"
)


# ============================================================
# SAVE FEATURE CONFIG
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "feature_groups.json"
    ),
    "w"
) as f:

    json.dump(
        feature_groups,
        f,
        indent=2
    )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print("\nResults saved to:")

print(
    OUTPUT_DIR
)

print("\nMain file:")

print(
    results_path
)