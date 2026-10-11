from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction"
)

FEATURE_FILE = (
    BASE
    / "fatigue_paper_features_v2"
    / "paper_style_features.csv"
)

OUT = (
    BASE
    / "fatigue_deployable_model"
)

OUT.mkdir(
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    FEATURE_FILE
)

print("=" * 75)
print("THERMORA DEPLOYABLE FATIGUE MODEL")
print("=" * 75)

print(
    f"Rows         : {len(df)}"
)

print(
    f"Participants : "
    f"{df['participant_id'].nunique()}"
)

print()


# ============================================================
# EXACT IMU FEATURE SET
# ============================================================

META = {
    "participant_id",
    "repetition",
    "window_start",
    "window_end",
    "fatigue"
}

ALL_FEATURES = [
    c for c in df.columns
    if c not in META
]


IMU_FEATURES = [
    c for c in ALL_FEATURES
    if c.startswith("IMU4_")
]


# Safety check
if len(IMU_FEATURES) != 156:

    raise RuntimeError(
        f"Expected 156 IMU4 features, "
        f"found {len(IMU_FEATURES)}"
    )


print(
    f"IMU4 features: "
    f"{len(IMU_FEATURES)}"
)

print()


# ============================================================
# INPUT / TARGET
# ============================================================

X = df[
    IMU_FEATURES
]

y = df[
    "fatigue"
]


# ============================================================
# IMPUTER
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_imp = imputer.fit_transform(
    X
)


# ============================================================
# FINAL RANDOM FOREST
# ============================================================

print(
    "Training final Random Forest..."
)

model = RandomForestRegressor(

    n_estimators=500,

    min_samples_leaf=3,

    max_features="sqrt",

    random_state=RANDOM_STATE,

    n_jobs=-1
)

model.fit(
    X_imp,
    y
)


# ============================================================
# TRAINING PERFORMANCE
# ============================================================

train_pred = model.predict(
    X_imp
)

train_mae = np.mean(
    np.abs(
        train_pred - y.to_numpy()
    )
)

train_rmse = np.sqrt(
    np.mean(
        (
            train_pred
            -
            y.to_numpy()
        ) ** 2
    )
)


print()

print(
    f"Training MAE  : "
    f"{train_mae:.4f}"
)

print(
    f"Training RMSE : "
    f"{train_rmse:.4f}"
)

print()

print(
    "NOTE: training metrics are NOT "
    "the generalization benchmark."
)

print(
    "Use the previous GroupKFold result:"
)

print(
    "MAE = 1.6728 ± 0.4865"
)

print()


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    OUT
    / "thermora_fatigue_rf.pkl"
)

imputer_file = (
    OUT
    / "thermora_fatigue_imputer.pkl"
)

features_file = (
    OUT
    / "thermora_fatigue_features.json"
)

model_info_file = (
    OUT
    / "model_info.json"
)


joblib.dump(
    model,
    model_file
)

joblib.dump(
    imputer,
    imputer_file
)


# ============================================================
# SAVE FEATURE ORDER
# ============================================================

with open(
    features_file,
    "w"
) as f:

    json.dump(
        IMU_FEATURES,
        f,
        indent=2
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

info = {

    "model":
        "RandomForestRegressor",

    "n_estimators":
        500,

    "min_samples_leaf":
        3,

    "max_features":
        "sqrt",

    "random_state":
        RANDOM_STATE,

    "input_features":
        len(IMU_FEATURES),

    "training_rows":
        len(df),

    "training_participants":
        int(
            df[
                "participant_id"
            ].nunique()
        ),

    "validated_mae":
        1.6728,

    "validated_mae_std":
        0.4865,

    "validated_rmse":
        2.0253,

    "validated_r2":
        0.0528,

    "target":
        "continuous fatigue score 0-10",

    "sensor":
        "BMI270 IMU4-equivalent",

    "window_seconds":
        180,

    "window_stride_seconds":
        30
}


with open(
    model_info_file,
    "w"
) as f:

    json.dump(
        info,
        f,
        indent=2
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        IMU_FEATURES,

    "importance":
        model.feature_importances_
})

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)

importance.to_csv(
    OUT
    / "feature_importance.csv",
    index=False
)


# ============================================================
# MODEL SIZE
# ============================================================

print("=" * 75)
print("MODEL ARTIFACTS")
print("=" * 75)

print()

print(
    f"Model:\n{model_file}"
)

print(
    f"Imputer:\n{imputer_file}"
)

print(
    f"Features:\n{features_file}"
)

print(
    f"Info:\n{model_info_file}"
)

print()

print(
    "Trees:",
    model.n_estimators
)

print(
    "Features:",
    model.n_features_in_
)


# ============================================================
# TREE STATISTICS
# ============================================================

nodes = [
    estimator.tree_.node_count
    for estimator in model.estimators_
]

depths = [
    estimator.tree_.max_depth
    for estimator in model.estimators_
]

print()

print(
    f"Average tree nodes : "
    f"{np.mean(nodes):.1f}"
)

print(
    f"Maximum tree nodes : "
    f"{np.max(nodes)}"
)

print(
    f"Average tree depth : "
    f"{np.mean(depths):.1f}"
)

print(
    f"Maximum tree depth : "
    f"{np.max(depths)}"
)


# ============================================================
# TOP FEATURES
# ============================================================

print()

print("=" * 75)
print("TOP 20 FEATURES")
print("=" * 75)

print()

print(
    importance.head(20).to_string(
        index=False
    )
)

print()

print("=" * 75)
print("DONE")
print("=" * 75)