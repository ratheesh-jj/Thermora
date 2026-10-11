from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import GroupKFold


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction"
)

DATA_FILE = (
    BASE
    / "mxd_imu4_windows"
    / "imu4_3min_windows.csv"
)

OUT = BASE / "imu4_window_models"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("IMU4 3-MINUTE FATIGUE MODEL")
print("=" * 70)

print(
    "Rows:",
    len(df)
)

print(
    "Participants:",
    df["participant_id"].nunique()
)

print(
    "Repetitions:",
    df[
        ["participant_id", "repetition"]
    ].drop_duplicates().shape[0]
)


# ============================================================
# FEATURES
# ============================================================

DROP = {
    "participant_id",
    "repetition",
    "window_start",
    "window_end",
    "fatigue"
}

features = [
    c for c in df.columns
    if c not in DROP
]

X = df[features]

y = df["fatigue"]

groups = df["participant_id"]


print(
    "Features:",
    len(features)
)


# ============================================================
# MODELS
# ============================================================

def make_hgb():

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )


def make_rf():

    return RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


models = {
    "HGB": make_hgb,
    "RF": make_rf
}


# ============================================================
# 5-FOLD GROUP CV
# ============================================================

gkf = GroupKFold(
    n_splits=5
)

all_results = []

all_predictions = []


for model_name, model_factory in models.items():

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        train_groups = groups.iloc[
            train_idx
        ]

        test_groups = groups.iloc[
            test_idx
        ]

        # ----------------------------------------------------
        # Fit preprocessing only on training participants
        # ----------------------------------------------------

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_imp = (
            imputer
            .fit_transform(X_train)
        )

        X_test_imp = (
            imputer
            .transform(X_test)
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = model_factory()

        model.fit(
            X_train_imp,
            y_train
        )

        prediction = model.predict(
            X_test_imp
        )

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prediction
            )
        )

        r2 = r2_score(
            y_test,
            prediction
        )

        print(
            f"Fold {fold}: "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R²={r2:.4f} | "
            f"Train={len(train_idx)} | "
            f"Test={len(test_idx)}"
        )

        all_results.append({
            "model": model_name,
            "fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_participants":
                train_groups.nunique(),
            "test_participants":
                test_groups.nunique()
        })

        pred = df.iloc[
            test_idx
        ][
            [
                "participant_id",
                "repetition",
                "window_start",
                "window_end",
                "fatigue"
            ]
        ].copy()

        pred["prediction"] = prediction

        pred["absolute_error"] = np.abs(
            prediction -
            y_test.to_numpy()
        )

        pred["model"] = model_name
        pred["fold"] = fold

        all_predictions.append(
            pred
        )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

results.to_csv(
    OUT / "fold_results.csv",
    index=False
)

predictions.to_csv(
    OUT / "predictions.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

summary = (
    results
    .groupby("model")
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std")
    )
    .reset_index()
)

print()

for _, row in summary.iterrows():

    print(
        row["model"]
    )

    print(
        f"  MAE  : "
        f"{row['MAE_mean']:.4f} "
        f"± {row['MAE_std']:.4f}"
    )

    print(
        f"  RMSE : "
        f"{row['RMSE_mean']:.4f} "
        f"± {row['RMSE_std']:.4f}"
    )

    print(
        f"  R²   : "
        f"{row['R2_mean']:.4f} "
        f"± {row['R2_std']:.4f}"
    )

    print()

summary.to_csv(
    OUT / "summary.csv",
    index=False
)

print(
    "Saved:",
    OUT
)