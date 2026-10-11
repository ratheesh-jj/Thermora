from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


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
    / "fatigue_temporal_v1"
)

OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Number of historical windows
HISTORY = 4


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("=" * 75)
print("THERMORA TEMPORAL FATIGUE MODEL")
print("=" * 75)

print(
    f"Input rows       : {len(df)}"
)

print(
    f"Participants     : "
    f"{df['participant_id'].nunique()}"
)

print()


# ============================================================
# IDENTIFY FEATURES
# ============================================================

META = {
    "participant_id",
    "repetition",
    "window_start",
    "window_end",
    "fatigue"
}

FEATURES = [
    c for c in df.columns
    if c not in META
]


# Only numerical features
FEATURES = [
    c for c in FEATURES
    if pd.api.types.is_numeric_dtype(
        df[c]
    )
]

print(
    f"Base features    : {len(FEATURES)}"
)

print()


# ============================================================
# SORT TEMPORALLY
# ============================================================

df = df.sort_values(
    [
        "participant_id",
        "window_start"
    ]
).reset_index(drop=True)


# ============================================================
# BUILD TEMPORAL SEQUENCES
# ============================================================

print(
    f"Building {HISTORY}-window temporal sequences..."
)

temporal_rows = []

expected_step = 30000  # 30 seconds


for pid, group in df.groupby(
    "participant_id"
):

    group = group.sort_values(
        "window_start"
    ).reset_index(drop=True)

    X_values = group[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    fatigue = group[
        "fatigue"
    ].to_numpy(
        dtype=float
    )

    times = group[
        "window_start"
    ].to_numpy(
        dtype=float
    )

    repetitions = group[
        "repetition"
    ].to_numpy()

    for i in range(
        HISTORY - 1,
        len(group)
    ):

        start = i - HISTORY + 1

        history_times = times[
            start:i + 1
        ]

        # ----------------------------------------------------
        # Require consecutive 30-second windows
        # ----------------------------------------------------

        diffs = np.diff(
            history_times
        )

        if len(diffs):

            if not np.all(
                np.isclose(
                    diffs,
                    expected_step,
                    atol=100
                )
            ):
                continue

        # ----------------------------------------------------
        # Flatten temporal history
        # ----------------------------------------------------

        sequence = X_values[
            start:i + 1
        ].reshape(-1)

        row = {}

        for h in range(HISTORY):

            for j, feature in enumerate(
                FEATURES
            ):

                row[
                    f"h{h+1}_{feature}"
                ] = sequence[
                    h * len(FEATURES) + j
                ]

        # ----------------------------------------------------
        # Add simple temporal change features
        # ----------------------------------------------------

        current = X_values[i]

        previous = X_values[i - 1]

        delta = current - previous

        for j, feature in enumerate(
            FEATURES
        ):

            row[
                f"delta_{feature}"
            ] = delta[j]

        # ----------------------------------------------------
        # Metadata / target
        # ----------------------------------------------------

        row[
            "participant_id"
        ] = pid

        row[
            "repetition"
        ] = repetitions[i]

        row[
            "window_start"
        ] = times[i]

        row[
            "fatigue"
        ] = fatigue[i]

        temporal_rows.append(
            row
        )


temporal = pd.DataFrame(
    temporal_rows
)


# ============================================================
# CHECK
# ============================================================

print()

print("=" * 75)
print("TEMPORAL DATASET")
print("=" * 75)

print(
    f"Rows             : {len(temporal)}"
)

print(
    f"Participants     : "
    f"{temporal['participant_id'].nunique()}"
)

print(
    f"Features          : "
    f"{len(temporal.columns) - 4}"
)

print()


# ============================================================
# SAVE TEMPORAL DATASET
# ============================================================

temporal_file = (
    OUT
    / "temporal_sequences.csv"
)

temporal.to_csv(
    temporal_file,
    index=False
)

print(
    f"Saved:\n{temporal_file}"
)

print()


# ============================================================
# X / Y / GROUPS
# ============================================================

META_TEMP = {
    "participant_id",
    "repetition",
    "window_start",
    "fatigue"
}

TEMP_FEATURES = [
    c for c in temporal.columns
    if c not in META_TEMP
]

X = temporal[
    TEMP_FEATURES
]

y = temporal[
    "fatigue"
]

groups = temporal[
    "participant_id"
]


# ============================================================
# MODELS
# ============================================================

models = {

    "RF": RandomForestRegressor(

        n_estimators=500,

        min_samples_leaf=3,

        max_features="sqrt",

        random_state=RANDOM_STATE,

        n_jobs=-1
    ),

    "HGB": HistGradientBoostingRegressor(

        max_iter=300,

        learning_rate=0.04,

        max_leaf_nodes=15,

        l2_regularization=1.0,

        random_state=RANDOM_STATE
    )
}


# ============================================================
# GROUP K-FOLD
# ============================================================

gkf = GroupKFold(
    n_splits=5
)

results = []

predictions = []


for model_name, model_template in models.items():

    print()
    print("=" * 75)
    print(model_name)
    print("=" * 75)

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
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


        # ----------------------------------------------------
        # IMPUTATION
        # ----------------------------------------------------

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_imp = (
            imputer.fit_transform(
                X_train
            )
        )

        X_test_imp = (
            imputer.transform(
                X_test
            )
        )


        # ----------------------------------------------------
        # Fresh model
        # ----------------------------------------------------

        if model_name == "RF":

            model = RandomForestRegressor(

                n_estimators=500,

                min_samples_leaf=3,

                max_features="sqrt",

                random_state=RANDOM_STATE,

                n_jobs=-1
            )

        else:

            model = HistGradientBoostingRegressor(

                max_iter=300,

                learning_rate=0.04,

                max_leaf_nodes=15,

                l2_regularization=1.0,

                random_state=RANDOM_STATE
            )


        model.fit(
            X_train_imp,
            y_train
        )

        pred = model.predict(
            X_test_imp
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        r2 = r2_score(
            y_test,
            pred
        )


        print(
            f"Fold {fold}: "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R²={r2:.4f}"
        )


        results.append({

            "model":
                model_name,

            "fold":
                fold,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2,

            "test_participants":
                groups.iloc[
                    test_idx
                ].nunique(),

            "test_samples":
                len(test_idx)
        })


        p = temporal.iloc[
            test_idx
        ][
            [
                "participant_id",
                "repetition",
                "window_start",
                "fatigue"
            ]
        ].copy()

        p[
            "prediction"
        ] = pred

        p[
            "absolute_error"
        ] = np.abs(
            pred -
            y_test.to_numpy()
        )

        p[
            "model"
        ] = model_name

        p[
            "fold"
        ] = fold

        predictions.append(
            p
        )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.concat(
    predictions,
    ignore_index=True
)


results_df.to_csv(
    OUT / "fold_results.csv",
    index=False
)

predictions_df.to_csv(
    OUT / "predictions.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    results_df
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

summary.to_csv(
    OUT / "summary.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 75)
print("TEMPORAL MODEL SUMMARY")
print("=" * 75)

for _, row in summary.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"  MAE  = "
        f"{row['MAE_mean']:.4f} "
        f"± "
        f"{row['MAE_std']:.4f}"
    )

    print(
        f"  RMSE = "
        f"{row['RMSE_mean']:.4f} "
        f"± "
        f"{row['RMSE_std']:.4f}"
    )

    print(
        f"  R²   = "
        f"{row['R2_mean']:.4f} "
        f"± "
        f"{row['R2_std']:.4f}"
    )


# ============================================================
# BEST
# ============================================================

best = (
    summary
    .sort_values("MAE_mean")
    .iloc[0]
)

baseline_mae = 1.6728

improvement = (
    (
        baseline_mae -
        best["MAE_mean"]
    )
    /
    baseline_mae
) * 100


print()

print("=" * 75)
print("BEST TEMPORAL MODEL")
print("=" * 75)

print(
    f"Model : {best['model']}"
)

print(
    f"MAE   : "
    f"{best['MAE_mean']:.4f} "
    f"± "
    f"{best['MAE_std']:.4f}"
)

print(
    f"RMSE  : "
    f"{best['RMSE_mean']:.4f} "
    f"± "
    f"{best['RMSE_std']:.4f}"
)

print(
    f"R²    : "
    f"{best['R2_mean']:.4f} "
    f"± "
    f"{best['R2_std']:.4f}"
)

print()

print(
    f"MAE change vs "
    f"1.6728 baseline: "
    f"{improvement:+.2f}%"
)

print()

print(
    f"Output directory:\n{OUT}"
)

print()

print("=" * 75)
print("DONE")
print("=" * 75)