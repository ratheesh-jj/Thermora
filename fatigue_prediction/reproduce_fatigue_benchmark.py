from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction"
)

DATA_DIR = (
    BASE_DIR
    / "fatigue_personalization_results"
    / "extracted"
    / "MxD_Data_User_Study"
    / "MxD_Data_User_Study"
    / "composite"
)

OUT_DIR = BASE_DIR / "fatigue_5fold_results"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# EXACT 39 PARTICIPANTS
# ============================================================

PARTICIPANTS = [
    "P001", "P002", "P003", "P004", "P005",
    "P006", "P007", "P008", "P010", "P011",
    "P012", "P013", "P014", "P016", "P017",
    "P018", "P019", "P020", "P021", "P022",
    "P023", "P024", "P025", "P026", "P028",
    "P029", "P030", "P031", "P032", "P033",
    "P034", "P036", "P037", "P038", "P039",
    "P040", "P041", "P042", "P043"
]


# ============================================================
# HELPERS
# ============================================================

def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def safe_stats(x, prefix):

    x = numeric(x).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(x) == 0:
        return {}

    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": x.std(),
        f"{prefix}_min": x.min(),
        f"{prefix}_max": x.max(),
        f"{prefix}_median": x.median(),
        f"{prefix}_range": x.max() - x.min(),
        f"{prefix}_abs_mean": np.abs(x).mean(),
        f"{prefix}_rms": np.sqrt(np.mean(x ** 2)),
    }


# ============================================================
# IMU4
# ============================================================

def extract_imu4_features(df):

    cols = {
        "ax": "IMU_4_ax_g_",
        "ay": "IMU_4_ay_g_",
        "az": "IMU_4_az_g_",
        "gx": "IMU_4_gx_dps_",
        "gy": "IMU_4_gy_dps_",
        "gz": "IMU_4_gz_dps_",
    }

    features = {}

    for axis, col in cols.items():

        if col not in df.columns:
            raise ValueError(
                f"Missing IMU4 column: {col}"
            )

        features.update(
            safe_stats(
                df[col],
                f"imu4_{axis}"
            )
        )

    ax = numeric(df[cols["ax"]])
    ay = numeric(df[cols["ay"]])
    az = numeric(df[cols["az"]])

    gx = numeric(df[cols["gx"]])
    gy = numeric(df[cols["gy"]])
    gz = numeric(df[cols["gz"]])

    acc_mag = np.sqrt(
        ax ** 2 +
        ay ** 2 +
        az ** 2
    )

    gyro_mag = np.sqrt(
        gx ** 2 +
        gy ** 2 +
        gz ** 2
    )

    features.update(
        safe_stats(
            acc_mag,
            "imu4_acc_mag"
        )
    )

    features.update(
        safe_stats(
            gyro_mag,
            "imu4_gyro_mag"
        )
    )

    valid = acc_mag.dropna()

    if len(valid) > 1:

        features["acc_mag_diff_mean"] = (
            valid.diff()
            .abs()
            .mean()
        )

        features["acc_mag_diff_std"] = (
            valid.diff()
            .std()
        )

    return features


# ============================================================
# BASELINE
# ============================================================

def extract_baseline_features(df):

    features = {}

    physiological = [
        ("HR_Processed", "hr"),
        ("HR", "hr_raw"),
        ("HRV", "hrv"),
        ("RR", "rr"),
        ("Temperature", "temp"),
    ]

    for col, name in physiological:

        if col in df.columns:

            features.update(
                safe_stats(
                    df[col],
                    f"{name}"
                )
            )

    features.update(
        extract_imu4_features(df)
    )

    return features


def load_initial_baseline(participant):

    files = list(
        DATA_DIR.rglob(
            f"Initial_Rest_{participant}*.csv"
        )
    )

    if not files:
        raise RuntimeError(
            f"Initial rest missing for {participant}"
        )

    return extract_baseline_features(
        pd.read_csv(files[0])
    )


# ============================================================
# TARGET
# ============================================================

def fatigue_value(df, column):

    x = numeric(
        df[column]
    ).replace(
        [-2, -1],
        np.nan
    ).dropna()

    if len(x) == 0:
        return np.nan

    return float(
        x.iloc[0]
    )


def calculate_target(df):

    initial = fatigue_value(
        df,
        "Physical Fatigue - Initial"
    )

    final = fatigue_value(
        df,
        "Physical Fatigue - Final"
    )

    if np.isnan(initial) or np.isnan(final):
        return np.nan

    return float(
        np.clip(
            (initial + final) / 2,
            0,
            10
        )
    )


# ============================================================
# DATASET
# ============================================================

def build_dataset():

    rows = []

    print("=" * 70)
    print("BUILDING 195-SAMPLE DATASET")
    print("=" * 70)

    for participant in PARTICIPANTS:

        baseline = load_initial_baseline(
            participant
        )

        for rep in range(1, 6):

            files = list(
                DATA_DIR.rglob(
                    f"Rep{rep}_{participant}*.csv"
                )
            )

            if len(files) != 1:
                raise RuntimeError(
                    f"{participant} Rep{rep}: "
                    f"found {len(files)} files"
                )

            df = pd.read_csv(
                files[0]
            )

            target = calculate_target(
                df
            )

            if np.isnan(target):
                continue

            row = extract_imu4_features(
                df
            )

            row["participant_id"] = participant
            row["repetition"] = rep
            row["fatigue"] = target

            for key, value in baseline.items():

                row[
                    f"baseline_{key}"
                ] = value

            rows.append(row)

    data = pd.DataFrame(rows)

    print(
        "Samples:",
        len(data)
    )

    print(
        "Participants:",
        data["participant_id"].nunique()
    )

    assert len(data) == 195
    assert data["participant_id"].nunique() == 39

    return data


# ============================================================
# PERSONALIZED FEATURES
# ============================================================

def add_relative_features(data):

    data = data.copy()

    pairs = [
        (
            "imu4_acc_mag_mean",
            "baseline_imu4_acc_mag_mean",
            "rel_acc_mag"
        ),
        (
            "imu4_gyro_mag_mean",
            "baseline_imu4_gyro_mag_mean",
            "rel_gyro_mag"
        ),
        (
            "imu4_ax_mean",
            "baseline_imu4_ax_mean",
            "rel_ax"
        ),
        (
            "imu4_ay_mean",
            "baseline_imu4_ay_mean",
            "rel_ay"
        ),
        (
            "imu4_az_mean",
            "baseline_imu4_az_mean",
            "rel_az"
        ),
        (
            "imu4_gx_mean",
            "baseline_imu4_gx_mean",
            "rel_gx"
        ),
        (
            "imu4_gy_mean",
            "baseline_imu4_gy_mean",
            "rel_gy"
        ),
        (
            "imu4_gz_mean",
            "baseline_imu4_gz_mean",
            "rel_gz"
        ),
    ]

    for current, baseline, name in pairs:

        if (
            current in data.columns
            and baseline in data.columns
        ):

            data[name] = (
                data[current]
                - data[baseline]
            )

    return data


# ============================================================
# FEATURES
# ============================================================

def get_feature_sets(data):

    forbidden = {
        "fatigue",
        "participant_id",
        "repetition",
    }

    raw = [
        c for c in data.columns
        if c.startswith("imu4_")
        and c not in forbidden
    ]

    personalized = [
        c for c in data.columns
        if (
            c.startswith("imu4_")
            or c.startswith("baseline_")
            or c.startswith("rel_")
        )
        and c not in forbidden
    ]

    return raw, personalized


# ============================================================
# MODEL
# ============================================================

def create_model():

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )


# ============================================================
# 5-FOLD GROUP CROSS VALIDATION
# ============================================================

def evaluate_model(
    data,
    features,
    model_name,
    splits
):

    fold_results = []
    all_predictions = []

    groups = data[
        "participant_id"
    ].values

    X = data[
        features
    ]

    y = data[
        "fatigue"
    ]

    for fold, (train_idx, test_idx) in enumerate(
        splits,
        start=1
    ):

        train = data.iloc[
            train_idx
        ]

        test = data.iloc[
            test_idx
        ]

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
        # Imputation fitted ONLY on training participants
        # ----------------------------------------------------

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train = imputer.fit_transform(
            X_train
        )

        X_test = imputer.transform(
            X_test
        )

        model = create_model()

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
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

        fold_results.append({
            "model": model_name,
            "fold": fold,
            "train_participants":
                train["participant_id"].nunique(),
            "test_participants":
                test["participant_id"].nunique(),
            "train_samples":
                len(train),
            "test_samples":
                len(test),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        predictions = test[
            [
                "participant_id",
                "repetition",
                "fatigue"
            ]
        ].copy()

        predictions[
            "prediction"
        ] = prediction

        predictions[
            "absolute_error"
        ] = np.abs(
            prediction -
            y_test.to_numpy()
        )

        predictions[
            "fold"
        ] = fold

        all_predictions.append(
            predictions
        )

        print(
            f"{model_name} | "
            f"Fold {fold} | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R²={r2:.4f}"
        )

    fold_df = pd.DataFrame(
        fold_results
    )

    pred_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    fold_df.to_csv(
        OUT_DIR /
        f"{model_name}_fold_results.csv",
        index=False
    )

    pred_df.to_csv(
        OUT_DIR /
        f"{model_name}_predictions.csv",
        index=False
    )

    return fold_df


# ============================================================
# MAIN
# ============================================================

def main():

    data = build_dataset()

    data = add_relative_features(
        data
    )

    raw_features, personalized_features = (
        get_feature_sets(data)
    )

    print()
    print(
        "Raw features:",
        len(raw_features)
    )

    print(
        "Personalized features:",
        len(personalized_features)
    )

    # --------------------------------------------------------
    # EXACT SAME PARTICIPANT FOLDS FOR BOTH MODELS
    # --------------------------------------------------------

    groups = data[
        "participant_id"
    ]

    gkf = GroupKFold(
        n_splits=5
    )

    splits = list(
        gkf.split(
            data,
            data["fatigue"],
            groups
        )
    )

    print()
    print("=" * 70)
    print("5-FOLD PARTICIPANT-HELD-OUT EVALUATION")
    print("=" * 70)

    raw_results = evaluate_model(
        data,
        raw_features,
        "RAW_IMU4",
        splits
    )

    personalized_results = evaluate_model(
        data,
        personalized_features,
        "PERSONALIZED_IMU4",
        splits
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL 5-FOLD RESULTS")
    print("=" * 70)

    summary = []

    for name, results in [
        (
            "RAW_IMU4",
            raw_results
        ),
        (
            "PERSONALIZED_IMU4",
            personalized_results
        )
    ]:

        summary.append({
            "model": name,

            "MAE_mean":
                results["MAE"].mean(),

            "MAE_std":
                results["MAE"].std(),

            "RMSE_mean":
                results["RMSE"].mean(),

            "RMSE_std":
                results["RMSE"].std(),

            "R2_mean":
                results["R2"].mean(),

            "R2_std":
                results["R2"].std(),
        })

    summary_df = pd.DataFrame(
        summary
    )

    print()

    for _, row in summary_df.iterrows():

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

    raw_mae = summary_df.loc[
        summary_df["model"] == "RAW_IMU4",
        "MAE_mean"
    ].iloc[0]

    personalized_mae = summary_df.loc[
        summary_df["model"] == "PERSONALIZED_IMU4",
        "MAE_mean"
    ].iloc[0]

    improvement = (
        (raw_mae - personalized_mae)
        / raw_mae
        * 100
    )

    print(
        f"Mean MAE improvement: "
        f"{improvement:.2f}%"
    )

    summary_df.to_csv(
        OUT_DIR /
        "FINAL_5FOLD_SUMMARY.csv",
        index=False
    )

    print()
    print(
        "Saved results to:"
    )

    print(
        OUT_DIR
    )


if __name__ == "__main__":
    main()