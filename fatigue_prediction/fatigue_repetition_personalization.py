import os
import re
import glob
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"fatigue_personalization_results\extracted"

OUTPUT_DIR = r"fatigue_repetition_personalization_results"

IMU = "IMU_4"

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def clean_columns(df):
    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]
    return df


def find_col(df, exact=None, contains=None):

    if exact:
        for c in df.columns:
            if str(c).strip().lower() == exact.lower():
                return c

    if contains:
        for c in df.columns:
            lc = str(c).lower().replace(" ", "")
            if contains.lower().replace(" ", "") in lc:
                return c

    return None


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [-2, -999, -9999],
        np.nan
    )


# ============================================================
# PARTICIPANT EXTRACTION
# ============================================================

def participant_from_path(path):

    # IMPORTANT:
    # Use the Pxxx portion of the recording directory,
    # not the entire directory name.

    text = str(path)

    matches = re.findall(
        r'P(\d+)',
        text,
        flags=re.IGNORECASE
    )

    if matches:

        # Usually the first Pxxx is the participant.
        return f"P{int(matches[0]):02d}"

    # Fallback
    return os.path.basename(
        os.path.dirname(path)
    )


# ============================================================
# TASK
# ============================================================

def task_from_path(path):

    text = str(path).lower()

    if "composite" in text:
        return "composite"

    if "ziptie" in text:
        return "ziptie"

    return "unknown"


# ============================================================
# FIND IMU4 COLUMNS
# ============================================================

def find_imu_columns(df):

    result = {}

    for axis in [
        "ax", "ay", "az",
        "gx", "gy", "gz"
    ]:

        candidates = []

        for c in df.columns:

            lc = (
                str(c)
                .lower()
                .replace(" ", "")
            )

            # Match IMU_4_ax / IMU4_ax etc.
            if re.search(
                rf"imu[_\-]?4.*{axis}$",
                lc
            ):
                candidates.append(c)

            elif re.search(
                rf"imu[_\-]?4.*{axis}.*",
                lc
            ):
                candidates.append(c)

        if candidates:
            result[axis] = candidates[0]

    return result


# ============================================================
# FATIGUE LABEL
# ============================================================

def get_fatigue(df):

    initial_col = None
    final_col = None

    for c in df.columns:

        lc = str(c).lower()

        if (
            "physical fatigue" in lc
            and "initial" in lc
        ):
            initial_col = c

        if (
            "physical fatigue" in lc
            and "final" in lc
        ):
            final_col = c

    if initial_col is None or final_col is None:
        return None, None

    initial = numeric(
        df[initial_col]
    ).dropna()

    final = numeric(
        df[final_col]
    ).dropna()

    if len(initial) == 0 or len(final) == 0:
        return None, None

    initial_value = float(
        initial.iloc[0]
    )

    final_value = float(
        final.iloc[0]
    )

    if not (
        0 <= initial_value <= 10
        and
        0 <= final_value <= 10
    ):
        return None, None

    return (
        initial_value,
        final_value
    )


# ============================================================
# SIGNAL FEATURES
# ============================================================

def signal_features(x, prefix):

    x = np.asarray(x, dtype=float)

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {}

    d = {}

    d[f"{prefix}_mean"] = np.mean(x)
    d[f"{prefix}_std"] = np.std(x)
    d[f"{prefix}_min"] = np.min(x)
    d[f"{prefix}_max"] = np.max(x)

    d[f"{prefix}_median"] = np.median(x)

    d[f"{prefix}_range"] = (
        np.max(x) -
        np.min(x)
    )

    d[f"{prefix}_rms"] = np.sqrt(
        np.mean(x ** 2)
    )

    d[f"{prefix}_q25"] = np.percentile(
        x, 25
    )

    d[f"{prefix}_q75"] = np.percentile(
        x, 75
    )

    d[f"{prefix}_abs_mean"] = np.mean(
        np.abs(x)
    )

    # First difference activity
    if len(x) > 1:

        dx = np.diff(x)

        d[f"{prefix}_diff_mean"] = np.mean(
            np.abs(dx)
        )

        d[f"{prefix}_diff_std"] = np.std(
            dx
        )

    else:

        d[f"{prefix}_diff_mean"] = 0
        d[f"{prefix}_diff_std"] = 0

    return d


# ============================================================
# PROCESS REPETITION
# ============================================================

def process_rep(path):

    try:

        df = pd.read_csv(path)

    except Exception:
        return None

    df = clean_columns(df)

    imu_cols = find_imu_columns(df)

    if len(imu_cols) != 6:
        return None

    initial, final = get_fatigue(df)

    if initial is None:
        return None

    signals = {}

    for axis, col in imu_cols.items():

        signals[axis] = numeric(
            df[col]
        ).values

    # --------------------------------------------------------
    # Remove rows with incomplete IMU
    # --------------------------------------------------------

    imu_df = pd.DataFrame(signals)

    imu_df = imu_df.dropna()

    if len(imu_df) < 20:
        return None

    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    imu_df["acc_mag"] = np.sqrt(
        imu_df["ax"] ** 2 +
        imu_df["ay"] ** 2 +
        imu_df["az"] ** 2
    )

    imu_df["gyro_mag"] = np.sqrt(
        imu_df["gx"] ** 2 +
        imu_df["gy"] ** 2 +
        imu_df["gz"] ** 2
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = {}

    for axis in [
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "acc_mag",
        "gyro_mag"
    ]:

        features.update(
            signal_features(
                imu_df[axis].values,
                axis
            )
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    features["participant"] = (
        participant_from_path(path)
    )

    features["task"] = (
        task_from_path(path)
    )

    features["initial_fatigue"] = initial
    features["final_fatigue"] = final

    # ========================================================
    # IMPORTANT TARGET
    # ========================================================
    #
    # For repetition-level prediction we use the fatigue
    # state associated with the repetition.
    #
    # The safest target here is the interpolated fatigue
    # position between Initial and Final.
    #
    # Rep1 ... RepN are assigned progressively between
    # Initial and Final.
    #
    # ========================================================

    filename = os.path.basename(path)

    m = re.search(
        r'Rep(\d+)',
        filename,
        flags=re.IGNORECASE
    )

    if m:

        rep_number = int(
            m.group(1)
        )

    else:

        rep_number = 1

    features["rep_number"] = rep_number

    # --------------------------------------------------------
    # Determine total repetitions later.
    # For now store rep number.
    # --------------------------------------------------------

    return features


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    patterns = [
        os.path.join(
            DATA_DIR,
            "**",
            "Rep*.csv"
        )
    ]

    files = []

    for pattern in patterns:

        files.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    files = sorted(
        list(set(files))
    )

    print(
        f"\nRepetition CSV files found: "
        f"{len(files)}"
    )

    rows = []

    for i, path in enumerate(files):

        row = process_rep(path)

        if row is not None:
            rows.append(row)

        if (i + 1) % 50 == 0:

            print(
                f"Processed "
                f"{i+1}/{len(files)}"
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:

        raise RuntimeError(
            "No valid repetition files found."
        )

    # ========================================================
    # CREATE FATIGUE TARGET
    # ========================================================

    df["n_reps"] = (
        df.groupby(
            ["participant", "task"]
        )["rep_number"]
        .transform("max")
    )

    # Position of each repetition.
    #
    # Rep1 is near the initial rating.
    # Final repetition is near the final rating.

    position = (
        df["rep_number"] - 1
    ) / np.maximum(
        df["n_reps"] - 1,
        1
    )

    df["fatigue"] = (
        df["initial_fatigue"] +
        position *
        (
            df["final_fatigue"] -
            df["initial_fatigue"]
        )
    )

    df["fatigue"] = np.clip(
        df["fatigue"],
        0,
        10
    )

    return df


# ============================================================
# PERSONAL BASELINE
# ============================================================

def add_personalization_features(df):

    df = df.copy()

    # Personal baseline is estimated from the earliest
    # available repetition for each participant/task.
    #
    # This is a dataset approximation of calibration.
    #
    # IMPORTANT:
    # We do NOT use fatigue itself to calculate baseline.

    baseline_cols = [
        c for c in df.columns
        if (
            c.startswith("ax_")
            or c.startswith("ay_")
            or c.startswith("az_")
            or c.startswith("gx_")
            or c.startswith("gy_")
            or c.startswith("gz_")
            or c.startswith("acc_mag_")
            or c.startswith("gyro_mag_")
        )
    ]

    # Choose baseline features from first repetition
    baseline_source = (
        df.sort_values(
            ["participant", "task", "rep_number"]
        )
        .groupby(
            ["participant", "task"]
        )
        .first()
    )

    # --------------------------------------------------------
    # Add participant/task baseline
    # --------------------------------------------------------

    for col in baseline_cols:

        base_name = (
            f"baseline_{col}"
        )

        baseline_values = (
            baseline_source[col]
        )

        df[base_name] = [
            baseline_values.get(
                (p, t),
                np.nan
            )
            for p, t in zip(
                df["participant"],
                df["task"]
            )
        ]

    # --------------------------------------------------------
    # Baseline-relative
    # --------------------------------------------------------

    relative_cols = []

    for col in baseline_cols:

        base_col = (
            f"baseline_{col}"
        )

        rel_col = (
            f"relative_{col}"
        )

        df[rel_col] = (
            df[col] -
            df[base_col]
        )

        relative_cols.append(
            rel_col
        )

    return df


# ============================================================
# MODEL
# ============================================================

def make_model(name):

    if name == "RandomForest":

        return RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if name == "XGBoost":

        return XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if name == "HistGradientBoosting":

        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        )


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(
    train_df,
    test_df,
    features,
    model_name
):

    X_train = train_df[features]
    X_test = test_df[features]

    y_train = train_df["fatigue"]
    y_test = test_df["fatigue"]

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    model = make_model(
        model_name
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    pred = np.clip(
        pred,
        0,
        10
    )

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

    return (
        mae,
        rmse,
        r2,
        pred
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\n")
    print("=" * 60)
    print("THERMORA")
    print("IMU4 REPETITION-LEVEL PERSONALIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print("\nDataset:")
    print(
        "Shape:",
        df.shape
    )

    print(
        "Participants:",
        df["participant"].nunique()
    )

    print(
        "Tasks:",
        df["task"].value_counts().to_dict()
    )

    print(
        "Fatigue range:",
        df["fatigue"].min(),
        "to",
        df["fatigue"].max()
    )

    # --------------------------------------------------------
    # Personalization
    # --------------------------------------------------------

    df = add_personalization_features(
        df
    )

    dataset_path = os.path.join(
        OUTPUT_DIR,
        "repetition_features.csv"
    )

    df.to_csv(
        dataset_path,
        index=False
    )

    # --------------------------------------------------------
    # Participant split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["participant"]
        )
    )

    train_df = df.iloc[
        train_idx
    ].copy()

    test_df = df.iloc[
        test_idx
    ].copy()

    print("\n")
    print("=" * 60)
    print("PARTICIPANT-HELD-OUT SPLIT")
    print("=" * 60)

    print(
        "Train:",
        train_df["participant"].nunique()
    )

    print(
        "Test:",
        test_df["participant"].nunique()
    )

    print(
        "Test participants:",
        sorted(
            test_df[
                "participant"
            ].unique()
        )
    )

    # --------------------------------------------------------
    # Feature groups
    # --------------------------------------------------------

    metadata = [
        "participant",
        "task",
        "initial_fatigue",
        "final_fatigue",
        "rep_number",
        "n_reps",
        "fatigue"
    ]

    all_features = [
        c for c in df.columns
        if c not in metadata
    ]

    raw_features = [
        c for c in all_features
        if not c.startswith(
            "baseline_"
        )
        and not c.startswith(
            "relative_"
        )
    ]

    baseline_features = [
        c for c in all_features
        if c.startswith(
            "baseline_"
        )
    ]

    relative_features = [
        c for c in all_features
        if c.startswith(
            "relative_"
        )
    ]

    feature_sets = {

        "RAW_IMU4":
            raw_features,

        "RAW_PLUS_BASELINE":
            raw_features +
            baseline_features,

        "BASELINE_RELATIVE":
            relative_features
    }

    results = []

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for feature_name, features in feature_sets.items():

        print("\n")
        print("=" * 60)
        print(
            feature_name,
            "|",
            len(features),
            "features"
        )
        print("=" * 60)

        for model_name in [
            "RandomForest",
            "XGBoost",
            "HistGradientBoosting"
        ]:

            print(
                "\nTraining:",
                model_name
            )

            mae, rmse, r2, pred = \
                run_experiment(
                    train_df,
                    test_df,
                    features,
                    model_name
                )

            print(
                f"MAE  : {mae:.4f}"
            )

            print(
                f"RMSE : {rmse:.4f}"
            )

            print(
                f"R²   : {r2:.4f}"
            )

            results.append({
                "feature_set":
                    feature_name,

                "model":
                    model_name,

                "n_features":
                    len(features),

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "R2":
                    r2
            })

            pred_df = test_df[
                [
                    "participant",
                    "task",
                    "rep_number",
                    "initial_fatigue",
                    "final_fatigue",
                    "fatigue"
                ]
            ].copy()

            pred_df[
                "predicted_fatigue"
            ] = pred

            pred_df.to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    f"{feature_name}_{model_name}.csv"
                ),
                index=False
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    ).sort_values(
        "MAE"
    )

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best of each
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BEST MODEL PER FEATURE SET")
    print("=" * 60)

    for feature_set in feature_sets:

        subset = results_df[
            results_df["feature_set"]
            == feature_set
        ]

        best = subset.iloc[0]

        print(
            f"{feature_set:25s}"
            f" {best['model']:25s}"
            f" MAE={best['MAE']:.4f}"
            f" R²={best['R2']:.4f}"
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    benchmark_mae = 0.900452

    best = results_df.iloc[0]

    improvement = (
        (
            benchmark_mae -
            best["MAE"]
        )
        /
        benchmark_mae
    ) * 100

    print("\n")
    print("=" * 60)
    print("BENCHMARK")
    print("=" * 60)

    print(
        f"Previous benchmark MAE : "
        f"{benchmark_mae:.4f}"
    )

    print(
        f"New best MAE           : "
        f"{best['MAE']:.4f}"
    )

    print(
        f"Difference             : "
        f"{improvement:+.2f}%"
    )

    # --------------------------------------------------------
    # Personalization improvement
    # --------------------------------------------------------

    raw = results_df[
        results_df["feature_set"]
        == "RAW_IMU4"
    ].iloc[0]

    baseline = results_df[
        results_df["feature_set"]
        == "RAW_PLUS_BASELINE"
    ].iloc[0]

    relative = results_df[
        results_df["feature_set"]
        == "BASELINE_RELATIVE"
    ].iloc[0]

    raw_to_baseline = (
        (
            raw["MAE"] -
            baseline["MAE"]
        )
        /
        raw["MAE"]
    ) * 100

    raw_to_relative = (
        (
            raw["MAE"] -
            relative["MAE"]
        )
        /
        raw["MAE"]
    ) * 100

    print("\n")
    print("=" * 60)
    print("PERSONALIZATION EFFECT")
    print("=" * 60)

    print(
        f"Raw → Raw+Baseline : "
        f"{raw_to_baseline:+.2f}%"
    )

    print(
        f"Raw → Relative     : "
        f"{raw_to_relative:+.2f}%"
    )

    print("\nResults saved to:")
    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )


if __name__ == "__main__":
    main()