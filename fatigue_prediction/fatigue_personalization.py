import os
import zipfile
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ZIP_PATH = r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction\MxD_Data_User_Study.zip"

OUTPUT_DIR = Path("fatigue_personalization_results")
EXTRACT_DIR = OUTPUT_DIR / "extracted"

RANDOM_STATE = 42

# We specifically want IMU4 = lower arm.
IMU_NUMBER = 4

# Number of initial samples used to estimate personal baseline.
# This is a prototype approximation of the 5–10 minute calibration.
BASELINE_MINUTES = 5

# Dataset sampling appears to be approximately 2 Hz.
SAMPLE_RATE = 2

BASELINE_SAMPLES = BASELINE_MINUTES * 60 * SAMPLE_RATE


# ============================================================
# UTILITY
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def clean_column_names(df):
    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]
    return df


# ============================================================
# EXTRACT ZIP
# ============================================================

def extract_dataset():

    OUTPUT_DIR.mkdir(exist_ok=True)
    EXTRACT_DIR.mkdir(exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"\nZIP file not found:\n{ZIP_PATH}\n\n"
            "Change ZIP_PATH at the top of the script."
        )

    marker = EXTRACT_DIR / ".extracted"

    if marker.exists():
        print("Dataset already extracted.")
        return

    print("\nExtracting dataset...")
    print(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    marker.touch()

    print("Extraction complete.")


# ============================================================
# DISCOVER CSV FILES
# ============================================================

def discover_csvs():

    csv_files = list(EXTRACT_DIR.rglob("*.csv"))

    print(f"\nCSV files found: {len(csv_files)}")

    if len(csv_files) == 0:
        raise RuntimeError("No CSV files found.")

    # Print examples
    for f in csv_files[:10]:
        print(" ", f)

    return csv_files


# ============================================================
# IDENTIFY PARTICIPANT
# ============================================================

def get_participant_from_path(path):

    parts = [p.lower() for p in path.parts]

    # Look for P01 / P1 / participant01 etc.
    for part in parts:

        patterns = [
            r"p(\d+)",
            r"participant[_\-]?(\d+)",
            r"subject[_\-]?(\d+)"
        ]

        for pattern in patterns:
            m = re.search(pattern, part)

            if m:
                return f"P{int(m.group(1)):02d}"

    # Search complete path
    m = re.search(
        r"(?:p|participant|subject)[_\-]?(\d+)",
        str(path),
        flags=re.IGNORECASE
    )

    if m:
        return f"P{int(m.group(1)):02d}"

    return path.parent.name


# ============================================================
# IDENTIFY TASK
# ============================================================

def get_task_from_path(path):

    text = str(path).lower()

    if "composite" in text:
        return "composite"

    if "ziptie" in text:
        return "ziptie"

    return "unknown"


# ============================================================
# FIND FATIGUE COLUMNS
# ============================================================

def find_fatigue_columns(df):

    cols = list(df.columns)

    physical_initial = None
    physical_final = None

    for c in cols:

        lc = str(c).lower()

        if "physical fatigue" in lc:

            if "initial" in lc:
                physical_initial = c

            elif "final" in lc:
                physical_final = c

    return physical_initial, physical_final


# ============================================================
# FIND IMU4 COLUMNS
# ============================================================

def find_imu4_columns(df):

    cols = list(df.columns)

    mapping = {}

    for axis in ["ax", "ay", "az", "gx", "gy", "gz"]:

        candidates = []

        for c in cols:

            lc = str(c).lower().replace(" ", "")

            # Examples:
            # IMU_4_ax
            # IMU4_ax
            # IMU_4_ax_processed

            if (
                re.search(
                    rf"imu[_\-]?{IMU_NUMBER}.*{axis}",
                    lc
                )
            ):
                candidates.append(c)

        if candidates:
            mapping[axis] = candidates[0]

    return mapping


# ============================================================
# FIND OTHER PHYSIOLOGICAL COLUMNS
# ============================================================

def find_column(df, patterns):

    for c in df.columns:

        lc = str(c).lower().replace(" ", "")

        for pattern in patterns:

            if pattern in lc:
                return c

    return None


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_signal_features(signal, prefix):

    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) == 0:
        return {}

    features = {}

    features[f"{prefix}_mean"] = np.mean(signal)
    features[f"{prefix}_std"] = np.std(signal)
    features[f"{prefix}_min"] = np.min(signal)
    features[f"{prefix}_max"] = np.max(signal)
    features[f"{prefix}_median"] = np.median(signal)

    features[f"{prefix}_range"] = (
        np.max(signal) - np.min(signal)
    )

    features[f"{prefix}_rms"] = np.sqrt(
        np.mean(signal ** 2)
    )

    features[f"{prefix}_q25"] = np.percentile(signal, 25)
    features[f"{prefix}_q75"] = np.percentile(signal, 75)

    # Mean absolute value
    features[f"{prefix}_abs_mean"] = np.mean(
        np.abs(signal)
    )

    return features


# ============================================================
# IMU MAGNITUDE
# ============================================================

def magnitude(x, y, z):

    return np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )


# ============================================================
# PROCESS ONE CSV
# ============================================================

def process_file(csv_path):

    try:

        df = pd.read_csv(csv_path)

    except Exception as e:

        print(f"Could not read {csv_path}: {e}")
        return []

    df = clean_column_names(df)

    imu_cols = find_imu4_columns(df)

    if len(imu_cols) < 6:
        return []

    fatigue_initial_col, fatigue_final_col = \
        find_fatigue_columns(df)

    if fatigue_initial_col is None or fatigue_final_col is None:

        return []

    # --------------------------------------------------------
    # Convert IMU columns
    # --------------------------------------------------------

    for axis, col in imu_cols.items():

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    imu = pd.DataFrame({
        axis: df[col]
        for axis, col in imu_cols.items()
    })

    imu = imu.replace(
        [-2, -999, -9999],
        np.nan
    )

    imu = imu.dropna()

    if len(imu) < 20:
        return []

    # --------------------------------------------------------
    # Fatigue labels
    # --------------------------------------------------------

    initial_fatigue = pd.to_numeric(
        df[fatigue_initial_col],
        errors="coerce"
    )

    final_fatigue = pd.to_numeric(
        df[fatigue_final_col],
        errors="coerce"
    )

    initial_fatigue = initial_fatigue.replace(
        [-2, -999, -9999],
        np.nan
    )

    final_fatigue = final_fatigue.replace(
        [-2, -999, -9999],
        np.nan
    )

    # Usually these ratings are repeated on every row.
    valid_initial = initial_fatigue.dropna()
    valid_final = final_fatigue.dropna()

    if len(valid_initial) == 0 or len(valid_final) == 0:
        return []

    initial_value = float(valid_initial.iloc[0])
    final_value = float(valid_final.iloc[0])

    # Safety check
    if not (0 <= initial_value <= 10):
        return []

    if not (0 <= final_value <= 10):
        return []

    # --------------------------------------------------------
    # Participant/task
    # --------------------------------------------------------

    participant = get_participant_from_path(csv_path)
    task = get_task_from_path(csv_path)

    # --------------------------------------------------------
    # Calculate movement magnitude
    # --------------------------------------------------------

    imu["acc_mag"] = magnitude(
        imu["ax"],
        imu["ay"],
        imu["az"]
    )

    imu["gyro_mag"] = magnitude(
        imu["gx"],
        imu["gy"],
        imu["gz"]
    )

    # --------------------------------------------------------
    # Personal baseline
    # --------------------------------------------------------

    baseline_n = min(
        int(BASELINE_SAMPLES),
        len(imu)
    )

    baseline = imu.iloc[:baseline_n]

    baseline_values = {}

    for axis in [
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "acc_mag",
        "gyro_mag"
    ]:

        baseline_values[
            f"baseline_{axis}_mean"
        ] = baseline[axis].mean()

        baseline_values[
            f"baseline_{axis}_std"
        ] = baseline[axis].std()

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    WINDOW_SECONDS = 30

    window_size = int(
        WINDOW_SECONDS * SAMPLE_RATE
    )

    if len(imu) < window_size:
        return []

    rows = []

    # 50% overlap
    step = window_size // 2

    for start in range(
        0,
        len(imu) - window_size + 1,
        step
    ):

        window = imu.iloc[
            start:start + window_size
        ]

        if len(window) < window_size:
            continue

        features = {}

        # ----------------------------------------------------
        # Raw IMU features
        # ----------------------------------------------------

        for axis in [
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "acc_mag",
            "gyro_mag"
        ]:

            features.update(
                extract_signal_features(
                    window[axis].values,
                    axis
                )
            )

        # ----------------------------------------------------
        # Personal baseline features
        # ----------------------------------------------------

        features.update(baseline_values)

        # ----------------------------------------------------
        # Baseline-relative features
        # ----------------------------------------------------

        for axis in [
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "acc_mag",
            "gyro_mag"
        ]:

            base_mean = baseline_values[
                f"baseline_{axis}_mean"
            ]

            base_std = baseline_values[
                f"baseline_{axis}_std"
            ]

            current_mean = window[axis].mean()
            current_std = window[axis].std()

            features[
                f"relative_{axis}_mean"
            ] = current_mean - base_mean

            features[
                f"relative_{axis}_std"
            ] = current_std - base_std

            if base_std > 1e-8:

                features[
                    f"relative_{axis}_z"
                ] = (
                    current_mean - base_mean
                ) / base_std

            else:

                features[
                    f"relative_{axis}_z"
                ] = 0.0

        # ----------------------------------------------------
        # Fatigue target
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We do NOT assume initial fatigue = 0.
        #
        # We interpolate the fatigue rating through the
        # recording based on its relative position.
        #
        # This is consistent with the dataset design where
        # Initial and Final fatigue ratings define the change.
        # ----------------------------------------------------

        relative_position = (
            start + window_size / 2
        ) / len(imu)

        relative_position = np.clip(
            relative_position,
            0,
            1
        )

        fatigue_target = (
            initial_value +
            relative_position *
            (final_value - initial_value)
        )

        fatigue_target = np.clip(
            fatigue_target,
            0,
            10
        )

        features["fatigue"] = fatigue_target

        features["participant"] = participant
        features["task"] = task

        features["initial_fatigue"] = initial_value
        features["final_fatigue"] = final_value

        rows.append(features)

    return rows


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(csv_files):

    all_rows = []

    print("\nSearching for usable IMU4 recordings...")

    for i, csv_path in enumerate(csv_files):

        rows = process_file(csv_path)

        if rows:
            all_rows.extend(rows)

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i + 1}/{len(csv_files)} "
                f"| windows={len(all_rows)}"
            )

    if not all_rows:
        raise RuntimeError(
            "\nNo usable IMU4 files were found.\n"
            "Check the dataset structure and column names."
        )

    df = pd.DataFrame(all_rows)

    print("\n==========================================")
    print("DATASET CREATED")
    print("==========================================")

    print("Shape:", df.shape)
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

    return df


# ============================================================
# PREPARE FEATURE SETS
# ============================================================

def get_feature_sets(df):

    excluded = [
        "fatigue",
        "participant",
        "task",
        "initial_fatigue",
        "final_fatigue"
    ]

    all_features = [
        c for c in df.columns
        if c not in excluded
    ]

    raw_features = [
        c for c in all_features
        if not c.startswith("baseline_")
        and not c.startswith("relative_")
    ]

    baseline_features = [
        c for c in all_features
        if c.startswith("baseline_")
    ]

    relative_features = [
        c for c in all_features
        if c.startswith("relative_")
    ]

    # A: Raw IMU
    set_A = raw_features

    # B: Raw + personal baseline
    set_B = raw_features + baseline_features

    # C: Baseline-relative
    set_C = relative_features

    return {
        "RAW_IMU4": set_A,
        "RAW_PLUS_BASELINE": set_B,
        "BASELINE_RELATIVE": set_C
    }


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test
):

    if model_name == "RandomForest":

        model = RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2
        )

    elif model_name == "XGBoost":

        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "HistGradientBoosting":

        model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        )

    else:
        raise ValueError(model_name)

    # Imputation
    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_test_imp = imputer.transform(
        X_test
    )

    model.fit(
        X_train_imp,
        y_train
    )

    predictions = model.predict(
        X_test_imp
    )

    predictions = np.clip(
        predictions,
        0,
        10
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        model,
        imputer,
        predictions,
        mae,
        rmse,
        r2
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n")
    print("==========================================")
    print("THERMORA FATIGUE PERSONALIZATION")
    print("IMU4 / LOWER ARM EXPERIMENT")
    print("==========================================")

    extract_dataset()

    csv_files = discover_csvs()

    df = build_dataset(csv_files)

    # Save extracted feature dataset
    dataset_path = (
        OUTPUT_DIR /
        "imu4_fatigue_features.csv"
    )

    df.to_csv(
        dataset_path,
        index=False
    )

    print(
        "\nFeature dataset saved:",
        dataset_path
    )

    # --------------------------------------------------------
    # Participant-held-out split
    # --------------------------------------------------------

    groups = df["participant"]

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

    train_df = df.iloc[
        train_idx
    ].copy()

    test_df = df.iloc[
        test_idx
    ].copy()

    print("\n==========================================")
    print("PARTICIPANT-HELD-OUT SPLIT")
    print("==========================================")

    print(
        "Train participants:",
        train_df["participant"].nunique()
    )

    print(
        "Test participants:",
        test_df["participant"].nunique()
    )

    print(
        "Train participants:",
        sorted(
            train_df["participant"].unique()
        )
    )

    print(
        "Test participants:",
        sorted(
            test_df["participant"].unique()
        )
    )

    feature_sets = get_feature_sets(df)

    models = [
        "RandomForest",
        "XGBoost",
        "HistGradientBoosting"
    ]

    results = []

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for feature_set_name, feature_columns in feature_sets.items():

        print("\n")
        print("==========================================")
        print(feature_set_name)
        print("Features:", len(feature_columns))
        print("==========================================")

        X_train = train_df[
            feature_columns
        ]

        y_train = train_df[
            "fatigue"
        ]

        X_test = test_df[
            feature_columns
        ]

        y_test = test_df[
            "fatigue"
        ]

        for model_name in models:

            print(
                f"\nTraining {model_name}..."
            )

            (
                model,
                imputer,
                predictions,
                mae,
                rmse,
                r2
            ) = train_model(
                model_name,
                X_train,
                y_train,
                X_test,
                y_test
            )

            print(
                f"MAE  = {mae:.4f}"
            )

            print(
                f"RMSE = {rmse:.4f}"
            )

            print(
                f"R²   = {r2:.4f}"
            )

            results.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "n_features": len(
                    feature_columns
                ),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            # Save predictions for later analysis
            prediction_df = pd.DataFrame({
                "participant": test_df[
                    "participant"
                ].values,

                "task": test_df[
                    "task"
                ].values,

                "actual_fatigue": y_test.values,

                "predicted_fatigue": predictions
            })

            prediction_file = (
                OUTPUT_DIR /
                f"{feature_set_name}_{model_name}_predictions.csv"
            )

            prediction_df.to_csv(
                prediction_file,
                index=False
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "MAE"
    )

    results_path = (
        OUTPUT_DIR /
        "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n")
    print("==========================================")
    print("FINAL RESULTS")
    print("==========================================")

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Personalization improvement
    # --------------------------------------------------------

    print("\n")
    print("==========================================")
    print("PERSONALIZATION ANALYSIS")
    print("==========================================")

    raw_best = results_df[
        results_df["feature_set"] ==
        "RAW_IMU4"
    ].sort_values("MAE").iloc[0]

    baseline_best = results_df[
        results_df["feature_set"] ==
        "RAW_PLUS_BASELINE"
    ].sort_values("MAE").iloc[0]

    relative_best = results_df[
        results_df["feature_set"] ==
        "BASELINE_RELATIVE"
    ].sort_values("MAE").iloc[0]

    print(
        f"\nRaw IMU4 best:"
        f" {raw_best['model']}"
        f" | MAE={raw_best['MAE']:.4f}"
    )

    print(
        f"Raw + baseline best:"
        f" {baseline_best['model']}"
        f" | MAE={baseline_best['MAE']:.4f}"
    )

    print(
        f"Baseline-relative best:"
        f" {relative_best['model']}"
        f" | MAE={relative_best['MAE']:.4f}"
    )

    # Improvement calculations

    raw_mae = raw_best["MAE"]

    baseline_improvement = (
        (raw_mae - baseline_best["MAE"])
        / raw_mae
    ) * 100

    relative_improvement = (
        (raw_mae - relative_best["MAE"])
        / raw_mae
    ) * 100

    print(
        f"\nRaw → Raw+Baseline improvement:"
        f" {baseline_improvement:.2f}%"
    )

    print(
        f"Raw → Baseline-relative improvement:"
        f" {relative_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    benchmark_mae = 0.900452

    best_overall = results_df.iloc[0]

    benchmark_improvement = (
        (benchmark_mae - best_overall["MAE"])
        / benchmark_mae
    ) * 100

    print("\n")
    print("==========================================")
    print("CURRENT BENCHMARK COMPARISON")
    print("==========================================")

    print(
        f"Previous benchmark MAE: "
        f"{benchmark_mae:.4f}"
    )

    print(
        f"New best MAE: "
        f"{best_overall['MAE']:.4f}"
    )

    print(
        f"Improvement vs benchmark: "
        f"{benchmark_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "previous_benchmark_MAE": benchmark_mae,

        "best_feature_set":
            best_overall["feature_set"],

        "best_model":
            best_overall["model"],

        "best_MAE":
            best_overall["MAE"],

        "best_RMSE":
            best_overall["RMSE"],

        "best_R2":
            best_overall["R2"],

        "raw_best_MAE":
            raw_best["MAE"],

        "raw_plus_baseline_MAE":
            baseline_best["MAE"],

        "baseline_relative_MAE":
            relative_best["MAE"],

        "raw_to_baseline_improvement_pct":
            baseline_improvement,

        "raw_to_relative_improvement_pct":
            relative_improvement,

        "benchmark_improvement_pct":
            benchmark_improvement
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUTPUT_DIR /
        "personalization_summary.csv",
        index=False
    )

    print("\n")
    print("==========================================")
    print("DONE")
    print("==========================================")

    print(
        "\nAll outputs saved in:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()