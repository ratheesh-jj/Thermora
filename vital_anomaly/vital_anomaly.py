from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# THERMORA VITAL ANOMALY DETECTION
# Dataset: BIDMC Numerics CSV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "bidmc_csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WINDOW_SECONDS = 10


# ------------------------------------------------------------
# 1. LOAD ALL NUMERICS FILES
# ------------------------------------------------------------

def load_dataset():

    files = sorted(DATA_DIR.glob("*_Numerics.csv"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No *_Numerics.csv files found in:\n{DATA_DIR}"
        )

    print(f"Found {len(files)} Numerics files.")

    all_data = []

    for i, file in enumerate(files, 1):

        print(f"[{i:02d}/{len(files)}] {file.name}")

        df = pd.read_csv(file)

        # Normalize column names
        df.columns = [
            c.strip().replace(" ", "_")
            for c in df.columns
        ]

        # Rename time column
        if "Time_[s]" in df.columns:
            df.rename(
                columns={"Time_[s]": "time_sec"},
                inplace=True
            )

        df["record"] = file.stem

        all_data.append(df)

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    return data


# ------------------------------------------------------------
# 2. CLEAN DATA
# ------------------------------------------------------------

def clean_data(df):

    df = df.copy()

    ranges = {
        "HR": (30, 220),
        "PULSE": (30, 220),
        "RESP": (3, 50),
        "SpO2": (70, 100),
    }

    for column, (low, high) in ranges.items():

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df.loc[
            (df[column] < low) |
            (df[column] > high),
            column
        ] = np.nan

    return df


# ------------------------------------------------------------
# 3. CREATE 10-SECOND WINDOWS
# ------------------------------------------------------------

def create_windows(df):

    rows = []

    for record, group in df.groupby("record"):

        group = group.sort_values("time_sec").copy()

        group["window"] = (
            group["time_sec"] //
            WINDOW_SECONDS
        ).astype(int)

        for window_id, window in group.groupby("window"):

            row = {
                "record": record,
                "window": window_id,
                "time_sec": window["time_sec"].mean()
            }

            for column in [
                "HR",
                "PULSE",
                "RESP",
                "SpO2"
            ]:

                if column not in window.columns:
                    row[column] = np.nan
                    continue

                values = window[column].dropna()

                if len(values) == 0:
                    row[column] = np.nan
                else:
                    row[column] = values.median()

            rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 4. PERSONAL / RECORD BASELINES
# ------------------------------------------------------------

def calculate_baselines(windows):

    baselines = {}

    for record, group in windows.groupby("record"):

        baseline = {}

        for column in [
            "HR",
            "PULSE",
            "RESP",
            "SpO2"
        ]:

            values = group[column].dropna()

            if len(values) == 0:
                continue

            median = float(values.median())

            mad = float(
                np.median(
                    np.abs(values - median)
                )
            )

            robust_std = max(
                1.4826 * mad,
                0.5
            )

            baseline[column] = {
                "median": median,
                "robust_std": robust_std,
                "samples": int(len(values))
            }

        baselines[record] = baseline

    return baselines


# ------------------------------------------------------------
# 5. DEVIATION SCORE
# ------------------------------------------------------------

def deviation_score(
    value,
    baseline,
    direction="both"
):

    if pd.isna(value) or baseline is None:
        return 0.0

    median = baseline["median"]
    std = baseline["robust_std"]

    z = (float(value) - median) / std

    if direction == "both":
        z = abs(z)

    elif direction == "high":
        z = max(0, z)

    elif direction == "low":
        z = max(0, -z)

    return float(
        np.clip(z * 25, 0, 100)
    )


# ------------------------------------------------------------
# 6. VITAL ANOMALY ENGINE
# ------------------------------------------------------------

def calculate_scores(windows, baselines):

    result = windows.copy()

    scores = []

    for idx, row in result.iterrows():

        record = row["record"]
        baseline = baselines.get(record, {})

        component_scores = {}

        # HR
        if "HR" in baseline:
            component_scores["HR"] = deviation_score(
                row["HR"],
                baseline["HR"],
                "both"
            )

        # Pulse
        if "PULSE" in baseline:
            component_scores["PULSE"] = deviation_score(
                row["PULSE"],
                baseline["PULSE"],
                "both"
            )

        # Respiration
        if "RESP" in baseline:
            component_scores["RESP"] = deviation_score(
                row["RESP"],
                baseline["RESP"],
                "both"
            )

        # SpO2:
        # Low SpO2 is more concerning than high SpO2.
        if "SpO2" in baseline:
            component_scores["SpO2"] = deviation_score(
                row["SpO2"],
                baseline["SpO2"],
                "low"
            )

        # Remove zero-information components
        valid_scores = [
            value
            for value in component_scores.values()
            if value > 0
        ]

        if not valid_scores:
            anomaly = 0.0

        else:
            # Strongest deviation gets most weight
            valid_scores = sorted(
                valid_scores,
                reverse=True
            )

            anomaly = 0.7 * valid_scores[0]

            if len(valid_scores) > 1:
                anomaly += (
                    0.3 *
                    np.mean(valid_scores[1:])
                )

        scores.append(anomaly)

    result["anomaly_score"] = np.clip(
        scores,
        0,
        100
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    result["status"] = np.select(
        [
            result["anomaly_score"] >= 60,
            result["anomaly_score"] >= 30
        ],
        [
            "CRITICAL",
            "WARNING"
        ],
        default="NORMAL"
    )

    return result


# ------------------------------------------------------------
# 7. TREND DETECTION
# ------------------------------------------------------------

def add_trends(result):

    result = result.copy()

    for record, group in result.groupby("record"):

        group = group.sort_values("time_sec")

        for column in [
            "HR",
            "PULSE",
            "RESP",
            "SpO2"
        ]:

            if column not in result.columns:
                continue

            # Approximately one-minute change
            trend = group[column].diff(6)

            result.loc[
                group.index,
                f"{column}_trend_1min"
            ] = trend.values

    return result


# ------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("THERMORA - VITAL ANOMALY DETECTION")
    print("=" * 70)

    # Load
    data = load_dataset()

    print("\nRaw dataset:")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {list(data.columns)}")

    # Clean
    data = clean_data(data)

    print("\nCleaned vital statistics:")

    print(
        data[
            ["HR", "PULSE", "RESP", "SpO2"]
        ].describe()
    )

    # Windows
    windows = create_windows(data)

    print(
        f"\nCreated {len(windows):,} "
        f"10-second windows."
    )

    # Baselines
    baselines = calculate_baselines(
        windows
    )

    # Anomaly scoring
    result = calculate_scores(
        windows,
        baselines
    )

    # Trends
    result = add_trends(result)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    cleaned_path = (
        OUTPUT_DIR /
        "bidmc_clean_vitals.csv"
    )

    windows_path = (
        OUTPUT_DIR /
        "bidmc_vital_windows.csv"
    )

    scores_path = (
        OUTPUT_DIR /
        "thermora_vital_anomaly_scores.csv"
    )

    baseline_path = (
        OUTPUT_DIR /
        "personal_baselines.json"
    )

    data.to_csv(
        cleaned_path,
        index=False
    )

    windows.to_csv(
        windows_path,
        index=False
    )

    result.to_csv(
        scores_path,
        index=False
    )

    with open(
        baseline_path,
        "w"
    ) as f:
        json.dump(
            baselines,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("THERMORA RESULTS")
    print("=" * 70)

    print(
        f"\nRaw samples       : {len(data):,}"
    )

    print(
        f"10-sec windows    : {len(windows):,}"
    )

    print(
        f"Records           : "
        f"{data['record'].nunique()}"
    )

    print("\nStatus distribution:")

    print(
        result["status"]
        .value_counts()
        .to_string()
    )

    print("\nAnomaly score statistics:")

    print(
        result["anomaly_score"].describe()
    )

    print("\nExample highest-risk windows:")

    print(
        result[
            [
                "record",
                "time_sec",
                "HR",
                "PULSE",
                "RESP",
                "SpO2",
                "anomaly_score",
                "status"
            ]
        ]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nFiles saved:")

    print(cleaned_path)
    print(windows_path)
    print(scores_path)
    print(baseline_path)

    print(
        "\nTHERMORA vital anomaly pipeline complete."
    )


if __name__ == "__main__":
    main()