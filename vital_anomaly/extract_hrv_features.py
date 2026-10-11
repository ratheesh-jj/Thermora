import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# THERMORA
# Wearable Device Dataset - HRV Feature Extraction
# ============================================================

DATASET_ROOT = (
    r"C:\Users\OMEN\Desktop\Projects\SIH26"
    r"\vital_anomaly\wearable_stress"
)

STRESS_DIR = os.path.join(
    DATASET_ROOT,
    "STRESS"
)

OUTPUT_DIR = (
    r"C:\Users\OMEN\Desktop\Projects\SIH26"
    r"\vital_anomaly\outputs"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "wearable_stress_hrv_features.csv"
)

# HRV window
WINDOW_SECONDS = 60

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET FILES
# ============================================================

def load_signal_file(path, signal_type):

    """
    Load the special CSV format used by the
    Wearable Device Dataset.

    Format examples:

    HR.csv:

        2013-06-12 16:19:08
        1.0
        85.0
        84.0
        83.67

    TEMP.csv:

        2013-06-12 16:18:58
        4.0
        30.05
        30.05

    ACC.csv:

        2013-06-12 16:18:58,
        2013-06-12 16:18:58,
        2013-06-12 16:18:58

        32.0,32.0,32.0
        -8.0,-17.0,54.0

    IBI.csv:

        2013-06-12 16:18:58,
        2013-06-12 16:18:58

        12.8125,0.921875
        13.671875,0.859375

    """

    try:

        # ----------------------------------------------------
        # Read WITHOUT headers
        # ----------------------------------------------------

        raw = pd.read_csv(
            path,
            header=None
        )

        if raw.empty:

            print(
                f"  [WARNING] Empty file: {path}"
            )

            return None

        # ----------------------------------------------------
        # First row contains starting timestamp(s)
        # ----------------------------------------------------

        first_row = raw.iloc[0]

        start_timestamp = pd.to_datetime(
            str(first_row.iloc[0]).strip(),
            errors="coerce"
        )

        if pd.isna(start_timestamp):

            print(
                f"  [WARNING] Invalid start timestamp: {path}"
            )

            print(
                "  First row:",
                first_row.tolist()
            )

            return None

        # ----------------------------------------------------
        # Remove first row
        # ----------------------------------------------------

        data = raw.iloc[
            1:
        ].copy()

        data = data.reset_index(
            drop=True
        )

        # ====================================================
        # IBI
        # ====================================================

        if signal_type == "IBI":

            if data.shape[1] < 2:

                print(
                    "  [WARNING] IBI requires 2 columns"
                )

                return None

            # Column 0 = elapsed timestamp in seconds
            # Column 1 = IBI in seconds

            elapsed = pd.to_numeric(
                data.iloc[:, 0],
                errors="coerce"
            )

            ibi = pd.to_numeric(
                data.iloc[:, 1],
                errors="coerce"
            )

            result = pd.DataFrame()

            result["timestamp"] = (
                start_timestamp
                +
                pd.to_timedelta(
                    elapsed,
                    unit="s"
                )
            )

            result["ibi"] = ibi

            result = result.dropna(
                subset=[
                    "timestamp",
                    "ibi"
                ]
            )

            return result

        # ====================================================
        # HR
        # ====================================================

        elif signal_type == "HR":

            # First value is typically a sampling/metadata value.
            # The actual HR values follow.
            #
            # HR sampling is approximately 1 Hz.
            # We construct timestamps using the sample index.

            values = pd.to_numeric(
                data.iloc[:, 0],
                errors="coerce"
            )

            values = values.dropna()

            # Remove obvious metadata first value if present.
            if len(values) > 1:

                first_value = values.iloc[0]

                # Dataset commonly starts with 1.0
                # before actual HR measurements.
                if first_value <= 5:

                    values = values.iloc[1:]

            timestamps = pd.date_range(
                start=start_timestamp,
                periods=len(values),
                freq="1s"
            )

            result = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "hr": values.values
                }
            )

            return result

        # ====================================================
        # TEMP
        # ====================================================

        elif signal_type == "TEMP":

            values = pd.to_numeric(
                data.iloc[:, 0],
                errors="coerce"
            )

            values = values.dropna()

            # Dataset contains an initial sampling/metadata
            # value such as 4.0.
            if len(values) > 1:

                first_value = values.iloc[0]

                if first_value <= 10:

                    values = values.iloc[1:]

            # Empatica temperature is approximately 4 Hz.
            timestamps = pd.date_range(
                start=start_timestamp,
                periods=len(values),
                freq="250ms"
            )

            result = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "temperature": values.values
                }
            )

            return result

        # ====================================================
        # EDA
        # ====================================================

        elif signal_type == "EDA":

            values = pd.to_numeric(
                data.iloc[:, 0],
                errors="coerce"
            )

            values = values.dropna()

            # EDA is approximately 4 Hz.
            #
            # Unlike temperature, 0 can be a valid EDA
            # measurement, so do NOT remove based on value.

            # The first value is generally a sampling/
            # metadata value in this dataset.
            if len(values) > 1:

                first_value = values.iloc[0]

                if first_value <= 10:

                    # Only remove if it looks like the
                    # known metadata/sample-rate value.
                    values = values.iloc[1:]

            timestamps = pd.date_range(
                start=start_timestamp,
                periods=len(values),
                freq="250ms"
            )

            result = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "eda": values.values
                }
            )

            return result

        # ====================================================
        # ACC
        # ====================================================

        elif signal_type == "ACC":

            if data.shape[1] < 3:

                print(
                    "  [WARNING] ACC requires 3 columns"
                )

                return None

            x = pd.to_numeric(
                data.iloc[:, 0],
                errors="coerce"
            )

            y = pd.to_numeric(
                data.iloc[:, 1],
                errors="coerce"
            )

            z = pd.to_numeric(
                data.iloc[:, 2],
                errors="coerce"
            )

            valid = (
                x.notna() &
                y.notna() &
                z.notna()
            )

            x = x[valid]
            y = y[valid]
            z = z[valid]

            # ACC is approximately 32 Hz.
            timestamps = pd.date_range(
                start=start_timestamp,
                periods=len(x),
                freq=pd.Timedelta(
                    seconds=1 / 32
                )
            )

            result = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "x": x.values,
                    "y": y.values,
                    "z": z.values
                }
            )

            return result

        else:

            print(
                f"Unknown signal type: {signal_type}"
            )

            return None

    except Exception as e:

        print(
            f"ERROR loading {path}"
        )

        print(
            e
        )

        return None


# ============================================================
# HRV CALCULATION
# ============================================================

def calculate_hrv_features(
    ibi_values
):

    rr = np.asarray(
        ibi_values,
        dtype=float
    )

    rr = rr[
        np.isfinite(rr)
    ]

    # --------------------------------------------------------
    # IBI is in seconds
    # --------------------------------------------------------

    rr_ms = rr * 1000.0

    # --------------------------------------------------------
    # Physiological filtering
    #
    # 300 ms -> 200 BPM
    # 2000 ms -> 30 BPM
    # --------------------------------------------------------

    rr_ms = rr_ms[
        (rr_ms >= 300) &
        (rr_ms <= 2000)
    ]

    # Need enough intervals
    if len(rr_ms) < 5:

        return {
            "ibi_count": len(rr_ms),
            "mean_rr_ms": np.nan,
            "mean_hr_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "pnn50_pct": np.nan,
            "hrv_range_ms": np.nan
        }

    # --------------------------------------------------------
    # Mean RR
    # --------------------------------------------------------

    mean_rr = np.mean(
        rr_ms
    )

    # --------------------------------------------------------
    # HR from RR
    # --------------------------------------------------------

    mean_hr = (
        60000.0 /
        mean_rr
    )

    # --------------------------------------------------------
    # SDNN
    # --------------------------------------------------------

    sdnn = np.std(
        rr_ms,
        ddof=1
    )

    # --------------------------------------------------------
    # Successive differences
    # --------------------------------------------------------

    diff_rr = np.diff(
        rr_ms
    )

    # --------------------------------------------------------
    # RMSSD
    # --------------------------------------------------------

    rmssd = np.sqrt(
        np.mean(
            diff_rr ** 2
        )
    )

    # --------------------------------------------------------
    # pNN50
    # --------------------------------------------------------

    pnn50 = (
        np.mean(
            np.abs(diff_rr) > 50
        )
        * 100
    )

    # --------------------------------------------------------
    # RR range
    # --------------------------------------------------------

    rr_range = (
        np.max(rr_ms)
        -
        np.min(rr_ms)
    )

    return {

        "ibi_count":
            len(rr_ms),

        "mean_rr_ms":
            mean_rr,

        "mean_hr_bpm":
            mean_hr,

        "sdnn_ms":
            sdnn,

        "rmssd_ms":
            rmssd,

        "pnn50_pct":
            pnn50,

        "hrv_range_ms":
            rr_range
    }


# ============================================================
# SIGNAL FEATURES
# ============================================================

def signal_features(
    values,
    prefix
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {

            f"{prefix}_mean": np.nan,

            f"{prefix}_std": np.nan,

            f"{prefix}_min": np.nan,

            f"{prefix}_max": np.nan,

            f"{prefix}_median": np.nan
        }

    return {

        f"{prefix}_mean":
            np.mean(values),

        f"{prefix}_std":
            np.std(values),

        f"{prefix}_min":
            np.min(values),

        f"{prefix}_max":
            np.max(values),

        f"{prefix}_median":
            np.median(values)
    }


# ============================================================
# ACCELERATION FEATURES
# ============================================================

def acceleration_features(
    acc_window
):

    if len(acc_window) == 0:

        return {}

    x = acc_window["x"].values
    y = acc_window["y"].values
    z = acc_window["z"].values

    valid = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(z)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    if len(x) == 0:

        return {}

    magnitude = np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )

    features = {}

    features.update(
        signal_features(
            x,
            "acc_x"
        )
    )

    features.update(
        signal_features(
            y,
            "acc_y"
        )
    )

    features.update(
        signal_features(
            z,
            "acc_z"
        )
    )

    features.update(
        signal_features(
            magnitude,
            "acc_mag"
        )
    )

    return features


# ============================================================
# PROCESS PARTICIPANT
# ============================================================

def process_participant(
    participant_folder
):

    participant = os.path.basename(
        participant_folder
    )

    print()
    print(
        f"Processing {participant}..."
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    paths = {

        "IBI":
            os.path.join(
                participant_folder,
                "IBI.csv"
            ),

        "HR":
            os.path.join(
                participant_folder,
                "HR.csv"
            ),

        "TEMP":
            os.path.join(
                participant_folder,
                "TEMP.csv"
            ),

        "EDA":
            os.path.join(
                participant_folder,
                "EDA.csv"
            ),

        "ACC":
            os.path.join(
                participant_folder,
                "ACC.csv"
            )
    }

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    for signal, path in paths.items():

        if not os.path.exists(path):

            print(
                f"  [SKIP] Missing {signal}.csv"
            )

            return []

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    ibi = load_signal_file(
        paths["IBI"],
        "IBI"
    )

    hr = load_signal_file(
        paths["HR"],
        "HR"
    )

    temp = load_signal_file(
        paths["TEMP"],
        "TEMP"
    )

    eda = load_signal_file(
        paths["EDA"],
        "EDA"
    )

    acc = load_signal_file(
        paths["ACC"],
        "ACC"
    )

    if any(
        x is None
        for x in [
            ibi,
            hr,
            temp,
            eda,
            acc
        ]
    ):

        print(
            "  [SKIP] Failed to load one or more signals"
        )

        return []

    # --------------------------------------------------------
    # Print sizes
    # --------------------------------------------------------

    print(
        f"  IBI rows : {len(ibi)}"
    )

    print(
        f"  HR rows  : {len(hr)}"
    )

    print(
        f"  TEMP rows: {len(temp)}"
    )

    print(
        f"  EDA rows : {len(eda)}"
    )

    print(
        f"  ACC rows : {len(acc)}"
    )

    # --------------------------------------------------------
    # Common recording range
    # --------------------------------------------------------

    start_time = max(

        ibi["timestamp"].min(),

        hr["timestamp"].min(),

        temp["timestamp"].min(),

        eda["timestamp"].min(),

        acc["timestamp"].min()
    )

    end_time = min(

        ibi["timestamp"].max(),

        hr["timestamp"].max(),

        temp["timestamp"].max(),

        eda["timestamp"].max(),

        acc["timestamp"].max()
    )

    duration = (
        end_time -
        start_time
    ).total_seconds()

    print(
        f"  Common duration: "
        f"{duration:.1f} seconds"
    )

    if duration < WINDOW_SECONDS:

        print(
            "  [SKIP] Recording too short"
        )

        return []

    # ========================================================
    # CREATE WINDOWS
    # ========================================================

    rows = []

    current = start_time

    while (
        current +
        pd.Timedelta(
            seconds=WINDOW_SECONDS
        )
        <= end_time
    ):

        window_start = current

        window_end = (
            current +
            pd.Timedelta(
                seconds=WINDOW_SECONDS
            )
        )

        # ----------------------------------------------------
        # IBI
        # ----------------------------------------------------

        ibi_window = ibi[
            (ibi["timestamp"] >= window_start) &
            (ibi["timestamp"] < window_end)
        ]

        hrv = calculate_hrv_features(
            ibi_window["ibi"].values
        )

        # ----------------------------------------------------
        # HR
        # ----------------------------------------------------

        hr_window = hr[
            (hr["timestamp"] >= window_start) &
            (hr["timestamp"] < window_end)
        ]

        hr_feat = signal_features(
            hr_window["hr"].values,
            "hr"
        )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        temp_window = temp[
            (temp["timestamp"] >= window_start) &
            (temp["timestamp"] < window_end)
        ]

        temp_feat = signal_features(
            temp_window["temperature"].values,
            "temp"
        )

        # ----------------------------------------------------
        # EDA
        # ----------------------------------------------------

        eda_window = eda[
            (eda["timestamp"] >= window_start) &
            (eda["timestamp"] < window_end)
        ]

        eda_feat = signal_features(
            eda_window["eda"].values,
            "eda"
        )

        # ----------------------------------------------------
        # ACC
        # ----------------------------------------------------

        acc_window = acc[
            (acc["timestamp"] >= window_start) &
            (acc["timestamp"] < window_end)
        ]

        acc_feat = acceleration_features(
            acc_window
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        row = {

            "participant":
                participant,

            "window_start":
                window_start,

            "window_end":
                window_end
        }

        row.update(
            hrv
        )

        row.update(
            hr_feat
        )

        row.update(
            temp_feat
        )

        row.update(
            eda_feat
        )

        row.update(
            acc_feat
        )

        rows.append(
            row
        )

        current = window_end

    print(
        f"  Windows created: "
        f"{len(rows)}"
    )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "THERMORA - Wearable Dataset HRV Feature Extraction"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Dataset: {STRESS_DIR}"
    )

    # --------------------------------------------------------
    # Find participants
    # --------------------------------------------------------

    folders = sorted(
        glob.glob(
            os.path.join(
                STRESS_DIR,
                "f*"
            )
        )
    )

    print()

    print(
        f"Participant folders found: "
        f"{len(folders)}"
    )

    if not folders:

        print(
            "ERROR: No participant folders found."
        )

        return

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    all_rows = []

    for folder in folders:

        rows = process_participant(
            folder
        )

        all_rows.extend(
            rows
        )

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if not all_rows:

        print()
        print(
            "ERROR: No features extracted."
        )

        return

    df = pd.DataFrame(
        all_rows
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Rows before HRV filtering: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # HRV validity
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "mean_rr_ms",
            "mean_hr_bpm",
            "sdnn_ms",
            "rmssd_ms"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Rows after HRV filtering: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    df = df[
        (df["mean_hr_bpm"] >= 30) &
        (df["mean_hr_bpm"] <= 200) &
        (df["mean_rr_ms"] >= 300) &
        (df["mean_rr_ms"] <= 2000) &
        (df["sdnn_ms"] >= 0) &
        (df["rmssd_ms"] >= 0) &
        (df["pnn50_pct"] >= 0) &
        (df["pnn50_pct"] <= 100)
    ].reset_index(
        drop=True
    )

    print(
        f"Rows after sanity filtering: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Final shape:"
    )

    print(
        df.shape
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if len(df) > 0:

        print()

        print(
            "Participants:"
        )

        print(
            df["participant"].nunique()
        )

        print()

        print(
            "HRV summary:"
        )

        print(
            df[
                [
                    "ibi_count",
                    "mean_rr_ms",
                    "mean_hr_bpm",
                    "sdnn_ms",
                    "rmssd_ms",
                    "pnn50_pct",
                    "hrv_range_ms"
                ]
            ].describe()
        )

        print()

        print(
            "First 5 rows:"
        )

        print(
            df.head(5).to_string(
                index=False
            )
        )

    print()

    print(
        "=" * 70
    )

    print(
        "HRV extraction complete."
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()