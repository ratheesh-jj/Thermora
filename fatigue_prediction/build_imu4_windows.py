from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction"
)

DATA_DIR = (
    BASE
    / "fatigue_personalization_results"
    / "extracted"
    / "MxD_Data_User_Study"
    / "MxD_Data_User_Study"
    / "composite"
)

TARGET_FILE = (
    BASE
    / "mxd_continuous_fatigue"
    / "continuous_fatigue_targets.csv"
)

OUT_DIR = BASE / "mxd_imu4_windows"
OUT_DIR.mkdir(exist_ok=True)


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


WINDOW_MS = 180_000       # 3 minutes
STEP_MS = 30_000          # 30-second stride


# ============================================================
# STATISTICS
# ============================================================

def add_stats(features, x, prefix):

    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(x) < 5:
        return

    features[f"{prefix}_mean"] = x.mean()
    features[f"{prefix}_std"] = x.std()
    features[f"{prefix}_min"] = x.min()
    features[f"{prefix}_max"] = x.max()
    features[f"{prefix}_median"] = x.median()

    features[f"{prefix}_range"] = (
        x.max() - x.min()
    )

    features[f"{prefix}_rms"] = np.sqrt(
        np.mean(x ** 2)
    )

    features[f"{prefix}_abs_mean"] = (
        np.abs(x).mean()
    )

    features[f"{prefix}_q25"] = (
        x.quantile(0.25)
    )

    features[f"{prefix}_q75"] = (
        x.quantile(0.75)
    )


# ============================================================
# IMU4 FEATURES
# ============================================================

def extract_features(df):

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

        add_stats(
            features,
            df[col],
            f"imu4_{axis}"
        )

    ax = pd.to_numeric(
        df[cols["ax"]],
        errors="coerce"
    )

    ay = pd.to_numeric(
        df[cols["ay"]],
        errors="coerce"
    )

    az = pd.to_numeric(
        df[cols["az"]],
        errors="coerce"
    )

    gx = pd.to_numeric(
        df[cols["gx"]],
        errors="coerce"
    )

    gy = pd.to_numeric(
        df[cols["gy"]],
        errors="coerce"
    )

    gz = pd.to_numeric(
        df[cols["gz"]],
        errors="coerce"
    )

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

    add_stats(
        features,
        acc_mag,
        "imu4_acc_mag"
    )

    add_stats(
        features,
        gyro_mag,
        "imu4_gyro_mag"
    )

    # --------------------------------------------------------
    # Dynamic acceleration
    # --------------------------------------------------------

    acc_valid = acc_mag.dropna()

    if len(acc_valid) > 2:

        diff = acc_valid.diff().dropna()

        features[
            "imu4_acc_jerk_mean"
        ] = diff.abs().mean()

        features[
            "imu4_acc_jerk_std"
        ] = diff.std()

        features[
            "imu4_acc_energy"
        ] = np.mean(
            acc_valid ** 2
        )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    target = pd.read_csv(
        TARGET_FILE
    )

    target = target.sort_values(
        [
            "participant_id",
            "Time"
        ]
    )

    rows = []

    print("=" * 70)
    print("BUILDING 3-MINUTE IMU4 WINDOWS")
    print("=" * 70)

    for participant in PARTICIPANTS:

        print(
            f"Processing {participant}"
        )

        participant_target = target[
            target["participant_id"]
            == participant
        ]

        for rep in range(1, 6):

            files = list(
                DATA_DIR.rglob(
                    f"Rep{rep}_{participant}*.csv"
                )
            )

            if len(files) != 1:
                continue

            df = pd.read_csv(
                files[0]
            )

            df["Time"] = pd.to_numeric(
                df["Time"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["Time"]
            ).sort_values(
                "Time"
            )

            if len(df) < 20:
                continue

            start = float(
                df["Time"].iloc[0]
            )

            end = float(
                df["Time"].iloc[-1]
            )

            # ------------------------------------------------
            # 3-minute windows
            # ------------------------------------------------

            window_start = start

            while (
                window_start +
                WINDOW_MS
                <= end
            ):

                window_end = (
                    window_start +
                    WINDOW_MS
                )

                mask = (
                    (df["Time"] >= window_start)
                    &
                    (df["Time"] < window_end)
                )

                window = df.loc[
                    mask
                ]

                if len(window) >= 100:

                    features = extract_features(
                        window
                    )

                    # ----------------------------------------
                    # Match continuous fatigue target
                    # ----------------------------------------

                    tm = participant_target[
                        (
                            participant_target["Time"]
                            >= window_start
                        )
                        &
                        (
                            participant_target["Time"]
                            < window_end
                        )
                    ]

                    if len(tm) > 0:

                        fatigue = tm[
                            "fatigue"
                        ].mean()

                        features[
                            "participant_id"
                        ] = participant

                        features[
                            "repetition"
                        ] = rep

                        features[
                            "window_start"
                        ] = window_start

                        features[
                            "window_end"
                        ] = window_end

                        features[
                            "fatigue"
                        ] = fatigue

                        rows.append(
                            features
                        )

                window_start += STEP_MS

    result = pd.DataFrame(
        rows
    )

    output = (
        OUT_DIR /
        "imu4_3min_windows.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    print()
    print(
        "=" * 70
    )

    print(
        "WINDOW DATASET"
    )

    print(
        "Rows:",
        len(result)
    )

    print(
        "Participants:",
        result[
            "participant_id"
        ].nunique()
    )

    print(
        "Repetitions:",
        result[
            [
                "participant_id",
                "repetition"
            ]
        ].drop_duplicates().shape[0]
    )

    print(
        "Features:",
        len(result.columns)
    )

    print(
        "Fatigue:",
        result["fatigue"].min(),
        "->",
        result["fatigue"].max()
    )

    print()
    print(
        "Saved:",
        output
    )


if __name__ == "__main__":
    main()