from pathlib import Path
import re
import numpy as np
import pandas as pd


BASE = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction"
)

DATA = (
    BASE
    / "fatigue_personalization_results"
    / "extracted"
    / "MxD_Data_User_Study"
    / "MxD_Data_User_Study"
    / "composite"
)

OUT = BASE / "mxd_continuous_fatigue"
OUT.mkdir(exist_ok=True)

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


def get_value(df, column):
    x = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    x = x.replace(
        [-2, -1],
        np.nan
    ).dropna()

    if len(x) == 0:
        return np.nan

    return float(x.iloc[0])


def get_file(pattern):

    files = list(DATA.rglob(pattern))

    if len(files) != 1:
        raise RuntimeError(
            f"{pattern}: found {len(files)} files"
        )

    return files[0]


def get_segment(participant, name):

    f = get_file(
        f"{name}_{participant}*.csv"
    )

    df = pd.read_csv(f)

    if "Time" not in df.columns:
        raise RuntimeError(
            f"Missing Time column: {f}"
        )

    initial = get_value(
        df,
        "Physical Fatigue - Initial"
    )

    final = get_value(
        df,
        "Physical Fatigue - Final"
    )

    start_time = float(
        pd.to_numeric(
            df["Time"],
            errors="coerce"
        ).dropna().iloc[0]
    )

    end_time = float(
        pd.to_numeric(
            df["Time"],
            errors="coerce"
        ).dropna().iloc[-1]
    )

    return {
        "file": str(f),
        "start_time": start_time,
        "end_time": end_time,
        "initial": initial,
        "final": final,
    }


def build_participant(participant):

    segments = []

    # Initial rest
    segments.append(
        (
            "Initial_Rest",
            get_segment(
                participant,
                "Initial_Rest"
            )
        )
    )

    # Five repetitions
    for rep in range(1, 6):

        segments.append(
            (
                f"Rep{rep}",
                get_segment(
                    participant,
                    f"Rep{rep}"
                )
            )
        )

    # Final rest
    segments.append(
        (
            "Final_Rest",
            get_segment(
                participant,
                "Final_Rest"
            )
        )
    )

    # --------------------------------------------------------
    # Build anchor points.
    #
    # The participant provides a fatigue rating at the
    # boundaries of the repeated task segments.
    #
    # We use the reported score associated with each segment
    # as the temporal anchor.
    # --------------------------------------------------------

    anchors = []

    for name, s in segments:

        if np.isnan(s["initial"]):
            continue

        # Initial rating corresponds to the beginning
        # of the segment.
        anchors.append(
            (
                s["start_time"],
                s["initial"],
                f"{name}_initial"
            )
        )

        # Final rating corresponds to the end.
        if not np.isnan(s["final"]):

            anchors.append(
                (
                    s["end_time"],
                    s["final"],
                    f"{name}_final"
                )
            )

    # Sort chronologically
    anchors.sort(
        key=lambda x: x[0]
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps.
    # --------------------------------------------------------

    clean = []

    for t, y, label in anchors:

        if not clean:

            clean.append(
                (t, y, label)
            )

        elif t > clean[-1][0]:

            clean.append(
                (t, y, label)
            )

    # --------------------------------------------------------
    # Create a continuous fatigue function.
    # --------------------------------------------------------

    times = np.array(
        [x[0] for x in clean],
        dtype=float
    )

    values = np.array(
        [x[1] for x in clean],
        dtype=float
    )

    if len(times) < 2:
        return None

    # --------------------------------------------------------
    # Generate 500-ms samples over the task.
    # --------------------------------------------------------

    full_time = np.arange(
        times[0],
        times[-1] + 250,
        500
    )

    fatigue = np.interp(
        full_time,
        times,
        values
    )

    result = pd.DataFrame({
        "participant_id": participant,
        "Time": full_time,
        "fatigue": np.clip(
            fatigue,
            0,
            10
        )
    })

    return result


def main():

    all_data = []

    print("=" * 70)
    print("BUILDING CONTINUOUS MxD FATIGUE TARGET")
    print("=" * 70)

    for participant in PARTICIPANTS:

        print(
            f"Processing {participant}"
        )

        result = build_participant(
            participant
        )

        if result is not None:
            all_data.append(
                result
            )

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    output = (
        OUT /
        "continuous_fatigue_targets.csv"
    )

    data.to_csv(
        output,
        index=False
    )

    print()
    print(
        "Rows:",
        len(data)
    )

    print(
        "Participants:",
        data["participant_id"].nunique()
    )

    print(
        "Fatigue:",
        data["fatigue"].min(),
        "->",
        data["fatigue"].max()
    )

    print()
    print(
        "Saved:",
        output
    )


if __name__ == "__main__":
    main()