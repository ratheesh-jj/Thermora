"""
THERMORA UNIFIED RISK ENGINE
============================

Combines:
    1. Thermal risk
    2. Fatigue risk
    3. Fall risk

into one worker-safety decision.

Prototype / research use only.
Not a medical diagnostic system.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\OMEN\Desktop\Projects\SIH26")

THERMAL_FILE = (
    BASE_DIR
    / "results"
    / "thermal_risk_temporal_hysteresis_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "unified_risk_results.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "results"
    / "unified_risk_summary.csv"
)


# ============================================================
# RISK LEVELS
# ============================================================

RISK_VALUE = {
    "SAFE": 0,
    "CAUTION": 1,
    "HIGH": 2,
    "CRITICAL": 3,
    "EMERGENCY": 4,
}


def normalize_risk(value):
    """
    Convert different risk representations into:
        SAFE / CAUTION / HIGH / CRITICAL / EMERGENCY
    """

    if pd.isna(value):
        return "SAFE"

    if isinstance(value, str):
        v = value.strip().upper()

        if v in RISK_VALUE:
            return v

        # Handle common variants
        if "EMERGENCY" in v:
            return "EMERGENCY"

        if "CRITICAL" in v:
            return "CRITICAL"

        if "HIGH" in v:
            return "HIGH"

        if "CAUTION" in v or "MODERATE" in v:
            return "CAUTION"

        if "SAFE" in v or "LOW" in v:
            return "SAFE"

    # Numeric fallback
    try:
        x = float(value)

        if x >= 3:
            return "CRITICAL"
        elif x >= 2:
            return "HIGH"
        elif x >= 1:
            return "CAUTION"
        else:
            return "SAFE"

    except Exception:
        return "SAFE"


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, candidates, required=False):
    """
    Find a column using several possible names.
    """

    # Exact match first
    for c in candidates:
        if c in df.columns:
            return c

    # Case-insensitive match
    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:
        if str(c).lower() in lower_map:
            return lower_map[str(c).lower()]

    if required:
        raise ValueError(
            f"Required column not found.\n"
            f"Tried: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


# ============================================================
# LOAD THERMAL DATA
# ============================================================

def load_thermal():

    print("=" * 70)
    print("LOADING THERMAL RESULTS")
    print("=" * 70)

    if not THERMAL_FILE.exists():
        raise FileNotFoundError(
            f"Thermal results not found:\n{THERMAL_FILE}"
        )

    df = pd.read_csv(THERMAL_FILE)

    print(f"Thermal rows: {len(df)}")
    print(f"Thermal columns: {len(df.columns)}")

    return df


# ============================================================
# ADD PLACEHOLDER FATIGUE / FALL INPUTS
# ============================================================

def add_demo_inputs(df):

    """
    Temporary integration layer.

    Until we connect the real fatigue and fall inference
    pipelines, this creates demonstration inputs.

    IMPORTANT:
    These are NOT ML predictions.

    Replace this section later with:
        fatigue_model.predict(...)
        fall_model.predict_proba(...)
    """

    print()
    print("=" * 70)
    print("ADDING DEMONSTRATION FATIGUE / FALL INPUTS")
    print("=" * 70)

    # --------------------------------------------------------
    # FATIGUE
    # --------------------------------------------------------
    #
    # For now use a deterministic placeholder based on
    # thermal/physiological stress.
    #
    # This allows us to test the fusion engine itself.
    #

    phys_col = find_column(
        df,
        [
            "physiological_score",
            "phys_score",
            "thermal_physiological_score",
        ]
    )

    if phys_col:

        phys = pd.to_numeric(
            df[phys_col],
            errors="coerce"
        ).fillna(0)

        fatigue_score = np.clip(
            phys * 1.1,
            0,
            10
        )

    else:

        fatigue_score = pd.Series(
            0.0,
            index=df.index
        )

    df["fatigue_score"] = fatigue_score

    def fatigue_risk(x):

        if x >= 7:
            return "HIGH"

        if x >= 4:
            return "CAUTION"

        return "SAFE"

    df["fatigue_risk"] = (
        df["fatigue_score"]
        .apply(fatigue_risk)
    )

    # --------------------------------------------------------
    # FALL
    # --------------------------------------------------------
    #
    # No actual fall inference yet.
    #

    df["fall_probability"] = 0.0
    df["fall_detected"] = False
    df["fall_risk"] = "SAFE"

    print("Fatigue/fall demo inputs added.")
    print()
    print(
        "WARNING: fatigue and fall values are currently "
        "placeholders for integration testing."
    )

    return df


# ============================================================
# THERMAL COLUMN
# ============================================================

def identify_thermal_risk(df):

    thermal_col = find_column(
        df,
        [
            "final_risk",
            "risk_level",
            "thermal_risk",
            "risk",
        ],
        required=True
    )

    print(f"Thermal risk column: {thermal_col}")

    df["thermal_risk"] = (
        df[thermal_col]
        .apply(normalize_risk)
    )

    return df


# ============================================================
# UNIFIED SCORE
# ============================================================

def calculate_unified_score(row):

    thermal = RISK_VALUE[
        normalize_risk(row["thermal_risk"])
    ]

    fatigue = RISK_VALUE[
        normalize_risk(row["fatigue_risk"])
    ]

    fall = RISK_VALUE[
        normalize_risk(row["fall_risk"])
    ]

    # Normalize each risk to 0–1
    thermal_n = thermal / 3.0
    fatigue_n = fatigue / 3.0
    fall_n = fall / 3.0

    # Weighted score
    score = (
        0.50 * thermal_n
        + 0.30 * fatigue_n
        + 0.20 * fall_n
    )

    return round(score * 100, 2)


# ============================================================
# UNIFIED DECISION
# ============================================================

def determine_unified_risk(row):

    thermal = normalize_risk(row["thermal_risk"])
    fatigue = normalize_risk(row["fatigue_risk"])
    fall = normalize_risk(row["fall_risk"])

    fall_detected = bool(row["fall_detected"])

    # ========================================================
    # EMERGENCY OVERRIDE
    # ========================================================

    if fall_detected or fall == "EMERGENCY":
        return "EMERGENCY"

    # ========================================================
    # THERMAL CRITICAL
    # ========================================================

    if thermal == "CRITICAL":
        return "CRITICAL"

    # ========================================================
    # MULTI-FACTOR HIGH
    # ========================================================

    if thermal == "HIGH" and fatigue == "HIGH":
        return "HIGH"

    if thermal == "HIGH":
        return "HIGH"

    if fatigue == "HIGH":
        return "HIGH"

    # ========================================================
    # CAUTION
    # ========================================================

    if thermal == "CAUTION":
        return "CAUTION"

    if fatigue == "CAUTION":
        return "CAUTION"

    if fall == "CAUTION":
        return "CAUTION"

    return "SAFE"


# ============================================================
# ACTION
# ============================================================

def determine_action(risk):

    actions = {

        "SAFE":
            "Continue work",

        "CAUTION":
            "Monitor condition",

        "HIGH":
            "Recommend break",

        "CRITICAL":
            "Immediate rest and supervisor notification",

        "EMERGENCY":
            "Emergency response - check worker immediately",
    }

    return actions.get(
        risk,
        "Monitor worker"
    )


# ============================================================
# ALERTS
# ============================================================

def wristband_alert(risk):

    return risk in [
        "HIGH",
        "CRITICAL",
        "EMERGENCY"
    ]


def supervisor_alert(risk):

    return risk in [
        "CRITICAL",
        "EMERGENCY"
    ]


# ============================================================
# PROCESS
# ============================================================

def process(df):

    print()
    print("=" * 70)
    print("RUNNING UNIFIED RISK FUSION")
    print("=" * 70)

    df = identify_thermal_risk(df)

    df["unified_risk_score"] = (
        df.apply(
            calculate_unified_score,
            axis=1
        )
    )

    df["unified_risk"] = (
        df.apply(
            determine_unified_risk,
            axis=1
        )
    )

    df["recommended_action"] = (
        df["unified_risk"]
        .apply(determine_action)
    )

    df["wristband_alert"] = (
        df["unified_risk"]
        .apply(wristband_alert)
    )

    df["supervisor_alert"] = (
        df["unified_risk"]
        .apply(supervisor_alert)
    )

    return df


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(df):

    print()
    print("=" * 70)
    print("THERMORA UNIFIED RISK DISTRIBUTION")
    print("=" * 70)

    distribution = (
        df["unified_risk"]
        .value_counts()
        .reindex(
            [
                "SAFE",
                "CAUTION",
                "HIGH",
                "CRITICAL",
                "EMERGENCY"
            ],
            fill_value=0
        )
    )

    total = len(df)

    rows = []

    for risk, count in distribution.items():

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{risk:<12} "
            f"{count:>6} "
            f"({percentage:6.2f}%)"
        )

        rows.append({
            "risk_level": risk,
            "rows": count,
            "percentage": round(
                percentage,
                2
            )
        })

    print()
    print(
        f"Mean unified risk score: "
        f"{df['unified_risk_score'].mean():.2f}/100"
    )

    print(
        f"Wristband alerts: "
        f"{df['wristband_alert'].sum()}"
    )

    print(
        f"Supervisor alerts: "
        f"{df['supervisor_alert'].sum()}"
    )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("THERMORA UNIFIED RISK ENGINE")
    print("=" * 70)

    df = load_thermal()

    df = add_demo_inputs(df)

    df = process(df)

    summary = generate_summary(df)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)

    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE UNIFIED DECISIONS")
    print("=" * 70)

    cols = [
        c for c in [
            "participant_id",
            "condition_id",
            "input_end_time",
            "estimated_current_core_temp",
            "predicted_core_temp",
            "temperature_rise",
            "thermal_risk",
            "fatigue_score",
            "fatigue_risk",
            "fall_probability",
            "fall_risk",
            "unified_risk_score",
            "unified_risk",
            "recommended_action",
        ]
        if c in df.columns
    ]

    print(
        df[cols]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()