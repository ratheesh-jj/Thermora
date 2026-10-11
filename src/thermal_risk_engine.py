import pandas as pd
import numpy as np
import json

from pathlib import Path


# ============================================================
# THERMORA
# TEMPORAL-AWARE MULTI-FACTOR THERMAL RISK ENGINE
# ============================================================
#
# Features:
#   1. Physiological risk
#   2. Environmental stress
#   3. Activity/work state
#   4. Persistence
#   5. Hysteresis
#   6. Time/sequence awareness
#   7. Emergency overrides
#
# IMPORTANT:
# This is a prototype decision engine.
# It is NOT a clinical heat-stress diagnostic system.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    BASE_DIR
    / "results"
    / "integrated_10min_no_wind_test_predictions.csv"
)

ENVIRONMENT_FILE = (
    BASE_DIR
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "thermal_risk_temporal_hysteresis_results.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "results"
    / "thermal_risk_temporal_hysteresis_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Physiological thresholds
# ------------------------------------------------------------

SAFE_CORE_MAX = 38.0
CAUTION_CORE_MAX = 38.4
HIGH_CORE_MAX = 38.8


# ------------------------------------------------------------
# Predicted temperature rise
# ------------------------------------------------------------

CAUTION_RISE = 0.20
HIGH_RISE = 0.30
CRITICAL_RISE = 0.40


# ------------------------------------------------------------
# Temporal persistence
# ------------------------------------------------------------

ESCALATION_COUNT = 2

DEESCALATION_COUNT = 2


# Maximum allowed time gap between consecutive observations
# for persistence to continue.
#
# Example:
# 17 -> 19 -> 21 minutes
# is continuous if the data interval is <= this value.
#
# If a large gap occurs, the state machine resets.
# ------------------------------------------------------------

MAX_CONTINUOUS_GAP_MIN = 5.0


# ------------------------------------------------------------
# Emergency overrides
# ------------------------------------------------------------

EMERGENCY_CORE_TEMP = 39.0

EMERGENCY_RISE = 0.60


# ============================================================
# RISK LEVEL ORDER
# ============================================================

RISK_ORDER = {
    "SAFE": 0,
    "CAUTION": 1,
    "HIGH": 2,
    "CRITICAL": 3
}


def risk_name(level):

    reverse = {
        0: "SAFE",
        1: "CAUTION",
        2: "HIGH",
        3: "CRITICAL"
    }

    return reverse[level]


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

def safe_float(value, default=np.nan):

    try:

        if pd.isna(value):

            return default

        return float(value)

    except:

        return default


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):

        return ""

    return str(value).strip().lower()


# ============================================================
# ENVIRONMENTAL STRESS SCORE
# ============================================================
#
# Missing values DO NOT automatically increase risk.
#
# This is important because:
#
#   missing wind != zero wind
#   missing humidity != zero humidity
# ============================================================

def environmental_score(
    ambient_temp,
    humidity,
    wind,
    solar
):

    score = 0


    # --------------------------------------------------------
    # Ambient temperature
    # --------------------------------------------------------

    if not np.isnan(ambient_temp):

        if ambient_temp >= 40:

            score += 3

        elif ambient_temp >= 37:

            score += 2

        elif ambient_temp >= 34:

            score += 1


    # --------------------------------------------------------
    # Humidity
    # --------------------------------------------------------

    if not np.isnan(humidity):

        if humidity >= 70:

            score += 2

        elif humidity >= 55:

            score += 1


    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------
    #
    # Only apply wind modifier when wind is actually available.
    # --------------------------------------------------------

    if not np.isnan(wind):

        if wind <= 0.5:

            score += 1


    # --------------------------------------------------------
    # Solar radiation
    # --------------------------------------------------------

    if not np.isnan(solar):

        if solar >= 700:

            score += 2

        elif solar >= 400:

            score += 1


    return min(score, 5)


# ============================================================
# PHYSIOLOGICAL SCORE
# ============================================================

def physiological_score(
    estimated_core,
    predicted_core,
    rise
):

    score = 0


    # --------------------------------------------------------
    # Predicted core temperature
    # --------------------------------------------------------

    if predicted_core >= HIGH_CORE_MAX:

        score += 5

    elif predicted_core >= CAUTION_CORE_MAX:

        score += 3

    elif predicted_core >= SAFE_CORE_MAX:

        score += 1


    # --------------------------------------------------------
    # Predicted rise
    # --------------------------------------------------------

    if rise >= CRITICAL_RISE:

        score += 3

    elif rise >= HIGH_RISE:

        score += 2

    elif rise >= CAUTION_RISE:

        score += 1


    # --------------------------------------------------------
    # Current estimated core
    # --------------------------------------------------------

    if estimated_core >= HIGH_CORE_MAX:

        score += 4

    elif estimated_core >= CAUTION_CORE_MAX:

        score += 2

    elif estimated_core >= SAFE_CORE_MAX:

        score += 1


    return min(score, 10)


# ============================================================
# ACTIVITY / WORK SCORE
# ============================================================

def activity_work_score(
    activity,
    work_rest
):

    score = 0


    activity_text = normalize_text(
        activity
    )

    work_text = normalize_text(
        work_rest
    )


    # --------------------------------------------------------
    # Work state
    # --------------------------------------------------------

    if (

        work_text == "work"

        or

        work_text == "1"

        or

        "work" in work_text

    ):

        score += 1


    # --------------------------------------------------------
    # Numeric activity code
    # --------------------------------------------------------

    try:

        activity_numeric = float(
            activity
        )

        if activity_numeric >= 3:

            score += 1

    except:

        pass


    # --------------------------------------------------------
    # Text activity
    # --------------------------------------------------------

    active_words = [

        "walk",
        "walking",
        "run",
        "running",
        "exercise",
        "active",
        "heavy",
        "high",
        "work"
    ]


    if any(
        word in activity_text
        for word in active_words
    ):

        score += 1


    return min(score, 2)


# ============================================================
# RAW RISK CLASSIFICATION
# ============================================================

def classify_raw_risk(

    estimated_core,
    predicted_core,
    rise,

    phys_score,
    env_score,
    activity_score

):

    # ========================================================
    # CRITICAL OVERRIDES
    # ========================================================

    if (
        predicted_core >= EMERGENCY_CORE_TEMP
    ):

        return "CRITICAL"


    if (
        rise >= EMERGENCY_RISE
    ):

        return "CRITICAL"


    if (

        predicted_core >= HIGH_CORE_MAX

        and

        rise >= HIGH_RISE

    ):

        return "CRITICAL"


    if (

        estimated_core >= HIGH_CORE_MAX

        and

        predicted_core >= HIGH_CORE_MAX

    ):

        return "CRITICAL"


    # ========================================================
    # HIGH
    # ========================================================

    if (
        predicted_core >= HIGH_CORE_MAX
    ):

        return "HIGH"


    if (

        rise >= CRITICAL_RISE

        and

        (
            env_score >= 3
            or
            activity_score >= 1
        )

    ):

        return "HIGH"


    if (

        phys_score >= 6

        and

        env_score >= 3

    ):

        return "HIGH"


    if (
        phys_score >= 7
    ):

        return "HIGH"


    # ========================================================
    # CAUTION
    # ========================================================

    if (
        predicted_core >= CAUTION_CORE_MAX
    ):

        return "CAUTION"


    if (

        rise >= HIGH_RISE

        and

        env_score >= 2

    ):

        return "CAUTION"


    if (

        phys_score >= 3

        and

        env_score >= 2

    ):

        return "CAUTION"


    if (

        env_score >= 5

        and

        activity_score >= 1

        and

        rise >= CAUTION_RISE

    ):

        return "CAUTION"


    # ========================================================
    # SAFE
    # ========================================================

    return "SAFE"


# ============================================================
# TEMPORAL PERSISTENCE + HYSTERESIS
# ============================================================
#
# IMPORTANT DIFFERENCE FROM PREVIOUS VERSION:
#
# State is reset when:
#
#   - participant changes
#   - condition changes
#   - time goes backwards
#   - time gap becomes too large
#
# Therefore HIGH from one experiment cannot leak into another.
# ============================================================

def apply_temporal_hysteresis(

    group

):

    group = group.sort_values(
        "time_min"
    ).copy()


    final_states = []

    state_reasons = []


    current_level = 0

    escalation_counter = 0
    deescalation_counter = 0


    previous_time = None


    for _, row in group.iterrows():

        current_time = safe_float(
            row["time_min"]
        )

        raw_state = row[
            "raw_risk"
        ]

        raw_level = RISK_ORDER[
            raw_state
        ]


        predicted_core = safe_float(
            row[
                "predicted_future_core_temp_c"
            ],
            default=np.nan
        )


        rise = safe_float(
            row[
                "predicted_rise_c"
            ],
            default=np.nan
        )


        # ====================================================
        # CHECK TEMPORAL CONTINUITY
        # ====================================================

        continuous = True


        if previous_time is not None:

            gap = (
                current_time
                - previous_time
            )


            if (
                gap < 0
                or
                gap > MAX_CONTINUOUS_GAP_MIN
            ):

                continuous = False


        # ----------------------------------------------------
        # Reset if sequence is not continuous
        # ----------------------------------------------------

        if not continuous:

            current_level = 0

            escalation_counter = 0

            deescalation_counter = 0


            state_reasons.append(
                "SEQUENCE_RESET"
            )


        # ====================================================
        # EMERGENCY OVERRIDE
        # ====================================================

        emergency = False


        if not np.isnan(
            predicted_core
        ):

            if (
                predicted_core
                >= EMERGENCY_CORE_TEMP
            ):

                emergency = True


        if not np.isnan(
            rise
        ):

            if (
                rise
                >= EMERGENCY_RISE
            ):

                emergency = True


        if emergency:

            current_level = 3

            escalation_counter = 0

            deescalation_counter = 0


            final_states.append(
                "CRITICAL"
            )


            state_reasons.append(
                "EMERGENCY_OVERRIDE"
            )


            previous_time = current_time

            continue


        # ====================================================
        # ESCALATION
        # ====================================================

        if raw_level > current_level:

            escalation_counter += 1

            deescalation_counter = 0


            if (
                escalation_counter
                >= ESCALATION_COUNT
            ):

                current_level = raw_level

                escalation_counter = 0

                state_reasons.append(
                    "ESCALATED_AFTER_PERSISTENCE"
                )

            else:

                state_reasons.append(
                    "ESCALATION_PENDING"
                )


        # ====================================================
        # DE-ESCALATION
        # ====================================================

        elif raw_level < current_level:

            deescalation_counter += 1

            escalation_counter = 0


            if (
                deescalation_counter
                >= DEESCALATION_COUNT
            ):

                current_level = raw_level

                deescalation_counter = 0

                state_reasons.append(
                    "DEESCALATED_AFTER_PERSISTENCE"
                )

            else:

                state_reasons.append(
                    "DEESCALATION_PENDING"
                )


        # ====================================================
        # SAME LEVEL
        # ====================================================

        else:

            escalation_counter = 0

            deescalation_counter = 0

            state_reasons.append(
                "STATE_MAINTAINED"
            )


        final_states.append(
            risk_name(
                current_level
            )
        )


        previous_time = current_time


    group[
        "thermal_risk"
    ] = final_states


    group[
        "state_transition_reason"
    ] = state_reasons


    return group


# ============================================================
# ALERT ACTION
# ============================================================

def alert_action(risk):

    if risk == "CRITICAL":

        return "EMERGENCY_ALERT"

    if risk == "HIGH":

        return "IMMEDIATE_BREAK_AND_ALERT"

    if risk == "CAUTION":

        return "BREAK_SOON"

    return "NORMAL"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("THERMORA TEMPORAL-AWARE THERMAL RISK ENGINE")
print("PERSISTENCE + HYSTERESIS + SEQUENCE VALIDATION")
print("=" * 70)


print()
print("Loading prediction data...")


pred = pd.read_csv(
    PREDICTION_FILE
)


print(
    f"Prediction rows: "
    f"{len(pred)}"
)


# ============================================================
# REQUIRED PREDICTION COLUMNS
# ============================================================

required_prediction_columns = [

    "participant_id",

    "condition_id",

    "time_min",

    "estimated_core_temp_c",

    "predicted_future_core_temp_c"
]


missing = [

    c

    for c in required_prediction_columns

    if c not in pred.columns
]


if missing:

    raise ValueError(
        "Missing prediction columns: "
        + str(missing)
    )


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

print()
print("Loading environmental data...")


env = pd.read_csv(
    ENVIRONMENT_FILE
)


print(
    f"Environmental rows: "
    f"{len(env)}"
)


# ============================================================
# REQUIRED ENVIRONMENT COLUMNS
# ============================================================

required_env_columns = [

    "participant_id",

    "condition_id",

    "time_min",

    "ambient_temp_c",

    "humidity_pct",

    "wind_mps",

    "solar_radiation_w_m2",

    "activity_code",

    "work_rest"
]


missing_env = [

    c

    for c in required_env_columns

    if c not in env.columns
]


if missing_env:

    raise ValueError(
        "Missing environmental columns: "
        + str(missing_env)
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for dataframe in [
    pred,
    env
]:

    dataframe[
        "participant_id"
    ] = pd.to_numeric(
        dataframe[
            "participant_id"
        ],
        errors="coerce"
    )


    dataframe[
        "condition_id"
    ] = pd.to_numeric(
        dataframe[
            "condition_id"
        ],
        errors="coerce"
    )


    dataframe[
        "time_min"
    ] = pd.to_numeric(
        dataframe[
            "time_min"
        ],
        errors="coerce"
    )


# ============================================================
# ENVIRONMENT SUBSET
# ============================================================

env_small = env[
    [
        "participant_id",
        "condition_id",
        "time_min",
        "ambient_temp_c",
        "humidity_pct",
        "wind_mps",
        "solar_radiation_w_m2",
        "activity_code",
        "work_rest"
    ]
].copy()


# ============================================================
# MERGE
# ============================================================

print()
print(
    "Merging prediction + environmental data..."
)


merged = pred.merge(

    env_small,

    on=[
        "participant_id",
        "condition_id",
        "time_min"
    ],

    how="left"
)


print(
    f"Merged rows: "
    f"{len(merged)}"
)


# ============================================================
# CALCULATE PREDICTED RISE
# ============================================================

merged[
    "predicted_rise_c"
] = (

    merged[
        "predicted_future_core_temp_c"
    ]

    -

    merged[
        "estimated_core_temp_c"
    ]
)


# ============================================================
# CALCULATE SCORES
# ============================================================

print()
print(
    "Calculating multi-factor risk scores..."
)


phys_scores = []
env_scores = []
activity_scores = []
raw_states = []


for _, row in merged.iterrows():

    estimated_core = safe_float(
        row[
            "estimated_core_temp_c"
        ],
        default=np.nan
    )


    predicted_core = safe_float(
        row[
            "predicted_future_core_temp_c"
        ],
        default=np.nan
    )


    rise = safe_float(
        row[
            "predicted_rise_c"
        ],
        default=np.nan
    )


    ambient = safe_float(
        row[
            "ambient_temp_c"
        ],
        default=np.nan
    )


    humidity = safe_float(
        row[
            "humidity_pct"
        ],
        default=np.nan
    )


    wind = safe_float(
        row[
            "wind_mps"
        ],
        default=np.nan
    )


    solar = safe_float(
        row[
            "solar_radiation_w_m2"
        ],
        default=np.nan
    )


    activity = row[
        "activity_code"
    ]

    work_rest = row[
        "work_rest"
    ]


    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    phys = physiological_score(

        estimated_core,

        predicted_core,

        rise
    )


    env_score = environmental_score(

        ambient,

        humidity,

        wind,

        solar
    )


    activity_score = (
        activity_work_score(
            activity,
            work_rest
        )
    )


    # --------------------------------------------------------
    # Raw classification
    # --------------------------------------------------------

    raw_state = classify_raw_risk(

        estimated_core,

        predicted_core,

        rise,

        phys,

        env_score,

        activity_score
    )


    phys_scores.append(
        phys
    )

    env_scores.append(
        env_score
    )

    activity_scores.append(
        activity_score
    )

    raw_states.append(
        raw_state
    )


merged[
    "physiological_score"
] = phys_scores


merged[
    "environmental_score"
] = env_scores


merged[
    "activity_work_score"
] = activity_scores


merged[
    "raw_risk"
] = raw_states


# ============================================================
# TEMPORAL SORT
# ============================================================

merged = merged.sort_values(

    [
        "participant_id",
        "condition_id",
        "time_min"
    ]

).reset_index(
    drop=True
)


# ============================================================
# APPLY TEMPORAL HYSTERESIS
# ============================================================

print()
print(
    "Applying sequence-aware persistence + hysteresis..."
)


processed_groups = []


for (
    participant_id,
    condition_id
), group in merged.groupby(

    [
        "participant_id",
        "condition_id"
    ],

    sort=False

):

    processed = apply_temporal_hysteresis(
        group
    )


    processed_groups.append(
        processed
    )


merged = pd.concat(
    processed_groups,
    ignore_index=True
)


# ============================================================
# ALERT ACTION
# ============================================================

merged[
    "recommended_action"
] = merged[
    "thermal_risk"
].apply(
    alert_action
)


# ============================================================
# FINAL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("FINAL THERMAL RISK DISTRIBUTION")
print("=" * 70)


counts = merged[
    "thermal_risk"
].value_counts()


for level in [

    "SAFE",
    "CAUTION",
    "HIGH",
    "CRITICAL"

]:

    count = int(
        counts.get(
            level,
            0
        )
    )


    percentage = (
        count
        / len(merged)
        * 100
    )


    print(
        f"{level:<10}"
        f"{count:>6} rows "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# RAW VS FINAL
# ============================================================

changed = (

    merged[
        "raw_risk"
    ]

    !=

    merged[
        "thermal_risk"
    ]
)


changed_count = int(
    changed.sum()
)


print()
print(
    f"Rows changed by persistence/"
    f"hysteresis: {changed_count}"
)


print(
    f"Percentage changed: "
    f"{changed_count / len(merged) * 100:.2f}%"
)


# ============================================================
# TRANSITION REASONS
# ============================================================

print()
print("=" * 70)
print("STATE TRANSITION REASONS")
print("=" * 70)


reason_counts = (
    merged[
        "state_transition_reason"
    ]
    .value_counts()
)


for reason, count in (
    reason_counts.items()
):

    print(
        f"{reason:<35}"
        f"{int(count):>6}"
    )


# ============================================================
# THERMAL STATISTICS
# ============================================================

print()
print("=" * 70)
print("THERMAL STATISTICS")
print("=" * 70)


print(
    f"Mean estimated core: "
    f"{merged['estimated_core_temp_c'].mean():.3f} °C"
)


print(
    f"Mean predicted +10 min core: "
    f"{merged['predicted_future_core_temp_c'].mean():.3f} °C"
)


print(
    f"Mean predicted rise: "
    f"{merged['predicted_rise_c'].mean():.3f} °C"
)


print(
    f"Maximum predicted core: "
    f"{merged['predicted_future_core_temp_c'].max():.3f} °C"
)


print(
    f"Maximum predicted rise: "
    f"{merged['predicted_rise_c'].max():.3f} °C"
)


print(
    f"Mean physiological score: "
    f"{merged['physiological_score'].mean():.2f}"
)


print(
    f"Mean environmental score: "
    f"{merged['environmental_score'].mean():.2f}"
)


print(
    f"Mean activity/work score: "
    f"{merged['activity_work_score'].mean():.2f}"
)


# ============================================================
# ENVIRONMENTAL STATISTICS
# ============================================================

print()
print("=" * 70)
print("ENVIRONMENTAL STATISTICS")
print("=" * 70)


print(
    f"Mean ambient temperature: "
    f"{merged['ambient_temp_c'].mean():.2f} °C"
)


print(
    f"Mean humidity: "
    f"{merged['humidity_pct'].mean():.2f}%"
)


print(
    f"Mean wind speed: "
    f"{merged['wind_mps'].mean():.2f} m/s"
)


print(
    f"Mean solar radiation: "
    f"{merged['solar_radiation_w_m2'].mean():.2f} W/m²"
)


# ============================================================
# HIGH / CRITICAL EXAMPLES
# ============================================================

print()
print("=" * 70)
print("HIGH / CRITICAL EXAMPLES")
print("=" * 70)


examples = merged[
    merged[
        "thermal_risk"
    ].isin(
        [
            "HIGH",
            "CRITICAL"
        ]
    )
].head(10)


if len(examples) == 0:

    print(
        "No HIGH or CRITICAL examples."
    )

else:

    for _, row in examples.iterrows():

        print()

        print(
            f"Participant: "
            f"{int(row['participant_id'])}"
        )

        print(
            f"Condition: "
            f"{int(row['condition_id'])}"
        )

        print(
            f"Time: "
            f"{row['time_min']:.1f} min"
        )

        print(
            f"Current estimated core: "
            f"{row['estimated_core_temp_c']:.3f} °C"
        )

        print(
            f"Predicted +10 min core: "
            f"{row['predicted_future_core_temp_c']:.3f} °C"
        )

        print(
            f"Predicted rise: "
            f"{row['predicted_rise_c']:.3f} °C"
        )

        print(
            f"Ambient: "
            f"{row['ambient_temp_c']:.2f} °C"
        )

        print(
            f"Humidity: "
            f"{row['humidity_pct']:.2f}%"
        )

        wind_value = row[
            "wind_mps"
        ]

        if pd.isna(wind_value):

            wind_text = "N/A"

        else:

            wind_text = (
                f"{wind_value:.2f} m/s"
            )


        print(
            f"Wind: {wind_text}"
        )

        print(
            f"Physiological score: "
            f"{row['physiological_score']}"
        )

        print(
            f"Environmental score: "
            f"{row['environmental_score']}"
        )

        print(
            f"Activity/work score: "
            f"{row['activity_work_score']}"
        )

        print(
            f"Raw risk: "
            f"{row['raw_risk']}"
        )

        print(
            f"Final risk: "
            f"{row['thermal_risk']}"
        )

        print(
            f"Transition reason: "
            f"{row['state_transition_reason']}"
        )

        print(
            f"Action: "
            f"{row['recommended_action']}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "engine":
        "THERMORA temporal-aware multi-factor thermal risk engine",

    "persistence_enabled":
        True,

    "hysteresis_enabled":
        True,

    "sequence_awareness_enabled":
        True,

    "escalation_consecutive_readings":
        int(ESCALATION_COUNT),

    "deescalation_consecutive_readings":
        int(DEESCALATION_COUNT),

    "maximum_continuous_gap_minutes":
        float(MAX_CONTINUOUS_GAP_MIN),

    "emergency_core_temperature_c":
        float(EMERGENCY_CORE_TEMP),

    "emergency_predicted_rise_c":
        float(EMERGENCY_RISE),

    "rows":
        int(len(merged)),

    "risk_distribution": {

        level:
            int(
                counts.get(
                    level,
                    0
                )
            )

        for level in [

            "SAFE",
            "CAUTION",
            "HIGH",
            "CRITICAL"

        ]
    },

    "rows_changed_by_hysteresis":
        int(changed_count),

    "percentage_changed":
        float(
            changed_count
            / len(merged)
            * 100
        ),

    "transition_reasons": {

        str(reason):
            int(count)

        for reason, count
        in reason_counts.items()

    },

    "thermal_statistics": {

        "mean_estimated_core_c":
            float(
                merged[
                    "estimated_core_temp_c"
                ].mean()
            ),

        "mean_predicted_core_c":
            float(
                merged[
                    "predicted_future_core_temp_c"
                ].mean()
            ),

        "mean_predicted_rise_c":
            float(
                merged[
                    "predicted_rise_c"
                ].mean()
            ),

        "max_predicted_core_c":
            float(
                merged[
                    "predicted_future_core_temp_c"
                ].max()
            ),

        "max_predicted_rise_c":
            float(
                merged[
                    "predicted_rise_c"
                ].max()
            )
    },

    "mean_scores": {

        "physiological":
            float(
                merged[
                    "physiological_score"
                ].mean()
            ),

        "environmental":
            float(
                merged[
                    "environmental_score"
                ].mean()
            ),

        "activity_work":
            float(
                merged[
                    "activity_work_score"
                ].mean()
            )
    },

    "mean_environment": {

        "ambient_temp_c":
            float(
                merged[
                    "ambient_temp_c"
                ].mean()
            ),

        "humidity_pct":
            float(
                merged[
                    "humidity_pct"
                ].mean()
            ),

        "wind_mps":
            float(
                merged[
                    "wind_mps"
                ].mean()
            ),

        "solar_radiation_w_m2":
            float(
                merged[
                    "solar_radiation_w_m2"
                ].mean()
            )
    }
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)


print(
    OUTPUT_FILE
)


print(
    SUMMARY_FILE
)


print()
print("=" * 70)
print("THERMORA TEMPORAL THERMAL RISK ENGINE COMPLETE")
print("=" * 70)