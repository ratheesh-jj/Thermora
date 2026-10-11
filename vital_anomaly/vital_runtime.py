import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# THERMORA - DEPLOYABLE VITAL RISK ENGINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_FILE = OUTPUT_DIR / "runtime_baseline.json"


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

CALIBRATION_SECONDS = 300       # 5 minutes
WINDOW_SECONDS = 10

# These are prototype engineering thresholds.
# They are NOT clinical diagnostic thresholds.
SPO2_WARNING = 94
SPO2_CRITICAL = 90

TEMP_WARNING_DELTA = 1.0
TEMP_CRITICAL_DELTA = 2.0

HR_MIN = 40
HR_MAX = 180

# Persistence: abnormal state must persist across windows
WARNING_PERSISTENCE = 2
CRITICAL_PERSISTENCE = 2


# ============================================================
# PERSONAL BASELINE
# ============================================================

class PersonalBaseline:

    def __init__(self):
        self.hr = []
        self.spo2 = []
        self.hrv = []
        self.temperature = []

        self.calibrated = False

        self.hr_median = None
        self.hr_mad = None

        self.spo2_median = None
        self.spo2_mad = None

        self.hrv_median = None
        self.hrv_mad = None

        self.temperature_median = None
        self.temperature_mad = None

    def add_sample(
        self,
        hr=None,
        spo2=None,
        hrv=None,
        temperature=None
    ):
        if hr is not None and HR_MIN <= hr <= HR_MAX:
            self.hr.append(float(hr))

        if spo2 is not None and 70 <= spo2 <= 100:
            self.spo2.append(float(spo2))

        if hrv is not None and hrv > 0:
            self.hrv.append(float(hrv))

        if temperature is not None:
            self.temperature.append(float(temperature))

    @staticmethod
    def median_mad(values):
        if len(values) == 0:
            return None, None

        values = np.asarray(values, dtype=float)

        median = np.median(values)

        mad = np.median(
            np.abs(values - median)
        )

        # Avoid zero MAD causing division problems.
        mad = max(mad, 0.1)

        return float(median), float(mad)

    def calibrate(self):

        self.hr_median, self.hr_mad = self.median_mad(self.hr)

        self.spo2_median, self.spo2_mad = self.median_mad(
            self.spo2
        )

        self.hrv_median, self.hrv_mad = self.median_mad(
            self.hrv
        )

        self.temperature_median, self.temperature_mad = (
            self.median_mad(self.temperature)
        )

        self.calibrated = True

    def save(self):

        data = {
            "hr_median": self.hr_median,
            "hr_mad": self.hr_mad,
            "spo2_median": self.spo2_median,
            "spo2_mad": self.spo2_mad,
            "hrv_median": self.hrv_median,
            "hrv_mad": self.hrv_mad,
            "temperature_median": self.temperature_median,
            "temperature_mad": self.temperature_mad,
        }

        with open(BASELINE_FILE, "w") as f:
            json.dump(data, f, indent=4)


# ============================================================
# SIGNAL QUALITY
# ============================================================

def signal_quality(
    hr=None,
    spo2=None,
    hrv=None,
    temperature=None
):

    quality = 100

    if hr is None:
        quality -= 25
    elif not (HR_MIN <= hr <= HR_MAX):
        quality -= 40

    if spo2 is None:
        quality -= 25
    elif not (70 <= spo2 <= 100):
        quality -= 40

    if hrv is None:
        quality -= 15
    elif hrv <= 0:
        quality -= 20

    if temperature is None:
        quality -= 15

    return max(0, quality)


# ============================================================
# NORMALIZED DEVIATION
# ============================================================

def robust_deviation(value, median, mad):

    if value is None or median is None or mad is None:
        return 0.0

    return abs(value - median) / (1.4826 * mad)


# ============================================================
# VITAL RISK ENGINE
# ============================================================

class VitalRiskEngine:

    def __init__(self, baseline):

        self.baseline = baseline

        self.warning_count = 0
        self.critical_count = 0

        self.previous_spo2 = None
        self.previous_hr = None
        self.previous_temperature = None

    def evaluate(
        self,
        hr=None,
        spo2=None,
        hrv=None,
        temperature=None,
        activity="UNKNOWN"
    ):

        reasons = []

        quality = signal_quality(
            hr,
            spo2,
            hrv,
            temperature
        )

        score = 0.0

        # ----------------------------------------------------
        # HEART RATE
        # ----------------------------------------------------

        hr_score = 0

        if hr is not None:

            # Activity context
            if activity in ["REST", "SITTING", "STANDING"]:

                if hr >= 120:
                    hr_score = 70
                    reasons.append(
                        "High heart rate during low activity"
                    )

                elif hr >= 100:
                    hr_score = 40
                    reasons.append(
                        "Elevated heart rate during low activity"
                    )

            elif activity in ["WALKING", "STAIRS", "JOGGING"]:

                # Higher HR can be expected during exercise.
                hr_score = min(
                    30,
                    robust_deviation(
                        hr,
                        self.baseline.hr_median,
                        self.baseline.hr_mad
                    ) * 5
                )

            else:

                hr_score = min(
                    50,
                    robust_deviation(
                        hr,
                        self.baseline.hr_median,
                        self.baseline.hr_mad
                    ) * 10
                )

        score += hr_score


        # ----------------------------------------------------
        # SPO2
        # ----------------------------------------------------

        spo2_score = 0

        if spo2 is not None:

            if spo2 <= SPO2_CRITICAL:

                spo2_score = 100
                reasons.append(
                    "Very low SpO2"
                )

            elif spo2 <= SPO2_WARNING:

                spo2_score = 60
                reasons.append(
                    "Low SpO2"
                )

            elif self.previous_spo2 is not None:

                drop = self.previous_spo2 - spo2

                if drop >= 3:

                    spo2_score = 50

                    reasons.append(
                        "Rapid SpO2 decrease"
                    )

        score += spo2_score


        # ----------------------------------------------------
        # HRV
        # ----------------------------------------------------

        hrv_score = 0

        if hrv is not None:

            deviation = robust_deviation(
                hrv,
                self.baseline.hrv_median,
                self.baseline.hrv_mad
            )

            if deviation >= 4:

                hrv_score = 40

                reasons.append(
                    "Large HRV deviation from personal baseline"
                )

            elif deviation >= 2:

                hrv_score = 20

                reasons.append(
                    "HRV deviation from personal baseline"
                )

        score += hrv_score


        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        temp_score = 0

        if (
            temperature is not None
            and self.baseline.temperature_median is not None
        ):

            delta = (
                temperature
                - self.baseline.temperature_median
            )

            if delta >= TEMP_CRITICAL_DELTA:

                temp_score = 80

                reasons.append(
                    "Large temperature increase from baseline"
                )

            elif delta >= TEMP_WARNING_DELTA:

                temp_score = 40

                reasons.append(
                    "Temperature elevated from baseline"
                )

        score += temp_score


        # ----------------------------------------------------
        # COMBINATION EFFECT
        # ----------------------------------------------------

        abnormal_components = sum([
            hr_score >= 40,
            spo2_score >= 40,
            hrv_score >= 20,
            temp_score >= 40
        ])

        if abnormal_components >= 3:

            score += 20

            reasons.append(
                "Multiple physiological signals abnormal"
            )


        score = min(100, score)


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if score >= 70:

            self.critical_count += 1
            self.warning_count = 0

        elif score >= 35:

            self.warning_count += 1
            self.critical_count = 0

        else:

            self.warning_count = 0
            self.critical_count = 0


        # Persistence filter
        if self.critical_count >= CRITICAL_PERSISTENCE:

            status = "CRITICAL"

        elif self.warning_count >= WARNING_PERSISTENCE:

            status = "WARNING"

        else:

            status = "NORMAL"


        # ----------------------------------------------------
        # SAVE PREVIOUS VALUES
        # ----------------------------------------------------

        if spo2 is not None:
            self.previous_spo2 = spo2

        if hr is not None:
            self.previous_hr = hr

        if temperature is not None:
            self.previous_temperature = temperature


        return {
            "vital_score": round(score, 2),
            "status": status,
            "signal_quality": quality,
            "reasons": reasons
        }


# ============================================================
# BIDMC TEST
# ============================================================

def test_with_bidmc():

    csv_file = (
        OUTPUT_DIR /
        "bidmc_vital_windows.csv"
    )

    if not csv_file.exists():

        print(
            f"ERROR: {csv_file} not found."
        )

        return

    df = pd.read_csv(csv_file)

    print("=" * 70)
    print("THERMORA - DEPLOYABLE VITAL ENGINE TEST")
    print("=" * 70)

    print(f"Loaded {len(df)} windows")

    # --------------------------------------------------------
    # Build a baseline from the first 5 minutes
    # --------------------------------------------------------

    baseline = PersonalBaseline()

    first_record = df["record"].iloc[0]

    calibration = df[
        df["record"] == first_record
    ].head(30)

    for _, row in calibration.iterrows():

        baseline.add_sample(
            hr=row.get("HR"),
            spo2=row.get("SpO2"),
            hrv=row.get("HRV"),
            temperature=row.get("temperature")
        )

    baseline.calibrate()
    baseline.save()

    print("\nPersonal baseline:")
    print(
        f"HR:   {baseline.hr_median}"
    )
    print(
        f"SpO2: {baseline.spo2_median}"
    )
    print(
        f"HRV:  {baseline.hrv_median}"
    )
    print(
        f"Temp: {baseline.temperature_median}"
    )

    # --------------------------------------------------------
    # Run engine
    # --------------------------------------------------------

    engine = VitalRiskEngine(baseline)

    results = []

    for _, row in df.iterrows():

        result = engine.evaluate(
            hr=row.get("HR"),
            spo2=row.get("SpO2"),
            hrv=row.get("HRV"),
            temperature=row.get("temperature"),
            activity="UNKNOWN"
        )

        results.append(result)

    result_df = pd.DataFrame(results)

    output = pd.concat(
        [
            df.reset_index(drop=True),
            result_df
        ],
        axis=1
    )

    output_file = (
        OUTPUT_DIR /
        "thermora_vital_runtime_test.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        output["status"].value_counts()
    )

    print("\nScore statistics:")

    print(
        output["vital_score"].describe()
    )

    print("\nOutput saved:")
    print(output_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_with_bidmc()