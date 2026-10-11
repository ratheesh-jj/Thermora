import json
import numpy as np
import pandas as pd
from pathlib import Path
from vital_anomaly_inference import VitalAnomalyDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "vital_anomaly" / "bidmc_csv"

def get_window_data(sig, num, start_t, duration=30):
    end_t = start_t + duration
    s_mask = (sig["Time [s]"] >= start_t) & (sig["Time [s]"] < end_t)
    n_mask = (num["Time [s]"] >= start_t) & (num["Time [s]"] < end_t)
    
    ppg = sig.loc[s_mask, "PLETH"].to_numpy(float)
    ppg_time = sig.loc[s_mask, "Time [s]"].to_numpy(float)
    hr = num.loc[n_mask, "HR"].to_numpy(float)
    hr_time = num.loc[n_mask, "Time [s]"].to_numpy(float)
    pulse = num.loc[n_mask, "PULSE"].to_numpy(float)
    pulse_time = num.loc[n_mask, "Time [s]"].to_numpy(float)
    spo2 = num.loc[n_mask, "SpO2"].to_numpy(float)
    spo2_time = num.loc[n_mask, "Time [s]"].to_numpy(float)
    
    return ppg, ppg_time, hr, hr_time, pulse, pulse_time, spo2, spo2_time

def main():
    print("=" * 60)
    print("VITAL ANOMALY INFERENCE TEST")
    print("=" * 60)
    
    detector = VitalAnomalyDetector()
    
    rec_id = "bidmc_02"
    sig = pd.read_csv(DATA_DIR / f"{rec_id}_Signals.csv")
    num = pd.read_csv(DATA_DIR / f"{rec_id}_Numerics.csv")
    
    sig.columns = [c.strip() for c in sig.columns]
    num.columns = [c.strip() for c in num.columns]
    
    start_t = float(max(sig["Time [s]"].min(), num["Time [s]"].min()))
    
    print("\n--- 1. Baseline Calibration ---")
    for i in range(5):
        args = get_window_data(sig, num, start_t + i*30)
        res = detector.process_window(*args)
        print(f"Window {i+1}: {res['vital_status']}")
        
    print("\n--- 2. Normal Window ---")
    # Needs two consecutive to confirm anomalies normally, so let's just push normal first
    args = get_window_data(sig, num, start_t + 5*30)
    res = detector.process_window(*args)
    print(res)
    
    print("\n--- 3. HR Abnormal Window (Consecutive for confirmation) ---")
    args = list(get_window_data(sig, num, start_t + 6*30))
    args[2] = args[2] + 40  # HR + 40
    res1 = detector.process_window(*args)
    print(f"Window 1: {res1['vital_status']}")
    res2 = detector.process_window(*args)
    print(f"Window 2: {res2['vital_status']}")
    print(res2)
    
    print("\n--- 4. SpO2 Abnormal Window ---")
    detector.reset_state() # Reset to start fresh
    for i in range(5): detector.process_window(*get_window_data(sig, num, start_t + i*30))
    
    args = list(get_window_data(sig, num, start_t + 6*30))
    args[6] = args[6] - 6  # SpO2 - 6
    res1 = detector.process_window(*args)
    res2 = detector.process_window(*args)
    print(f"Window 2: {res2['vital_status']}")
    
    print("\n--- 5. Poor PPG Window ---")
    detector.reset_state()
    for i in range(5): detector.process_window(*get_window_data(sig, num, start_t + i*30))
    
    args = list(get_window_data(sig, num, start_t + 6*30))
    args[0] = args[0] * 0  # Flatline PPG
    res = detector.process_window(*args)
    print(f"Window: {res['vital_status']}")
    
    print("\n--- 6. Combined Abnormal Window ---")
    detector.reset_state()
    for i in range(5): detector.process_window(*get_window_data(sig, num, start_t + i*30))
    
    args = list(get_window_data(sig, num, start_t + 6*30))
    args[2] = args[2] + 50
    args[6] = args[6] - 10
    res1 = detector.process_window(*args)
    res2 = detector.process_window(*args)
    print(f"Window 2: {res2['vital_status']}")
    
    print("\nTest Complete!")

if __name__ == "__main__":
    main()
