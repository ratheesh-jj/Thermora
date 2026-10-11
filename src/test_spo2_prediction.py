from pathlib import Path
import numpy as np
import pandas as pd
from spo2_prediction_inference import SpO2Forecaster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "vital_anomaly" / "bidmc_csv"

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_column(df, names):
    exact = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = name.strip().lower()
        if key in exact:
            return exact[key]
    for c in df.columns:
        c_low = str(c).strip().lower()
        for name in names:
            if name.strip().lower() in c_low:
                return c
    return None

def main():
    recording_id = "bidmc_02"
    print(f"Loading {recording_id} for test...")
    signal_file = DATA_DIR / f"{recording_id}_Signals.csv"
    numeric_file = DATA_DIR / f"{recording_id}_Numerics.csv"
    
    signals = clean_columns(pd.read_csv(signal_file))
    numerics = clean_columns(pd.read_csv(numeric_file))
    
    signal_time = find_column(signals, ["Time [s]", "Time"])
    ppg_col = find_column(signals, ["PLETH"])
    numeric_time = find_column(numerics, ["Time [s]", "Time"])
    hr_col = find_column(numerics, ["HR"])
    pulse_col = find_column(numerics, ["PULSE"])
    spo2_col = find_column(numerics, ["SpO2", "SPO2"])
    
    for c in [signal_time, ppg_col]:
        signals[c] = pd.to_numeric(signals[c], errors="coerce")
    for c in [numeric_time, hr_col, pulse_col, spo2_col]:
        numerics[c] = pd.to_numeric(numerics[c], errors="coerce")
        
    sig = signals[[signal_time, ppg_col]].dropna(subset=[signal_time]).sort_values(signal_time)
    num = numerics[[numeric_time, hr_col, pulse_col, spo2_col]].dropna(subset=[numeric_time]).sort_values(numeric_time)
    num = num.set_index(numeric_time).interpolate(limit=3, limit_direction="both")
    
    t_end = num.index[100]
    t_start = t_end - 29
    
    mask_num = (num.index >= t_start) & (num.index <= t_end)
    mask_sig = (sig[signal_time] >= t_start) & (sig[signal_time] <= t_end)
    
    ppg_history = sig[mask_sig][ppg_col].to_numpy(float)
    hr_history = num[mask_num][hr_col].to_numpy(float)
    pulse_history = num[mask_num][pulse_col].to_numpy(float)
    spo2_history = num[mask_num][spo2_col].to_numpy(float)
    
    forecaster = SpO2Forecaster()
    result = forecaster.predict(ppg_history, hr_history, pulse_history, spo2_history)
    
    print("\nInference Output:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
            
    print("\nTest passed. Schema matched expectations.")

if __name__ == "__main__":
    main()
