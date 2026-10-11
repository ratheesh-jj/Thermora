import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"vital_anomaly\bidmc_csv"
OUTPUT_DIR = r"vital_anomaly\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vital_windows.csv")

WINDOW_SECONDS = 30
PPG_FS = 125

# ============================================================
# FEATURE EXTRACTORS
# ============================================================
def safe_slope(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    mask = np.isfinite(values) & np.isfinite(times)
    if mask.sum() < 2: return 0.0
    x, y = times[mask], values[mask]
    if np.ptp(x) == 0: return 0.0
    return float(np.polyfit(x, y, 1)[0])

def numeric_features(values, times, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {f"{prefix}_{k}": np.nan for k in ["latest", "mean", "median", "std", "min", "max", "range", "slope", "first_last_change"]}
        
    latest = float(values[-1])
    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values))
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    vrange = float(np.ptp(values))
    slope = safe_slope(values, times)
    change = float(values[-1] - values[0])
    
    feats = {
        f"{prefix}_latest": latest,
        f"{prefix}_mean": mean,
        f"{prefix}_median": median,
        f"{prefix}_std": std,
        f"{prefix}_min": vmin,
        f"{prefix}_max": vmax,
        f"{prefix}_range": vrange,
        f"{prefix}_slope": slope,
        f"{prefix}_first_last_change": change,
    }
    
    if prefix == "spo2":
        feats["spo2_pct_below_95"] = float(np.mean(values < 95) * 100)
        feats["spo2_pct_below_92"] = float(np.mean(values < 92) * 100)
        feats["spo2_pct_below_90"] = float(np.mean(values < 90) * 100)
        
    return feats

def ppg_features(ppg, time):
    ppg = np.asarray(ppg, dtype=float)
    if len(ppg) == 0:
        return {}
        
    diff = np.diff(ppg)
    abs_diff = np.abs(diff)
    
    feats = {
        "ppg_mean": float(np.mean(ppg)),
        "ppg_std": float(np.std(ppg)),
        "ppg_min": float(np.min(ppg)),
        "ppg_max": float(np.max(ppg)),
        "ppg_range": float(np.ptp(ppg)),
        "ppg_rms": float(np.sqrt(np.mean(ppg ** 2))),
        "ppg_median": float(np.median(ppg)),
        "ppg_p10": float(np.percentile(ppg, 10)),
        "ppg_p25": float(np.percentile(ppg, 25)),
        "ppg_p75": float(np.percentile(ppg, 75)),
        "ppg_p90": float(np.percentile(ppg, 90)),
        "ppg_slope": safe_slope(ppg, time),
        "ppg_first_last_change": float(ppg[-1] - ppg[0]) if len(ppg) > 0 else 0.0,
        "ppg_diff_abs_mean": float(np.mean(abs_diff)) if len(diff) else 0.0,
        "ppg_diff_std": float(np.std(diff)) if len(diff) else 0.0,
        "ppg_energy": float(np.mean((ppg - np.mean(ppg))**2))
    }
    
    # PPG Quality metrics
    amplitude = feats["ppg_max"] - feats["ppg_min"]
    feats["ppg_amplitude"] = amplitude
    
    if len(ppg) > 10:
        peaks, props = find_peaks(ppg, distance=int(PPG_FS*0.35), prominence=max(feats["ppg_std"]*0.25, 1e-6))
        feats["ppg_pulse_amplitude"] = float(np.mean(props["prominences"])) if len(peaks) > 0 else 0.0
        if len(peaks) >= 2:
            intervals = np.diff(time[peaks])
            feats["ppg_beat_regularity"] = float(np.std(intervals))
        else:
            feats["ppg_beat_regularity"] = np.nan
    else:
        feats["ppg_pulse_amplitude"] = 0.0
        feats["ppg_beat_regularity"] = np.nan

    feats["ppg_baseline_variation"] = feats["ppg_std"]
    ac = feats["ppg_std"]
    dc = abs(feats["ppg_mean"])
    feats["ppg_ac_dc_ratio"] = float(ac/dc) if dc > 1e-6 else 0.0
    
    feats["ppg_clipping_pct"] = float(np.mean((ppg >= 0.99*feats["ppg_max"]) | (ppg <= 1.01*feats["ppg_min"])) * 100) if amplitude > 1e-6 else 100.0
    feats["ppg_flatline_pct"] = float(np.mean(abs_diff < 1e-6) * 100) if len(diff) else 100.0
    feats["ppg_zero_pct"] = float(np.mean(np.abs(ppg) < 1e-6) * 100)
    
    # High freq proxy (rough zero crossings of diff)
    if len(diff) > 2:
        zc = np.sum(np.diff(np.sign(diff)) != 0)
        feats["ppg_hf_noise"] = float(zc / len(diff))
    else:
        feats["ppg_hf_noise"] = 0.0
        
    return feats


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    signal_files = sorted(glob.glob(os.path.join(BASE_DIR, "bidmc_*_Signals.csv")))
    numeric_files = sorted(glob.glob(os.path.join(BASE_DIR, "bidmc_*_Numerics.csv")))

    numeric_map = {os.path.basename(f).replace("_Numerics.csv", ""): f for f in numeric_files}

    all_rows = []

    for signal_file in signal_files:
        rec_id = os.path.basename(signal_file).replace("_Signals.csv", "")
        if rec_id not in numeric_map: continue
        
        print(f"Processing {rec_id}")
        sig = pd.read_csv(signal_file)
        num = pd.read_csv(numeric_map[rec_id])
        
        sig.columns = [c.strip() for c in sig.columns]
        num.columns = [c.strip() for c in num.columns]
        
        if not all(c in sig.columns for c in ["Time [s]", "PLETH"]): continue
        if not all(c in num.columns for c in ["Time [s]", "HR", "SpO2"]): continue
        
        sig["Time [s]"] = pd.to_numeric(sig["Time [s]"], errors="coerce")
        sig["PLETH"] = pd.to_numeric(sig["PLETH"], errors="coerce")
        num["Time [s]"] = pd.to_numeric(num["Time [s]"], errors="coerce")
        num["HR"] = pd.to_numeric(num["HR"], errors="coerce")
        num["SpO2"] = pd.to_numeric(num["SpO2"], errors="coerce")
        
        sig = sig.dropna(subset=["Time [s]", "PLETH"])
        num = num.dropna(subset=["Time [s]", "HR", "SpO2"])
        if len(sig) == 0 or len(num) == 0: continue
        
        start_t = max(sig["Time [s]"].min(), num["Time [s]"].min())
        end_t = min(sig["Time [s]"].max(), num["Time [s]"].max())
        
        w_start = float(start_t)
        
        rec_windows = []
        
        while w_start + WINDOW_SECONDS <= end_t:
            w_end = w_start + WINDOW_SECONDS
            
            p_mask = (sig["Time [s]"] >= w_start) & (sig["Time [s]"] < w_end)
            n_mask = (num["Time [s]"] >= w_start) & (num["Time [s]"] < w_end)
            
            if p_mask.sum() < 1000 or n_mask.sum() < 10:
                w_start += WINDOW_SECONDS
                continue
                
            ppg = sig.loc[p_mask, "PLETH"].to_numpy()
            ptimes = sig.loc[p_mask, "Time [s]"].to_numpy()
            
            hr = num.loc[n_mask, "HR"].to_numpy()
            spo2 = num.loc[n_mask, "SpO2"].to_numpy()
            ntimes = num.loc[n_mask, "Time [s]"].to_numpy()
            
            row = {
                "recording_id": rec_id,
                "window_start_s": w_start,
                "window_end_s": w_end
            }
            
            row.update(ppg_features(ppg, ptimes))
            row.update(numeric_features(hr, ntimes, "hr"))
            row.update(numeric_features(spo2, ntimes, "spo2"))
            
            rec_windows.append(row)
            w_start += WINDOW_SECONDS
            
        # Baseline Calculation
        if len(rec_windows) > 0:
            # find first 5 stable windows (proxy for stable: hr_range < 20, spo2_latest >= 90)
            stable = [w for w in rec_windows if w.get("hr_range", 100) < 20 and w.get("spo2_latest", 0) >= 90 and w.get("ppg_flatline_pct", 100) < 5]
            if len(stable) == 0:
                stable = rec_windows[:5]
            else:
                stable = stable[:5]
                
            base_hr = np.mean([w["hr_mean"] for w in stable])
            base_hr_std = np.mean([w["hr_std"] for w in stable])
            base_spo2 = np.mean([w["spo2_mean"] for w in stable])
            base_spo2_std = np.mean([w["spo2_std"] for w in stable])
            base_ppg_amp = np.mean([w["ppg_amplitude"] for w in stable])
            
            for w in rec_windows:
                w["baseline_hr"] = base_hr
                w["baseline_hr_std"] = base_hr_std
                w["baseline_spo2"] = base_spo2
                w["baseline_spo2_std"] = base_spo2_std
                w["baseline_ppg_amplitude"] = base_ppg_amp
                
                w["hr_dev_abs"] = w["hr_latest"] - base_hr
                w["hr_dev_rel"] = abs(w["hr_latest"] - base_hr) / max(base_hr, 1.0)
                w["spo2_dev_abs"] = base_spo2 - w["spo2_latest"] # positive means drop
                w["ppg_amp_dev_rel"] = (w["ppg_amplitude"] - base_ppg_amp) / max(base_ppg_amp, 1e-6)
                
                all_rows.append(w)

    dataset = pd.DataFrame(all_rows)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset complete! {len(dataset)} windows.")

if __name__ == "__main__":
    main()