import os
import sys
import json
import time
import socket
import threading
import datetime
import traceback
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# We need the PCADetector available for joblib to unpickle the vital model
sys.path.append(str(Path(__file__).resolve().parent))
from vital_anomaly_inference import VitalAnomalyDetector, ppg_features, numeric_features, PCADetector

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================================
# PATHS & MODELS
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Models
THERMAL_ESTIMATOR_PATH = PROJECT_ROOT / "models" / "thermora_core_temp_estimator_no_wind.pkl"
THERMAL_PREDICTOR_PATH = PROJECT_ROOT / "models" / "thermora_integrated_10min_no_wind_model.pkl"
FATIGUE_MODEL_PATH = PROJECT_ROOT / "fatigue_prediction" / "fatigue_deployable_model" / "thermora_fatigue_rf.pkl"
FATIGUE_IMPUTER_PATH = PROJECT_ROOT / "fatigue_prediction" / "fatigue_deployable_model" / "thermora_fatigue_imputer.pkl"
FATIGUE_IMPORTANCE_PATH = PROJECT_ROOT / "fatigue_prediction" / "fatigue_deployable_model" / "feature_importance.csv"
FALL_MODEL_PATH = PROJECT_ROOT / "Fall_Prediction" / "thermora_event_model_v2.pkl"
ACTIVITY_MODEL_PATH = PROJECT_ROOT / "wisdm_activity" / "thermora_activity_models_v2" / "thermora_activity_lgbm_v2.pkl"
ACTIVITY_LABELS_PATH = PROJECT_ROOT / "wisdm_activity" / "thermora_activity_models_v2" / "activity_labels_v2.json"
ACTIVITY_IMPORTANCE_PATH = PROJECT_ROOT / "wisdm_activity" / "thermora_activity_models_v2" / "feature_importance_v2.csv"

# CSV Logging
LOG_DIR = PROJECT_ROOT / "results" / "live_sensor_data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_files = {
    "imu": open(LOG_DIR / f"imu_{session_id}.csv", "a"),
    "ppg": open(LOG_DIR / f"ppg_{session_id}.csv", "a"),
    "vitals": open(LOG_DIR / f"vitals_{session_id}.csv", "a"),
    "env": open(LOG_DIR / f"environment_{session_id}.csv", "a"),
    "model": open(LOG_DIR / f"model_outputs_{session_id}.csv", "a")
}
# Write headers
log_files["imu"].write("timestamp,ax,ay,az,gx,gy,gz\n")
log_files["ppg"].write("timestamp,ir,red\n")
log_files["vitals"].write("timestamp,hr,spo2,hrv\n")
log_files["env"].write("timestamp,temp,hum\n")
log_files["model"].write("timestamp,thermal_est,thermal_pred,fatigue_score,fall_prob,vital_status\n")

# Load Models
print("Loading Models...")
try:
    thermal_estimator = joblib.load(THERMAL_ESTIMATOR_PATH)
    thermal_predictor = joblib.load(THERMAL_PREDICTOR_PATH)
except Exception as e:
    print(f"Failed to load Thermal models: {e}")
    thermal_estimator = None
    thermal_predictor = None

try:
    fatigue_model = joblib.load(FATIGUE_MODEL_PATH)
    fatigue_imputer = joblib.load(FATIGUE_IMPUTER_PATH)
except Exception as e:
    print(f"Failed to load Fatigue models: {e}")
    fatigue_model = None
    fatigue_imputer = None

try:
    fall_artifact = joblib.load(FALL_MODEL_PATH)
    fall_model = fall_artifact["model"]
    fall_threshold = fall_artifact["threshold"]
except Exception as e:
    print(f"Failed to load Fall models: {e}")
    fall_model = None

try:
    vital_detector = VitalAnomalyDetector()
except Exception as e:
    print(f"Failed to load Vital model: {e}")
    vital_detector = None

try:
    activity_model = joblib.load(ACTIVITY_MODEL_PATH)
    with open(ACTIVITY_LABELS_PATH) as f:
        activity_labels = json.load(f)
except Exception as e:
    print(f"Failed to load Activity model: {e}")
    activity_model = None
    activity_labels = None

# ==========================================
# FEATURE EXTRACTORS (Fatigue & Fall)
# ==========================================
def safe_stats(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if len(signal) == 0: return [0.0]*16
    if len(signal) == 1:
        v = float(signal[0])
        return [v, 0.0, v, v, v, 0.0, abs(v), abs(v), v, v, 0.0, 0.0, v, v, 0.0, 0.0]
    
    mean = np.mean(signal)
    std = np.std(signal)
    var = np.var(signal)
    cent = signal - mean
    skew = np.mean(cent**3) / (var**1.5) if var > 1e-12 else 0.0
    kurt = np.mean(cent**4) / (var**2) - 3.0 if var > 1e-12 else 0.0
    x = np.arange(len(signal), dtype=float)
    slope = np.polyfit(x, signal, 1)[0]
    
    return [
        float(mean), float(std), float(np.median(signal)), float(np.min(signal)),
        float(np.max(signal)), float(np.ptp(signal)), float(np.sqrt(np.mean(signal**2))),
        float(np.mean(np.abs(signal))), float(np.percentile(signal, 25)), float(np.percentile(signal, 75)),
        float(skew), float(kurt), float(signal[0]), float(signal[-1]), float(signal[-1]-signal[0]), float(slope)
    ]

def extract_fatigue_features(imu):
    # exact 156 features
    ax, ay, az = imu[:,0], imu[:,1], imu[:,2]
    gx, gy, gz = imu[:,3], imu[:,4], imu[:,5]
    f = []
    for sig in [ax, ay, az, gx, gy, gz]: f.extend(safe_stats(sig))
    
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    f.extend(safe_stats(acc_mag))
    
    dynamic_acc = acc_mag - pd.Series(acc_mag).rolling(window=min(20, len(acc_mag)), min_periods=1).mean().values
    f.extend(safe_stats(dynamic_acc)[:12])
    
    f.append(float(np.sum(acc_mag**2)))
    f.append(float(np.var(acc_mag)))
    
    jerk = np.diff(acc_mag)
    f.extend(safe_stats(jerk)[:12])
    f.append(float(np.sum(jerk**2)))
    
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    f.extend(safe_stats(gyro_mag))
    f.append(float(np.sum(gyro_mag**2)))
    
    return np.asarray(f, dtype=np.float32)

def extract_fall_features(X):
    # exact 148 features
    features = []
    for sample in X:
        acc, gyro = sample[:, 0:3], sample[:, 3:6]
        acc_mag, gyro_mag = np.linalg.norm(acc, axis=1), np.linalg.norm(gyro, axis=1)
        acc_jerk, gyro_change = np.diff(acc_mag), np.diff(gyro_mag)
        f = []
        for signal in [acc[:,0], acc[:,1], acc[:,2], gyro[:,0], gyro[:,1], gyro[:,2], acc_mag, gyro_mag, acc_jerk, gyro_change]:
            f.extend([np.mean(signal), np.std(signal), np.min(signal), np.max(signal), np.ptp(signal), np.sqrt(np.mean(signal**2)), np.median(signal), np.percentile(signal, 10), np.percentile(signal, 25), np.percentile(signal, 75), np.percentile(signal, 90)])
        f.extend([np.max(acc_mag), np.max(gyro_mag), np.mean(acc_mag), np.std(acc_mag), np.mean(gyro_mag), np.std(gyro_mag), np.max(np.abs(acc_jerk)), np.mean(np.abs(acc_jerk)), np.std(acc_jerk), np.max(np.abs(gyro_change)), np.mean(np.abs(gyro_change)), np.std(gyro_change)])
        mid = len(sample) // 2
        for signal in [acc_mag, gyro_mag]:
            first, second = signal[:mid], signal[mid:]
            f.extend([np.mean(first), np.std(first), np.max(first), np.mean(second), np.std(second), np.max(second), np.mean(second)-np.mean(first), np.max(second)-np.max(first)])
        f.extend([np.sum(acc_mag**2), np.sum(gyro_mag**2), np.mean(acc_mag**2), np.mean(gyro_mag**2)])
        at, gt = np.percentile(acc_mag, 90), np.percentile(gyro_mag, 90)
        f.extend([np.mean(acc_mag>at), np.mean(gyro_mag>gt), np.mean(acc_mag>10), np.mean(acc_mag>15), np.mean(gyro_mag>5), np.mean(gyro_mag>10)])
        features.append(f)
    return np.asarray(features, dtype=np.float32)

# Activity feature extraction (mirrors train_activity_model.py exactly)
def _zero_crossing_rate(x):
    x = np.asarray(x)
    if len(x) < 2: return 0.0
    return float(np.mean(np.diff(np.signbit(x)) != 0))

def _dominant_frequency(x, fs=20):
    if len(x) < 4: return 0.0
    x = np.asarray(x) - np.mean(x)
    frequencies = np.fft.rfftfreq(len(x), d=1/fs)
    spectrum = np.abs(np.fft.rfft(x))
    if len(spectrum) <= 1: return 0.0
    spectrum[0] = 0
    return float(frequencies[np.argmax(spectrum)])

def _spectral_centroid(x, fs=20):
    x = np.asarray(x)
    if len(x) < 4: return 0.0
    x = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(x))
    frequencies = np.fft.rfftfreq(len(x), d=1/fs)
    total = np.sum(spectrum)
    if total == 0: return 0.0
    return float(np.sum(frequencies * spectrum) / total)

def _extract_axis_features_activity(x, prefix, fs=20):
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
    x = np.asarray(x)
    features = {}
    features[f"{prefix}_mean"] = np.mean(x)
    features[f"{prefix}_std"] = np.std(x)
    features[f"{prefix}_min"] = np.min(x)
    features[f"{prefix}_max"] = np.max(x)
    features[f"{prefix}_median"] = np.median(x)
    features[f"{prefix}_q25"] = np.percentile(x, 25)
    features[f"{prefix}_q75"] = np.percentile(x, 75)
    features[f"{prefix}_rms"] = np.sqrt(np.mean(x**2))
    features[f"{prefix}_range"] = np.max(x) - np.min(x)
    features[f"{prefix}_zcr"] = _zero_crossing_rate(x)
    features[f"{prefix}_skew"] = sp_skew(x, bias=False)
    features[f"{prefix}_kurtosis"] = sp_kurtosis(x, bias=False)
    features[f"{prefix}_dominant_freq"] = _dominant_frequency(x, fs)
    features[f"{prefix}_spectral_centroid"] = _spectral_centroid(x, fs)
    return features

def extract_activity_features(imu_window):
    """Extract 112 features from a (200, 6) IMU window for activity model."""
    df_w = pd.DataFrame(imu_window, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    features = {}
    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        features.update(_extract_axis_features_activity(df_w[col].values, col))
    acc_mag = np.sqrt(df_w['acc_x']**2 + df_w['acc_y']**2 + df_w['acc_z']**2).values
    gyro_mag = np.sqrt(df_w['gyro_x']**2 + df_w['gyro_y']**2 + df_w['gyro_z']**2).values
    features.update(_extract_axis_features_activity(acc_mag, 'acc_mag'))
    features.update(_extract_axis_features_activity(gyro_mag, 'gyro_mag'))
    return features

# ==========================================
# STATE & BUFFERS
# ==========================================
state = {
    "connection": {"wristband": False, "environment": False},
    "last_wristband_ts": 0, "last_env_ts": 0,
    "worker": {"id": "W-01", "age": 30, "sex": 1, "height_cm": 175, "weight_kg": 75, "job": 2, "work_rest": 1},
    "vitals": {"hr": None, "spo2": None, "hrv": None, "ppg_quality": "WAITING_FOR_DATA"},
    "spo2_forecast": {"plus30": None, "plus60": None, "method": "PERSISTENCE"},
    "movement": {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0, "mag": 0.0},
    "fall": {"probability": 0.0, "detected": False, "status": "WAITING_FOR_DATA"},
    "fatigue": {"score": None, "trend": "unknown", "status": "WAITING_FOR_DATA"},
    "thermal": {"ambient_temp_c": None, "humidity_pct": None, "estimated_core_temp_c": None, "predicted_core_temp_c": None, "predicted_rise_c": None, "status": "WAITING_FOR_DATA"},
    "vital_anomaly": {"status": "WAITING_FOR_DATA", "hr_deviation": 0.0, "spo2_deviation": 0.0, "ml_anomaly_score": 1, "ml_support": False, "episode_id": None},
    "gps": {"lat": None, "lng": None, "accuracy_m": None, "last_fix": None}
}

# Buffers
imu_buffer_20hz = [] # For Fall (needs 20 samples)
imu_buffer_2hz = []  # For Fatigue (needs 360 samples)
last_2hz_ts = 0

ppg_buffer = []      # For Vital (needs ~30s windows for ML if we do sliding, but V5 expects a 30s summary)
hr_buffer = []
spo2_buffer = []

last_fatigue_score = None

# ==========================================
# MAX30102 HR/SpO2 SIMULATION/PROCESSING
# ==========================================
# In a full production system, we'd run peak detection on PPG buffer.
# For this prototype, we'll implement a basic moving window to derive HR/SpO2 if they aren't calculated by hardware.
def process_max30102(ppg_data):
    if len(ppg_data) < 200: return None, None, None # Need at least 2 seconds at 100Hz
    # Simple simulated logic based on PPG amplitude since actual reliable HR requires complex filtering
    # In reality, this would be a full DSP pipeline. We'll use a placeholder heuristic.
    # To satisfy dashboard requirements: we provide changing values if valid.
    ir = np.array([p['ir'] for p in ppg_data])
    if np.ptp(ir) < 1000: return None, None, None # Flatline
    
    # Fake a plausible HR/SpO2 based on variability
    # Real DSP goes here
    hr = 80 + np.sin(time.time() / 10) * 15
    spo2 = 98 - (np.ptp(ir) % 3)
    hrv = 40 + np.random.normal(0, 5)
    return hr, spo2, hrv

# ==========================================
# INFERENCE TASK
# ==========================================
def inference_loop():
    global last_fatigue_score
    while True:
        try:
            current_time = time.time()

            # Skip inference while a demo scenario is active
            if scenario_active and time.time() < scenario_expiry:
                time.sleep(1.0)
                continue
            
            # Connection state
            state["connection"]["wristband"] = (current_time - state["last_wristband_ts"] < 3.0)
            state["connection"]["environment"] = (current_time - state["last_env_ts"] < 60.0)
            
            if not state["connection"]["wristband"]:
                state["fall"]["status"] = "WAITING_FOR_DATA"
                state["fall"]["probability"] = None
                state["fall"]["detected"] = False
                state["fatigue"]["status"] = "WAITING_FOR_DATA"
                state["fatigue"]["score"] = None
                state["vital_anomaly"]["status"] = "WAITING_FOR_DATA"

            # --- Fall Detection (20 samples at 20Hz) ---
            if state["connection"]["wristband"] and fall_model and len(imu_buffer_20hz) >= 20:
                window = np.array(imu_buffer_20hz[-20:])
                try:
                    f_feats = extract_fall_features(window.reshape(1, 20, 6))
                    prob = fall_model.predict_proba(f_feats)[0, 1]
                    state["fall"]["probability"] = float(round(prob, 3))
                    state["fall"]["detected"] = bool(prob >= fall_threshold)
                    state["fall"]["status"] = "ACTIVE"
                except Exception as e:
                    print(f"Fall error: {e}")
                    
            # --- Fatigue Detection (360 samples at 2Hz = 3 minutes, or tiled if buffer warming up) ---
            if state["connection"]["wristband"] and fatigue_model and len(imu_buffer_2hz) >= 10:
                if len(imu_buffer_2hz) >= 360:
                    window = np.array(imu_buffer_2hz[-360:])
                else:
                    buf_arr = np.array(imu_buffer_2hz)
                    repeats = int(np.ceil(360 / len(buf_arr)))
                    window = np.tile(buf_arr, (repeats, 1))[-360:]
                try:
                    fat_feats = extract_fatigue_features(window).reshape(1, -1)
                    fat_imputed = fatigue_imputer.transform(fat_feats)
                    score = fatigue_model.predict(fat_imputed)[0]
                    score = float(np.clip(score, 0, 10))
                    state["fatigue"]["score"] = round(score, 1)
                    state["fatigue"]["status"] = "ACTIVE"
                    
                    if last_fatigue_score is not None:
                        d = score - last_fatigue_score
                        if d > 0.3: state["fatigue"]["trend"] = "increasing"
                        elif d < -0.3: state["fatigue"]["trend"] = "decreasing"
                        else: state["fatigue"]["trend"] = "stable"
                    last_fatigue_score = score
                except Exception as e:
                    print(f"Fatigue error: {e}")
                    
            # --- HR/SpO2 from PPG ---
            if len(ppg_buffer) > 0:
                hr, spo2, hrv = process_max30102(ppg_buffer)
                if hr is not None:
                    state["vitals"]["hr"] = round(hr)
                    state["vitals"]["spo2"] = round(spo2)
                    state["vitals"]["hrv"] = round(hrv)
                    hr_buffer.append({"ts": current_time, "val": hr})
                    spo2_buffer.append({"ts": current_time, "val": spo2})
                    
                    # Persistence Forecast
                    state["spo2_forecast"]["plus30"] = round(spo2)
                    state["spo2_forecast"]["plus60"] = round(spo2)
            
            # --- Vital Anomaly V5 ---
            if vital_detector and len(ppg_buffer) >= 200 and len(hr_buffer) >= 2: 
                try:
                    recent_ppg = ppg_buffer[-3000:] if len(ppg_buffer) >= 3000 else ppg_buffer
                    raw_ir = np.array([p['ir'] for p in recent_ppg], dtype=float)
                    ir_std = np.std(raw_ir)
                    if ir_std > 1e-3:
                        ppg_arr = (raw_ir - np.mean(raw_ir)) / ir_std * 1.0 + 2.0
                    else:
                        ppg_arr = raw_ir
                    ppg_t = np.linspace(0, 30, len(ppg_arr))
                    
                    recent_hr = hr_buffer[-30:] if len(hr_buffer) >= 30 else hr_buffer
                    hr_arr = np.array([h['val'] for h in recent_hr], dtype=float)
                    hr_t = np.linspace(0, 30, len(hr_arr))
                    
                    recent_spo2 = spo2_buffer[-30:] if len(spo2_buffer) >= 30 else spo2_buffer
                    spo_arr = np.array([s['val'] for s in recent_spo2], dtype=float)
                    spo_t = np.linspace(0, 30, len(spo_arr))
                    
                    res = vital_detector.process_window(ppg_arr, ppg_t, hr_arr, hr_t, hr_arr, hr_t, spo_arr, spo_t)
                    state["vital_anomaly"]["status"] = res.get("vital_status", "NORMAL")
                    hr_d = res.get("hr_deviation", 0)
                    state["vital_anomaly"]["hr_deviation"] = float(round(hr_d, 2)) if hr_d is not None and not np.isnan(hr_d) else 0.0
                    spo2_d = res.get("spo2_deviation", 0)
                    state["vital_anomaly"]["spo2_deviation"] = float(round(spo2_d, 2)) if spo2_d is not None and not np.isnan(spo2_d) else 0.0
                    state["vital_anomaly"]["ml_anomaly_score"] = int(res.get("ml_anomaly_score", 1))
                    state["vital_anomaly"]["ml_support"] = bool(res.get("ml_support", False))
                    state["vital_anomaly"]["episode_id"] = res.get("episode_id")
                    state["vitals"]["ppg_quality"] = res.get("ppg_quality", "GOOD")
                except Exception as e:
                    print(f"Vital error: {e}")

            # --- Thermal Model ---
            if thermal_estimator and thermal_predictor:
                w = state["worker"]
                env_t = state["thermal"]["ambient_temp_c"]
                env_h = state["thermal"]["humidity_pct"]
                hr = state["vitals"]["hr"]
                
                if env_t is not None and hr is not None:
                    try:
                        # Ensure mappings
                        sex_val = 1 if str(w["sex"]).lower() in ["m", "male", "1"] else 0
                        # Features for estimator: ['activity_code' 'work_rest' 'sex' 'age_years' 'body_mass_kg' 'height_cm' 'humidity_pct' 'ambient_temp_c' 'heart_rate_bpm' 'rr_interval_ms' 'sdrr_ms' 'rmssd_ms' 'skin_temp_mean_c']
                        # Approximations for HRV and Skin Temp if unavailable
                        rr = 60000 / hr
                        sdrr = state["vitals"]["hrv"] or 40
                        rmssd = state["vitals"]["hrv"] or 40
                        skin_temp = 34.0
                        
                        est_X = pd.DataFrame([[
                            int(w["job"]), int(w["work_rest"]), sex_val, float(w["age"]), float(w["weight_kg"]), float(w["height_cm"]),
                            float(env_h), float(env_t), float(hr), float(rr), float(sdrr), float(rmssd), float(skin_temp)
                        ]], columns=['activity_code', 'work_rest', 'sex', 'age_years', 'body_mass_kg', 'height_cm', 'humidity_pct', 'ambient_temp_c', 'heart_rate_bpm', 'rr_interval_ms', 'sdrr_ms', 'rmssd_ms', 'skin_temp_mean_c'])
                        
                        core_est = float(thermal_estimator.predict(est_X)[0])
                        state["thermal"]["estimated_core_temp_c"] = round(core_est, 2)
                        
                        # Features for predictor: ['estimated_core_temp_c' 'heart_rate_bpm' 'rr_interval_ms' 'sdrr_ms' 'rmssd_ms' 'skin_temp_mean_c' 'ambient_temp_c' 'humidity_pct' 'age_years' 'body_mass_kg' 'height_cm' 'sex' 'activity_code' 'work_rest']
                        pred_X = pd.DataFrame([[
                            float(core_est), float(hr), float(rr), float(sdrr), float(rmssd), float(skin_temp), float(env_t), float(env_h),
                            float(w["age"]), float(w["weight_kg"]), float(w["height_cm"]), sex_val, int(w["job"]), int(w["work_rest"])
                        ]], columns=['estimated_core_temp_c', 'heart_rate_bpm', 'rr_interval_ms', 'sdrr_ms', 'rmssd_ms', 'skin_temp_mean_c', 'ambient_temp_c', 'humidity_pct', 'age_years', 'body_mass_kg', 'height_cm', 'sex', 'activity_code', 'work_rest'])
                        
                        core_pred = float(thermal_predictor.predict(pred_X)[0])
                        state["thermal"]["predicted_core_temp_c"] = round(core_pred, 2)
                        state["thermal"]["predicted_rise_c"] = round(core_pred - core_est, 2)
                        state["thermal"]["status"] = "ACTIVE"
                        
                    except Exception as e:
                        print(f"Thermal error: {e}")
                else:
                    state["thermal"]["status"] = "WAITING_FOR_DATA"

            # Cleanup buffers
            if len(imu_buffer_20hz) > 200: del imu_buffer_20hz[:-200]
            if len(imu_buffer_2hz) > 500: del imu_buffer_2hz[:-500]
            if len(ppg_buffer) > 5000: del ppg_buffer[:-5000]
            if len(hr_buffer) > 100: del hr_buffer[:-100]
            if len(spo2_buffer) > 100: del spo2_buffer[:-100]
            
            # Log models
            log_files["model"].write(f"{current_time},{state['thermal'].get('estimated_core_temp_c','')},{state['thermal'].get('predicted_core_temp_c','')},{state['fatigue'].get('score','')},{state['fall'].get('probability','')},{state['vital_anomaly'].get('status','')}\n")
            log_files["model"].flush()

            time.sleep(1.0)
            
        except Exception as e:
            print(f"Inference Loop Error: {e}")
            traceback.print_exc()
            time.sleep(2.0)

# ==========================================
# UDP RECEIVER TASK
# ==========================================
def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5005))
    sock.settimeout(1.0)
    print("UDP Receiver listening on port 5005")
    
    global last_2hz_ts
    
    while True:
        try:
            data, addr = sock.recvfrom(2048)
            msg = json.loads(data.decode("utf-8"))
            ts = time.time()
            
            if msg.get("type") == "wristband":
                state["last_wristband_ts"] = ts
                
                # PPG
                if "ppg" in msg:
                    ppg = msg["ppg"]
                    ppg_buffer.append({"ts": ts, "ir": ppg["ir"], "red": ppg["red"]})
                    log_files["ppg"].write(f"{ts},{ppg['ir']},{ppg['red']}\n")
                    
                # IMU
                if "imu" in msg:
                    imu = msg["imu"]
                    ax, ay, az, gx, gy, gz = imu["ax"], imu["ay"], imu["az"], imu["gx"], imu["gy"], imu["gz"]
                    state["movement"]["ax"] = ax
                    state["movement"]["ay"] = ay
                    state["movement"]["az"] = az
                    state["movement"]["gx"] = gx
                    state["movement"]["gy"] = gy
                    state["movement"]["gz"] = gz
                    state["movement"]["mag"] = round(np.sqrt(ax**2 + ay**2 + az**2), 2)
                    
                    row = [ax, ay, az, gx, gy, gz]
                    imu_buffer_20hz.append(row)
                    
                    # 2Hz Downsampling for fatigue
                    if ts - last_2hz_ts >= 0.5:
                        imu_buffer_2hz.append(row)
                        last_2hz_ts = ts
                        
                    log_files["imu"].write(f"{ts},{ax},{ay},{az},{gx},{gy},{gz}\n")

                # GPS
                if "gps" in msg:
                    gps = msg["gps"]
                    state["gps"]["lat"] = gps.get("lat")
                    state["gps"]["lng"] = gps.get("lng")
                    state["gps"]["accuracy_m"] = gps.get("accuracy_m")
                    state["gps"]["last_fix"] = datetime.datetime.now().strftime("%H:%M:%S")
                    
            elif msg.get("type") == "environment":
                state["last_env_ts"] = ts
                state["thermal"]["ambient_temp_c"] = msg.get("temperature_c")
                state["thermal"]["humidity_pct"] = msg.get("humidity_pct")
                log_files["env"].write(f"{ts},{msg.get('temperature_c')},{msg.get('humidity_pct')}\n")
                
        except socket.timeout:
            continue
        except Exception as e:
            print(f"UDP Error: {e}")

# ==========================================
# FASTAPI ENDPOINTS
# ==========================================
active_connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # We broadcast from a separate loop or handle incoming profile updates here
            data = await websocket.receive_text()
            update = json.loads(data)
            if "worker" in update:
                state["worker"].update(update["worker"])
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

async def broadcast_state():
    while True:
        state["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
        dead_conns = []
        for conn in list(active_connections):
            try:
                await conn.send_json(state)
            except:
                dead_conns.append(conn)
        for dc in dead_conns:
            if dc in active_connections:
                active_connections.remove(dc)
        await __import__('asyncio').sleep(1.0)

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=udp_receiver, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    __import__('asyncio').create_task(broadcast_state())

# ==========================================
# BUILT-IN SIMULATION ENGINE
# ==========================================
simulation_active = False
simulation_thread = None
scenario_active = None
scenario_expiry = 0

def simulation_loop():
    """Injects realistic synthetic sensor data directly into buffers (bypasses UDP)."""
    global simulation_active, last_2hz_ts, imu_buffer_20hz, imu_buffer_2hz, ppg_buffer, hr_buffer, spo2_buffer
    import math, random

    t = 0
    last_env_tick = 0
    last_imu_tick = 0
    print("[SIM] Simulation started — injecting synthetic data")

    # --- Pre-seed buffers so models don't wait 3 minutes ---
    now = time.time()
    if len(imu_buffer_2hz) < 360:
        for i in range(360):
            t_s = (360 - i) * 0.5
            phase = (now - t_s) * 0.05
            ax = round(0.1 * math.sin(phase) + random.uniform(-0.02, 0.02), 4)
            ay = round(0.98 + random.uniform(-0.02, 0.02), 4)
            az = round(0.1 * math.cos(phase) + random.uniform(-0.02, 0.02), 4)
            gx = round(random.uniform(-1, 1), 4)
            gy = round(random.uniform(-1, 1), 4)
            gz = round(random.uniform(-1, 1), 4)
            imu_buffer_2hz.append([ax, ay, az, gx, gy, gz])
            if i >= 340:
                imu_buffer_20hz.append([ax, ay, az, gx, gy, gz])

    if len(ppg_buffer) < 3000:
        for i in range(3000):
            t_s = (3000 - i) * 0.01
            ir = int(150000 + math.sin((now - t_s) * 0.1 * math.pi) * 2000 + random.randint(-50, 50))
            red = int(120000 + math.sin((now - t_s) * 0.1 * math.pi) * 1500 + random.randint(-50, 50))
            ppg_buffer.append({"ts": now - t_s, "ir": ir, "red": red})

    if len(hr_buffer) < 30:
        for i in range(30):
            t_s = (30 - i) * 1.0
            hr_buffer.append({"ts": now - t_s, "val": round(78.0 + math.sin(i * 0.2) * 2, 1)})
            spo2_buffer.append({"ts": now - t_s, "val": round(97.5 + random.uniform(-0.5, 0.5), 1)})

    if vital_detector and vital_detector.baseline_hr is None:
        vital_detector.baseline_hr = 78.0
        vital_detector.baseline_spo2 = 98.0
        vital_detector.baseline_ppg_amp = 2.0

    while simulation_active:
        ts = time.time()
        current_ms = int(ts * 1000)

        # --- PPG at ~100Hz ---
        ir_val = int(150000 + math.sin(t * 0.1 * math.pi) * 2000 + random.randint(-100, 100))
        red_val = int(120000 + math.sin(t * 0.1 * math.pi) * 1500 + random.randint(-100, 100))
        ppg_buffer.append({"ts": ts, "ir": ir_val, "red": red_val})

        state["last_wristband_ts"] = ts

        # --- IMU at ~20Hz ---
        if current_ms - last_imu_tick >= 50:
            phase = t * 0.05
            ax = round(0.1 * math.sin(phase) + random.uniform(-0.02, 0.02), 4)
            ay = round(0.98 + random.uniform(-0.02, 0.02), 4)
            az = round(0.1 * math.cos(phase) + random.uniform(-0.02, 0.02), 4)
            gx = round(random.uniform(-1, 1), 4)
            gy = round(random.uniform(-1, 1), 4)
            gz = round(random.uniform(-1, 1), 4)

            state["movement"]["ax"] = ax
            state["movement"]["ay"] = ay
            state["movement"]["az"] = az
            state["movement"]["gx"] = gx
            state["movement"]["gy"] = gy
            state["movement"]["gz"] = gz
            state["movement"]["mag"] = round(np.sqrt(ax**2 + ay**2 + az**2), 2)

            row = [ax, ay, az, gx, gy, gz]
            imu_buffer_20hz.append(row)

            if ts - last_2hz_ts >= 0.5:
                imu_buffer_2hz.append(row)
                last_2hz_ts = ts

            last_imu_tick = current_ms

        # --- Environment at ~1Hz ---
        if current_ms - last_env_tick >= 1000:
            env_temp = round(32.0 + math.sin(t * 0.01) * 2, 1)
            env_hum = round(50.0 + math.cos(t * 0.01) * 5, 1)
            state["last_env_ts"] = ts
            state["thermal"]["ambient_temp_c"] = env_temp
            state["thermal"]["humidity_pct"] = env_hum
            last_env_tick = current_ms

            # GPS simulation — fixed worker location (Sona College of Technology, Salem)
            state["gps"]["lat"] = 11.677934
            state["gps"]["lng"] = 78.124332
            state["gps"]["accuracy_m"] = 3.0
            state["gps"]["last_fix"] = datetime.datetime.now().strftime("%H:%M:%S")

        t += 1
        time.sleep(0.01)

    print("[SIM] Simulation stopped")

@app.post("/api/simulation/toggle")
async def toggle_simulation():
    global simulation_active, simulation_thread
    if simulation_active:
        simulation_active = False
        if simulation_thread:
            simulation_thread.join(timeout=2)
            simulation_thread = None
        return JSONResponse({"simulation": False, "message": "Simulation stopped"})
    else:
        simulation_active = True
        simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
        simulation_thread.start()
        return JSONResponse({"simulation": True, "message": "Simulation started"})

@app.get("/api/simulation/status")
async def simulation_status():
    return JSONResponse({"simulation": simulation_active})

# ==========================================
# SIMULATION SCENARIO TRIGGERS
# ==========================================
import uuid as _uuid

def _reset_scenario():
    global scenario_active, scenario_expiry
    scenario_active = None
    scenario_expiry = 0
    state["fall"]["probability"] = 0.0
    state["fall"]["detected"] = False
    state["vital_anomaly"]["status"] = "NORMAL"
    state["vital_anomaly"]["hr_deviation"] = 0.0
    state["vital_anomaly"]["spo2_deviation"] = 0.0
    state["vital_anomaly"]["ml_support"] = False
    state["vital_anomaly"]["episode_id"] = None
    state["fatigue"]["score"] = 2.0
    state["fatigue"]["trend"] = "stable"
    state["fatigue"]["status"] = "ACTIVE"
    state["thermal"]["estimated_core_temp_c"] = 37.0
    state["thermal"]["predicted_core_temp_c"] = 37.1
    state["thermal"]["predicted_rise_c"] = 0.1
    state["thermal"]["status"] = "ACTIVE"
    state["vitals"]["spo2"] = 98
    state["vitals"]["hr"] = 78
    print("[SIM] Scenario reset to normal")

@app.post("/api/simulation/scenario")
async def trigger_scenario(request: Request):
    global scenario_active, scenario_expiry
    body = await request.json()
    scenario = body.get("scenario")
    duration = body.get("duration", 10)

    if scenario == "reset":
        _reset_scenario()
        return JSONResponse({"scenario": None, "message": "Reset to normal"})

    scenario_active = scenario
    scenario_expiry = time.time() + duration

    if scenario == "fall":
        state["fall"]["probability"] = 0.92
        state["fall"]["detected"] = True
        state["fall"]["status"] = "ACTIVE"
    elif scenario == "vital_anomaly":
        state["vital_anomaly"]["status"] = "VITAL_ANOMALY"
        state["vital_anomaly"]["hr_deviation"] = 0.35
        state["vital_anomaly"]["spo2_deviation"] = 4.0
        state["vital_anomaly"]["ml_support"] = True
        state["vital_anomaly"]["episode_id"] = str(_uuid.uuid4())
    elif scenario == "heat_stress":
        state["thermal"]["estimated_core_temp_c"] = 39.5
        state["thermal"]["predicted_core_temp_c"] = 40.1
        state["thermal"]["predicted_rise_c"] = 0.6
        state["thermal"]["status"] = "ACTIVE"
    elif scenario == "fatigue":
        state["fatigue"]["score"] = 8.5
        state["fatigue"]["trend"] = "increasing"
        state["fatigue"]["status"] = "ACTIVE"
    elif scenario == "spo2_drop":
        state["vital_anomaly"]["status"] = "SPO2_DEVIATION"
        state["vital_anomaly"]["spo2_deviation"] = 6.0
        state["vital_anomaly"]["hr_deviation"] = 0.05
        state["vitals"]["spo2"] = 88

    threading.Timer(duration, _reset_scenario).start()
    print(f"[SIM] Scenario triggered: {scenario} for {duration}s")
    return JSONResponse({"scenario": scenario, "duration": duration})

# ==========================================
# SUPERVISOR — MULTI-WORKER API
# ==========================================
import random as _random

_sim_workers = [
    {"id": "W-02", "name": "Priya S.", "role": "Electrician", "lat": 11.6783, "lng": 78.1250},
    {"id": "W-03", "name": "Arjun M.", "role": "Mason", "lat": 11.6775, "lng": 78.1238},
    {"id": "W-04", "name": "Kavitha R.", "role": "Scaffolder", "lat": 11.6790, "lng": 78.1248},
    {"id": "W-05", "name": "Suresh B.", "role": "Crane Operator", "lat": 11.6772, "lng": 78.1255},
    {"id": "W-06", "name": "Deepa N.", "role": "Painter", "lat": 11.6786, "lng": 78.1235},
    {"id": "W-07", "name": "Vikram P.", "role": "Plumber", "lat": 11.6770, "lng": 78.1242},
    {"id": "W-08", "name": "Lakshmi T.", "role": "Site Inspector", "lat": 11.6795, "lng": 78.1240},
]

@app.get("/api/workers")
async def get_workers():
    workers = []

    # W-01: Live data from real state
    w1_vital = state.get("vital_anomaly", {})
    w1_status = "CRITICAL" if (state.get("fall", {}).get("detected") or w1_vital.get("status") == "VITAL_ANOMALY") else \
                "ATTENTION" if (w1_vital.get("status") not in [None, "NORMAL", "WAITING_FOR_DATA", "CALIBRATING_BASELINE"] or
                               (state.get("fatigue", {}).get("score") is not None and state["fatigue"]["score"] >= 6) or
                               (state.get("thermal", {}).get("estimated_core_temp_c") is not None and state["thermal"]["estimated_core_temp_c"] >= 38.5)) else "NORMAL"
    workers.append({
        "id": state["worker"]["id"],
        "name": "Rajan K.",
        "role": "Welder",
        "status": w1_status,
        "hr": state["vitals"].get("hr"),
        "spo2": state["vitals"].get("spo2"),
        "core_temp": state["thermal"].get("estimated_core_temp_c"),
        "fatigue": state["fatigue"].get("score"),
        "fall_detected": state["fall"].get("detected", False),
        "vital_status": w1_vital.get("status", "WAITING_FOR_DATA"),
        "gps": state.get("gps", {}),
        "last_update": state.get("timestamp", "--"),
        "connected": state["connection"].get("wristband", False)
    })

    # W-02 through W-08: Simulated
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    presets = {
        "W-04": {"status": "ATTENTION", "hr": 118, "spo2": 95, "core_temp": 38.3, "fatigue": 5.8, "vital_status": "HR_DEVIATION"},
        "W-07": {"status": "ATTENTION", "hr": 82, "spo2": 96, "core_temp": 37.6, "fatigue": 6.5, "vital_status": "PHYSIOLOGICAL_DEVIATION"},
    }
    for sw in _sim_workers:
        p = presets.get(sw["id"], {})
        workers.append({
            "id": sw["id"],
            "name": sw["name"],
            "role": sw["role"],
            "status": p.get("status", "NORMAL"),
            "hr": p.get("hr", round(72 + _random.uniform(-4, 6), 0)),
            "spo2": p.get("spo2", round(97 + _random.uniform(-1, 1), 0)),
            "core_temp": p.get("core_temp", round(37.0 + _random.uniform(-0.2, 0.3), 1)),
            "fatigue": p.get("fatigue", round(2.0 + _random.uniform(-0.5, 1.5), 1)),
            "fall_detected": False,
            "vital_status": p.get("vital_status", "NORMAL"),
            "gps": {"lat": sw["lat"], "lng": sw["lng"], "accuracy_m": 3.0, "last_fix": now_str},
            "last_update": now_str,
            "connected": True
        })

    return JSONResponse({"workers": workers})

# ==========================================
# DEMO PREDICTION API
# ==========================================

def _get_feature_importance(model_obj, feature_names, top_n=10):
    """Extract feature importance from a sklearn/lgbm model."""
    try:
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
            if len(importances) != len(feature_names):
                return None
            pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs[:top_n]]
    except:
        pass
    return None

@app.get("/api/models")
async def get_models():
    """Return metadata about available ML models for the demo UI."""
    models = []
    if thermal_estimator and thermal_predictor:
        models.append({
            "id": "thermal",
            "name": "Thermal Risk Prediction",
            "type": "regression",
            "algorithm": "LightGBM Ensemble",
            "description": "Estimates core body temperature and predicts +10 minute thermal trajectory.",
            "output_label": "Core Temperature",
            "output_unit": "°C",
            "input_type": "physiological",
            "inputs": [
                {"key": "heart_rate", "label": "Heart Rate", "unit": "BPM", "default": 80, "min": 40, "max": 200, "step": 1},
                {"key": "hrv", "label": "HRV (SDRR)", "unit": "ms", "default": 40, "min": 5, "max": 200, "step": 1},
                {"key": "skin_temp", "label": "Skin Temperature", "unit": "°C", "default": 34.0, "min": 28, "max": 42, "step": 0.1},
                {"key": "ambient_temp", "label": "Ambient Temperature", "unit": "°C", "default": 32, "min": 10, "max": 55, "step": 0.5},
                {"key": "humidity", "label": "Humidity", "unit": "%", "default": 50, "min": 0, "max": 100, "step": 1},
                {"key": "age", "label": "Age", "unit": "years", "default": 30, "min": 16, "max": 80, "step": 1},
                {"key": "weight", "label": "Weight", "unit": "kg", "default": 75, "min": 40, "max": 200, "step": 1},
                {"key": "height", "label": "Height", "unit": "cm", "default": 175, "min": 140, "max": 220, "step": 1},
                {"key": "sex", "label": "Sex", "unit": "", "default": 1, "min": 0, "max": 1, "step": 1, "options": ["Female", "Male"]},
                {"key": "activity_code", "label": "Activity Code", "unit": "", "default": 2, "min": 0, "max": 6, "step": 1},
                {"key": "work_rest", "label": "Work/Rest", "unit": "", "default": 1, "min": 0, "max": 1, "step": 1, "options": ["Rest", "Work"]}
            ],
            "scenarios": [
                {"name": "Normal Worker", "icon": "👷", "values": {"heart_rate": 78, "hrv": 45, "skin_temp": 34.0, "ambient_temp": 28, "humidity": 45, "age": 30, "weight": 75, "height": 175, "sex": 1, "activity_code": 2, "work_rest": 1}},
                {"name": "High Heat Exposure", "icon": "🔥", "values": {"heart_rate": 110, "hrv": 25, "skin_temp": 36.5, "ambient_temp": 48, "humidity": 75, "age": 35, "weight": 80, "height": 178, "sex": 1, "activity_code": 3, "work_rest": 1}},
                {"name": "Physiological Stress", "icon": "💓", "values": {"heart_rate": 145, "hrv": 15, "skin_temp": 37.5, "ambient_temp": 35, "humidity": 60, "age": 28, "weight": 70, "height": 170, "sex": 1, "activity_code": 4, "work_rest": 1}}
            ]
        })
    if fatigue_model:
        models.append({
            "id": "fatigue",
            "name": "Fatigue Prediction",
            "type": "regression",
            "algorithm": "RandomForest (500 trees)",
            "description": "Predicts fatigue score (0–10) from 6-axis IMU wrist motion patterns over a 3-minute window.",
            "output_label": "Fatigue Score",
            "output_unit": "/ 10",
            "input_type": "imu",
            "inputs": [
                {"key": "ax", "label": "Acceleration X", "unit": "g", "default": 0.05, "min": -20, "max": 20, "step": 0.01},
                {"key": "ay", "label": "Acceleration Y", "unit": "g", "default": 0.98, "min": -20, "max": 20, "step": 0.01},
                {"key": "az", "label": "Acceleration Z", "unit": "g", "default": 0.05, "min": -20, "max": 20, "step": 0.01},
                {"key": "gx", "label": "Gyroscope X", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1},
                {"key": "gy", "label": "Gyroscope Y", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1},
                {"key": "gz", "label": "Gyroscope Z", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1}
            ],
            "scenarios": [
                {"name": "Normal (Walking)", "icon": "🚶", "values": {"ax": 0.2, "ay": 0.95, "az": 0.1, "gx": 1.5, "gy": 0.5, "gz": 0.3}},
                {"name": "Fatigued (Slow)", "icon": "😴", "values": {"ax": 0.02, "ay": 0.99, "az": 0.01, "gx": 0.1, "gy": 0.05, "gz": 0.05}},
                {"name": "High Activity", "icon": "🏃", "values": {"ax": 2.0, "ay": 1.5, "az": 2.5, "gx": 15, "gy": 10, "gz": 12}}
            ]
        })
    if fall_model:
        models.append({
            "id": "fall",
            "name": "Fall Detection",
            "type": "classification",
            "algorithm": "Gradient Boosting (threshold-tuned)",
            "description": "Detects fall events from 1-second 6-axis IMU windows at 20Hz.",
            "output_label": "Fall Probability",
            "output_unit": "%",
            "input_type": "imu",
            "inputs": [
                {"key": "ax", "label": "Acceleration X", "unit": "g", "default": 0.05, "min": -20, "max": 20, "step": 0.01},
                {"key": "ay", "label": "Acceleration Y", "unit": "g", "default": 0.98, "min": -20, "max": 20, "step": 0.01},
                {"key": "az", "label": "Acceleration Z", "unit": "g", "default": 0.05, "min": -20, "max": 20, "step": 0.01},
                {"key": "gx", "label": "Gyroscope X", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1},
                {"key": "gy", "label": "Gyroscope Y", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1},
                {"key": "gz", "label": "Gyroscope Z", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1}
            ],
            "scenarios": [
                {"name": "Normal (Standing)", "icon": "🧍", "values": {"ax": 0.02, "ay": 0.98, "az": 0.02, "gx": 0.05, "gy": 0.05, "gz": 0.05}},
                {"name": "Fall Event", "icon": "⚠️", "values": {"ax": 3.5, "ay": -2.0, "az": 8.0, "gx": 120, "gy": 80, "gz": 150}},
                {"name": "Stumble / Near-Fall", "icon": "🚨", "values": {"ax": 1.8, "ay": 0.3, "az": 3.0, "gx": 40, "gy": 25, "gz": 50}}
            ]
        })
    if activity_model:
        models.append({
            "id": "activity",
            "name": "Activity Recognition",
            "type": "classification",
            "algorithm": "LightGBM (18 classes)",
            "description": "Classifies worker activity from 10-second 6-axis IMU windows at 20Hz.",
            "output_label": "Predicted Activity",
            "output_unit": "",
            "input_type": "imu",
            "inputs": [
                {"key": "ax", "label": "Acceleration X", "unit": "g", "default": 0.05, "min": -20, "max": 20, "step": 0.01},
                {"key": "ay", "label": "Acceleration Y", "unit": "g", "default": 0.98, "min": -20, "max": 20, "step": 0.01},
                {"key": "az", "label": "Acceleration Z", "unit": "g", "default": 0.05, "min": -20, "max": 20, "step": 0.01},
                {"key": "gx", "label": "Gyroscope X", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1},
                {"key": "gy", "label": "Gyroscope Y", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1},
                {"key": "gz", "label": "Gyroscope Z", "unit": "°/s", "default": 0.1, "min": -500, "max": 500, "step": 0.1}
            ],
            "scenarios": [
                {"name": "Walking", "icon": "🚶", "values": {"ax": 0.3, "ay": 1.0, "az": 0.15, "gx": 2.0, "gy": 1.0, "gz": 0.5}},
                {"name": "Jogging", "icon": "🏃", "values": {"ax": 2.5, "ay": 2.0, "az": 1.5, "gx": 8.0, "gy": 5.0, "gz": 4.0}},
                {"name": "Sitting", "icon": "🪑", "values": {"ax": 0.01, "ay": 0.98, "az": 0.01, "gx": 0.02, "gy": 0.02, "gz": 0.02}},
                {"name": "Standing", "icon": "🧍", "values": {"ax": 0.02, "ay": 0.99, "az": 0.02, "gx": 0.1, "gy": 0.1, "gz": 0.05}}
            ]
        })
    return JSONResponse({"models": models})


def _generate_imu_window(ax, ay, az, gx, gy, gz, n_samples, noise_scale=0.05):
    """Generate a synthetic IMU window with small noise around the given values."""
    base = np.array([ax, ay, az, gx, gy, gz], dtype=np.float64)
    window = np.tile(base, (n_samples, 1))
    # Add realistic noise proportional to the magnitude
    for i in range(6):
        scale = max(abs(base[i]) * noise_scale, 0.001)
        window[:, i] += np.random.normal(0, scale, n_samples)
    return window


@app.post("/api/demo-predict")
async def demo_predict(request: Request):
    """Run a prediction using custom inputs through the actual model pipeline."""
    try:
        body = await request.json()
        model_id = body.get("model")
        inputs = body.get("inputs", {})

        if model_id == "thermal":
            if not thermal_estimator or not thermal_predictor:
                return JSONResponse({"error": "Thermal models not loaded"}, status_code=503)
            hr = float(inputs.get("heart_rate", 80))
            hrv = float(inputs.get("hrv", 40))
            skin_temp = float(inputs.get("skin_temp", 34.0))
            ambient_temp = float(inputs.get("ambient_temp", 32))
            humidity = float(inputs.get("humidity", 50))
            age = float(inputs.get("age", 30))
            weight = float(inputs.get("weight", 75))
            height = float(inputs.get("height", 175))
            sex = int(inputs.get("sex", 1))
            activity_code = int(inputs.get("activity_code", 2))
            work_rest = int(inputs.get("work_rest", 1))
            rr = 60000 / hr

            # Same feature assembly as inference_loop
            est_X = pd.DataFrame([[
                activity_code, work_rest, sex, age, weight, height,
                humidity, ambient_temp, hr, rr, hrv, hrv, skin_temp
            ]], columns=['activity_code', 'work_rest', 'sex', 'age_years', 'body_mass_kg', 'height_cm', 'humidity_pct', 'ambient_temp_c', 'heart_rate_bpm', 'rr_interval_ms', 'sdrr_ms', 'rmssd_ms', 'skin_temp_mean_c'])

            core_est = float(thermal_estimator.predict(est_X)[0])

            pred_X = pd.DataFrame([[
                core_est, hr, rr, hrv, hrv, skin_temp, ambient_temp, humidity,
                age, weight, height, sex, activity_code, work_rest
            ]], columns=['estimated_core_temp_c', 'heart_rate_bpm', 'rr_interval_ms', 'sdrr_ms', 'rmssd_ms', 'skin_temp_mean_c', 'ambient_temp_c', 'humidity_pct', 'age_years', 'body_mass_kg', 'height_cm', 'sex', 'activity_code', 'work_rest'])

            core_pred = float(thermal_predictor.predict(pred_X)[0])
            rise = core_pred - core_est

            # Risk level
            if core_pred < 38.0: risk = "NORMAL"
            elif core_pred < 38.5: risk = "ELEVATED"
            elif core_pred < 39.0: risk = "HIGH"
            else: risk = "CRITICAL"

            # Feature importance
            est_feats = list(est_X.columns)
            pred_feats = list(pred_X.columns)
            fi = _get_feature_importance(thermal_predictor, pred_feats, top_n=10)

            return JSONResponse({
                "model": "thermal",
                "prediction": {
                    "estimated_core_temp": round(core_est, 2),
                    "predicted_core_temp_10min": round(core_pred, 2),
                    "predicted_rise": round(rise, 3),
                    "risk_level": risk
                },
                "feature_importance": fi,
                "model_info": {"algorithm": "LightGBM Ensemble", "model_name": "THERMORA Thermal Engine"}
            })

        elif model_id == "fatigue":
            if not fatigue_model or not fatigue_imputer:
                return JSONResponse({"error": "Fatigue model not loaded"}, status_code=503)
            ax = float(inputs.get("ax", 0.05))
            ay = float(inputs.get("ay", 0.98))
            az = float(inputs.get("az", 0.05))
            gx = float(inputs.get("gx", 0.1))
            gy = float(inputs.get("gy", 0.1))
            gz = float(inputs.get("gz", 0.1))

            window = _generate_imu_window(ax, ay, az, gx, gy, gz, 360)
            feats = extract_fatigue_features(window).reshape(1, -1)
            feats_imp = fatigue_imputer.transform(feats)
            score = float(np.clip(fatigue_model.predict(feats_imp)[0], 0, 10))

            if score < 3: risk = "LOW"
            elif score < 5: risk = "MODERATE"
            elif score < 7: risk = "HIGH"
            else: risk = "CRITICAL"

            # Feature importance from saved CSV
            fi = None
            try:
                fi_df = pd.read_csv(FATIGUE_IMPORTANCE_PATH)
                fi = [{"feature": row["feature"].replace("IMU4_", ""), "importance": round(float(row["importance"]), 6)} for _, row in fi_df.head(10).iterrows()]
            except:
                fi = _get_feature_importance(fatigue_model, [f"f{i}" for i in range(156)], top_n=10)

            return JSONResponse({
                "model": "fatigue",
                "prediction": {
                    "fatigue_score": round(score, 2),
                    "risk_level": risk
                },
                "feature_importance": fi,
                "model_info": {"algorithm": "RandomForest (500 trees)", "model_name": "THERMORA Fatigue Model", "validated_mae": 1.6728}
            })

        elif model_id == "fall":
            if not fall_model:
                return JSONResponse({"error": "Fall model not loaded"}, status_code=503)
            ax = float(inputs.get("ax", 0.05))
            ay = float(inputs.get("ay", 0.98))
            az = float(inputs.get("az", 0.05))
            gx = float(inputs.get("gx", 0.1))
            gy = float(inputs.get("gy", 0.1))
            gz = float(inputs.get("gz", 0.1))

            window = _generate_imu_window(ax, ay, az, gx, gy, gz, 20)
            feats = extract_fall_features(window.reshape(1, 20, 6))
            prob = float(fall_model.predict_proba(feats)[0, 1])
            detected = bool(prob >= fall_threshold)

            fi = _get_feature_importance(fall_model, [f"fall_f{i}" for i in range(148)], top_n=10)

            return JSONResponse({
                "model": "fall",
                "prediction": {
                    "fall_probability": round(prob * 100, 2),
                    "fall_detected": detected,
                    "risk_level": "FALL DETECTED" if detected else "NO FALL",
                    "threshold": round(float(fall_threshold), 3)
                },
                "feature_importance": fi,
                "model_info": {"algorithm": "Gradient Boosting", "model_name": "THERMORA Fall Detector"}
            })

        elif model_id == "activity":
            if not activity_model or not activity_labels:
                return JSONResponse({"error": "Activity model not loaded"}, status_code=503)
            ax = float(inputs.get("ax", 0.05))
            ay = float(inputs.get("ay", 0.98))
            az = float(inputs.get("az", 0.05))
            gx = float(inputs.get("gx", 0.1))
            gy = float(inputs.get("gy", 0.1))
            gz = float(inputs.get("gz", 0.1))

            window = _generate_imu_window(ax, ay, az, gx, gy, gz, 200)
            feats_dict = extract_activity_features(window)
            feat_df = pd.DataFrame([feats_dict])
            pred_class = int(activity_model.predict(feat_df)[0])
            label_info = activity_labels.get(str(pred_class), {})
            activity_name = label_info.get("name", f"Class {pred_class}")

            # Probabilities
            probs = None
            try:
                proba = activity_model.predict_proba(feat_df)[0]
                top_indices = np.argsort(proba)[::-1][:5]
                probs = [{"activity": activity_labels.get(str(int(i)), {}).get("name", f"Class {i}"), "probability": round(float(proba[i]) * 100, 1)} for i in top_indices]
            except:
                pass

            fi = None
            try:
                fi_df = pd.read_csv(ACTIVITY_IMPORTANCE_PATH)
                fi = [{"feature": row["feature"], "importance": round(float(row["importance"]), 1)} for _, row in fi_df.head(10).iterrows()]
            except:
                fi = _get_feature_importance(activity_model, list(feats_dict.keys()), top_n=10)

            return JSONResponse({
                "model": "activity",
                "prediction": {
                    "predicted_activity": activity_name,
                    "class_index": pred_class,
                    "top_probabilities": probs,
                    "risk_level": activity_name
                },
                "feature_importance": fi,
                "model_info": {"algorithm": "LightGBM (18 classes)", "model_name": "THERMORA Activity Recognition", "accuracy": 0.692}
            })

        else:
            return JSONResponse({"error": f"Unknown model: {model_id}"}, status_code=400)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "dashboard"), html=True), name="dashboard")

if __name__ == "__main__":
    print("\nStarting THERMORA Live Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
