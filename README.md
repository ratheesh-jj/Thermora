# THERMORA 🌡️

> **Multimodal Industrial Worker Health, Safety & Heat-Stress Monitoring Platform**  
> *Developed for Smart India Hackathon (SIH)*

THERMORA is an end-to-end wearable IoT and edge-intelligence platform designed to safeguard industrial workers from heat strain, acute fatigue, sudden falls, and vital anomalies in hazardous working environments.

---

## 🚀 Key Highlights & Capabilities

- **Core Temperature Estimation & Heat Strain Forecasting**: Real-time core body temperature estimation and 10-minute predictive thermal forecasting using multimodal physiological and ambient inputs (no wind dependencies needed).
- **Fatigue & Exertion Monitoring**: Real-time fatigue scoring leveraging IMU window dynamics and physiological signatures.
- **Fall Detection Engine**: High-fidelity, low-latency impact and post-fall immobility classifier.
- **Vital Anomaly Detection**: Unsupervised and supervised anomaly detection on continuous PPG, Heart Rate, and SpO2 signals.
- **Activity Recognition**: 6-axis IMU human activity classification (LightGBM/XGBoost).
- **Dual Dashboard Architecture**:
  - **Live Worker Dashboard**: Real-time websocket telemetry, biometric trends, and alerts.
  - **Supervisor Operations Dashboard**: Site-wide overview, SOS emergency tracking, and safety metrics.
- **Hardware Integration**: Wi-Fi UDP wireless telemetry transmission from ESP32 wearable nodes (Wristband & Ambient Station).

---

## 🛠️ System Architecture

```text
[ESP32 Wristband] (MPU6050 + MAX30102) \
                                         ---> [UDP Port 5005] ---> [THERMORA Server (FastAPI)]
[ESP32 Environment Node] (DHT11)       /                               |
                                                                        +---> ML Multi-modal Inference
                                                                        |     (Thermal, Fatigue, Fall, Vitals)
                                                                        |
                                                                        +---> WebSocket (Port 8000)
                                                                        |
                                                                        v
                                                            [Interactive Dashboards]
                                                            (Worker Live View & Supervisor Portal)
```

---

## 📂 Repository Layout

```text
.
├── api/                        # Flask backend services
├── dashboard/                  # Live telemetry & supervisor operations dashboards
│   ├── index.html              # Real-time worker health monitor
│   ├── supervisor.html         # Plant-wide supervisor console
│   ├── app.js                  # Worker WebSocket & chart controller
│   └── supervisor.js           # Operations & SOS event dispatcher
├── data/                       # PROSPIE dataset sequences and dictionaries
├── Fall_Prediction/            # Fall detection training scripts & deployable models
├── fatigue_prediction/         # Fatigue modeling and deployable feature extractors
├── firmware/                   # Microcontroller C++ sketches
│   ├── wristband/              # ESP32 wearable firmware (IMU + Pulse Oximeter)
│   └── environment_node/       # ESP32 environmental station (Temp + Humidity)
├── models/                     # Core temperature estimation & 10-min forecast models
├── results/                    # Model validation benchmarks & live session logs
├── src/                        # Core Python pipelines & servers
│   ├── thermora_live_server.py # Central FastAPI UDP receiver & WebSocket broadcaster
│   ├── thermora_dashboard_test_mode.py # Synthetic live hardware simulator
│   └── vital_anomaly_inference.py # PPG and vital anomaly inference pipeline
├── vital_anomaly/              # Vital signal calibration and anomaly models
├── web/                        # React + Vite web dashboard application
├── README_LIVE_DASHBOARD.md    # Dedicated hardware bring-up guide
├── requirements.txt            # Python dependencies
└── run_thermora_dashboard.bat  # 1-click launcher for Windows
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (for `web/` React portal)
- Arduino IDE (if compiling ESP32 firmware)

### 2. Python Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launching the Live Server & Dashboard

#### Option A: One-Click Launch (Windows)
Double-click `run_thermora_dashboard.bat`.

#### Option B: Terminal Launch
```powershell
python src\thermora_live_server.py
```
Then open `dashboard\index.html` or visit [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🧪 Simulation Mode (No Hardware Needed)

You can run full multi-modal inference and dashboard streaming without physical hardware:

1. Start the live server:
   ```powershell
   python src\thermora_live_server.py
   ```
2. In a separate terminal, run the sensor simulator:
   ```powershell
   python src\thermora_dashboard_test_mode.py
   ```
3. Open `dashboard\index.html` to see synthetic worker telemetry streaming in real time.

---

## 🌐 React Web Portal (Optional)

If running the React frontend in `web/`:
```powershell
cd web
npm install
npm run dev
```

---

## 📡 Hardware Deployment

1. Open `firmware/wristband/wristband.ino` and `firmware/environment_node/environment_node.ino`.
2. Configure your Wi-Fi credentials (`ssid`, `password`) and workstation IP (`LAPTOP_IP`).
3. Flash the ESP32 boards and power them on.
4. Telemetry will automatically stream via UDP port `5005` to the live pipeline.
For detailed wiring and pin mapping, see [README_LIVE_DASHBOARD.md](README_LIVE_DASHBOARD.md).
