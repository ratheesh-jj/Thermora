# THERMORA Live Dashboard & End-to-End Pipeline

This document explains how to boot up and run the live THERMORA dashboard using actual hardware sensors.

## Step 1: Find Laptop IP
Open your terminal (Command Prompt or PowerShell) and run:
```cmd
ipconfig
```
Look for `IPv4 Address` under your active Wi-Fi adapter (e.g., `192.168.1.x`).

## Step 2: Configure ESP32 Sketches
Open the two ESP32 firmware sketches in the Arduino IDE:
1. `firmware\wristband\wristband.ino`
2. `firmware\environment_node\environment_node.ino`

Change the following variables at the top of both files:
```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* LAPTOP_IP = "YOUR_LAPTOP_IP_FROM_STEP_1"; 
```

## Step 3: Flash Hardware
1. Connect the Wristband ESP32 to your PC and upload `wristband.ino`.
2. Connect the Environment Node ESP32 to your PC and upload `environment_node.ino`.
*(Ensure the MPU6050, MAX30102, and DHT11 are correctly wired to the pins specified in the sketches).*

## Step 4: Start the Server
Double click the script:
`run_thermora_dashboard.bat`

This will automatically start the FastAPI server with the machine learning inference pipeline on UDP port `5005` and WebSocket port `8000`.

## Step 5: Open the Dashboard
Open your web browser and navigate to:
`C:\Users\OMEN\Desktop\Projects\SIH26\dashboard\index.html` (or `http://localhost:8000` if serving the html).

## Step 6: Verify Operation
- The dashboard should show **CONNECTED** for both the Wristband and Environment.
- Live values should stream in.
- ML models (Thermal, Fatigue, Fall, Vital Anomaly) will transition from `WAITING FOR DATA` to active outputs once sufficient data accumulates in their rolling buffers.
- All raw CSV logs are automatically saved to `results\live_sensor_data\`.

---

### Simulation Mode (No Hardware)
If you don't have the hardware connected but want to test the dashboard, you can run the simulator:
```cmd
.venv\Scripts\python.exe src\thermora_dashboard_test_mode.py
```
This will inject synthetic UDP packets into the pipeline. Note that since the simulator generates highly erratic/random IMU and PPG data, some models may trigger anomalies.
