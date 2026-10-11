import json
import time
import socket
import random
import math

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

print("==========================================")
print("  THERMORA DASHBOARD - SIMULATION MODE    ")
print("==========================================")
print(f"Sending synthetic data to {UDP_IP}:{UDP_PORT}")
print("Press Ctrl+C to stop.")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Simulation state
t = 0
last_env_send = 0
last_imu_send = 0

try:
    while True:
        current_ms = int(time.time() * 1000)
        
        # 1. Wristband (PPG @ 100Hz, IMU @ 20Hz appended)
        doc_wrist = {
            "device": "WRISTBAND_SIM",
            "type": "wristband",
            "timestamp_ms": current_ms,
            "ppg": {
                "ir": int(150000 + math.sin(t * 0.1 * math.pi) * 2000 + random.randint(-100, 100)),
                "red": int(120000 + math.sin(t * 0.1 * math.pi) * 1500 + random.randint(-100, 100))
            }
        }
        
        # Add IMU at ~20Hz
        if current_ms - last_imu_send >= 50:
            # Simulate walking motion
            ax = 0.1 * math.sin(t * 0.05) + random.uniform(-0.02, 0.02)
            ay = 0.98 + random.uniform(-0.02, 0.02)
            az = 0.1 * math.cos(t * 0.05) + random.uniform(-0.02, 0.02)
            doc_wrist["imu"] = {
                "ax": ax, "ay": ay, "az": az,
                "gx": random.uniform(-1, 1), "gy": random.uniform(-1, 1), "gz": random.uniform(-1, 1)
            }
            last_imu_send = current_ms
            
        sock.sendto(json.dumps(doc_wrist).encode("utf-8"), (UDP_IP, UDP_PORT))
        
        # 2. Environment (1Hz)
        if current_ms - last_env_send >= 1000:
            doc_env = {
                "device": "ENV_SIM",
                "type": "environment",
                "timestamp_ms": current_ms,
                "temperature_c": round(32.0 + math.sin(t * 0.01) * 2, 1),
                "humidity_pct": round(50.0 + math.cos(t * 0.01) * 5, 1)
            }
            sock.sendto(json.dumps(doc_env).encode("utf-8"), (UDP_IP, UDP_PORT))
            last_env_send = current_ms
            print(f"Sent {current_ms}: Env & IMU tick")
            
        t += 1
        time.sleep(0.01) # ~100Hz loop
        
except KeyboardInterrupt:
    print("\nSimulator stopped.")
