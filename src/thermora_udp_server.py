import socket
import json
from datetime import datetime

HOST = "0.0.0.0"
PORT = 5005

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((HOST, PORT))

print("=" * 60)
print("             THERMORA UDP SENSOR SERVER")
print("=" * 60)
print(f"[UDP] Listening on {HOST}:{PORT}")
print("[UDP] Waiting for wristband data...")
print()

while True:
    data, address = server.recvfrom(1024)

    try:
        packet = json.loads(data.decode())

        hr = packet["heart_rate"]
        spo2 = packet["spo2"]
        hrv = packet["hrv"]

        timestamp = datetime.now().strftime("%H:%M:%S")

        print(f"[{timestamp}] PACKET RECEIVED")
        print(f"  SOURCE   : {address[0]}:{address[1]}")
        print(f"  HR       : {hr} BPM")
        print(f"  SpO2     : {spo2:.1f} %")
        print(f"  HRV      : {hrv} ms")
        print("  STATUS   : NORMAL")
        print("-" * 60)

    except Exception as e:
        print("[ERROR] Invalid packet:", e)