import socket
import json
import time
import random

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

heart_rate = 82
spo2 = 98.0
hrv = 52

print("THERMORA WRISTBAND")
print("Starting UDP transmission...")
print()

for _ in range(20):

    # Smoothly change synthetic values
    heart_rate += random.uniform(-2.5, 2.5)
    spo2 += random.uniform(-0.2, 0.1)
    hrv += random.uniform(-3, 3)

    heart_rate = max(65, min(120, heart_rate))
    spo2 = max(94, min(99, spo2))
    hrv = max(30, min(75, hrv))

    packet = {
        "heart_rate": round(heart_rate),
        "spo2": round(spo2, 1),
        "hrv": round(hrv)
    }

    message = json.dumps(packet)

    sock.sendto(
        message.encode(),
        (SERVER_IP, SERVER_PORT)
    )

    time.sleep(1)

sock.close()
print("UDP transmission stopped.")