import time
import random
import os

# Synthetic THERMORA vital-sign demonstration
# NOTE: These are artificially generated values for demonstration only.

duration = 15
start_time = time.time()

# Starting values
heart_rate = 82
spo2 = 98
hrv = 52

while time.time() - start_time < duration:

    # Smoothly vary the values
    heart_rate += random.uniform(-2.5, 2.5)
    spo2 += random.uniform(-0.3, 0.2)
    hrv += random.uniform(-3, 3)

    # Keep values within realistic demonstration ranges
    heart_rate = max(65, min(120, heart_rate))
    spo2 = max(94, min(99, spo2))
    hrv = max(30, min(75, hrv))

    # Clear terminal
    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 55)
    print("              THERMORA")
    print("        WRISTBAND VITAL MONITOR")
    print("=" * 55)

    print()
    print(f"  HEART RATE       : {heart_rate:.0f} BPM")
    print(f"  SpO2             : {spo2:.1f} %")
    print(f"  HRV              : {hrv:.0f} ms")

    print()
    print("-" * 55)
    print("  STATUS           : MONITORING")
    print("  SENSOR           : MAX30102")
    print("-" * 55)

    time.sleep(1)

print()
print("=" * 55)
print("             MONITORING COMPLETE")
print("=" * 55)