import numpy as np
import time

FS = 20
DT = 1 / FS

print("=" * 60)
print("THERMORA IMU SIMULATOR")
print("=" * 60)

print()
print("Generating simulated 6-axis wrist IMU data...")
print("Press Ctrl+C to stop.")
print()

rng = np.random.default_rng(42)

while True:

    # Normal wrist movement
    ax = rng.normal(0, 1.2)
    ay = rng.normal(0, 1.2)
    az = 9.81 + rng.normal(0, 1.0)

    gx = rng.normal(0, 2.0)
    gy = rng.normal(0, 2.0)
    gz = rng.normal(0, 2.0)

    print(
        f"acc=({ax:6.2f}, {ay:6.2f}, {az:6.2f})  "
        f"gyro=({gx:6.2f}, {gy:6.2f}, {gz:6.2f})"
    )

    time.sleep(DT)