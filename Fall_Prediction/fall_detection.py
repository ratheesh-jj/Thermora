import os
import numpy as np

# Locate dataset
project_path = os.path.dirname(__file__)
data_path = os.path.join(project_path, "bits2_imu_windows.npz")

# Load dataset
data = np.load(data_path)

X = data["X"]
y = data["y"]

print("========== DATASET ==========")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\n========== LABEL DISTRIBUTION ==========")

unique_labels, counts = np.unique(y, return_counts=True)

for label, count in zip(unique_labels, counts):
    percentage = (count / len(y)) * 100
    print(f"Label {label}: {count} windows ({percentage:.2f}%)")

print("\n========== FIRST 20 LABELS ==========")
print(y[:20])

print("\n========== DATA RANGE ==========")
print(f"Minimum value in X: {X.min():.4f}")
print(f"Maximum value in X: {X.max():.4f}")

print("\n========== COMPLETE ==========")