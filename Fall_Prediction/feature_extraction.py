import os
import numpy as np


# Load dataset
project_path = os.path.dirname(__file__)
data_path = os.path.join(project_path, "bits2_imu_windows.npz")

data = np.load(data_path)

X = data["X"]
y = data["y"]


# Channel names
channel_names = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z"
]

feature_types = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms"
]


# Create feature names
feature_names = []

for channel in channel_names:
    for feature in feature_types:
        feature_names.append(channel + "_" + feature)


# Extract features
features = []

for window in X:

    window_features = []

    for channel in range(6):

        signal = window[:, channel]

        mean_value = np.mean(signal)
        std_value = np.std(signal)
        min_value = np.min(signal)
        max_value = np.max(signal)
        range_value = max_value - min_value
        rms_value = np.sqrt(np.mean(signal ** 2))

        window_features.extend([
            mean_value,
            std_value,
            min_value,
            max_value,
            range_value,
            rms_value
        ])

    features.append(window_features)


features = np.array(features)


# Results
print("======================================")
print("THERMORA FALL DETECTION")
print("FEATURE EXTRACTION")
print("======================================")

print("\nOriginal X shape:", X.shape)
print("Feature matrix shape:", features.shape)

print("\nNumber of features per window:", features.shape[1])

print("\nFeature names:")
print(feature_names)

print("\nFirst window features:")
print(features[0])

print("\nFirst window label:")
print(y[0])