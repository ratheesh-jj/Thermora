import os
import numpy as np
import pandas as pd


# --------------------------------------------------
# Load data
# --------------------------------------------------

project_path = os.path.dirname(__file__)

data_path = os.path.join(
    project_path,
    "bits2_imu_windows.npz"
)

metadata_path = os.path.join(
    project_path,
    "window_metadata.csv"
)

data = np.load(data_path)

X = data["X"]
y = data["y"]

metadata = pd.read_csv(metadata_path)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

features = []

for window in X:

    acc_x = window[:, 0]
    acc_y = window[:, 1]
    acc_z = window[:, 2]

    gyro_x = window[:, 3]
    gyro_y = window[:, 4]
    gyro_z = window[:, 5]

    # Acceleration magnitude
    acc_mag = np.sqrt(
        acc_x**2 + acc_y**2 + acc_z**2
    )

    # Gyroscope magnitude
    gyro_mag = np.sqrt(
        gyro_x**2 + gyro_y**2 + gyro_z**2
    )

    # Jerk: change in acceleration
    jerk = np.diff(acc_mag)

    window_features = []

    # Basic features for each individual channel
    for signal in [
        acc_x, acc_y, acc_z,
        gyro_x, gyro_y, gyro_z
    ]:

        window_features.extend([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.max(signal) - np.min(signal),
            np.sqrt(np.mean(signal**2))
        ])

    # Acceleration magnitude features
    window_features.extend([
        np.mean(acc_mag),
        np.std(acc_mag),
        np.min(acc_mag),
        np.max(acc_mag),
        np.max(acc_mag) - np.min(acc_mag),
        np.sqrt(np.mean(acc_mag**2))
    ])

    # Gyroscope magnitude features
    window_features.extend([
        np.mean(gyro_mag),
        np.std(gyro_mag),
        np.min(gyro_mag),
        np.max(gyro_mag),
        np.max(gyro_mag) - np.min(gyro_mag),
        np.sqrt(np.mean(gyro_mag**2))
    ])

    # Jerk features
    window_features.extend([
        np.mean(jerk),
        np.std(jerk),
        np.min(jerk),
        np.max(jerk),
        np.max(jerk) - np.min(jerk),
        np.sqrt(np.mean(jerk**2))
    ])

    features.append(window_features)


features = np.array(features)

print("\nFeature matrix shape:", features.shape)


# --------------------------------------------------
# Participant-independent split
# --------------------------------------------------

train_users = list(range(1, 30))
val_users = list(range(30, 36))
test_users = list(range(36, 42))

train_mask = metadata["user"].isin(train_users)
val_mask = metadata["user"].isin(val_users)
test_mask = metadata["user"].isin(test_users)

X_train = features[train_mask]
y_train = y[train_mask]

X_val = features[val_mask]
y_val = y[val_mask]

X_test = features[test_mask]
y_test = y[test_mask]


# --------------------------------------------------
# Display split information
# --------------------------------------------------

print("======================================")
print("THERMORA FALL DETECTION")
print("DATA SPLIT")
print("======================================")

print("\nTraining:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTest:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nClass distribution:")

print("\nTraining")
print(np.bincount(y_train))

print("\nValidation")
print(np.bincount(y_val))

print("\nTest")
print(np.bincount(y_test))
# --------------------------------------------------
# Random Forest model
# --------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

print("\n======================================")
print("RANDOM FOREST TRAINING")
print("======================================")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# Validation
# --------------------------------------------------

val_accuracy = model.score(X_val, y_val)

print("\nValidation accuracy:")
print(val_accuracy)
# --------------------------------------------------
# Test evaluation
# --------------------------------------------------

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print("\n======================================")
print("TEST SET EVALUATION")
print("======================================")

y_test_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["No Fall", "Fall"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns


# Feature names
feature_names = []

channels = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z"
]

basic_features = [
    "mean", "std", "min", "max", "range", "rms"
]

for channel in channels:
    for feature in basic_features:
        feature_names.append(f"{channel}_{feature}")

for feature in basic_features:
    feature_names.append(f"acc_mag_{feature}")

for feature in basic_features:
    feature_names.append(f"gyro_mag_{feature}")

for feature in basic_features:
    feature_names.append(f"jerk_{feature}")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Fall", "Fall"],
    yticklabels=["No Fall", "Fall"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("THERMORA Fall Detection - Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n======================================")
print("TOP 15 IMPORTANT FEATURES")
print("======================================")

print(
    importance_df.head(15).to_string(index=False)
)


# Plot top 15
top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("THERMORA Fall Detection - Top 15 Features")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()
# --------------------------------------------------
# Threshold tuning
# --------------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score

print("\n======================================")
print("FALL DETECTION THRESHOLD TEST")
print("======================================")

# Get probability of FALL
val_prob = model.predict_proba(X_val)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

print("\nThreshold | Precision | Recall | F1")
print("--------------------------------------")

for threshold in thresholds:

    val_pred = (val_prob >= threshold).astype(int)

    precision = precision_score(y_val, val_pred)
    recall = recall_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.3f} | "
        f"{recall:6.3f} | "
        f"{f1:5.3f}"
    )
    # --------------------------------------------------
# Final threshold evaluation on TEST SET
# --------------------------------------------------

threshold = 0.45

test_prob = model.predict_proba(X_test)[:, 1]

test_threshold_pred = (
    test_prob >= threshold
).astype(int)


print("\n======================================")
print("TEST RESULTS AT THRESHOLD 0.45")
print("======================================")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_threshold_pred,
        target_names=["No Fall", "Fall"]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_threshold_pred
    )
)