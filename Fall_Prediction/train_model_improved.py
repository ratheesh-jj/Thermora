import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")

# ============================================================
# THERMORA FALL DETECTION - IMPROVED TRAINING PIPELINE
# ============================================================
# IMPORTANT:
# - This script keeps the participant-independent split.
# - It adds substantially stronger temporal/statistical features.
# - It compares multiple tree models and selects the best model
#   using VALIDATION F1 (not the test set).
# - It tunes the fall threshold on validation data.
# - It saves the selected model + threshold + feature names.
#
# 98-99% accuracy cannot be honestly guaranteed from BITS-2.
# BITS-2 uses recording-level fall labels, so some "fall" windows
# may not contain the actual fall event. The goal here is to
# maximize real generalization without test-set leakage.
# ============================================================

project_path = os.path.dirname(__file__)

data_path = os.path.join(project_path, "bits2_imu_windows.npz")
metadata_path = os.path.join(project_path, "window_metadata.csv")

print("\nLoading dataset...")
data = np.load(data_path)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int8)
metadata = pd.read_csv(metadata_path)

print("Raw X:", X.shape)
print("Raw y:", y.shape)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def safe_corr(a, b):
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def add_signal_features(out, prefix, s):
    """Strong statistical + temporal features for one signal."""
    s = np.asarray(s, dtype=np.float32)

    mean = np.mean(s)
    std = np.std(s)
    mn = np.min(s)
    mx = np.max(s)
    rng = mx - mn
    rms = np.sqrt(np.mean(s * s))
    median = np.median(s)
    q10, q25, q75, q90 = np.percentile(s, [10, 25, 75, 90])

    # Temporal change
    d1 = np.diff(s)
    d2 = np.diff(s, n=2)

    out.extend([
        mean, std, mn, mx, rng, rms, median,
        q10, q25, q75, q90,
        np.mean(np.abs(s)),
        np.mean(np.abs(d1)),
        np.std(d1),
        np.max(np.abs(d1)),
        np.sqrt(np.mean(d1 * d1)),
        np.mean(np.abs(d2)),
        np.std(d2),
        np.max(np.abs(d2)),
        np.sqrt(np.mean(d2 * d2)),
        np.max(np.abs(s - mean)),
    ])


def extract_features(window):
    ax, ay, az = window[:, 0], window[:, 1], window[:, 2]
    gx, gy, gz = window[:, 3], window[:, 4], window[:, 5]

    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    # Magnitude jerk / change
    jerk = np.diff(acc_mag)

    # Useful derived signals
    horiz_acc = np.sqrt(ax * ax + ay * ay)
    gyro_energy = gx * gx + gy * gy + gz * gz

    f = []

    signals = [
        ("acc_x", ax), ("acc_y", ay), ("acc_z", az),
        ("gyro_x", gx), ("gyro_y", gy), ("gyro_z", gz),
        ("acc_mag", acc_mag), ("gyro_mag", gyro_mag),
        ("jerk", jerk), ("horiz_acc", horiz_acc),
        ("gyro_energy", gyro_energy),
    ]

    for _, s in signals:
        add_signal_features(f, "", s)

    # --------------------------------------------------------
    # Peak / impact features
    # --------------------------------------------------------
    for s in [acc_mag, gyro_mag, np.abs(jerk)]:
        f.extend([
            np.max(s),
            np.argmax(s) / max(1, len(s) - 1),
            np.percentile(s, 95),
            np.percentile(s, 99),
            np.mean(s > np.percentile(s, 90)),
            np.mean(s > np.percentile(s, 95)),
        ])

    # --------------------------------------------------------
    # First-half vs second-half dynamics
    # --------------------------------------------------------
    n = len(acc_mag)
    mid = max(1, n // 2)

    for s in [acc_mag, gyro_mag, jerk]:
        a = s[:mid]
        b = s[mid:]
        f.extend([
            np.mean(a),
            np.mean(b),
            np.std(a),
            np.std(b),
            np.max(a),
            np.max(b),
            np.mean(b) - np.mean(a),
            np.std(b) - np.std(a),
        ])

    # --------------------------------------------------------
    # Axis relationships
    # --------------------------------------------------------
    f.extend([
        safe_corr(ax, ay),
        safe_corr(ax, az),
        safe_corr(ay, az),
        safe_corr(gx, gy),
        safe_corr(gx, gz),
        safe_corr(gy, gz),
        safe_corr(ax, gx),
        safe_corr(ay, gy),
        safe_corr(az, gz),
    ])

    # --------------------------------------------------------
    # Cross-axis energy / ratios
    # --------------------------------------------------------
    acc_energy = np.mean(ax * ax + ay * ay + az * az)
    gyro_energy_mean = np.mean(gyro_energy)

    f.extend([
        acc_energy,
        gyro_energy_mean,
        np.sqrt(acc_energy),
        np.sqrt(gyro_energy_mean),
        np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)),
        np.mean(np.abs(gx) + np.abs(gy) + np.abs(gz)),
    ])

    return f


print("\nExtracting improved features...")
features = np.asarray(
    [extract_features(w) for w in X],
    dtype=np.float32
)

print("Improved feature matrix:", features.shape)


# ============================================================
# PARTICIPANT-INDEPENDENT SPLIT
# ============================================================

train_users = list(range(1, 30))
val_users = list(range(30, 36))
test_users = list(range(36, 42))

train_mask = metadata["user"].isin(train_users).to_numpy()
val_mask = metadata["user"].isin(val_users).to_numpy()
test_mask = metadata["user"].isin(test_users).to_numpy()

X_train, y_train = features[train_mask], y[train_mask]
X_val, y_val = features[val_mask], y[val_mask]
X_test, y_test = features[test_mask], y[test_mask]

print("\n======================================")
print("DATA SPLIT")
print("======================================")
print("Train:", X_train.shape, np.bincount(y_train))
print("Val:  ", X_val.shape, np.bincount(y_val))
print("Test: ", X_test.shape, np.bincount(y_test))


# ============================================================
# MODELS
# ============================================================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )
}


# ============================================================
# TRAIN + VALIDATE
# ============================================================

results = []

print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

best_model = None
best_name = None
best_val_f1 = -1.0

for name, candidate in models.items():

    print(f"\nTraining {name}...")
    candidate.fit(X_train, y_train)

    val_prob = candidate.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.50).astype(np.int8)

    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    rec = recall_score(y_val, val_pred)
    prec = precision_score(y_val, val_pred)

    results.append([name, acc, prec, rec, f1])

    print(
        f"{name}: "
        f"accuracy={acc:.4f}, "
        f"precision={prec:.4f}, "
        f"recall={rec:.4f}, "
        f"F1={f1:.4f}"
    )

    if f1 > best_val_f1:
        best_val_f1 = f1
        best_model = candidate
        best_name = name

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

print("\nValidation model ranking:")
print(results_df.sort_values("F1", ascending=False).to_string(index=False))

print("\nSelected model:", best_name)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

val_prob = best_model.predict_proba(X_val)[:, 1]

threshold_results = []

# Fine threshold search
for threshold in np.arange(0.20, 0.801, 0.01):

    pred = (val_prob >= threshold).astype(np.int8)

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)

# Select threshold by F1, then accuracy as tie-breaker.
threshold_df = threshold_df.sort_values(
    ["f1", "accuracy"],
    ascending=[False, False]
)

threshold = float(threshold_df.iloc[0]["threshold"])

print("\n======================================")
print("BEST VALIDATION THRESHOLD")
print("======================================")
print(threshold_df.head(10).to_string(index=False))
print("\nSelected threshold:", threshold)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(np.int8)

test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred, zero_division=0)
test_recall = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)

print("\n======================================")
print("FINAL TEST RESULTS")
print("======================================")

print(f"Model:     {best_name}")
print(f"Threshold: {threshold:.2f}")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1:        {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=["No Fall", "Fall"],
    zero_division=0
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, test_pred)
print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

feature_names = [f"feature_{i}" for i in range(features.shape[1])]

model_path = os.path.join(project_path, "thermora_fall_model.pkl")

artifact = {
    "model": best_model,
    "threshold": threshold,
    "feature_names": feature_names,
    "n_features": features.shape[1],
    "window_samples": X.shape[1],
    "channels": [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"
    ],
    "sampling_rate_hz": 20,
    "label_mapping": {
        0: "No Fall",
        1: "Fall"
    },
    "selected_model": best_name,
    "validation_f1": best_val_f1,
    "test_accuracy": test_accuracy,
    "test_f1": test_f1
}

joblib.dump(artifact, model_path)

print("\n======================================")
print("MODEL SAVED")
print("======================================")
print(model_path)

print("\nIMPORTANT:")
print("Do NOT manually enter feature values for THERMORA.")
print("The next prediction script will accept raw 20x6 IMU windows")
print("and reproduce this exact feature extraction pipeline.")
