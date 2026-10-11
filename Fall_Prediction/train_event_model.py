import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

PROJECT_DIR = r"C:\Thermora Fall Prediction Model"

DATA_FILE = os.path.join(
    PROJECT_DIR,
    "thermora_full_event_dataset.npz"
)

META_FILE = os.path.join(
    PROJECT_DIR,
    "thermora_full_event_metadata.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_DIR,
    "thermora_event_model.pkl"
)


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_features(X):

    features = []

    for window in X:

        row = []

        # -------------------------------------------------
        # Raw channels
        # -------------------------------------------------

        for ch in range(6):

            s = window[:, ch]

            row.extend([
                np.mean(s),
                np.std(s),
                np.min(s),
                np.max(s),
                np.max(s) - np.min(s),
                np.sqrt(np.mean(s ** 2)),
                np.mean(np.abs(s)),
                np.percentile(s, 10),
                np.percentile(s, 25),
                np.percentile(s, 50),
                np.percentile(s, 75),
                np.percentile(s, 90),
            ])

        # -------------------------------------------------
        # Acceleration magnitude
        # -------------------------------------------------

        acc = window[:, :3]
        acc_mag = np.linalg.norm(acc, axis=1)

        row.extend([
            np.mean(acc_mag),
            np.std(acc_mag),
            np.min(acc_mag),
            np.max(acc_mag),
            np.ptp(acc_mag),
            np.sqrt(np.mean(acc_mag ** 2)),
            np.mean(np.abs(acc_mag)),
            np.percentile(acc_mag, 90),
            np.percentile(acc_mag, 95),
            np.percentile(acc_mag, 99),
        ])

        # -------------------------------------------------
        # Gyroscope magnitude
        # -------------------------------------------------

        gyro = window[:, 3:]
        gyro_mag = np.linalg.norm(gyro, axis=1)

        row.extend([
            np.mean(gyro_mag),
            np.std(gyro_mag),
            np.min(gyro_mag),
            np.max(gyro_mag),
            np.ptp(gyro_mag),
            np.sqrt(np.mean(gyro_mag ** 2)),
            np.mean(np.abs(gyro_mag)),
            np.percentile(gyro_mag, 90),
            np.percentile(gyro_mag, 95),
            np.percentile(gyro_mag, 99),
        ])

        # -------------------------------------------------
        # Jerk
        # -------------------------------------------------

        jerk = np.diff(acc_mag)

        if len(jerk) > 0:

            row.extend([
                np.mean(jerk),
                np.std(jerk),
                np.min(jerk),
                np.max(jerk),
                np.sqrt(np.mean(jerk ** 2)),
                np.mean(np.abs(jerk)),
                np.percentile(np.abs(jerk), 90),
                np.percentile(np.abs(jerk), 95),
                np.percentile(np.abs(jerk), 99),
            ])

        else:

            row.extend([0] * 9)

        # -------------------------------------------------
        # First half / second half
        # -------------------------------------------------

        mid = len(acc_mag) // 2

        first = acc_mag[:mid]
        second = acc_mag[mid:]

        row.extend([
            np.mean(first),
            np.std(first),
            np.max(first),

            np.mean(second),
            np.std(second),
            np.max(second),

            np.mean(second) - np.mean(first),
            np.max(second) - np.max(first),
        ])

        # -------------------------------------------------
        # High acceleration fractions
        # -------------------------------------------------

        for threshold in [8, 10, 12, 15, 20]:

            row.append(
                np.mean(acc_mag > threshold)
            )

        # -------------------------------------------------
        # High gyro fractions
        # -------------------------------------------------

        for threshold in [2, 4, 6, 8, 10]:

            row.append(
                np.mean(gyro_mag > threshold)
            )

        # -------------------------------------------------
        # Axis correlations
        # -------------------------------------------------

        try:
            corr_acc = np.corrcoef(acc.T)

            row.extend([
                corr_acc[0, 1],
                corr_acc[0, 2],
                corr_acc[1, 2],
            ])

        except:
            row.extend([0, 0, 0])

        try:
            corr_gyro = np.corrcoef(gyro.T)

            row.extend([
                corr_gyro[0, 1],
                corr_gyro[0, 2],
                corr_gyro[1, 2],
            ])

        except:
            row.extend([0, 0, 0])

        features.append(row)

    return np.asarray(features, dtype=np.float32)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("THERMORA EVENT MODEL TRAINING")
    print("=" * 60)

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    data = np.load(DATA_FILE)

    X = data["X"]
    y = data["y"]

    meta = pd.read_csv(META_FILE)

    print("\nDataset:")
    print("X:", X.shape)
    print("y:", y.shape)

    # -----------------------------------------------------
    # PARTICIPANT-INDEPENDENT SPLIT
    # -----------------------------------------------------

    train_users = list(range(1, 30))
    val_users = list(range(30, 36))
    test_users = list(range(36, 42))

    train_mask = meta["user"].isin(train_users).to_numpy()
    val_mask = meta["user"].isin(val_users).to_numpy()
    test_mask = meta["user"].isin(test_users).to_numpy()

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("\nSplits:")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    print("\nLabels:")
    print("Train:", np.bincount(y_train))
    print("Val  :", np.bincount(y_val))
    print("Test :", np.bincount(y_test))

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    print("\nExtracting features...")

    F_train = extract_features(X_train)
    F_val = extract_features(X_val)
    F_test = extract_features(X_test)

    print("Feature matrix:")
    print("Train:", F_train.shape)
    print("Val  :", F_val.shape)
    print("Test :", F_test.shape)

    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    models = {

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    best_model = None
    best_f1 = -1
    best_name = None

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(F_train, y_train)

        prob = model.predict_proba(F_val)[:, 1]

        pred = (prob >= 0.5).astype(int)

        accuracy = accuracy_score(y_val, pred)
        precision = precision_score(
            y_val,
            pred,
            zero_division=0
        )
        recall = recall_score(
            y_val,
            pred,
            zero_division=0
        )
        f1 = f1_score(
            y_val,
            pred,
            zero_division=0
        )

        print(
            f"{name}: "
            f"Accuracy={accuracy:.4f} "
            f"Precision={precision:.4f} "
            f"Recall={recall:.4f} "
            f"F1={f1:.4f}"
        )

        results.append(
            (name, accuracy, precision, recall, f1)
        )

        if f1 > best_f1:

            best_f1 = f1
            best_model = model
            best_name = name

    # -----------------------------------------------------
    # THRESHOLD SEARCH
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("THRESHOLD SEARCH")
    print("=" * 60)

    val_prob = best_model.predict_proba(
        F_val
    )[:, 1]

    best_threshold = 0.5
    best_threshold_f1 = -1

    for threshold in np.arange(
        0.20,
        0.81,
        0.01
    ):

        pred = (
            val_prob >= threshold
        ).astype(int)

        f1 = f1_score(
            y_val,
            pred,
            zero_division=0
        )

        if f1 > best_threshold_f1:

            best_threshold_f1 = f1
            best_threshold = threshold

    print(
        f"\nBest threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_threshold_f1:.4f}"
    )

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    test_prob = best_model.predict_proba(
        F_test
    )[:, 1]

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        test_pred
    )

    precision = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    print("\nModel:", best_name)
    print("Threshold:", round(best_threshold, 2))
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            test_pred,
            target_names=[
                "Non-event",
                "Fall-event"
            ],
            zero_division=0
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            test_pred
        )
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    artifact = {

        "model": best_model,

        "threshold": float(
            best_threshold
        ),

        "feature_count": F_train.shape[1],

        "window_size": 20,

        "sampling_rate": 20,

        "channels": [
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z"
        ],

        "label_mapping": {
            0: "non_event",
            1: "fall_event_candidate"
        },

        "validation_f1": float(
            best_threshold_f1
        ),

        "test_accuracy": float(
            accuracy
        ),

        "test_precision": float(
            precision
        ),

        "test_recall": float(
            recall
        ),

        "test_f1": float(
            f1
        )
    }

    joblib.dump(
        artifact,
        MODEL_FILE
    )

    print("\nModel saved:")
    print(MODEL_FILE)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()