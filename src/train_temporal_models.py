import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "data/sequences/train_10min.npz"
VAL_FILE = "data/sequences/validation_10min.npz"
TEST_FILE = "data/sequences/test_10min.npz"

RESULTS_DIR = "results"
MODEL_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def load_sequence_file(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    metadata = data["metadata"]

    return X, y, metadata


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def print_results(name, y_true, y_pred):

    mae, rmse, r2 = metrics(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE :  {mae:.4f} °C")
    print(f"RMSE:  {rmse:.4f} °C")
    print(f"R²  :  {r2:.4f}")

    return mae, rmse, r2


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

FEATURE_NAMES = [
    "age_years",
    "body_mass_kg",
    "ambient_temp_c",
    "humidity_pct",
    "heart_rate_bpm",
    "insulated_skin_temp_c",
    "sex_male"
]


def calculate_slope(values):

    """
    Calculate linear slope over the 20-minute window.
    """

    n = values.shape[1]

    x = np.arange(n, dtype=np.float32)

    x_mean = x.mean()

    denominator = np.sum((x - x_mean) ** 2)

    slopes = []

    for i in range(values.shape[0]):

        y = values[i]

        y_mean = y.mean()

        numerator = np.sum((x - x_mean) * (y - y_mean))

        slopes.append(numerator / denominator)

    return np.array(slopes, dtype=np.float32)


def engineer_features(X):

    """
    Convert:

        samples × 20 time steps × 7 features

    into:

        samples × engineered temporal features
    """

    # --------------------------------------------------------
    # Missing-value handling
    # --------------------------------------------------------

    n_samples, timesteps, n_features = X.shape

    X_clean = X.copy()

    # Feature-wise median imputation across time
    for f in range(n_features):

        values = X_clean[:, :, f]

        median = np.nanmedian(values)

        values[np.isnan(values)] = median

        X_clean[:, :, f] = values

    features = []

    # --------------------------------------------------------
    # Statistics for every feature
    # --------------------------------------------------------

    for f, name in enumerate(FEATURE_NAMES):

        values = X_clean[:, :, f]

        first = values[:, 0]
        last = values[:, -1]

        mean = values.mean(axis=1)
        std = values.std(axis=1)
        minimum = values.min(axis=1)
        maximum = values.max(axis=1)

        delta = last - first

        slope = calculate_slope(values)

        features.extend([
            first,
            last,
            mean,
            std,
            minimum,
            maximum,
            delta,
            slope
        ])

    # --------------------------------------------------------
    # Important physiological interaction features
    # --------------------------------------------------------

    hr = X_clean[:, :, 4]
    skin = X_clean[:, :, 5]
    ambient = X_clean[:, :, 2]
    humidity = X_clean[:, :, 3]

    # HR change
    hr_delta = hr[:, -1] - hr[:, 0]

    # Skin temperature change
    skin_delta = skin[:, -1] - skin[:, 0]

    # Ambient temperature change
    ambient_delta = ambient[:, -1] - ambient[:, 0]

    # Skin temperature relative to ambient
    skin_ambient_gap = skin[:, -1] - ambient[:, -1]

    # HR / skin temperature interaction
    hr_skin_interaction = hr[:, -1] * skin[:, -1]

    features.extend([
        hr_delta,
        skin_delta,
        ambient_delta,
        skin_ambient_gap,
        hr_skin_interaction
    ])

    return np.column_stack(features).astype(np.float32)


# ============================================================
# GRU MODEL
# ============================================================

class ThermoSentinelGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last_output = output[:, -1, :]

        return self.regressor(last_output).squeeze(1)


def prepare_gru_data(X_train, X_val, X_test):

    # --------------------------------------------------------
    # Fit imputer ONLY on training data
    # --------------------------------------------------------

    n_features = X_train.shape[-1]

    imputer = SimpleImputer(strategy="median")

    train_2d = X_train.reshape(-1, n_features)

    imputer.fit(train_2d)

    def transform(X):

        shape = X.shape

        X2 = X.reshape(-1, n_features)

        X2 = imputer.transform(X2)

        return X2.reshape(shape).astype(np.float32)

    X_train = transform(X_train)
    X_val = transform(X_val)
    X_test = transform(X_test)

    # --------------------------------------------------------
    # Standardize using training data only
    # --------------------------------------------------------

    scaler = StandardScaler()

    train_2d = X_train.reshape(-1, n_features)

    scaler.fit(train_2d)

    def scale(X):

        shape = X.shape

        X2 = X.reshape(-1, n_features)

        X2 = scaler.transform(X2)

        return X2.reshape(shape).astype(np.float32)

    X_train = scale(X_train)
    X_val = scale(X_val)
    X_test = scale(X_test)

    return X_train, X_val, X_test


def train_gru(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nGRU device:", device)

    X_train, X_val, X_test = prepare_gru_data(
        X_train,
        X_val,
        X_test
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    model = ThermoSentinelGRU(
        input_size=X_train.shape[-1]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    criterion = nn.HuberLoss()

    best_val_loss = float("inf")

    patience = 10
    patience_counter = 0

    print("\nTraining GRU...")

    for epoch in range(1, 101):

        model.train()

        train_losses = []

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            predictions = model(xb)

            loss = criterion(
                predictions,
                yb
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_losses.append(loss.item())

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                predictions = model(xb)

                loss = criterion(
                    predictions,
                    yb
                )

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                f"{MODEL_DIR}/thermora_gru_10min.pt"
            )

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print("\nEarly stopping.")

            break

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            f"{MODEL_DIR}/thermora_gru_10min.pt",
            map_location=device
        )
    )

    model.eval()

    predictions = []

    with torch.no_grad():

        for xb, _ in test_loader:

            xb = xb.to(device)

            pred = model(xb)

            predictions.extend(
                pred.cpu().numpy()
            )

    return np.array(predictions)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("THERMORA — TEMPORAL MODEL COMPARISON")
    print("=" * 80)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    X_train, y_train, meta_train = load_sequence_file(
        TRAIN_FILE
    )

    X_val, y_val, meta_val = load_sequence_file(
        VAL_FILE
    )

    X_test, y_test, meta_test = load_sequence_file(
        TEST_FILE
    )

    print("\nDataset shapes:")

    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    # --------------------------------------------------------
    # Persistence baseline
    #
    # Metadata column 4 = current core temperature
    # --------------------------------------------------------

    current_core_test = meta_test[:, 4].astype(np.float32)

    persistence_predictions = current_core_test

    persistence_metrics = print_results(
        "PERSISTENCE BASELINE",
        y_test,
        persistence_predictions
    )

    # --------------------------------------------------------
    # Engineer temporal features
    # --------------------------------------------------------

    print("\nEngineering temporal features...")

    X_train_engineered = engineer_features(X_train)

    X_val_engineered = engineer_features(X_val)

    X_test_engineered = engineer_features(X_test)

    print(
        "Engineered feature shape:",
        X_train_engineered.shape
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("XGBOOST")
    print("=" * 80)

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=SEED,
        n_jobs=-1
    )

    xgb_model.fit(
        X_train_engineered,
        y_train,
        eval_set=[
            (X_val_engineered, y_val)
        ],
        verbose=False
    )

    xgb_predictions = xgb_model.predict(
        X_test_engineered
    )

    xgb_metrics = print_results(
        "XGBOOST",
        y_test,
        xgb_predictions
    )

    xgb_model.save_model(
        f"{MODEL_DIR}/thermora_xgboost_10min.json"
    )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("LIGHTGBM")
    print("=" * 80)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=-1
    )

    lgb_model.fit(
        X_train_engineered,
        y_train,
        eval_set=[
            (X_val_engineered, y_val)
        ],
        callbacks=[
            lgb.early_stopping(
                30,
                verbose=False
            )
        ]
    )

    lgb_predictions = lgb_model.predict(
        X_test_engineered
    )

    lgb_metrics = print_results(
        "LIGHTGBM",
        y_test,
        lgb_predictions
    )

    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("GRU")
    print("=" * 80)

    gru_predictions = train_gru(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    gru_metrics = print_results(
        "GRU",
        y_test,
        gru_predictions
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    results = pd.DataFrame({

        "model": [
            "Persistence",
            "XGBoost",
            "LightGBM",
            "GRU"
        ],

        "MAE_C": [
            persistence_metrics[0],
            xgb_metrics[0],
            lgb_metrics[0],
            gru_metrics[0]
        ],

        "RMSE_C": [
            persistence_metrics[1],
            xgb_metrics[1],
            lgb_metrics[1],
            gru_metrics[1]
        ],

        "R2": [
            persistence_metrics[2],
            xgb_metrics[2],
            lgb_metrics[2],
            gru_metrics[2]
        ]
    })

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
        results.to_string(
            index=False
        )
    )

    results.to_csv(
        f"{RESULTS_DIR}/temporal_model_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions = pd.DataFrame({

        "participant_id":
            meta_test[:, 0],

        "condition_id":
            meta_test[:, 1],

        "input_end_time":
            meta_test[:, 2],

        "target_time":
            meta_test[:, 3],

        "current_core_temp":
            current_core_test,

        "actual_future_core_temp":
            y_test,

        "persistence_prediction":
            persistence_predictions,

        "xgboost_prediction":
            xgb_predictions,

        "lightgbm_prediction":
            lgb_predictions,

        "gru_prediction":
            gru_predictions
    })

    predictions.to_csv(
        f"{RESULTS_DIR}/temporal_model_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Improvement over persistence
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER PERSISTENCE")
    print("=" * 80)

    baseline_mae = persistence_metrics[0]

    for name, model_metrics in [
        ("XGBoost", xgb_metrics),
        ("LightGBM", lgb_metrics),
        ("GRU", gru_metrics)
    ]:

        improvement = (
            (baseline_mae - model_metrics[0])
            / baseline_mae
        ) * 100

        print(
            f"{name:10s}: "
            f"{improvement:+.2f}%"
        )

    print("\nResults saved to:")

    print(
        f"{RESULTS_DIR}/temporal_model_comparison.csv"
    )

    print(
        f"{RESULTS_DIR}/temporal_model_predictions.csv"
    )


if __name__ == "__main__":
    main()