from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "sequences"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_PATH = (
    SEQUENCE_DIR
    / "train_10min.npz"
)

VAL_PATH = (
    SEQUENCE_DIR
    / "validation_10min.npz"
)

TEST_PATH = (
    SEQUENCE_DIR
    / "test_10min.npz"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64

EPOCHS = 100

LEARNING_RATE = 0.001

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.20

PATIENCE = 10

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {DEVICE}"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_sequence_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Sequence file not found:\n{path}"
        )

    data = np.load(path)

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.float32
    )

    metadata = data["metadata"]

    return X, y, metadata


# ============================================================
# FIT TRAINING-ONLY PREPROCESSORS
# ============================================================

def fit_preprocessors(
    X_train,
):

    samples, timesteps, features = (
        X_train.shape
    )

    # Flatten only for fitting the sklearn
    # preprocessing objects.
    flattened = X_train.reshape(
        -1,
        features,
    )

    imputer = SimpleImputer(
        strategy="median",
    )

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit only on training data.
    flattened = imputer.fit_transform(
        flattened
    )

    scaler.fit(
        flattened
    )

    return imputer, scaler


# ============================================================
# TRANSFORM SEQUENCES
# ============================================================

def transform_sequences(
    X,
    imputer,
    scaler,
):

    samples, timesteps, features = (
        X.shape
    )

    flattened = X.reshape(
        -1,
        features,
    )

    flattened = imputer.transform(
        flattened
    )

    flattened = scaler.transform(
        flattened
    )

    transformed = flattened.reshape(
        samples,
        timesteps,
        features,
    )

    return transformed.astype(
        np.float32
    )


# ============================================================
# LSTM MODEL
# ============================================================

class ThermoSentinelLSTM(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.20,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                hidden_size,
                32,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        output, _ = self.lstm(x)

        # Use the final time step.
        final_state = output[
            :, -1, :
        ]

        prediction = self.regressor(
            final_state
        )

        return prediction.squeeze(1)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            output = model(
                X_batch
            )

            predictions.append(
                output
                .cpu()
                .numpy()
            )

            targets.append(
                y_batch.numpy()
            )

    predictions = np.concatenate(
        predictions
    )

    targets = np.concatenate(
        targets
    )

    mae = mean_absolute_error(
        targets,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            targets,
            predictions,
        )
    )

    r2 = r2_score(
        targets,
        predictions,
    )

    return (
        mae,
        rmse,
        r2,
        predictions,
        targets,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "THERMOSENTINEL LSTM "
        "— 10-MINUTE CORE TEMPERATURE FORECAST"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X_train, y_train, train_meta = (
        load_sequence_file(
            TRAIN_PATH
        )
    )

    X_val, y_val, val_meta = (
        load_sequence_file(
            VAL_PATH
        )
    )

    X_test, y_test, test_meta = (
        load_sequence_file(
            TEST_PATH
        )
    )

    print("\nRaw data:")
    print(
        f"Train:      {X_train.shape}"
    )
    print(
        f"Validation: {X_val.shape}"
    )
    print(
        f"Test:       {X_test.shape}"
    )

    # --------------------------------------------------------
    # Missing-value statistics
    # --------------------------------------------------------

    print("\nMissing values:")

    print(
        f"Train: "
        f"{np.isnan(X_train).sum():,}"
    )

    print(
        f"Validation: "
        f"{np.isnan(X_val).sum():,}"
    )

    print(
        f"Test: "
        f"{np.isnan(X_test).sum():,}"
    )

    # --------------------------------------------------------
    # Fit preprocessing ONLY on training data
    # --------------------------------------------------------

    print(
        "\nFitting preprocessing "
        "on training data only..."
    )

    imputer, scaler = (
        fit_preprocessors(
            X_train
        )
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    X_train = transform_sequences(
        X_train,
        imputer,
        scaler,
    )

    X_val = transform_sequences(
        X_val,
        imputer,
        scaler,
    )

    X_test = transform_sequences(
        X_test,
        imputer,
        scaler,
    )

    print(
        "Preprocessing complete."
    )

    # --------------------------------------------------------
    # Verify no NaNs remain
    # --------------------------------------------------------

    print(
        "\nNaNs after preprocessing:"
    )

    print(
        "Train:",
        np.isnan(X_train).sum(),
    )

    print(
        "Validation:",
        np.isnan(X_val).sum(),
    )

    print(
        "Test:",
        np.isnan(X_test).sum(),
    )

    # --------------------------------------------------------
    # Save preprocessing objects
    # --------------------------------------------------------

    joblib.dump(
        imputer,
        MODEL_DIR
        / "lstm_imputer.joblib",
    )

    joblib.dump(
        scaler,
        MODEL_DIR
        / "lstm_scaler.joblib",
    )

    # --------------------------------------------------------
    # PyTorch datasets
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32,
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_val,
            dtype=torch.float32,
        ),
    )

    test_dataset = TensorDataset(
        torch.tensor(
            X_test,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_test,
            dtype=torch.float32,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = ThermoSentinelLSTM(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(
        "\nModel:"
    )

    print(model)

    # --------------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    best_model_path = (
        MODEL_DIR
        / "thermosentinel_lstm_10min.pt"
    )

    print(
        "\nStarting training..."
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        train_losses = []

        for (
            X_batch,
            y_batch,
        ) in train_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(
                train_losses
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for (
                X_batch,
                y_batch,
            ) in val_loader:

                X_batch = X_batch.to(
                    DEVICE
                )

                y_batch = y_batch.to(
                    DEVICE
                )

                predictions = model(
                    X_batch
                )

                loss = criterion(
                    predictions,
                    y_batch,
                )

                val_losses.append(
                    loss.item()
                )

        val_loss = float(
            np.mean(
                val_losses
            )
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= PATIENCE
            ):

                print(
                    "\nEarly stopping."
                )

                break

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading best model..."
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
        )
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    (
        mae,
        rmse,
        r2,
        predictions,
        targets,
    ) = evaluate(
        model,
        test_loader,
    )

    print("\n" + "=" * 75)
    print(
        "FINAL TEST RESULTS"
    )
    print("=" * 75)

    print(
        f"MAE :  {mae:.4f} °C"
    )

    print(
        f"RMSE:  {rmse:.4f} °C"
    )

    print(
        f"R²  :  {r2:.4f}"
    )

    # --------------------------------------------------------
    # Compare against persistence
    # --------------------------------------------------------

    current_core_temp = (
        test_meta[:, 4]
        .astype(float)
    )

    persistence_mae = (
        mean_absolute_error(
            targets,
            current_core_temp,
        )
    )

    persistence_rmse = np.sqrt(
        mean_squared_error(
            targets,
            current_core_temp,
        )
    )

    persistence_r2 = r2_score(
        targets,
        current_core_temp,
    )

    print("\n" + "=" * 75)
    print(
        "PERSISTENCE BASELINE"
    )
    print("=" * 75)

    print(
        f"MAE :  {persistence_mae:.4f} °C"
    )

    print(
        f"RMSE:  {persistence_rmse:.4f} °C"
    )

    print(
        f"R²  :  {persistence_r2:.4f}"
    )

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    improvement = (
        (
            persistence_mae
            - mae
        )
        / persistence_mae
    ) * 100

    print("\n" + "=" * 75)
    print(
        "LSTM IMPROVEMENT"
    )
    print("=" * 75)

    print(
        f"MAE improvement over "
        f"persistence: "
        f"{improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = np.column_stack(
        [
            test_meta[:, 0],
            test_meta[:, 1],
            test_meta[:, 2],
            test_meta[:, 3],
            current_core_temp,
            targets,
            predictions,
        ]
    )

    prediction_path = (
        RESULTS_DIR
        / "lstm_10min_predictions.csv"
    )

    header = (
        "participant_id,"
        "condition_id,"
        "input_end_min,"
        "target_min,"
        "current_core_temp,"
        "actual_future_core_temp,"
        "predicted_future_core_temp"
    )

    np.savetxt(
        prediction_path,
        prediction_df,
        delimiter=",",
        header=header,
        comments="",
    )

    print(
        f"\nPredictions saved to:"
    )

    print(
        prediction_path
    )

    print(
        "\nModel saved to:"
    )

    print(
        best_model_path
    )


if __name__ == "__main__":
    main()