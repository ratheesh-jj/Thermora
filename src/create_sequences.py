from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "ThermoSentinel_PROSPIE_ModelData_clean.csv"
)

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "sequences"
)

SEQUENCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of previous minutes given to the model.
LOOKBACK = 20

# Predict core temperature this many minutes into the future.
HORIZON = 10


# ============================================================
# FEATURES
# ============================================================

# These are the measurements we can realistically obtain
# from the first ThermoSentinel prototype.

NUMERIC_FEATURES = [
    "age_years",
    "body_mass_kg",
    "ambient_temp_c",
    "humidity_pct",
    "heart_rate_bpm",
    "insulated_skin_temp_c",
]

CATEGORICAL_FEATURES = [
    "sex",
]


# ============================================================
# PARTICIPANT SPLIT
# ============================================================

def create_participant_split(df):

    from sklearn.model_selection import GroupShuffleSplit

    # First split:
    # 80% participants -> train/validation
    # 20% participants -> final test

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_val_idx, test_idx = next(
        outer_split.split(
            df,
            groups=df["participant_id"],
        )
    )

    train_val_df = df.iloc[
        train_val_idx
    ].copy()

    test_df = df.iloc[
        test_idx
    ].copy()

    # Second split:
    # 80% of train_val -> actual train
    # 20% of train_val -> validation

    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        inner_split.split(
            train_val_df,
            groups=train_val_df[
                "participant_id"
            ],
        )
    )

    train_df = train_val_df.iloc[
        train_idx
    ].copy()

    val_df = train_val_df.iloc[
        val_idx
    ].copy()

    print("\nPARTICIPANT SPLIT")
    print("=" * 70)

    print(
        "Train participants:",
        sorted(
            train_df["participant_id"]
            .unique()
            .tolist()
        ),
    )

    print(
        "Validation participants:",
        sorted(
            val_df["participant_id"]
            .unique()
            .tolist()
        ),
    )

    print(
        "Test participants:",
        sorted(
            test_df["participant_id"]
            .unique()
            .tolist()
        ),
    )

    # Safety checks.

    train_ids = set(
        train_df["participant_id"]
    )

    val_ids = set(
        val_df["participant_id"]
    )

    test_ids = set(
        test_df["participant_id"]
    )

    if train_ids & val_ids:
        raise RuntimeError(
            "Participant leakage between train and validation."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Participant leakage between train and test."
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Participant leakage between validation and test."
        )

    return train_df, val_df, test_df


# ============================================================
# ENCODE CATEGORICAL VARIABLES
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    # Binary sex feature.
    # Keep this simple for the first model.

    df["sex_male"] = (
        df["sex"]
        .astype(str)
        .str.lower()
        .eq("male")
        .astype(float)
    )

    numeric_features = NUMERIC_FEATURES.copy()

    numeric_features.append(
        "sex_male"
    )

    return df, numeric_features


# ============================================================
# CONTIGUOUS SEQUENCE EXTRACTION
# ============================================================

def create_sequences(
    df,
    feature_columns,
    lookback,
    horizon,
):

    X = []
    y = []

    metadata = []

    # IMPORTANT:
    # Every participant + condition is a separate sequence.
    # We NEVER allow a window to cross a condition boundary.

    grouped = df.groupby(
        [
            "participant_id",
            "condition_id",
        ],
        sort=False,
    )

    total_groups = 0
    skipped_groups = 0

    for (
        participant_id,
        condition_id,
    ), group in grouped:

        total_groups += 1

        group = group.sort_values(
            "time_min"
        ).reset_index(drop=True)

        # We need target values to exist.
        required_target = group[
            "core_temp_c"
        ].notna()

        # Check actual sampling gaps.
        time_difference = (
            group["time_min"]
            .diff()
            .dropna()
        )

        # Median was already confirmed as 1 minute,
        # but here we make sure every step in this
        # sequence is exactly one minute.

        if not time_difference.empty:

            invalid_gap = ~np.isclose(
                time_difference,
                1.0,
                atol=0.05,
            ).any()

            if invalid_gap:
                skipped_groups += 1
                continue

        # Convert features to numeric.

        features = group[
            feature_columns
        ].copy()

        target = group[
            "core_temp_c"
        ]

        # Build windows.

        max_start = (
            len(group)
            - lookback
            - horizon
            + 1
        )

        if max_start <= 0:
            skipped_groups += 1
            continue

        for start in range(
            max_start
        ):

            end = (
                start
                + lookback
            )

            target_index = (
                end
                + horizon
                - 1
            )

            # Make sure target exists.

            if pd.isna(
                target.iloc[target_index]
            ):
                continue

            window = features.iloc[
                start:end
            ].values.astype(
                np.float32
            )

            target_value = float(
                target.iloc[
                    target_index
                ]
            )

            # Don't allow NaNs into the sequence.
            # We'll handle missing features later
            # using training-set statistics.

            X.append(window)

            y.append(
                target_value
            )
            current_core_temp = group[
    "core_temp_c"
].iloc[end - 1]

            metadata.append(
                [
                    participant_id,
                    condition_id,
                    float(
                        group[
                            "time_min"
                        ].iloc[end - 1]
                    ),
                    float(
                        group[
                            "time_min"
                        ].iloc[target_index]
                    ),
                    float(current_core_temp)
                ]
            )

    print("\nSEQUENCE CREATION")
    print("=" * 70)

    print(
        f"Participant-condition groups: "
        f"{total_groups}"
    )

    print(
        f"Skipped groups: "
        f"{skipped_groups}"
    )

    print(
        f"Sequences created: "
        f"{len(X):,}"
    )

    if not X:

        raise RuntimeError(
            "No sequences were created."
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    metadata = np.asarray(
        metadata
    )

    return X, y, metadata


# ============================================================
# FILTER BY PARTICIPANT SET
# ============================================================

def filter_participants(
    df,
    participants,
):

    return df[
        df["participant_id"].isin(
            participants
        )
    ].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("THERMOSENTINEL TEMPORAL DATASET BUILDER")
    print("=" * 70)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"\nOriginal dataset: "
        f"{df.shape}"
    )

    # Remove rows without target.
    df = df.dropna(
        subset=[
            "core_temp_c"
        ]
    ).copy()

    # Sort.
    df = df.sort_values(
        [
            "participant_id",
            "condition_id",
            "time_min",
        ]
    ).reset_index(
        drop=True
    )

    # Add encoded sex.
    df, feature_columns = (
        prepare_dataframe(df)
    )

    # --------------------------------------------------------
    # Create fixed participant split.
    # --------------------------------------------------------

    train_df, val_df, test_df = (
        create_participant_split(df)
    )

    # --------------------------------------------------------
    # Create sequences independently within each split.
    # --------------------------------------------------------

    train_X, train_y, train_meta = (
        create_sequences(
            train_df,
            feature_columns,
            LOOKBACK,
            HORIZON,
        )
    )

    val_X, val_y, val_meta = (
        create_sequences(
            val_df,
            feature_columns,
            LOOKBACK,
            HORIZON,
        )
    )

    test_X, test_y, test_meta = (
        create_sequences(
            test_df,
            feature_columns,
            LOOKBACK,
            HORIZON,
        )
    )

    # --------------------------------------------------------
    # Print shapes.
    # --------------------------------------------------------

    print("\nFINAL SHAPES")
    print("=" * 70)

    print(
        "X_train:",
        train_X.shape,
    )

    print(
        "y_train:",
        train_y.shape,
    )

    print(
        "X_val:",
        val_X.shape,
    )

    print(
        "y_val:",
        val_y.shape,
    )

    print(
        "X_test:",
        test_X.shape,
    )

    print(
        "y_test:",
        test_y.shape,
    )

    print(
        "\nFeatures:",
        feature_columns,
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    train_path = (
        SEQUENCE_DIR
        / "train_10min.npz"
    )

    val_path = (
        SEQUENCE_DIR
        / "validation_10min.npz"
    )

    test_path = (
        SEQUENCE_DIR
        / "test_10min.npz"
    )

    np.savez_compressed(
        train_path,
        X=train_X,
        y=train_y,
        metadata=train_meta,
    )

    np.savez_compressed(
        val_path,
        X=val_X,
        y=val_y,
        metadata=val_meta,
    )

    np.savez_compressed(
        test_path,
        X=test_X,
        y=test_y,
        metadata=test_meta,
    )

    # Save feature names separately.

    feature_path = (
        SEQUENCE_DIR
        / "feature_names.txt"
    )

    with open(
        feature_path,
        "w",
        encoding="utf-8",
    ) as file:

        for feature in feature_columns:
            file.write(
                feature + "\n"
            )

    print("\nSaved files:")

    print(train_path)
    print(val_path)
    print(test_path)
    print(feature_path)


if __name__ == "__main__":
    main()