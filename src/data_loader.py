from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.preprocessing import clean_features


def check_dataset_exists() -> None:
    missing = [path for path in (config.TRAIN_CSV, config.TEST_CSV) if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            "Missing dataset files: "
            f"{names}. Download ECG Heartbeat Categorization Dataset from "
            "https://www.kaggle.com/datasets/shayanfazeli/heartbeat, extract it, "
            "then place mitbih_train.csv and mitbih_test.csv in data/raw/."
        )


def _read_csv(path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] < config.INPUT_LENGTH + 1:
        raise ValueError(
            f"{path} must contain at least {config.INPUT_LENGTH + 1} columns, got {frame.shape[1]}."
        )
    frame = frame.iloc[:, : config.INPUT_LENGTH + 1]
    x = clean_features(frame.iloc[:, : config.INPUT_LENGTH].to_numpy())
    y = frame.iloc[:, config.INPUT_LENGTH].to_numpy().astype(np.int64)
    return x, y


def load_raw_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    check_dataset_exists()
    x_train_full, y_train_full = _read_csv(config.TRAIN_CSV)
    x_test, y_test = _read_csv(config.TEST_CSV)
    return x_train_full, y_train_full, x_test, y_test


def create_data_splits(
    validation_size: float = config.VALIDATION_SIZE,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_full, y_train_full, x_test, y_test = load_raw_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=random_seed,
        stratify=y_train_full,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=config.NUM_CLASSES)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (config.NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _to_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def create_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_loader = DataLoader(
        _to_dataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        _to_dataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        _to_dataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader, compute_class_weights(y_train)
