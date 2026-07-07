from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src import config
from src.utils import save_json


def clean_features(features: np.ndarray, clip_value: Optional[float] = None) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    if clip_value is not None:
        features = np.clip(features, -clip_value, clip_value)
    return features


def class_distribution(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels.astype(np.int64), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(values, counts)}


def save_data_metadata(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "num_features": int(x_train.shape[1]),
        "samples": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "class_distribution": {
            "train": class_distribution(y_train),
            "validation": class_distribution(y_val),
            "test": class_distribution(y_test),
        },
        "class_names": {str(k): v for k, v in config.CLASS_NAMES.items()},
        "class_descriptions": {str(k): v for k, v in config.CLASS_DESCRIPTIONS.items()},
        "note": "Kaggle ECG heartbeat CSV is already normalized; only float conversion and finite-value checks are applied.",
    }
    save_json(metadata, config.REPORTS_DIR / "data_metadata.json")
    return metadata
