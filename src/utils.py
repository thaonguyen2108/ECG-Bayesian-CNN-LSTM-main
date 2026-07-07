import json
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from src import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dirs() -> None:
    for path in (
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.CHECKPOINT_DIR,
        config.FIGURE_DIR,
        config.METRICS_DIR,
        config.REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return config.DEVICE


def save_json(data: dict[str, Any], path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
