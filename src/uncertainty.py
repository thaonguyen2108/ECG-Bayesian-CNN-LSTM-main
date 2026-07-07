from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Union

from src import config


def enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d)):
            module.train()


def _loader_from_tensor(inputs: torch.Tensor) -> DataLoader:
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(1)
    dummy_labels = torch.zeros(inputs.size(0), dtype=torch.long)
    return DataLoader(TensorDataset(inputs.float(), dummy_labels), batch_size=config.BATCH_SIZE)


def predict_with_uncertainty(
    model: nn.Module,
    dataloader_or_tensor: Union[DataLoader, torch.Tensor],
    device: torch.device,
    mc_samples: int = config.MC_SAMPLES,
) -> dict[str, np.ndarray]:
    loader = _loader_from_tensor(dataloader_or_tensor) if isinstance(dataloader_or_tensor, torch.Tensor) else dataloader_or_tensor
    sample_probabilities: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(mc_samples):
            enable_dropout_only(model)
            batches: list[torch.Tensor] = []
            for batch in loader:
                features = batch[0].to(device)
                logits = model(features)
                batches.append(torch.softmax(logits, dim=1).cpu())
            sample_probabilities.append(torch.cat(batches, dim=0))

    stacked = torch.stack(sample_probabilities, dim=0)
    mean_probs = stacked.mean(dim=0)
    variance = stacked.var(dim=0)
    confidence, pred_class = mean_probs.max(dim=1)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1)
    normalized_entropy = entropy / math.log(config.NUM_CLASSES)

    return {
        "mean_probs": mean_probs.numpy(),
        "variance": variance.numpy(),
        "pred_class": pred_class.numpy(),
        "confidence": confidence.numpy(),
        "predictive_entropy": entropy.numpy(),
        "uncertainty_score": normalized_entropy.numpy(),
    }


def uncertainty_level(score: float) -> str:
    if score < 0.25:
        return "Tin cậy cao"
    if score < 0.5:
        return "Tin cậy trung bình"
    return "Tin cậy thấp, cần kiểm tra lại"
