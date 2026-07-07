from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from src import config
from src.model_factory import checkpoint_path


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for features, labels in tqdm(dataloader, desc="Train", leave=False):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validate", leave=False):
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def _checkpoint_payload(
    model_name: str,
    model: nn.Module,
    best_val_loss: float,
    best_epoch: int,
    training_time_seconds: float,
) -> dict:
    return {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "input_length": config.INPUT_LENGTH,
        "num_classes": config.NUM_CLASSES,
        "class_names": config.CLASS_NAMES,
        "config": {
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "batch_size": config.BATCH_SIZE,
        },
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time_seconds,
    }


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    epochs: int = config.EPOCHS,
) -> dict:
    model.to(device)
    weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []
    started = time.perf_counter()
    ckpt_path = checkpoint_path(model_name)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"{model_name} | epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model_name,
                    model,
                    best_val_loss,
                    best_epoch,
                    time.perf_counter() - started,
                ),
                ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    training_time = time.perf_counter() - started
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        payload["training_time_seconds"] = training_time
        torch.save(payload, ckpt_path)

    history_path = config.METRICS_DIR / f"{model_name}_history.csv"
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time,
        "history_path": str(history_path),
        "checkpoint_path": str(ckpt_path),
    }
