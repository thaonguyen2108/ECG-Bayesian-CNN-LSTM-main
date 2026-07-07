from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.model_factory import build_model, checkpoint_path
from src.uncertainty import predict_with_uncertainty
from src.utils import save_json


def load_checkpoint_model(model_name: str, device: torch.device) -> nn.Module:
    path = checkpoint_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train the model first.")
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint.get("model_name", model_name)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_model(
    model_name: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    save_outputs: bool = True,
) -> dict:
    criterion = criterion or nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    started = time.perf_counter()

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            losses.append(loss.item() * labels.size(0))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    inference_time = time.perf_counter() - started
    total_samples = max(len(y_true), 1)
    cm = confusion_matrix(y_true, y_pred, labels=list(config.CLASS_NAMES.keys()))
    metrics = {
        "model_name": model_name,
        "loss": float(sum(losses) / total_samples),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "inference_time_seconds": float(inference_time),
        "inference_time_per_sample_ms": float(inference_time * 1000 / total_samples),
    }

    if model_name == "bayesian_cnn_bilstm_attention":
        uncertainty = predict_with_uncertainty(model, dataloader, device)
        scores = uncertainty["uncertainty_score"]
        confidence = uncertainty["confidence"]
        metrics.update(
            {
                "mean_confidence": float(np.mean(confidence)),
                "mean_uncertainty": float(np.mean(scores)),
                "high_confidence_ratio": float(np.mean(scores < 0.25)),
                "medium_confidence_ratio": float(np.mean((scores >= 0.25) & (scores < 0.5))),
                "low_confidence_ratio": float(np.mean(scores >= 0.5)),
            }
        )
        if save_outputs:
            save_json(
                {
                    "mean_confidence": metrics["mean_confidence"],
                    "mean_uncertainty": metrics["mean_uncertainty"],
                    "high_confidence_ratio": metrics["high_confidence_ratio"],
                    "medium_confidence_ratio": metrics["medium_confidence_ratio"],
                    "low_confidence_ratio": metrics["low_confidence_ratio"],
                },
                config.METRICS_DIR / "bayesian_uncertainty_summary.json",
            )
            pd.DataFrame(
                {
                    "confidence": confidence,
                    "uncertainty_score": scores,
                    "predictive_entropy": uncertainty["predictive_entropy"],
                    "pred_class": uncertainty["pred_class"],
                }
            ).to_csv(config.METRICS_DIR / "bayesian_uncertainty_predictions.csv", index=False)

    if save_outputs:
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        save_json(metrics, config.METRICS_DIR / f"{model_name}_test_metrics.json")
        report = classification_report(
            y_true,
            y_pred,
            labels=list(config.CLASS_NAMES.keys()),
            target_names=[config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)],
            zero_division=0,
        )
        (config.METRICS_DIR / f"{model_name}_classification_report.txt").write_text(report, encoding="utf-8")
        pd.DataFrame(cm, index=config.CLASS_NAMES.values(), columns=config.CLASS_NAMES.values()).to_csv(
            config.METRICS_DIR / f"{model_name}_confusion_matrix.csv"
        )
    return metrics


def compare_models(model_names: Sequence[str] = config.ALL_MODEL_NAMES) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        path = config.METRICS_DIR / f"{model_name}_test_metrics.json"
        if path.exists():
            rows.append(pd.read_json(path, typ="series").to_dict())
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    return comparison
