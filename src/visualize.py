from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_history(model_name: str) -> None:
    path = config.METRICS_DIR / f"{model_name}_history.csv"
    if not path.exists():
        print(f"Chưa có history cho {model_name}. Cần train trước.")
        return
    history = pd.read_csv(path)
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{config.MODEL_DISPLAY_NAMES.get(model_name, model_name)} - Loss")
    plt.legend()
    _save_current(config.FIGURE_DIR / f"{model_name}_loss.png")

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    plt.plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{config.MODEL_DISPLAY_NAMES.get(model_name, model_name)} - Accuracy")
    plt.legend()
    _save_current(config.FIGURE_DIR / f"{model_name}_accuracy.png")


def plot_confusion_matrix(model_name: str) -> None:
    path = config.METRICS_DIR / f"{model_name}_confusion_matrix.csv"
    if not path.exists():
        print(f"Chưa có confusion matrix cho {model_name}. Cần evaluate trước.")
        return
    matrix = pd.read_csv(path, index_col=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix.values, cmap="Blues")
    plt.title(f"{config.MODEL_DISPLAY_NAMES.get(model_name, model_name)} - Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(config.NUM_CLASSES), matrix.columns)
    plt.yticks(range(config.NUM_CLASSES), matrix.index)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, int(matrix.values[row, col]), ha="center", va="center", fontsize=8)
    plt.colorbar()
    _save_current(config.FIGURE_DIR / f"{model_name}_confusion_matrix.png")


def plot_metrics_comparison() -> None:
    path = config.METRICS_DIR / "model_comparison.csv"
    if not path.exists():
        print("Chưa có model_comparison.csv. Cần evaluate trước.")
        return
    data = pd.read_csv(path)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    available = [metric for metric in metrics if metric in data.columns]
    if not available:
        return
    x = np.arange(len(data))
    width = 0.18
    plt.figure(figsize=(9, 5))
    for idx, metric in enumerate(available):
        plt.bar(x + (idx - len(available) / 2) * width, data[metric], width, label=metric)
    plt.xticks(x, data["model_name"], rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("So sánh metrics giữa các mô hình")
    plt.legend()
    _save_current(config.FIGURE_DIR / "model_metrics_comparison.png")


def plot_sample_ecg(signal, label, prediction=None, output_name: str = "sample_ecg_signal.png") -> None:
    plt.figure(figsize=(9, 3))
    plt.plot(np.asarray(signal).reshape(-1))
    title = f"ECG sample - true: {config.CLASS_NAMES.get(int(label), label)}"
    if prediction is not None:
        title += f" - pred: {config.CLASS_NAMES.get(int(prediction), prediction)}"
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    _save_current(config.FIGURE_DIR / output_name)


def plot_uncertainty_distribution() -> None:
    path = config.METRICS_DIR / "bayesian_uncertainty_predictions.csv"
    if not path.exists():
        print("Chưa có dữ liệu uncertainty của Bayesian model.")
        return
    data = pd.read_csv(path)
    if data.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(data["confidence"], bins=30, alpha=0.7, label="Confidence")
    plt.hist(data["uncertainty_score"], bins=30, alpha=0.7, label="Uncertainty")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Phân phối confidence và uncertainty của Bayesian model")
    plt.legend()
    _save_current(config.FIGURE_DIR / "bayesian_uncertainty_distribution.png")


def generate_all_figures(model_names: Sequence[str] = config.ALL_MODEL_NAMES) -> None:
    for model_name in model_names:
        plot_training_history(model_name)
        plot_confusion_matrix(model_name)
    plot_metrics_comparison()
    plot_uncertainty_distribution()
