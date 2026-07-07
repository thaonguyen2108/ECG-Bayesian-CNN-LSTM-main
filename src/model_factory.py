from torch import nn

from src import config
from src.models import BayesianCNNBiLSTMAttention, CNNBaseline, CNNBiLSTMAttention


def build_model(model_name: str) -> nn.Module:
    if model_name == "cnn_baseline":
        return CNNBaseline()
    if model_name == "cnn_bilstm_attention":
        return CNNBiLSTMAttention()
    if model_name == "bayesian_cnn_bilstm_attention":
        return BayesianCNNBiLSTMAttention()
    raise ValueError(f"Unknown model name: {model_name}. Expected one of {config.ALL_MODEL_NAMES}.")


def checkpoint_path(model_name: str):
    return config.CHECKPOINT_DIR / f"{model_name}_best.pt"
