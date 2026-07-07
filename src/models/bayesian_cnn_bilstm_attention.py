import torch
from torch import nn

from src import config
from src.models.cnn_bilstm_attention import AttentionPooling


class BayesianCNNBiLSTMAttention(nn.Module):
    """CNN-BiLSTM-Attention model with dropout layers used for MC inference."""

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden_size: int = 64,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(dropout * 0.5),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(dropout * 0.5),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = AttentionPooling(hidden_size * 2)
        self.dropout_context = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).transpose(1, 2)
        sequence, _ = self.lstm(features)
        context, _ = self.attention(sequence)
        context = self.dropout_context(context)
        return self.classifier(context)
