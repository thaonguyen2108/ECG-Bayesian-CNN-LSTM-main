import torch
from torch import nn

from src import config


class AttentionPooling(nn.Module):
    """Single-vector attention pooling over LSTM time steps."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_size, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return context, weights


class CNNBiLSTMAttention(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
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
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        features = self.cnn(x).transpose(1, 2)
        sequence, _ = self.lstm(features)
        context, weights = self.attention(sequence)
        logits = self.classifier(context)
        if return_attention:
            return logits, weights
        return logits
