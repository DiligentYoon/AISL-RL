import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.model import Model


class GRU(Model):
    """
    SafeFall GRU fall predictor.
    References: https://arxiv.org/abs/2511.18509

    Input:
        obs_seq: (B, T, obs_dim)

    Output:
        logits: (B, T, 2)
            class 0: safe
            class 1: falling
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, obs_seq: torch.Tensor, h0: torch.Tensor | None = None):
        """
        Args:
            obs_seq: (B, T, obs_dim)
            h0: optional hidden state, (num_layers, B, hidden_dim)

        Returns:
            logits: (B, T, 2)
            hn: final hidden state
        """
        gru_out, hn = self.gru(obs_seq, h0)
        logits = self.head(gru_out)
        return logits, hn, None

    @torch.no_grad()
    def predict_fall_prob(self, obs_seq: torch.Tensor, h0: torch.Tensor | None = None):
        logits, hn, _ = self.forward(obs_seq, h0)
        prob = torch.softmax(logits, dim=-1)[..., 1]
        return prob, hn