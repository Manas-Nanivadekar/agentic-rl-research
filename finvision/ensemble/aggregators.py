from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
import torch


@dataclass
class RidgeAggregator:
    alpha: float = 1.0
    _model: Ridge | None = None

    def fit(self, preds: np.ndarray, y: np.ndarray) -> None:
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(preds, y)

    def predict(self, preds: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Aggregator not fitted"
        return self._model.predict(preds)


class _AggMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPAggregator:
    hidden_size: int = 32
    lr: float = 1e-3
    epochs: int = 10
    device: str = "cpu"

    _model: _AggMLP | None = None

    def fit(self, preds: np.ndarray, y: np.ndarray) -> None:
        in_dim = preds.shape[1]
        m = _AggMLP(in_dim, self.hidden_size)
        m.to(self.device)
        opt = torch.optim.AdamW(m.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        X = torch.from_numpy(preds).float().to(self.device)
        Y = torch.from_numpy(y).float().to(self.device)
        for _ in range(self.epochs):
            m.train()
            opt.zero_grad()
            out = m(X).squeeze(-1)
            loss = loss_fn(out, Y)
            loss.backward()
            opt.step()
        self._model = m.cpu()

    def predict(self, preds: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Aggregator not fitted"
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(preds).float()
            out = self._model(x).squeeze(-1).numpy()
        return out

