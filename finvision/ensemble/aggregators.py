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


@dataclass
class ConstrainedAggregator:
    """Learn non-negative, sum-to-one weights via projected gradient on validation set.

    Minimizes MSE(preds @ w, y) with constraints w >= 0, sum(w)=1 (configurable).
    """
    lr: float = 1e-1
    steps: int = 500
    nonneg: bool = True
    sum_to_one: bool = True
    w_: np.ndarray | None = None

    def _project(self, w: np.ndarray) -> np.ndarray:
        x = w.copy()
        if self.nonneg:
            x = np.maximum(0.0, x)
        if self.sum_to_one:
            s = x.sum()
            if s <= 0:
                # if all zeros, set uniform
                x = np.ones_like(x) / len(x)
            else:
                x = x / s
        return x

    def fit(self, preds: np.ndarray, y: np.ndarray) -> None:
        n_agents = preds.shape[1]
        w = np.ones(n_agents, dtype=np.float64) / n_agents
        X = preds.astype(np.float64)
        Y = y.astype(np.float64)
        for _ in range(self.steps):
            # gradient of 0.5 * ||Xw - Y||^2 = X^T (Xw - Y)
            r = X @ w - Y
            g = X.T @ r / len(Y)
            w = w - self.lr * g
            w = self._project(w)
        self.w_ = w.astype(np.float32)

    def predict(self, preds: np.ndarray) -> np.ndarray:
        assert self.w_ is not None, "Aggregator not fitted"
        return preds @ self.w_
