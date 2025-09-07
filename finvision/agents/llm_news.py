from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class _MLP1D(torch.nn.Module):
    def __init__(self, lookback: int, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(lookback, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 1] -> flatten T
        b, t, f = x.shape
        x = x.view(b, t)
        return self.net(x)


@dataclass
class LLMNewsRegressor:
    name: str
    lookback: int
    hidden_size: int = 64
    dropout: float = 0.1
    device: str = "cpu"
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 128

    _model: _MLP1D | None = None

    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = _MLP1D(self.lookback, self.hidden_size, self.dropout)
        return self._model

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        m = self.model().to(self.device)
        opt = torch.optim.AdamW(m.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        best_val = float("inf")
        best_state = None
        for _ in range(self.epochs):
            m.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).float()
                pred = m(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if val_loader is not None:
                m.eval()
                vloss = 0.0
                n = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device).float()
                        pred = m(xb).squeeze(-1)
                        vloss += loss_fn(pred, yb).item() * len(yb)
                        n += len(yb)
                vloss /= max(n, 1)
                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: v.cpu() for k, v in m.state_dict().items()}
        if best_state is not None:
            self.model().load_state_dict(best_state)

    def predict(self, loader: DataLoader) -> torch.Tensor:
        m = self.model().to(self.device)
        m.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                pred = m(xb).squeeze(-1).detach().cpu()
                preds.append(pred)
        return torch.cat(preds, dim=0)

    def save(self, path: str) -> None:
        torch.save(self.model().state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        self.model().load_state_dict(state)

