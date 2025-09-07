from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, AgentConfig
from .data.dataset import load_ohlcv_csvs, make_panel
from .data.dataset import TimeSeriesDataset, sliding_windows
from .metrics import mse, mae, directional_accuracy
from .agents.lstm import LSTMRegressor
from .agents.mlp import MLPRegressor
from .ensemble.aggregators import RidgeAggregator, MLPAggregator


def _build_agent(cfg: AgentConfig, input_size: int, lookback: int):
    if cfg.type.lower() == "lstm":
        return LSTMRegressor(
            name=cfg.name,
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            device=cfg.device,
            lr=cfg.lr,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
        )
    elif cfg.type.lower() == "mlp":
        return MLPRegressor(
            name=cfg.name,
            input_dim=input_size * lookback,
            hidden_size=cfg.hidden_size,
            dropout=cfg.dropout,
            device=cfg.device,
            lr=cfg.lr,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
        )
    else:
        raise ValueError(f"Unknown agent type: {cfg.type}")


def _build_aggregator(agg_cfg):
    if agg_cfg.type == "ridge":
        return RidgeAggregator(alpha=agg_cfg.alpha)
    elif agg_cfg.type == "mlp":
        return MLPAggregator(hidden_size=agg_cfg.hidden_size, lr=agg_cfg.lr, epochs=agg_cfg.epochs)
    else:
        raise ValueError(f"Unknown aggregator type: {agg_cfg.type}")


def _make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    ds = TimeSeriesDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_walk_forward(cfg: ExperimentConfig) -> Dict[str, float]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    frames = load_ohlcv_csvs(cfg.data.data_dir, cfg.data.tickers, cfg.backtest.start, cfg.backtest.end)
    Xdf, yser = make_panel(
        frames,
        features=cfg.data.features,
        target=cfg.data.target,
        horizon=cfg.data.horizon,
        use_returns=cfg.data.use_returns,
    )
    Xw, yw, idx = sliding_windows(Xdf, yser, cfg.data.lookback)
    n = len(Xw)
    f = Xw.shape[-1]

    start = 0
    metrics = []
    while start + cfg.backtest.train_span + cfg.backtest.val_span + cfg.backtest.test_span <= n:
        tr0 = start
        tr1 = start + cfg.backtest.train_span
        va0 = tr1
        va1 = tr1 + cfg.backtest.val_span
        te0 = va1
        te1 = va1 + cfg.backtest.test_span

        Xtr, ytr = Xw[tr0:tr1], yw[tr0:tr1]
        Xva, yva = Xw[va0:va1], yw[va0:va1]
        Xte, yte = Xw[te0:te1], yw[te0:te1]

        # Build agents
        agents = [
            _build_agent(a, input_size=f, lookback=cfg.data.lookback)
            for a in cfg.agents
        ]

        # Train agents
        val_loaders = []
        test_loaders = []
        for a in agents:
            train_loader = _make_loaders(Xtr, ytr, a.batch_size)
            val_loader = _make_loaders(Xva, yva, a.batch_size, shuffle=False)
            test_loader = _make_loaders(Xte, yte, a.batch_size, shuffle=False)
            a.fit(train_loader, val_loader)
            val_loaders.append(val_loader)
            test_loaders.append(test_loader)

        # Collect validation predictions to fit aggregator
        val_preds = []
        for a, vloader in zip(agents, val_loaders):
            pred = a.predict(vloader).numpy()
            val_preds.append(pred)
        val_preds = np.stack(val_preds, axis=1)  # [N_val, n_agents]

        # Fit aggregator
        agg = _build_aggregator(cfg.aggregator)
        agg.fit(val_preds, yva)

        # Test predictions
        test_preds = []
        for a, tloader in zip(agents, test_loaders):
            pred = a.predict(tloader).numpy()
            test_preds.append(pred)
        test_preds = np.stack(test_preds, axis=1)
        yhat = agg.predict(test_preds)

        seg_metrics = {
            "mse": mse(yte, yhat),
            "mae": mae(yte, yhat),
            "directional_acc": directional_accuracy(yte, yhat),
        }
        metrics.append(seg_metrics)

        start += cfg.backtest.test_span

    # Aggregate metrics
    if not metrics:
        return {"mse": float("nan"), "mae": float("nan"), "directional_acc": float("nan")}

    keys = metrics[0].keys()
    out = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
    return out


def run_walk_forward_with_preds(cfg: ExperimentConfig) -> Tuple[Dict[str, float], pd.DataFrame]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    frames = load_ohlcv_csvs(cfg.data.data_dir, cfg.data.tickers, cfg.backtest.start, cfg.backtest.end)
    Xdf, yser = make_panel(
        frames,
        features=cfg.data.features,
        target=cfg.data.target,
        horizon=cfg.data.horizon,
        use_returns=cfg.data.use_returns,
    )
    Xw, yw, idx = sliding_windows(Xdf, yser, cfg.data.lookback)
    n = len(Xw)
    f = Xw.shape[-1]

    rows = []
    metrics = []
    start = 0
    while start + cfg.backtest.train_span + cfg.backtest.val_span + cfg.backtest.test_span <= n:
        tr0 = start
        tr1 = start + cfg.backtest.train_span
        va0 = tr1
        va1 = tr1 + cfg.backtest.val_span
        te0 = va1
        te1 = va1 + cfg.backtest.test_span

        Xtr, ytr = Xw[tr0:tr1], yw[tr0:tr1]
        Xva, yva = Xw[va0:va1], yw[va0:va1]
        Xte, yte = Xw[te0:te1], yw[te0:te1]

        # Build & train agents
        agents = [
            _build_agent(a, input_size=f, lookback=cfg.data.lookback)
            for a in cfg.agents
        ]
        val_loaders = []
        test_loaders = []
        for a in agents:
            train_loader = _make_loaders(Xtr, ytr, a.batch_size)
            val_loader = _make_loaders(Xva, yva, a.batch_size, shuffle=False)
            test_loader = _make_loaders(Xte, yte, a.batch_size, shuffle=False)
            a.fit(train_loader, val_loader)
            val_loaders.append(val_loader)
            test_loaders.append(test_loader)

        # Aggregator fit on validation
        val_preds = []
        for a, vloader in zip(agents, val_loaders):
            val_preds.append(a.predict(vloader).numpy())
        val_preds = np.stack(val_preds, axis=1)
        agg = _build_aggregator(cfg.aggregator)
        agg.fit(val_preds, yva)

        # Test predictions
        test_preds = []
        for a, tloader in zip(agents, test_loaders):
            test_preds.append(a.predict(tloader).numpy())
        test_preds = np.stack(test_preds, axis=1)
        yhat = agg.predict(test_preds)

        # Segment metrics
        seg_metrics = {
            "mse": mse(yte, yhat),
            "mae": mae(yte, yhat),
            "directional_acc": directional_accuracy(yte, yhat),
        }
        metrics.append(seg_metrics)

        # Collect rows with timestamp index
        seg_idx = idx[te0:te1]
        for tstamp, yt, yp in zip(seg_idx, yte, yhat):
            rows.append({"timestamp": tstamp, "y_true": float(yt), "y_pred": float(yp)})

        start += cfg.backtest.test_span

    # Aggregate metrics
    if not metrics:
        return {"mse": float("nan"), "mae": float("nan"), "directional_acc": float("nan")}, pd.DataFrame(columns=["timestamp","y_true","y_pred"]).set_index("timestamp")

    keys = metrics[0].keys()
    agg_metrics = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return agg_metrics, df
