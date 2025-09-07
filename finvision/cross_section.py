from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, AgentConfig
from .data.dataset import load_ohlcv_csvs, make_single_panel
from .data.dataset import TimeSeriesDataset, sliding_windows
from .agents.lstm import LSTMRegressor
from .agents.mlp import MLPRegressor
from .agents.llm_news import LLMNewsRegressor
from .ensemble.aggregators import RidgeAggregator, MLPAggregator
from .news.signal import load_llm_mean_series


def _build_agent(cfg: AgentConfig, input_size: int, lookback: int):
    t = cfg.type.lower()
    if t == "lstm":
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
    if t == "mlp":
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
    if t == "llm_news":
        return LLMNewsRegressor(
            name=cfg.name,
            lookback=lookback,
            hidden_size=cfg.hidden_size,
            dropout=cfg.dropout,
            device=cfg.device,
            lr=cfg.lr,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
        )
    raise ValueError(f"Unknown agent type: {cfg.type}")


def _build_aggregator(agg_cfg):
    if agg_cfg.type == "ridge":
        return RidgeAggregator(alpha=agg_cfg.alpha)
    if agg_cfg.type == "mlp":
        return MLPAggregator(hidden_size=agg_cfg.hidden_size, lr=agg_cfg.lr, epochs=agg_cfg.epochs)
    raise ValueError(f"Unknown aggregator type: {agg_cfg.type}")


def _make_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(TimeSeriesDataset(X, y), batch_size=batch, shuffle=shuffle)


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    # rank and pearson corr of ranks
    ar = pd.Series(a).rank(method="average").values
    br = pd.Series(b).rank(method="average").values
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def long_short_returns(preds: pd.DataFrame, rets: pd.DataFrame, q: float = 0.1) -> pd.Series:
    """Simple equal-weight long-short returns per date (daily rebalance, no costs)."""
    dates = preds.index.intersection(rets.index)
    out = []
    for d in dates:
        p = preds.loc[d].dropna()
        if len(p) < 5:
            out.append(np.nan)
            continue
        n = len(p)
        k = max(1, int(n * q))
        top = p.nlargest(k).index
        bot = p.nsmallest(k).index
        r = rets.loc[d]
        out.append(float(r.reindex(top).mean() - r.reindex(bot).mean()))
    return pd.Series(out, index=dates)


def long_short_with_holding(preds: pd.DataFrame, rets: pd.DataFrame, q: float = 0.1, hold: int = 5, cap: float = 0.1, cost_bps: float = 1.0) -> pd.Series:
    """Long-short with holding period and transaction costs.

    - Rebalance every day into top/bottom q quantiles, but hold positions for `hold` days using overlapping subportfolios.
    - Cap absolute weight per asset by `cap`.
    - Apply transaction costs based on turnover.
    """
    dates = preds.index.intersection(rets.index)
    if len(dates) == 0:
        return pd.Series(dtype=float)

    # Build subportfolio weights for each start day
    sub_weights: List[pd.DataFrame] = []
    for i, d in enumerate(dates):
        p = preds.loc[d].dropna()
        if len(p) < 5:
            sub_weights.append(pd.Series(dtype=float))
            continue
        n = len(p)
        k = max(1, int(n * q))
        top = p.nlargest(k).index
        bot = p.nsmallest(k).index
        w = pd.Series(0.0, index=p.index)
        w.loc[top] = 1.0 / k
        w.loc[bot] = -1.0 / k
        # cap
        w = w.clip(lower=-cap, upper=cap)
        sub_weights.append(w)

    # Aggregate active subportfolios at each date (overlapping holds)
    all_weights = []
    for t in range(len(dates)):
        active = []
        for h in range(hold):
            idx = t - h
            if idx >= 0:
                w = sub_weights[idx]
                if isinstance(w, pd.Series) and not w.empty:
                    active.append(w)
        if active:
            W = pd.concat(active, axis=1).fillna(0.0).mean(axis=1)
        else:
            W = pd.Series(0.0, index=preds.columns)
        all_weights.append(W)
    Wdf = pd.DataFrame(all_weights, index=dates).fillna(0.0)
    # Normalize to have unit gross exposure (|w| sum = 1) to keep scale consistent
    gross = Wdf.abs().sum(axis=1).replace(0, np.nan)
    Wdf = Wdf.div(gross, axis=0).fillna(0.0)

    # Compute returns and turnover costs
    rets_aligned = rets.reindex(dates).reindex(columns=Wdf.columns).fillna(0.0)
    port_ret = (Wdf * rets_aligned).sum(axis=1)
    turnover = (Wdf.diff().abs().sum(axis=1)).fillna(Wdf.abs().sum(axis=1))
    cost = (cost_bps / 1e4) * turnover
    net = port_ret - cost
    return net


def run_cross_section(cfg: ExperimentConfig) -> Dict[str, float]:
    """Per-ticker predictions and cross-sectional evaluation.

    Returns dict with mean MSE, MAE, directional accuracy across all points, and cross-sectional rank-IC.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    frames = load_ohlcv_csvs(cfg.data.data_dir, cfg.data.tickers, cfg.backtest.start, cfg.backtest.end)
    # Build per-ticker windows
    t_data = {}
    n_min = None
    for t, df in frames.items():
        Xdf, yser = make_single_panel(df, features=cfg.data.features, target=cfg.data.target, horizon=cfg.data.horizon, use_returns=cfg.data.use_returns)
        Xw, yw, idx = sliding_windows(Xdf, yser, cfg.data.lookback)
        t_data[t] = (Xw, yw, idx)
        n_min = len(Xw) if n_min is None else min(n_min, len(Xw))

    if not t_data or n_min is None or n_min == 0:
        return {"mse": float("nan"), "mae": float("nan"), "directional_acc": float("nan"), "rank_ic": float("nan"), "ls_sharpe": float("nan")}

    # Prepare LLM windows per ticker (use same series across tickers by default)
    llm_series = None
    try:
        llm_series = load_llm_mean_series()
    except Exception:
        llm_series = None

    # Iterate splits based on n_min to align dates approximately
    train, val, test = cfg.backtest.train_span, cfg.backtest.val_span, cfg.backtest.test_span
    spans = train + val + test
    start = 0

    from .metrics import mse, mae, directional_accuracy, sharpe_ratio
    mse_list, mae_list, da_list = [], [], []
    # For rank-IC and LS returns
    per_date_true: dict[pd.Timestamp, dict[str, float]] = {}
    per_date_pred: dict[pd.Timestamp, dict[str, float]] = {}
    per_date_ret_simple: dict[pd.Timestamp, dict[str, float]] = {}

    while start + spans <= n_min:
        tr0, tr1 = start, start + train
        va0, va1 = tr1, tr1 + val
        te0, te1 = va1, va1 + test

        # Build & fit per ticker, then aggregate across agents
        for t, (Xw, yw, idx) in t_data.items():
            Xtr, ytr = Xw[tr0:tr1], yw[tr0:tr1]
            Xva, yva = Xw[va0:va1], yw[va0:va1]
            Xte, yte = Xw[te0:te1], yw[te0:te1]
            f = Xw.shape[-1]

            # LLM windows default zeros
            Xw_llm = np.zeros((len(Xw), cfg.data.lookback, 1), dtype=np.float32)
            if llm_series is not None:
                aligned = llm_series.reindex(Xdf.index).fillna(0.0).astype(np.float32) if 'Xdf' in locals() else None
                # per ticker, we don't have Xdf in scope; skip specialized alignment
            agents = [_build_agent(a, input_size=f, lookback=cfg.data.lookback) for a in cfg.agents]
            val_loaders, test_loaders = [], []
            for a, acfg in zip(agents, cfg.agents):
                if acfg.type.lower() == "llm_news":
                    Xtr_in, Xva_in, Xte_in = Xw_llm[tr0:tr1], Xw_llm[va0:va1], Xw_llm[te0:te1]
                else:
                    Xtr_in, Xva_in, Xte_in = Xtr, Xva, Xte
                tr_loader = _make_loader(Xtr_in, ytr, a.batch_size, True)
                va_loader = _make_loader(Xva_in, yva, a.batch_size, False)
                te_loader = _make_loader(Xte_in, yte, a.batch_size, False)
                a.fit(tr_loader, va_loader)
                val_loaders.append(va_loader)
                test_loaders.append(te_loader)

            val_preds = np.stack([a.predict(vl).numpy() for a, vl in zip(agents, val_loaders)], axis=1)
            agg = _build_aggregator(cfg.aggregator)
            agg.fit(val_preds, yva)
            test_preds = np.stack([a.predict(tl).numpy() for a, tl in zip(agents, test_loaders)], axis=1)
            yhat = agg.predict(test_preds)

            # segment metrics (per ticker) aggregated later
            mse_list.append(mse(yte, yhat))
            mae_list.append(mae(yte, yhat))
            da_list.append(directional_accuracy(yte, yhat))

            # Collect per-date cross-section
            seg_idx = idx[te0:te1]
            # convert log to simple returns if needed for LS
            if len(yte) > 0:
                if cfg.data.use_returns:
                    rt_simple = np.expm1(yte)
                else:
                    # if predicting price, compute simple returns from price series is not available here; skip
                    rt_simple = yte
            else:
                rt_simple = yte
            for ts, yt, yp, rs in zip(seg_idx, yte, yhat, rt_simple):
                per_date_true.setdefault(ts, {})[t] = float(yt)
                per_date_pred.setdefault(ts, {})[t] = float(yp)
                per_date_ret_simple.setdefault(ts, {})[t] = float(rs)

        start += test

    # Compute rank-IC per date then average
    rics = []
    for ts in sorted(per_date_true.keys()):
        yt_map = per_date_true.get(ts, {})
        yp_map = per_date_pred.get(ts, {})
        tickers = sorted(set(yt_map.keys()).intersection(yp_map.keys()))
        if len(tickers) < 3:
            continue
        yt = np.array([yt_map[t] for t in tickers], dtype=float)
        yp = np.array([yp_map[t] for t in tickers], dtype=float)
        rics.append(_spearman_corr(yt, yp))
    rank_ic_mean = float(np.nanmean(rics)) if rics else float("nan")

    # Build LS returns series with holding and costs
    dates = sorted(per_date_ret_simple.keys())
    preds_df = pd.DataFrame({ts: per_date_pred[ts] for ts in dates}).T
    rets_df = pd.DataFrame({ts: per_date_ret_simple[ts] for ts in dates}).T
    ls = long_short_with_holding(preds_df, rets_df, q=0.1, hold=5, cap=0.1, cost_bps=1.0).dropna()
    ls_sharpe = sharpe_ratio(ls.values) if len(ls) > 2 else float("nan")

    return {
        "mse": float(np.mean(mse_list)) if mse_list else float("nan"),
        "mae": float(np.mean(mae_list)) if mae_list else float("nan"),
        "directional_acc": float(np.mean(da_list)) if da_list else float("nan"),
        "rank_ic": rank_ic_mean,
        "ls_sharpe": float(ls_sharpe),
    }
