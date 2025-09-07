from __future__ import annotations

from typing import Dict
import numpy as np

from ..config import ExperimentConfig
from ..data.dataset import load_ohlcv_csvs, make_panel, sliding_windows
from ..metrics import sharpe_ratio, max_drawdown, cumulative_return
from .env import TradingEnv
from .agent_pg import PolicyGradientAgent
from .agent_ppo import PPOAgent


def _make_env_series(cfg: ExperimentConfig):
    frames = load_ohlcv_csvs(cfg.data.data_dir, cfg.data.tickers, cfg.backtest.start, cfg.backtest.end)
    Xdf, yser = make_panel(
        frames,
        features=cfg.data.features,
        target=cfg.data.target,
        horizon=cfg.data.horizon,
        use_returns=cfg.data.use_returns,
    )
    Xw, yw, idx = sliding_windows(Xdf, yser, cfg.data.lookback)
    return Xw, yw


def run_rl(cfg: ExperimentConfig) -> Dict[str, float]:
    Xw, yw = _make_env_series(cfg)
    n = len(Xw)
    spans = cfg.backtest.train_span + cfg.backtest.val_span + cfg.backtest.test_span
    if n < spans:
        raise ValueError("Not enough data for RL split; reduce spans or lookback")

    start = (n - spans)
    tr0 = start
    tr1 = start + cfg.backtest.train_span + cfg.backtest.val_span
    te0 = tr1
    te1 = tr1 + cfg.backtest.test_span

    Xtr, ytr = Xw[tr0:tr1], yw[tr0:tr1]
    Xte, yte = Xw[te0:te1], yw[te0:te1]

    # If y are log returns, convert to simple returns for reward
    if cfg.data.use_returns:
        ytr = np.expm1(ytr)
        yte = np.expm1(yte)

    obs_dim = Xtr.shape[1] * Xtr.shape[2]
    agent = PolicyGradientAgent(obs_dim=obs_dim, hidden=128, lr=1e-3, gamma=0.99, device="cpu")
    env_train = TradingEnv(Xtr, ytr, cost_bps=cfg.cost_bps)
    env_test = TradingEnv(Xte, yte, cost_bps=cfg.cost_bps)

    agent.fit(env_train, epochs=10)

    # Evaluate on test: follow stochastic policy with greedy sampling (argmax of probs)
    obs = env_test.reset()
    rets = []
    done = False
    while not done:
        # greedy action from policy
        net = agent.net()
        import torch
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0)
            logits = net(x)
            a = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
        obs, r, done, _ = env_test.step(a)
        rets.append(r)

    rets = np.array(rets, dtype=np.float32)
    metrics = {
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(rets),
        "cumulative_return": cumulative_return(rets),
    }
    return metrics


def run_rl_ppo(cfg: ExperimentConfig) -> Dict[str, float]:
    Xw, yw = _make_env_series(cfg)
    n = len(Xw)
    spans = cfg.backtest.train_span + cfg.backtest.val_span + cfg.backtest.test_span
    results = []
    for start in range(0, n - spans + 1, cfg.backtest.test_span):
        tr0 = start
        tr1 = start + cfg.backtest.train_span + cfg.backtest.val_span
        te0 = tr1
        te1 = tr1 + cfg.backtest.test_span
        Xtr, ytr = Xw[tr0:tr1], yw[tr0:tr1]
        Xte, yte = Xw[te0:te1], yw[te0:te1]
        if cfg.data.use_returns:
            ytr = np.expm1(ytr)
            yte = np.expm1(yte)
        obs_dim = Xtr.shape[1] * Xtr.shape[2]
        agent = PPOAgent(obs_dim=obs_dim, hidden=128, device="cpu")
        env_train = TradingEnv(Xtr, ytr, cost_bps=cfg.cost_bps)
        env_test = TradingEnv(Xte, yte, cost_bps=cfg.cost_bps)
        agent.fit(env_train, epochs=10)

        obs = env_test.reset()
        rets = []
        done = False
        import torch
        with torch.no_grad():
            while not done:
                x = torch.from_numpy(obs).float().unsqueeze(0)
                logits = agent.policy()(x)
                a = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
                obs, r, done, _ = env_test.step(a)
                rets.append(r)
        rets = np.array(rets, dtype=np.float32)
        results.append({
            "sharpe": sharpe_ratio(rets),
            "max_drawdown": max_drawdown(rets),
            "cumulative_return": cumulative_return(rets),
        })
    # Average across splits
    if not results:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "cumulative_return": float("nan")}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
