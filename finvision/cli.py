import argparse
import os
import yaml
from dataclasses import asdict
import numpy as np

from .config import (
    ExperimentConfig,
    DataConfig,
    AgentConfig,
    AggregatorConfig,
    BacktestConfig,
)
from .backtest import run_walk_forward, run_walk_forward_with_preds
from .metrics import sharpe_ratio, max_drawdown, cumulative_return
from .portfolio import positions_from_predictions, strategy_returns
from .data.download import download_stooq_daily
from .rl.pipeline import run_rl
from .data.synthetic import make_synthetic_ohlcv


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    data = cfg.get("data", {})
    agents = cfg.get("agents", [])
    agg = cfg.get("aggregator", {})
    bt = cfg.get("backtest", {})

    exp = ExperimentConfig(
        seed=cfg.get("seed", 42),
        data=DataConfig(**data),
        agents=[AgentConfig(**a) for a in agents],
        aggregator=AggregatorConfig(**agg),
        backtest=BacktestConfig(**bt),
    )
    return exp


def cmd_backtest(args):
    exp = load_config(args.config)
    metrics = run_walk_forward(exp)
    print("Backtest metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v:.6f}")


def main():
    parser = argparse.ArgumentParser(prog="finvision")
    sub = parser.add_subparsers(dest="cmd")

    bt = sub.add_parser("backtest", help="Run walk-forward backtest from config")
    bt.add_argument("--config", required=True, help="Path to YAML config")
    bt.set_defaults(func=cmd_backtest)

    tr = sub.add_parser("trade", help="Run strategy backtest with transaction costs")
    tr.add_argument("--config", required=True, help="Path to YAML config")
    tr.set_defaults(func=cmd_trade)

    dl = sub.add_parser(
        "download", help="Download daily OHLCV from Stooq for config tickers"
    )
    dl.add_argument("--config", required=True, help="Path to YAML config")
    dl.set_defaults(func=cmd_download)

    rl = sub.add_parser(
        "rl", help="Train an RL policy (REINFORCE) and evaluate on test split"
    )
    rl.add_argument("--config", required=True, help="Path to YAML config")
    rl.set_defaults(func=cmd_rl)

    sy = sub.add_parser("make-synth", help="Generate synthetic OHLCV data")
    sy.add_argument("--ticker", default="SYNTH")
    sy.add_argument("--start", default="2015-01-01")
    sy.add_argument("--days", type=int, default=2000)
    sy.add_argument("--seed", type=int, default=42)
    sy.add_argument("--outdir", default="data")
    sy.set_defaults(func=cmd_make_synth)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)

def cmd_trade(args):
    exp = load_config(args.config)
    metrics, df = run_walk_forward_with_preds(exp)

    y_true = df["y_true"].astype(float)
    # If configured target is log returns, convert to simple returns for trading
    if exp.data.use_returns:
        y_true = y_true.apply(np.expm1)
    y_pred = df["y_pred"].astype(float)
    pos = positions_from_predictions(y_pred, threshold=exp.threshold)
    rets = strategy_returns(y_true, pos, cost_bps=exp.cost_bps)

    print("Backtest metrics (prediction):")
    for k, v in metrics.items():
        print(f"- {k}: {v:.6f}")

    print("\nStrategy metrics (net of costs):")
    print(f"- sharpe: {sharpe_ratio(rets.values):.4f}")
    print(f"- max_drawdown: {max_drawdown(rets.values):.4f}")
    print(f"- cumulative_return: {cumulative_return(rets.values):.4f}")


def cmd_download(args):
    exp = load_config(args.config)
    download_stooq_daily(exp.data.tickers, exp.data.data_dir)
    print(f"Downloaded: {', '.join(exp.data.tickers)} -> {exp.data.data_dir}")


def cmd_rl(args):
    exp = load_config(args.config)
    metrics = run_rl(exp)
    print("RL strategy metrics (test split):")
    for k, v in metrics.items():
        print(f"- {k}: {v:.6f}")


def cmd_make_synth(args):
    path = make_synthetic_ohlcv(
        ticker=args.ticker, start=args.start, days=args.days, seed=args.seed, data_dir=args.outdir
    )
    print(f"Wrote synthetic data: {path}")


if __name__ == "__main__":
    main()
