#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import json
from datetime import datetime
import shutil

from finvision.cli import load_config
from finvision.data.synthetic import make_synthetic_ohlcv
from finvision.backtest import run_walk_forward, run_walk_forward_with_preds
from finvision.news.signal import build_llm_news_signal
from finvision.metrics import sharpe_ratio, max_drawdown, cumulative_return
from finvision.portfolio import positions_from_predictions, strategy_returns
from finvision.rl.pipeline import run_rl


def ensure_price_data(data_dir: str, tickers: List[str], synth_days: int = 2000) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for t in tickers:
        path = os.path.join(data_dir, f"{t}.csv")
        if not os.path.exists(path):
            print(f"[data] Missing {path}; generating synthetic OHLCV ({synth_days} days)...")
            make_synthetic_ohlcv(ticker=t, days=synth_days, data_dir=data_dir)
        else:
            print(f"[data] Found {path}")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline: data -> LLM -> backtest -> trade -> RL")
    parser.add_argument("--config", default="configs/example.yaml", help="Path to experiment YAML")
    parser.add_argument("--synth-days", type=int, default=2000, help="If data missing, synthesize this many days")
    parser.add_argument("--skip-llm", action="store_true", help="Skip building LLM news signal")
    args = parser.parse_args()

    # 1) Load config
    exp = load_config(args.config)
    print(f"[config] Loaded config: {args.config}")

    # 2) Ensure price data exists (create synthetic if missing)
    ensure_price_data(exp.data.data_dir, exp.data.tickers, synth_days=args.synth_days)

    # 3) Build LLM news signal (if enabled)
    if not args.skip_llm:
        out = build_llm_news_signal(exp, out_path=os.path.join("data", "features", "llm_signal.csv"))
        print(f"[llm] Built LLM daily signal at: {out}")
    else:
        print("[llm] Skipping LLM news signal build per flag")

    # Prepare results directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", run_id)
    os.makedirs(results_dir, exist_ok=True)
    try:
        shutil.copy(args.config, os.path.join(results_dir, "config.yaml"))
    except Exception:
        pass

    # 4) Backtest (prediction metrics)
    pred_metrics = run_walk_forward(exp)
    print("[backtest] Prediction metrics (avg across splits):")
    for k, v in pred_metrics.items():
        print(f"  - {k}: {v:.6f}")

    # 5) Strategy backtest (PnL with costs)
    metrics, df = run_walk_forward_with_preds(exp)
    y_true = df["y_true"].astype(float)
    if exp.data.use_returns:
        y_true = y_true.apply(np.expm1)
    y_pred = df["y_pred"].astype(float)
    pos = positions_from_predictions(y_pred, threshold=exp.threshold)
    rets = strategy_returns(y_true, pos, cost_bps=exp.cost_bps)

    print("[trade] Prediction metrics (avg across splits):")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.6f}")
    print("[trade] Strategy metrics (net of costs):")
    print(f"  - sharpe: {sharpe_ratio(rets.values):.4f}")
    print(f"  - max_drawdown: {max_drawdown(rets.values):.4f}")
    print(f"  - cumulative_return: {cumulative_return(rets.values):.4f}")

    # 6) RL evaluation
    try:
        rl_metrics = run_rl(exp)
        print("[rl] Strategy metrics (test split):")
        for k, v in rl_metrics.items():
            print(f"  - {k}: {v:.6f}")
    except Exception as e:
        print(f"[rl] Skipped due to error: {e}")
        rl_metrics = {"error": str(e)}

    # 7) Persist results
    summary = {
        "config": os.path.abspath(args.config),
        "run_id": run_id,
        "backtest_metrics": pred_metrics,
        "trade_prediction_metrics": metrics,
        "trade_strategy_metrics": {
            "sharpe": float(sharpe_ratio(rets.values)),
            "max_drawdown": float(max_drawdown(rets.values)),
            "cumulative_return": float(cumulative_return(rets.values)),
        },
        "rl_metrics": rl_metrics,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # Save predictions and returns
    try:
        df.to_csv(os.path.join(results_dir, "predictions.csv"))
    except Exception:
        pass
    try:
        rets.to_frame(name="return").to_csv(os.path.join(results_dir, "strategy_returns.csv"))
    except Exception:
        pass

    print(f"[results] Saved to {results_dir}")


if __name__ == "__main__":
    main()
