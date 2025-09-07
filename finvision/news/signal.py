from __future__ import annotations

import os
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..config import ExtendedExperimentConfig
from .fetch_yahoo import fetch_yahoo_news
from ..llm.openai_scorer import GPT4oScorer, score_article_to_json


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _articles_to_daily_scores(arts: List[dict], max_per_day: int = 25) -> pd.DataFrame:
    rows: List[Tuple[pd.Timestamp, float, float, str]] = []
    # Sort by time to keep first N if needed
    arts_sorted = sorted(arts, key=lambda a: a.get("published_at", ""))
    counts: Dict[str, int] = defaultdict(int)
    for a in arts_sorted:
        ts = pd.to_datetime(a.get("published_at"))
        day = ts.normalize()
        key = f"{a.get('ticker','')}-{day.date()}"
        if counts[key] >= max_per_day:
            continue
        counts[key] += 1
        rows.append((day, float(a.get("score", 0.0)), float(a.get("confidence", 0.0)), a.get("ticker", "")))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "ticker", "score", "confidence"]).set_index("timestamp")
    df = pd.DataFrame(rows, columns=["timestamp", "score", "confidence", "ticker"]).set_index("timestamp")
    return df


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # confidence-weighted mean per ticker-day
    def agg_func(g):
        w = g["confidence"].values
        s = g["score"].values
        if w.sum() > 0:
            m = (w * s).sum() / w.sum()
        else:
            m = s.mean() if len(s) else 0.0
        return pd.Series({"signal": m})

    out = df.groupby([df.index, "ticker"]).apply(agg_func)
    out.index.names = ["timestamp", "ticker"]
    return out


def build_llm_news_signal(cfg: ExtendedExperimentConfig, out_path: str = "data/features/llm_signal.csv") -> str:
    """Fetch Yahoo news for each ticker, score with GPT-4o, aggregate daily signals, and save combined CSV.

    Output CSV columns:
      - timestamp, ticker, signal
      - also writes a pivoted mean across tickers as 'LLM_MEAN' in a separate file for convenience.
    """
    _ensure_dir(os.path.dirname(out_path))

    all_scored: List[dict] = []
    scorer = GPT4oScorer(model=cfg.llm.model, cache_dir=cfg.llm.cache_dir, max_retries=cfg.llm.max_retries, timeout_s=cfg.llm.timeout_s)

    # Fetch & score per ticker
    for t in cfg.data.tickers:
        arts = fetch_yahoo_news(t, cache_dir=cfg.news.cache_dir, start=cfg.news.start or cfg.backtest.start, end=cfg.news.end or cfg.backtest.end)
        for a in arts:
            j = score_article_to_json(scorer, t, a)
            all_scored.append(j)

    # Build daily confidence-weighted signal per ticker
    daily_df = _articles_to_daily_scores(all_scored, max_per_day=cfg.news.max_articles_per_day)
    agg_df = _aggregate_daily(daily_df)
    if agg_df.empty:
        # write empty stub
        pd.DataFrame(columns=["timestamp", "ticker", "signal"]).to_csv(out_path, index=False)
        return out_path

    # Save long-form
    long_df = agg_df.reset_index()
    long_df.to_csv(out_path, index=False)

    # Also save mean across tickers per day for convenience
    pivot = long_df.pivot(index="timestamp", columns="ticker", values="signal").sort_index()
    pivot["LLM_MEAN"] = pivot.mean(axis=1)
    pivot_out = os.path.join(os.path.dirname(out_path), "llm_signal_pivot.csv")
    pivot.to_csv(pivot_out)
    return out_path


def load_llm_mean_series(features_dir: str = "data/features", fname: str = "llm_signal_pivot.csv") -> pd.Series:
    path = os.path.join(features_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM pivot signal not found: {path}. Run 'finvision news' first.")
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if "LLM_MEAN" not in df.columns:
        # if not present, compute
        df["LLM_MEAN"] = df.mean(axis=1)
    return df["LLM_MEAN"].astype(float)

