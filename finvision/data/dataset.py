from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_ohlcv_csvs(
    data_dir: str,
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = pd.read_csv(f"{data_dir}/{t}.csv")
        # standardize column names
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").set_index("timestamp")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
        else:
            raise ValueError("CSV must contain a timestamp/date column")
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        frames[t] = df
    return frames


def _default_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols].copy()


def make_panel(
    frames: Dict[str, pd.DataFrame],
    features: Optional[List[str]] = None,
    target: str = "close",
    horizon: int = 1,
    use_returns: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    # align on timestamps; create multi-ticker feature panel by concatenating columns with ticker suffixes
    feats: List[pd.DataFrame] = []
    targs: List[pd.Series] = []
    for t, df in frames.items():
        fdf = df[features] if features else _default_features(df)
        fdf = fdf.add_prefix(f"{t}_")
        feats.append(fdf)

        base = df[target]
        if use_returns:
            # log return over horizon
            y = np.log(base.shift(-horizon) / base).rename(t)
        else:
            y = base.shift(-horizon).rename(t)
        targs.append(y)

    X = pd.concat(feats, axis=1, join="inner").dropna()
    Y = pd.concat(targs, axis=1, join="inner").reindex(X.index)
    # average across tickers for a single target
    y = Y.mean(axis=1)
    # drop rows where target is NaN (e.g., due to shift at the tail)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


def make_single_panel(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = "close",
    horizon: int = 1,
    use_returns: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build features and target for a single ticker DataFrame.

    Returns X (features) and y (target) aligned, dropping NaNs from horizon shift.
    """
    fdf = df[features] if features else _default_features(df)
    base = df[target]
    if use_returns:
        y = np.log(base.shift(-horizon) / base)
    else:
        y = base.shift(-horizon)
    mask = y.notna()
    X = fdf.loc[mask].copy()
    y = y.loc[mask].copy()
    return X, y


def sliding_windows(X: pd.DataFrame, y: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    values = X.values.astype(np.float32)
    target = y.values.astype(np.float32)
    n = len(X)
    xs, ys, idx = [], [], []
    for i in range(lookback, n):
        xs.append(values[i - lookback : i])
        ys.append(target[i])
        idx.append(X.index[i])
    return np.stack(xs), np.stack(ys), idx


@dataclass
class TimeSeriesDataset(Dataset):
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y
