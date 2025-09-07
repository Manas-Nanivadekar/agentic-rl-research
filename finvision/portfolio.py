from __future__ import annotations

import numpy as np
import pandas as pd


def positions_from_predictions(y_pred: pd.Series, threshold: float = 0.0) -> pd.Series:
    # simple long/short with threshold; 1 for long, -1 for short, 0 otherwise
    pos = y_pred.copy()
    pos[:] = 0.0
    pos[y_pred > threshold] = 1.0
    pos[y_pred < -threshold] = -1.0
    return pos


def turnover(positions: pd.Series) -> pd.Series:
    # absolute change in position day-over-day
    return positions.diff().abs().fillna(positions.abs())


def strategy_returns(y_true: pd.Series, positions: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    # assume y_true are simple returns per period
    pos_aligned = positions.reindex(y_true.index).fillna(0.0)
    gross = pos_aligned * y_true
    tc = (cost_bps / 1e4) * turnover(pos_aligned)
    net = gross - tc
    return net

