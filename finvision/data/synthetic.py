from __future__ import annotations

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def make_synthetic_ohlcv(
    ticker: str = "SYNTH",
    start: str = "2015-01-01",
    days: int = 2000,
    seed: int = 42,
    data_dir: str = "data",
) -> str:
    rng = np.random.default_rng(seed)
    dates = []
    d0 = datetime.fromisoformat(start)
    # generate business-day-like sequence (skip weekends)
    t = d0
    while len(dates) < days:
        if t.weekday() < 5:
            dates.append(t)
        t += timedelta(days=1)

    mu = 0.07 / 252
    sigma = 0.2 / np.sqrt(252)
    price = [100.0]
    for _ in range(1, len(dates)):
        r = rng.normal(mu, sigma)
        price.append(price[-1] * float(np.exp(r)))
    price = np.array(price)

    close = price
    # generate OHLC around close
    spread = rng.normal(0, 0.005, size=len(close))
    openp = close * (1 + rng.normal(0, 0.002, size=len(close)))
    high = np.maximum(openp, close) * (1 + np.abs(spread))
    low = np.minimum(openp, close) * (1 - np.abs(spread))
    vol = rng.lognormal(mean=12.0, sigma=0.3, size=len(close)).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, f"{ticker}.csv")
    df.to_csv(out, index=False)
    return out

