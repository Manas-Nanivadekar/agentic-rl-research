from __future__ import annotations

import os
from typing import List
from urllib.request import urlretrieve


def download_stooq_daily(tickers: List[str], data_dir: str = "data") -> None:
    os.makedirs(data_dir, exist_ok=True)
    base = "https://stooq.com/q/d/l/?i=d&s={ticker}"
    for t in tickers:
        url = base.format(ticker=t.lower())
        out = os.path.join(data_dir, f"{t.upper()}.csv")
        urlretrieve(url, out)

