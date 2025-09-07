from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dt_from_epoch(sec: int) -> datetime:
    return datetime.fromtimestamp(sec, tz=timezone.utc)


def _cache_path(cache_dir: str, ticker: str) -> str:
    _ensure_dir(cache_dir)
    return os.path.join(cache_dir, f"{ticker.upper()}_news.jsonl")


def fetch_yahoo_news(ticker: str, cache_dir: str, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch Yahoo Finance news via yfinance (best-effort) and append to cache.

    Returns a list of normalized articles with fields:
      - ticker, title, publisher, url, published_at (ISO), summary (optional)
    """
    try:
        import yfinance as yf
    except Exception:  # pragma: no cover
        raise RuntimeError("yfinance is required for Yahoo news fetching. Please install requirements.")

    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    t = yf.Ticker(ticker)
    items = t.news or []

    out: List[Dict[str, Any]] = []
    for it in items:
        # normalize
        pub_ts = it.get("providerPublishTime") or it.get("published_at")
        if pub_ts is None:
            continue
        if isinstance(pub_ts, (int, float)):
            pub_dt = _dt_from_epoch(int(pub_ts))
        else:
            try:
                pub_dt = datetime.fromisoformat(str(pub_ts))
            except Exception:
                continue
        if start_dt and pub_dt < start_dt:
            continue
        if end_dt and pub_dt > end_dt:
            continue

        out.append(
            {
                "ticker": ticker.upper(),
                "title": it.get("title") or "",
                "publisher": it.get("publisher") or it.get("provider") or "",
                "url": it.get("link") or it.get("thumbnail", {}).get("link") or "",
                "published_at": pub_dt.astimezone(timezone.utc).isoformat(),
                "summary": it.get("summary") or "",
            }
        )

    # append to cache
    cpath = _cache_path(cache_dir, ticker)
    existing = set()
    if os.path.exists(cpath):
        with open(cpath, "r") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    existing.add((j.get("title"), j.get("published_at")))
                except Exception:
                    pass
    with open(cpath, "a") as f:
        for art in out:
            key = (art.get("title"), art.get("published_at"))
            if key in existing:
                continue
            f.write(json.dumps(art) + "\n")

    return out

