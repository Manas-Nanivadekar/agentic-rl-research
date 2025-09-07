from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict


SCHEMA_INSTRUCTIONS = (
    "You are a financial news analyst. Given the news item and target ticker, "
    "estimate the next-day (t+1) price impact direction and magnitude for the specific ticker, "
    "and your confidence in that estimate. Return strict JSON only."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "horizon_days": {"type": "integer", "minimum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["score", "confidence", "horizon_days"],
}

DEFAULT_PROMPT = (
    "Analyze the following news in relation to ticker {ticker}. "
    "Output JSON with fields: score in [-1,1] (bearish to bullish) for next-day impact, "
    "confidence in [0,1], horizon_days=1, and a brief rationale.\n\n"
    "Title: {title}\n"
    "Summary: {summary}\n"
    "URL: {url}"
)


def _make_key(ticker: str, title: str, summary: str, url: str) -> str:
    h = hashlib.sha256()
    h.update(ticker.encode("utf-8"))
    h.update((title or "").encode("utf-8"))
    h.update((summary or "").encode("utf-8"))
    h.update((url or "").encode("utf-8"))
    return h.hexdigest()


@dataclass
class GPT4oScorer:
    model: str = "gpt-4o"
    cache_dir: str = "data/llm_cache"
    max_retries: int = 3
    timeout_s: int = 60

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def score(self, ticker: str, title: str, summary: str = "", url: str = "") -> Dict[str, Any]:
        key = _make_key(ticker, title, summary, url)
        cpath = self._cache_path(key)
        if os.path.exists(cpath):
            with open(cpath, "r") as f:
                return json.load(f)

        payload = self._query_openai(ticker, title, summary, url)
        with open(cpath, "w") as f:
            json.dump(payload, f)
        return payload

    def _query_openai(self, ticker: str, title: str, summary: str, url: str) -> Dict[str, Any]:
        """Call OpenAI responses API with JSON output mode. Requires OPENAI_API_KEY."""
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            raise RuntimeError("openai package is required. Please install requirements.") from e

        client = OpenAI(timeout=self.timeout_s)
        prompt = DEFAULT_PROMPT.format(ticker=ticker, title=title, summary=summary, url=url)

        # Use JSON response format for structured outputs
        retries = 0
        while True:
            try:
                resp = client.responses.create(
                    model=self.model,
                    input=prompt,
                    system=SCHEMA_INSTRUCTIONS,
                    response_format={"type": "json_object"},
                )
                text = resp.output_text
                data = json.loads(text)
                # basic validation
                score = float(max(-1.0, min(1.0, data.get("score", 0.0))))
                conf = float(max(0.0, min(1.0, data.get("confidence", 0.0))))
                horizon = int(data.get("horizon_days", 1))
                rationale = str(data.get("rationale", ""))
                return {
                    "score": score,
                    "confidence": conf,
                    "horizon_days": horizon,
                    "rationale": rationale,
                    "model": self.model,
                }
            except Exception as e:  # pragma: no cover
                retries += 1
                if retries >= self.max_retries:
                    # fallback neutral
                    return {
                        "score": 0.0,
                        "confidence": 0.0,
                        "horizon_days": 1,
                        "rationale": f"error: {e}",
                        "model": self.model,
                    }
                time.sleep(1.0 * retries)


def score_article_to_json(scorer: GPT4oScorer, ticker: str, article: Dict[str, Any]) -> Dict[str, Any]:
    res = scorer.score(
        ticker=ticker,
        title=article.get("title", ""),
        summary=article.get("summary", ""),
        url=article.get("url", ""),
    )
    return {
        **article,
        "score": res["score"],
        "confidence": res["confidence"],
        "horizon_days": res["horizon_days"],
        "rationale": res.get("rationale", ""),
        "model": res.get("model", ""),
    }

