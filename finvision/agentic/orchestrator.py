from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from ..llm.openai_scorer import GPT4oScorer
from ..llm.prompts import (
    NEWS_SUMMARIZER_PROMPT,
    CHART_ANALYST_PROMPT,
    REFLECTION_PROMPT,
    TRADING_SIGNAL_CHART_REFLECTION_PROMPT,
    PREDICTION_AGENT_PROMPT,
)
from ..indicators.technicals import macd, rsi, kdj


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@dataclass
class Portfolio:
    cash: float = 10000.0
    shares: float = 0.0
    avg_price: float = 0.0

    def value(self, price: float) -> float:
        return self.cash + self.shares * price

    def buy(self, price: float, pct: float):
        amt = self.cash * pct
        if amt <= 0:
            return
        qty = amt / price
        new_total_cost = self.avg_price * self.shares + qty * price
        self.shares += qty
        self.avg_price = new_total_cost / max(self.shares, 1e-9)
        self.cash -= qty * price

    def sell(self, price: float, pct: float):
        qty = self.shares * pct
        if qty <= 0:
            return
        self.cash += qty * price
        self.shares -= qty
        if self.shares <= 1e-9:
            self.shares = 0.0
            self.avg_price = 0.0


@dataclass
class AgenticState:
    news_summary: List[str] = field(default_factory=list)
    chart_analysis: List[str] = field(default_factory=list)
    reflection_insights: Dict[str, str] = field(default_factory=lambda: {"short_term": "", "medium_term": ""})
    market_intelligence: str = ""
    decisions: List[Dict[str, Any]] = field(default_factory=list)


def _call_openai_json(scorer: GPT4oScorer, system_prompt: str, user_prompt: str) -> str:
    # Reuse scorer client for JSON; but here we want free-form text, so call responses without json format
    from openai import OpenAI
    client = OpenAI(timeout=scorer.timeout_s)
    resp = client.responses.create(model=scorer.model, input=user_prompt, system=system_prompt)
    return resp.output_text


def _recent_context(df: pd.DataFrame, lookback: int = 60) -> str:
    tail = df.tail(lookback)
    cols = [c for c in ["open","high","low","close","volume"] if c in tail.columns]
    return tail[cols].round(4).to_csv(index=True)


def _build_technicals(df: pd.DataFrame) -> Dict[str, Any]:
    m = macd(df["close"]) if "close" in df.columns else pd.DataFrame()
    kdj_df = kdj(df["high"], df["low"], df["close"]) if set(["high","low","close"]).issubset(df.columns) else pd.DataFrame()
    r = rsi(df["close"]) if "close" in df.columns else pd.Series(dtype=float)
    out = {}
    if not m.empty:
        out["macd"] = m.round(4).tail(5).to_dict(orient="records")
    if not kdj_df.empty:
        out["kdj"] = kdj_df.round(2).tail(5).to_dict(orient="records")
    if not r.empty:
        out["rsi"] = list(r.round(2).tail(5).values)
    return out


def _format_portfolio_for_prompt(p: Portfolio, price: float) -> Dict[str, Any]:
    total_val = p.value(price)
    cash_pct = 100.0 * (p.cash / total_val) if total_val > 0 else 0.0
    unrl = (price - (p.avg_price or price)) * p.shares
    unrl_pct = 100.0 * ((price - (p.avg_price or price)) / (p.avg_price or price)) if p.avg_price > 0 else 0.0
    return dict(
        current_shares=p.shares,
        current_price=price,
        avg_purchase_price=p.avg_price if p.avg_price > 0 else price,
        total_value=total_val,
        cash_reserve=p.cash,
        cash_percentage=cash_pct,
        unrealized_profit_percentage=unrl_pct,
    )


def run_agentic_day(scorer: GPT4oScorer, ticker: str, date: pd.Timestamp, df_hist: pd.DataFrame, daily_news_text: str, state: AgenticState, portfolio: Portfolio) -> Dict[str, Any]:
    # 1) News Summarizer
    ns_prompt = NEWS_SUMMARIZER_PROMPT.format(ticker=ticker, news_data=daily_news_text)
    news_summary = _call_openai_json(scorer, system_prompt="", user_prompt=ns_prompt)
    state.news_summary.append(news_summary)

    # 2) Chart Analyst (with computed technicals context)
    context = _recent_context(df_hist)
    tech = _build_technicals(df_hist)
    tech_text = json.dumps(tech)
    ca_prompt = CHART_ANALYST_PROMPT.format(ticker=ticker, context=context + "\nIndicators:" + tech_text)
    chart_analysis = _call_openai_json(scorer, system_prompt="", user_prompt=ca_prompt)
    state.chart_analysis.append(chart_analysis)

    # 3) Reflections (short and medium)
    hist_json = json.dumps(state.decisions[-30:], default=str)
    short_reflection = _call_openai_json(scorer, system_prompt="", user_prompt=REFLECTION_PROMPT.format(len_data=min(30, len(state.decisions)), ticker=ticker, json_data=hist_json))
    medium_hist_json = json.dumps(state.decisions[-90:], default=str)
    medium_reflection = _call_openai_json(scorer, system_prompt="", user_prompt=REFLECTION_PROMPT.format(len_data=min(90, len(state.decisions)), ticker=ticker, json_data=medium_hist_json))
    state.reflection_insights["short_term"] = short_reflection
    state.reflection_insights["medium_term"] = medium_reflection

    # 4) Trading Signal Chart Reflection
    signal_context = json.dumps({
        "closes": list(df_hist["close"].tail(60).round(4).values),
        "decisions": state.decisions[-60:],
    })
    tscr = _call_openai_json(scorer, system_prompt="", user_prompt=TRADING_SIGNAL_CHART_REFLECTION_PROMPT.format(ticker=ticker, context=signal_context))

    # 5) Prediction Agent
    price = float(df_hist["close"].iloc[-1])
    port_vars = _format_portfolio_for_prompt(portfolio, price)
    pred_prompt = PREDICTION_AGENT_PROMPT.format(
        ticker=ticker,
        current_date=str(date.date()),
        technical_analysis=chart_analysis,
        news_summary=news_summary,
        short_term_reflection=short_reflection,
        medium_term_reflection=medium_reflection,
        market_intelligence=tscr,
        json_data=json.dumps({"closes": list(df_hist["close"].tail(30).round(4).values)}),
        len_history=min(30, len(df_hist)),
        **port_vars,
    )
    pred_text = _call_openai_json(scorer, system_prompt="", user_prompt=pred_prompt)

    # Parse recommendation
    rec = "HOLD"
    size = 0
    for line in pred_text.splitlines():
        L = line.strip()
        if L.lower().startswith("recommendation:"):
            rec = L.split(":", 1)[1].strip().upper()
        if L.lower().startswith("position size:"):
            try:
                size = int("".join(ch for ch in L if ch.isdigit()))
            except Exception:
                size = 0
    size = int(np.clip(size, 0, 10))

    # Map size bucket to percent
    pct = 0.0
    if rec == "BUY":
        pct = 0.01 * max(1, size)  # 1-10%
        portfolio.buy(price, pct)
    elif rec == "SELL":
        pct = 0.01 * max(1, size)
        portfolio.sell(price, pct)
    else:
        pct = 0.0

    decision = {
        "date": str(date.date()),
        "price": price,
        "recommendation": rec,
        "size": size,
        "pct": pct,
        "portfolio_value": portfolio.value(price),
        "shares": portfolio.shares,
        "cash": portfolio.cash,
        "news_summary": news_summary[:500],
        "chart_analysis": chart_analysis[:500],
    }
    state.decisions.append(decision)
    return decision


def run_agentic_pipeline(ticker: str, df: pd.DataFrame, news_by_day: Dict[pd.Timestamp, str], model: str = "gpt-4o", cache_dir: str = "data/llm_cache") -> List[Dict[str, Any]]:
    scorer = GPT4oScorer(model=model, cache_dir=cache_dir)
    portfolio = Portfolio()
    state = AgenticState()
    out: List[Dict[str, Any]] = []

    for date in df.index:
        # Require at least 60 bars to start
        idx = df.index.get_loc(date)
        if idx < 60:
            continue
        hist = df.iloc[: idx + 1]
        news_text = news_by_day.get(date.normalize(), "")
        dec = run_agentic_day(scorer, ticker, date, hist, news_text, state, portfolio)
        out.append(dec)
    return out

