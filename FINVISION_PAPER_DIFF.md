# FinVision (ACM DOI: 10.1145/3677052.3698688)
# Paper vs. Current Implementation

This document tracks the differences between the FinVision paper and this clean‑room repo. It is designed to be updated as we confirm details from the paper and extend the codebase.

Note: In this environment, the PDF text could not be programmatically extracted. Items marked “Confirm from paper” should be verified (sections, figures, hyperparameters, dataset definitions). Everything else is grounded in the current repo code.

## Summary
- Theme alignment: yes — multi‑agent stock prediction with an ensemble and trading evaluation.
- Exact reproduction: no — several paper‑specific components, datasets, and evaluation protocols are not present.

## Differences by Area

### 1) Data & Preprocessing
- Paper: Confirm universe (tickers/regions), frequency (daily/intraday), date ranges, and vendors; confirm factor/indicator sets and any corporate actions handling; confirm leakage controls and universe reconstitution.
- Repo: Daily per‑ticker OHLCV CSVs or synthetic data; optional hand‑selected features; simple alignment by timestamp. Leakage mitigation only for news via session bucketing to US/Eastern close.
  - Code: `finvision/data/dataset.py`, `finvision/data/synthetic.py`, `finvision/news/signal.py`.
- Gap: No paper‑specific dataset loaders, universe definitions, survivorship/corporate actions handling, or strict leakage checks across the full pipeline.
- Action: Add dataset scripts matching the paper; implement precise splits, calendar/market hours, and leakage controls.

### 2) Agents (Supervised Predictors)
- Paper: Confirm the exact agent types, architectures, and objectives (e.g., temporal Transformers, CNNs for charts, knowledge/graph modules, cross‑asset agents, multi‑task or multi‑horizon losses).
- Repo: LSTM and MLP over sliding windows; optional LLM news agent (1D MLP). All trained with MSE on 1‑step target.
  - Code: `finvision/agents/{lstm.py, mlp.py, llm_news.py}`, training loops inline per agent.
- Gap: No paper‑specific backbones or specialized objectives; no multi‑horizon or cross‑sectional joint objectives (beyond optional cross‑section eval tool).
- Action: Implement paper’s agent architectures and loss functions; add training/eval parity.

### 3) Multi‑Agent Interaction / Orchestration
- Paper: Confirm the communication/coordination protocol (e.g., specialized roles, negotiation, arbitration, tool‑use, or message passing) and learning thereof.
- Repo: A simple sequential LLM prompt pipeline (“agentic” workflow) for narrative decisions; not tied to predictive training.
  - Code: `finvision/agentic/orchestrator.py`, `finvision/llm/prompts.py`.
- Gap: No learned inter‑agent communication, arbitration policy, or formal coordination matching the paper.
- Action: Reproduce the paper’s multi‑agent protocol and integration with predictive models.

### 4) Ensemble / Aggregation
- Paper: Confirm ensemble method (e.g., constrained meta‑learner, dynamic weights, multi‑stage stacking) and fitting protocol.
- Repo: Ridge, tiny MLP, and a constrained non‑negative sum‑to‑one projector fitted on validation predictions.
  - Code: `finvision/ensemble/aggregators.py`.
- Gap: If the paper uses a different or adaptive/meta policy, it is not present.
- Action: Implement the paper’s ensemble/arbiter exactly and match constraints/training.

### 5) Targets & Horizons
- Paper: Confirm prediction target(s) (returns vs prices), horizons (t+1, multi‑horizon), and whether cross‑sectional ranking is primary.
- Repo: Default to 1‑day log return; average target across tickers in multi‑ticker panels.
  - Code: `finvision/data/dataset.py`.
- Gap: No multi‑horizon/multi‑task heads or explicit cross‑sectional training objective.
- Action: Extend dataset/heads to match paper targets and objectives.

### 6) Training Protocol & Hyperparameters
- Paper: Confirm optimizer, schedules, early stopping, CV, hyperparameter search, seeds, and ablation plan.
- Repo: Minimal loops (AdamW + MSE, fixed epochs); best‑val snapshot; no schedulers, early stopping, or sweeps.
- Gap: No search/ablation infra; no variance reporting across seeds.
- Action: Add early stopping, schedulers, sweeps, and ablation scripts to match tables.

### 7) Evaluation & Metrics
- Paper: Confirm reported metrics (e.g., MSE/MAE/DA, IC/ICIR, rank‑IC, hit@k), statistical tests, and split design.
- Repo: Walk‑forward with MSE/MAE/directional acc; strategy metrics (Sharpe, MDD, cumret). Cross‑section tool provides rank‑IC and long‑short Sharpe with holding period and costs.
  - Code: `finvision/backtest.py`, `finvision/cross_section.py`, `finvision/metrics.py`, `finvision/portfolio.py`.
- Gap: Any additional paper metrics/tests; exact split sizes and alignment; table‑ready reporting.
- Action: Implement exact metrics + significance testing; mirror paper’s split windows.

### 8) Strategy / Portfolio Construction
- Paper: Confirm portfolio rules (e.g., sector‑neutrality, risk targets, leverage, turnover caps, execution frictions).
- Repo: Threshold sign trading for single series; cross‑section long‑short with holding/caps/costs.
  - Code: `finvision/portfolio.py`, `finvision/cross_section.py`.
- Gap: No sector/industry neutrality, risk budgets, leverage or more realistic execution simulation.
- Action: Add portfolio constraints and execution logic per paper.

### 9) RL Components
- Paper: Confirm whether RL is a core component (state, action, reward, constraints) and how it integrates with agents.
- Repo: Minimal env with actions {-1,0,1} and costs; REINFORCE and PPO baselines trained on final or rolled splits; evaluation greedy policy.
  - Code: `finvision/rl/{env.py, agent_pg.py, agent_ppo.py, pipeline.py}`.
- Gap: If paper uses richer state/action/constraints or evaluation protocol, it’s not reproduced.
- Action: Align RL to paper definitions and evaluation.

### 10) News / LLM Signals
- Paper: Confirm source(s), entity linking, signal aggregation, LLM model or finetune, and leakage handling.
- Repo: Yahoo/yfinance news; GPT‑4o JSON scoring per article; daily confidence‑weighted mean; session bucketing to avoid look‑ahead.
  - Code: `finvision/news/{fetch_yahoo.py, signal.py}`, `finvision/llm/openai_scorer.py`.
- Gap: No curated event dataset, no entity‑level aggregation beyond simple mean, no paper‑specified LLM or finetune.
- Action: Recreate the paper’s event pipeline and scoring.

### 11) Reproducibility Artifacts
- Paper: Confirm tables/figures and experiment grids; environment pinning.
- Repo: CLI scripts and `run.py` save metrics and CSVs; no figure/table regeneration, no Docker/conda lockfiles.
- Gap: No one‑shot scripts to rebuild every table/figure; no environment pinning.
- Action: Add scripts to regenerate tables/plots and environment lockfiles.

## Current Implementation Map (for reference)
- Agents: `finvision/agents/*`
- Ensemble: `finvision/ensemble/aggregators.py`
- Backtests: `finvision/backtest.py`, `finvision/portfolio.py`, `finvision/cross_section.py`
- RL: `finvision/rl/*`
- News/LLM: `finvision/news/*`, `finvision/llm/*`
- CLI/Runner: `finvision/cli.py`, `run.py`
- Config: `finvision/config.py`
- Data: `finvision/data/*`

## Verification Checklist (Update as you confirm paper details)
- [ ] Dataset universe, vendors, frequency, and dates match paper
- [ ] Feature/indicator sets match paper
- [ ] Targets and horizons match paper (single/multi‑horizon)
- [ ] Agent architectures and losses match paper
- [ ] Multi‑agent protocol matches paper
- [ ] Ensemble method, constraints, and fitting match paper
- [ ] Training schedules, early stopping, and hyperparameter search match paper
- [ ] Evaluation metrics and statistical tests match paper
- [ ] Portfolio rules, constraints, and execution frictions match paper
- [ ] RL state/action/reward/constraints match paper (if applicable)
- [ ] News/LLM data source, scoring, and aggregation match paper
- [ ] All tables/figures regenerated from scripts and match paper values (within tolerance)

## How to Use This File
- As you verify a paper component, replace “Confirm from paper” notes with exact citations (section/figure) and mark the corresponding checklist items.
- Create issues or TODOs for each gap and link them to commits that close the gap.

---
Maintainers: update this document alongside code changes to keep the reproduction status current.
