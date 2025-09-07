# agentic-rl-research

This project provides a clean-room, from-scratch implementation of a modular multi‑agent framework for stock market prediction and trading, inspired by:
- FinVision: A Multi-Agent Framework for Stock Market Prediction (DOI: 10.1145/3677052.3698688)

It includes supervised agents (LSTM/MLP), a learnable ensemble aggregator, walk‑forward evaluation, a transaction‑cost‑aware trading backtest, and a lightweight RL policy (REINFORCE) trained in a simple trading environment.

Contents
- finvision: Python package with data, models, ensemble, backtests, RL
- configs: Ready-to-run example config
- data: Place your CSVs here (or generate synthetic data)
- requirements.txt: Python dependencies
 - Optional: Precompute LLM news signal (GPT-4o) via CLI

Install
- Use Python 3.10+.
- Create and activate a virtualenv.
  - macOS/Linux: `python3 -m venv venv && source venv/bin/activate`
  - Windows (PowerShell): `py -m venv venv; venv\\Scripts\\Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`
- Optional (LLM): set your OpenAI API key: `export OPENAI_API_KEY=...`

Getting Data
- Option A – Synthetic data (recommended to start):
  - `python -m finvision.cli make-synth --ticker SYNTH --days 2000 --outdir data`
  - Produces `data/SYNTH.csv` (daily OHLCV) ready for backtests.
- Option B – Download from Stooq (daily OHLCV):
  - Edit `configs/example.yaml` and set `data.tickers`.
  - `python -m finvision.cli download --config configs/example.yaml`
  - Note: Stooq access can be flaky behind some networks; retry or use VPN.
- Option C – Bring your own CSVs:
  - Place per‑ticker CSVs in `data/` with columns: `timestamp,open,high,low,close,volume` (case‑insensitive), sorted by timestamp.

Step‑by‑Step Usage
1) Configure experiment
- Open `configs/example.yaml` and set:
  - `data:` (tickers, lookback, target/horizon).
  - `agents:` include `lstm`, `mlp`, and optionally `llm_news`.
  - `aggregator:` choose `ridge` or `mlp`.
  - `backtest:` training/validation/test spans and optional `start`/`end`.
  - `trading:` threshold and `cost_bps`.
  - `news:` cache dir and optional date bounds.
  - `llm:` model `gpt-4o`, cache dir, retry policy.

2) Prepare price data
- Either generate synthetic or download Stooq data (see above).

3) Build LLM news signal (only if using `llm_news` agent)
- Ensure `OPENAI_API_KEY` is exported.
- Run: `python -m finvision.cli news --config configs/example.yaml --out data/features/llm_signal.csv`
- Outputs:
  - Long form: `data/features/llm_signal.csv` (timestamp, ticker, signal)
  - Pivot with mean: `data/features/llm_signal_pivot.csv` (columns per ticker + `LLM_MEAN`)

4) Run walk‑forward backtest (prediction metrics)
- `python -m finvision.cli backtest --config configs/example.yaml`
- Trains each agent per split, fits aggregator on validation, evaluates on test.

5) Run strategy backtest (PnL with costs)
- `python -m finvision.cli trade --config configs/example.yaml`
- Uses ensemble predictions to form positions with `threshold`, computes net returns with transaction costs (`cost_bps`).

6) Train and evaluate RL policy (optional)
- `python -m finvision.cli rl --config configs/example.yaml`
- Trains a REINFORCE agent on the final split (train+val), evaluates on test.

7) Cross-sectional evaluation (optional)
- `python -m finvision.cli xsect --config configs/example.yaml`
- Reports cross-sectional metrics including rank-IC and long-short Sharpe across tickers.

8) PPO RL across multiple splits (optional)
- `python -m finvision.cli rl-ppo --config configs/example.yaml`
- Trains PPO on each rolling split (train+val), evaluates on test, and averages metrics.

One‑shot end‑to‑end run
- Ensure `OPENAI_API_KEY` is exported.
- `python run.py --config configs/example.yaml`
  - If price CSVs for configured tickers are missing, it synthesizes data (2000 days).
  - Builds LLM news signal (skip with `--skip-llm`).
  - Runs backtest (prediction), strategy backtest (PnL), and RL evaluation.

Configuration
- File: `configs/example.yaml` controls data, agents, aggregator, backtest windows, and trading params.
- Key fields:
  - data: `data_dir`, `tickers`, `target`, `horizon`, `use_returns`, `lookback`
  - agents: list of models with `type` (`lstm` or `mlp`), capacity, and training hyperparameters
  - aggregator: `ridge` (alpha) or `mlp` (hidden_size, lr, epochs)
  - backtest: `train_span`, `val_span`, `test_span`, optional `start`/`end`
  - trading: `threshold` (entry), `cost_bps` (basis points per turnover)

Code Structure
- finvision/config.py: Dataclasses for experiment config
- finvision/data
  - dataset.py: CSV loader, feature panel builder, sliding window creation
  - utils.py: Utility split helpers
  - download.py: Simple Stooq daily downloader
  - synthetic.py: Deterministic synthetic OHLCV generator
- finvision/agents
  - base.py: Shared training loop (MSE) and predict
  - lstm.py: `LSTMRegressor` for sequence modeling over windows
  - mlp.py: `MLPRegressor` with flattened window features
- finvision/ensemble
  - aggregators.py: `RidgeAggregator` and small `MLPAggregator`
- finvision/news
  - fetch_yahoo.py: Fetch Yahoo Finance news (via yfinance) with caching
  - signal.py: Score with GPT‑4o, aggregate daily confidence‑weighted signals, save CSVs
- finvision/llm
  - openai_scorer.py: GPT‑4o scorer with JSON output and on‑disk cache
- finvision/backtest.py: Walk‑forward pipeline and prediction metrics
- finvision/portfolio.py: Positions from predictions and transaction costs
- finvision/metrics.py: MSE, MAE, directional accuracy, Sharpe, max drawdown, cumulative return
- finvision/rl
  - env.py: Minimal trading environment over windowed features and returns
  - agent_pg.py: REINFORCE policy gradient agent
  - pipeline.py: RL training on train+val and evaluation on test split
- finvision/cli.py: CLI commands (backtest, trade, download, rl, make‑synth)
  - Also `news` to build LLM signals

Data Flow
- Load CSVs per ticker -> align by timestamp -> build feature panel per ticker with standardized columns -> concatenate across tickers -> compute targets (1‑step log returns by default) -> create sliding windows of length `lookback` -> feed into agents.

Supervised Multi‑Agent Predictors
- LSTMRegressor: sequence model over windowed features
  - Input: `x ∈ R^{B×T×F}`
  - Architecture: LSTM(`hidden_size`, `num_layers`) -> Linear(1)
  - Training: AdamW, MSE on 1‑step target; best‑val snapshot
- MLPRegressor: feed‑forward on flattened windows
  - Input: flattened to `R^{B×(T·F)}`
  - Architecture: 2 hidden layers with ReLU+Dropout -> Linear(1)
- LLMNewsRegressor: sentiment‑driven agent from daily LLM signal windows
  - Input: `R^{B×T×1}` built from daily confidence‑weighted scores
  - Architecture: small MLP over lookback window -> Linear(1)
- Agents are trained independently on the train window and validated on the val window.

Ensemble Aggregator
- Fit on validation predictions to learn combination weights; apply to test predictions.
- Options: `ridge` (closed‑form via scikit‑learn) or `mlp` (tiny Torch MLP).

Walk‑Forward Evaluation
- Windows: roll by `test_span` across the time series.
- Per split: train agents on train; fit aggregator on val; evaluate on test.
- Metrics: MSE, MAE, directional accuracy averaged across splits.

Strategy Backtest (Costs)
- Positions: sign of ensemble prediction with optional `threshold`.
- Returns: `position × simple_return − cost_bps/1e4 × turnover`.
- Metrics: Sharpe (annualized), max drawdown, cumulative return.
- If `use_returns: true`, prediction targets are log returns; strategy automatically converts to simple returns for PnL.

RL Component (REINFORCE)
- Environment: `TradingEnv` emits flattened window features; actions are {-1, 0, +1} (short/flat/long); reward is `position × simple_return − costs × turnover`.
- Agent: `PolicyGradientAgent` (MLP policy over logits for 3 actions) trained with REINFORCE and reward‑to‑go.
- Training/Eval: uses the last available split (train+val to fit the policy; evaluate on test) with metrics Sharpe, max drawdown, cumulative return.
- Run: `python -m finvision.cli rl --config configs/example.yaml`

Repro Tips
- Fix seeds via `seed` in config.
- Scale data: current defaults rely on raw OHLCV. You can extend feature engineering (e.g., returns, tech indicators) by passing explicit `features` or modifying `make_panel`.
- GPU: set `device: cuda` per agent if available.
 - LLM keys: export `OPENAI_API_KEY` before running the `news` command.
 - Caching: News fetched via `yfinance` is cached under `data/news_cache`; LLM responses under `data/llm_cache`.

Extending
- Add agent types: implement a new class with `.model()` returning a `torch.nn.Module` and add a builder in `backtest._build_agent`.
- Add features: extend `make_panel` to compute returns/indicators; normalize features.
- Richer RL: add baselines (A2C/DQN), transaction constraints, or multi‑asset portfolios.

Troubleshooting
- Empty downloads from Stooq: retry with VPN; Stooq can return `No data` behind certain networks. Use `make-synth` to validate the pipeline.
- Not enough data: reduce `lookback` or `train/val/test` spans.
- NaNs during training: ensure CSVs are clean and aligned; `make_panel` drops rows with missing values after alignment.
 - LLM/news not found: If using `llm_news`, run `python -m finvision.cli news ...` first so `data/features/llm_signal_pivot.csv` exists.
 - OpenAI errors: verify `OPENAI_API_KEY`, model access, and network. The scorer caches outputs; retries are automatic.

Note: This is a clean-room implementation aligned with the paper’s theme (multi-agent stock prediction). It is designed to be extendable to the paper’s specific architectures and protocols once details are provided.
LLM integration uses GPT‑4o for article‑level scoring and aggregates per‑day per‑ticker signals; ensure `OPENAI_API_KEY` is set. Historical news availability via Yahoo/yfinance can be limited; consider alternative feeds for fuller replication.

Leakage controls and horizons
- News session bucketing: Articles are assigned to the session date whose US/Eastern market close (16:00) follows their publish time, preventing look‑ahead.
- Prediction horizons: Targets default to t+1; adjust `data.horizon` for longer horizons.

Cross‑sectional portfolios
- Long‑short backtest supports equal‑weight top/bottom quantiles, holding period (default 5 days), weight caps, and turnover‑based costs (bps).

Ensemble allocation
- In addition to ridge/MLP, a constrained aggregator enforces non‑negative weights that sum to 1; set `aggregator: {type: constrained, nonneg: true, sum_to_one: true}`.
