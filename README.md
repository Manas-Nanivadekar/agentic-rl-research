# agentic-rl-research

This project provides a clean-room, from-scratch implementation of a modular multi‑agent framework for stock market prediction and trading, inspired by:
- FinVision: A Multi-Agent Framework for Stock Market Prediction (DOI: 10.1145/3677052.3698688)

It includes supervised agents (LSTM/MLP), a learnable ensemble aggregator, walk‑forward evaluation, a transaction‑cost‑aware trading backtest, and a lightweight RL policy (REINFORCE) trained in a simple trading environment.

Contents
- finvision: Python package with data, models, ensemble, backtests, RL
- configs: Ready-to-run example config
- data: Place your CSVs here (or generate synthetic data)
- requirements.txt: Python dependencies

Install
- Create a virtual environment with Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

Getting Data
- Option 1 – Synthetic data (recommended to start):
  - `python -m finvision.cli make-synth --ticker SYNTH --days 2000 --outdir data`
  - This writes `data/SYNTH.csv` (daily OHLCV) usable out of the box.
- Option 2 – Download from Stooq (daily OHLCV):
  - Edit `configs/example.yaml` with `tickers` you want.
  - `python -m finvision.cli download --config configs/example.yaml` (uses Stooq; may require VPN or retries if blocked by Cloudflare.)
- Option 3 – Use your own CSVs:
  - Put per‑ticker CSVs in `data/` with columns: `timestamp,open,high,low,close,volume` (header case‑insensitive), sorted by timestamp.

Quickstart
- Backtest prediction quality (walk‑forward):
  - `python -m finvision.cli backtest --config configs/example.yaml`
- Strategy backtest (with transaction costs):
  - `python -m finvision.cli trade --config configs/example.yaml`
- RL training and evaluation (REINFORCE, single split):
  - `python -m finvision.cli rl --config configs/example.yaml`

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
- finvision/backtest.py: Walk‑forward pipeline and prediction metrics
- finvision/portfolio.py: Positions from predictions and transaction costs
- finvision/metrics.py: MSE, MAE, directional accuracy, Sharpe, max drawdown, cumulative return
- finvision/rl
  - env.py: Minimal trading environment over windowed features and returns
  - agent_pg.py: REINFORCE policy gradient agent
  - pipeline.py: RL training on train+val and evaluation on test split
- finvision/cli.py: CLI commands (backtest, trade, download, rl, make‑synth)

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

Extending
- Add agent types: implement a new class with `.model()` returning a `torch.nn.Module` and add a builder in `backtest._build_agent`.
- Add features: extend `make_panel` to compute returns/indicators; normalize features.
- Richer RL: add baselines (A2C/DQN), transaction constraints, or multi‑asset portfolios.

Troubleshooting
- Empty downloads from Stooq: retry with VPN; Stooq can return `No data` behind certain networks. Use `make-synth` to validate the pipeline.
- Not enough data: reduce `lookback` or `train/val/test` spans.
- NaNs during training: ensure CSVs are clean and aligned; `make_panel` drops rows with missing values after alignment.

Note: This is a clean-room implementation aligned with the paper’s theme (multi-agent stock prediction). It is designed to be extendable to the paper’s specific architectures and protocols once details are provided.
