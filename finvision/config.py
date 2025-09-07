from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    data_dir: str = "data"
    tickers: List[str] = field(default_factory=lambda: ["AAPL"])
    target: str = "close"
    horizon: int = 1  # predict t+h horizon return or price
    use_returns: bool = True  # if True, target is log return
    lookback: int = 60  # sliding window length
    features: Optional[List[str]] = None  # if None, uses [open, high, low, close, volume]


@dataclass
class AgentConfig:
    name: str
    type: str  # "lstm" | "mlp"
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 128
    device: str = "cpu"


@dataclass
class AggregatorConfig:
    type: str = "ridge"  # ridge | mlp
    alpha: float = 1.0
    hidden_size: int = 32
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 256


@dataclass
class BacktestConfig:
    start: Optional[str] = None  # ISO date to start
    end: Optional[str] = None  # ISO date to end
    train_span: int = 252 * 2  # trading days in training window
    val_span: int = 252 // 4
    test_span: int = 252 // 4


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    agents: List[AgentConfig] = field(default_factory=list)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    # trading settings
    threshold: float = 0.0  # prediction threshold for entering positions
    cost_bps: float = 1.0   # round-trip cost in basis points applied on turnover
