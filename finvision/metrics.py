import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.sign(y_true)
    b = np.sign(y_pred)
    return float(np.mean((a == b).astype(np.float32)))


def cumulative_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def sharpe_ratio(
    returns: np.ndarray, periods_per_year: int = 252, risk_free: float = 0.0
) -> float:
    r = returns - risk_free / periods_per_year
    mu = np.mean(r)
    sd = np.std(r, ddof=1) if len(r) > 1 else 0.0
    if sd == 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    # returns are simple returns per period
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity / peaks) - 1.0
    return float(drawdowns.min()) if len(drawdowns) else float("nan")
