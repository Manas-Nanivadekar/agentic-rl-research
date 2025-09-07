from __future__ import annotations

import numpy as np


class TradingEnv:
    """Minimal trading environment over a fixed feature/return series.

    - Observations: flattened window features (vector of shape [T*F])
    - Actions: {-1: short, 0: flat, +1: long}
    - Reward: position * return_t - cost_per_unit * turnover
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, cost_bps: float = 1.0):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.cost = cost_bps / 1e4
        self.t = 0
        self.pos = 0.0

    def reset(self):
        self.t = 0
        self.pos = 0.0
        return self._obs()

    def _obs(self):
        x = self.X[self.t]
        return x.reshape(-1)

    def step(self, action: int):
        # map action {0,1,2} to {-1,0,1} if needed
        if action in (0, 1, 2):
            act = action - 1
        else:
            act = int(np.clip(action, -1, 1))

        prev_pos = self.pos
        self.pos = float(act)
        ret = float(self.pos * self.y[self.t])
        tc = self.cost * abs(self.pos - prev_pos)
        reward = ret - tc

        self.t += 1
        done = self.t >= len(self.X)
        obs = None if done else self._obs()
        return obs, reward, done, {"pos": self.pos, "ret": ret, "tc": tc}

