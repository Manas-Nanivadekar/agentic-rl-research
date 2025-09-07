from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


class _PolicyNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 3),  # logits for {-1,0,1}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PolicyGradientAgent:
    obs_dim: int
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    device: str = "cpu"

    _net: _PolicyNet | None = None

    def net(self) -> _PolicyNet:
        if self._net is None:
            self._net = _PolicyNet(self.obs_dim, self.hidden)
        return self._net

    def select_action(self, obs: np.ndarray, require_grad: bool = False):
        net = self.net().to(self.device)
        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        if require_grad:
            logits = net(x)
            probs = torch.softmax(logits, dim=-1)
        else:
            with torch.no_grad():
                logits = net(x)
                probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs.squeeze(0))
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), logp

    def fit(self, env, epochs: int = 5):
        net = self.net().to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        for _ in range(epochs):
            obs = env.reset()
            logps = []
            rewards = []

            done = False
            while not done:
                a, logp = self.select_action(obs, require_grad=True)
                obs, r, done, _ = env.step(a)
                logps.append(logp)
                rewards.append(r)

            # returns with discount
            R = 0.0
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.append(R)
            returns.reverse()
            G = torch.tensor(returns, dtype=torch.float32, device=self.device)
            # normalize advantages
            if len(G) > 1:
                G = (G - G.mean()) / (G.std() + 1e-8)

            loss = 0.0
            for logp, g in zip(logps, G):
                loss = loss - logp.to(self.device) * g

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
