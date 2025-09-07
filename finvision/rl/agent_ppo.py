from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


class _Policy(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, act_dim: int = 3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Value(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PPOAgent:
    obs_dim: int
    hidden: int = 128
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_iters: int = 40
    target_kl: float = 0.02
    device: str = "cpu"

    _pi: _Policy | None = None
    _vf: _Value | None = None

    def _ensure(self):
        if self._pi is None:
            self._pi = _Policy(self.obs_dim).to(self.device)
        if self._vf is None:
            self._vf = _Value(self.obs_dim).to(self.device)

    def policy(self):
        self._ensure()
        return self._pi

    def value_fn(self):
        self._ensure()
        return self._vf

    def _dist(self, logits):
        return torch.distributions.Categorical(logits=logits)

    def gather_trajectory(self, env):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
        obs = env.reset()
        done = False
        with torch.no_grad():
            while not done:
                x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                logits = self.policy()(x)
                dist = self._dist(logits.squeeze(0))
                a = dist.sample()
                logp = dist.log_prob(a)
                v = self.value_fn()(x).squeeze(0)
                next_obs, r, done, _ = env.step(int(a.item()))
                obs_buf.append(obs)
                act_buf.append(int(a.item()))
                logp_buf.append(logp.cpu().numpy())
                rew_buf.append(r)
                val_buf.append(v.cpu().numpy())
                obs = next_obs
        return (
            np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.int64),
            np.array(logp_buf, dtype=np.float32),
            np.array(rew_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
        )

    def fit(self, env, epochs: int = 10, gamma: float = 0.99, lam: float = 0.95):
        self._ensure()
        pi_opt = torch.optim.Adam(self.policy().parameters(), lr=self.pi_lr)
        vf_opt = torch.optim.Adam(self.value_fn().parameters(), lr=self.vf_lr)

        for _ in range(epochs):
            obs, act, logp_old, rew, val = self.gather_trajectory(env)
            # GAE-Lambda advantages
            adv = np.zeros_like(rew)
            lastgaelam = 0
            for t in reversed(range(len(rew))):
                nextv = val[t + 1] if t + 1 < len(val) else 0.0
                delta = rew[t] + gamma * nextv - val[t]
                adv[t] = lastgaelam = delta + gamma * lam * lastgaelam
            ret = adv + val
            # normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obs_t = torch.from_numpy(obs).float().to(self.device)
            act_t = torch.from_numpy(act).long().to(self.device)
            logp_old_t = torch.from_numpy(logp_old).float().to(self.device)
            adv_t = torch.from_numpy(adv).float().to(self.device)
            ret_t = torch.from_numpy(ret).float().to(self.device)

            for _ in range(self.train_iters):
                logits = self.policy()(obs_t)
                dist = self._dist(logits)
                logp = dist.log_prob(act_t)
                ratio = torch.exp(logp - logp_old_t)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t
                loss_pi = -(torch.min(ratio * adv_t, clip_adv)).mean()

                v = self.value_fn()(obs_t).squeeze(-1)
                loss_v = ((v - ret_t) ** 2).mean()

                kl = (logp_old_t - logp).mean().item()
                pi_opt.zero_grad()
                loss_pi.backward()
                pi_opt.step()
                vf_opt.zero_grad()
                loss_v.backward()
                vf_opt.step()
                if kl > 1.5 * self.target_kl:
                    break

