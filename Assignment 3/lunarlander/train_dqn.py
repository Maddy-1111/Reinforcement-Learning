"""DQN on LunarLander-v3 (discrete).  Q2.2.4(c).

Standard DQN with Double-Q trick.  Writes progress.csv every 10K env steps in
the same schema as the SAC runs so plot.py can compare curves apples-to-apples.

Usage (single seed):
    python lunarlander/train_dqn.py --seed 1 --num-train-steps 500000

15-seed sweep:
    for seed in $(seq 1 15); do
        python lunarlander/train_dqn.py --seed $seed --num-train-steps 500000
    done
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sac_core import utils          # set_seed_everywhere
from lander_env import make_discrete


# ── Q-network ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int,
                 hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_actions))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Flat numpy-backed circular buffer for discrete actions."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self._cap = capacity
        self._device = device
        self._obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._not_done = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, terminated: bool) -> None:
        i = self._idx
        self._obs[i]      = obs
        self._next_obs[i] = next_obs
        self._actions[i]  = int(action)
        self._rewards[i]  = float(reward)
        self._not_done[i] = 1.0 - float(terminated)
        self._idx  = (i + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self._obs[idx],      device=self._device),
            torch.as_tensor(self._actions[idx],  device=self._device),
            torch.as_tensor(self._rewards[idx],  device=self._device),
            torch.as_tensor(self._next_obs[idx], device=self._device),
            torch.as_tensor(self._not_done[idx], device=self._device),
        )

    def __len__(self) -> int:
        return self._size


# ── Deterministic evaluation ──────────────────────────────────────────────────

def evaluate(q_net: QNetwork, eval_env, num_episodes: int,
             device: torch.device) -> tuple[float, float]:
    returns = []
    for _ in range(num_episodes):
        obs  = eval_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t  = torch.as_tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(q_net(obs_t).argmax(dim=1).item())
            obs, reward, done, _ = eval_env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


# ── Argument parser ───────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DQN on LunarLander-v3 discrete (Q2.2.4c)")
    p.add_argument('--seed',   type=int,   default=1)
    p.add_argument('--out-dir',            default='runs')
    p.add_argument('--device',             default='cuda')

    p.add_argument('--num-train-steps',         type=int,   default=500_000)
    p.add_argument('--num-seed-steps',          type=int,   default=10_000,
                   help='Pure random-action warm-up steps before learning.')
    p.add_argument('--replay-buffer-capacity',  type=int,   default=1_000_000)
    p.add_argument('--max-episode-steps',       type=int,   default=1000)
    p.add_argument('--eval-frequency',          type=int,   default=10_000)
    p.add_argument('--num-eval-episodes',       type=int,   default=20)

    p.add_argument('--batch-size',       type=int,   default=256)
    p.add_argument('--hidden-dim',       type=int,   default=256)
    p.add_argument('--hidden-depth',     type=int,   default=2)
    p.add_argument('--discount',         type=float, default=0.99)
    p.add_argument('--lr',               type=float, default=3e-4)
    p.add_argument('--target-update-freq', type=int, default=2_000,
                   help='Hard-copy online → target net every N env steps.')

    p.add_argument('--eps-start',        type=float, default=1.0)
    p.add_argument('--eps-end',          type=float, default=0.05)
    p.add_argument('--eps-decay-steps',  type=int,   default=200_000,
                   help='Linearly anneal epsilon over this many steps '
                        '(counting from end of seed phase).')
    return p


# ── Training ──────────────────────────────────────────────────────────────────

def main() -> None:
    args   = build_argparser().parse_args()
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available())
        else 'cpu')
    utils.set_seed_everywhere(args.seed)

    env      = make_discrete(max_episode_steps=args.max_episode_steps,
                             seed=args.seed)
    eval_env = make_discrete(max_episode_steps=args.max_episode_steps,
                             seed=args.seed + 10_000)

    obs_dim     = env.observation_space.shape[0]
    num_actions = env.num_actions

    q_net      = QNetwork(obs_dim, num_actions,
                          args.hidden_dim, args.hidden_depth).to(device)
    target_net = QNetwork(obs_dim, num_actions,
                          args.hidden_dim, args.hidden_depth).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    rb        = ReplayBuffer(args.replay_buffer_capacity, obs_dim, device)

    log_dir  = Path(args.out_dir) / f"dqn_seed{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / 'progress.csv'

    eps_slope = (args.eps_start - args.eps_end) / max(1, args.eps_decay_steps)

    obs        = env.reset()
    start_time = time.time()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'return_mean', 'return_std', 'alpha', 'wall_time'])

        # Step-0 eval: policy is randomly initialised, no updates yet.
        ret_mean0, ret_std0 = evaluate(q_net, eval_env, args.num_eval_episodes, device)
        writer.writerow([0, f'{ret_mean0:.4f}', f'{ret_std0:.4f}', '', '0.00'])
        f.flush()
        print(f'[DQN seed={args.seed}] step=      0  '
              f'return={ret_mean0:>8.2f} ± {ret_std0:.2f}  t=0s')

        for step in range(1, args.num_train_steps + 1):

            # ── collect ──────────────────────────────────────────────────────
            if step <= args.num_seed_steps:
                action = env.action_space.sample()
            else:
                eps = max(args.eps_end,
                          args.eps_start
                          - eps_slope * (step - args.num_seed_steps))
                if np.random.random() < eps:
                    action = env.action_space.sample()
                else:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                            device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = int(q_net(obs_t).argmax(dim=1).item())

            next_obs, reward, done, info = env.step(action)

            # Only mask bootstrapping on true termination, not timeouts.
            terminated = info.get('terminated', done)
            rb.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if done:
                obs = env.reset()

            # ── learn ─────────────────────────────────────────────────────────
            if step > args.num_seed_steps and len(rb) >= args.batch_size:
                obs_b, act_b, rew_b, next_b, not_done_b = rb.sample(args.batch_size)

                with torch.no_grad():
                    # Double DQN: online net picks action, target net evaluates it
                    next_acts = q_net(next_b).argmax(dim=1, keepdim=True)
                    next_q    = target_net(next_b).gather(1, next_acts).squeeze(1)
                    target_q  = rew_b + args.discount * not_done_b * next_q

                current_q = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

            # ── hard target update ────────────────────────────────────────────
            if step % args.target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

            # ── periodic deterministic eval ───────────────────────────────────
            if step % args.eval_frequency == 0:
                ret_mean, ret_std = evaluate(q_net, eval_env,
                                             args.num_eval_episodes, device)
                wall_time = time.time() - start_time
                writer.writerow([step, f'{ret_mean:.4f}', f'{ret_std:.4f}',
                                  '', f'{wall_time:.2f}'])
                f.flush()
                print(f'[DQN seed={args.seed}] step={step:>7d}  '
                      f'return={ret_mean:>8.2f} ± {ret_std:.2f}  '
                      f't={wall_time:.0f}s')

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
