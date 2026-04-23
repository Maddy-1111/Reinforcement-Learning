"""Discrete-SAC on LunarLander-v3 (discrete). Q2.2.4(b)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sac_core import DiscreteSACAgent, ReplayBuffer, TrainConfig, train, utils  # noqa: E402

from lander_env import make_discrete  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out-dir', default='runs')
    p.add_argument('--device', default='cuda')

    p.add_argument('--num-train-steps', type=int, default=500_000)
    p.add_argument('--num-seed-steps', type=int, default=10_000)
    p.add_argument('--replay-buffer-capacity', type=int, default=1_000_000)
    p.add_argument('--max-episode-steps', type=int, default=1000)
    p.add_argument('--eval-frequency', type=int, default=10_000)
    p.add_argument('--num-eval-episodes', type=int, default=20)

    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--hidden-depth', type=int, default=2)
    p.add_argument('--discount', type=float, default=0.99)
    p.add_argument('--actor-lr', type=float, default=3e-4)
    p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--alpha-lr', type=float, default=3e-4)
    p.add_argument('--critic-tau', type=float, default=0.005)
    p.add_argument('--init-temperature', type=float, default=0.2)
    p.add_argument('--target-entropy-ratio', type=float, default=0.98)
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available())
        else 'cpu')
    utils.set_seed_everywhere(args.seed)

    env = make_discrete(max_episode_steps=args.max_episode_steps, seed=args.seed)
    eval_env = make_discrete(max_episode_steps=args.max_episode_steps,
                             seed=args.seed + 10_000)

    obs_dim = env.observation_space.shape[0]
    num_actions = env.num_actions

    agent = DiscreteSACAgent(
        obs_dim=obs_dim, num_actions=num_actions, device=device,
        discount=args.discount, init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        critic_tau=args.critic_tau, batch_size=args.batch_size,
        learnable_temperature=True,
        hidden_dim=args.hidden_dim, hidden_depth=args.hidden_depth,
        target_entropy_ratio=args.target_entropy_ratio,
    )

    rb = ReplayBuffer(obs_shape=env.observation_space.shape,
                      action_shape=(),
                      capacity=args.replay_buffer_capacity, device=device,
                      action_dtype=np.int64)

    cfg = TrainConfig(
        num_train_steps=args.num_train_steps,
        num_seed_steps=args.num_seed_steps,
        eval_frequency=args.eval_frequency,
        num_eval_episodes=args.num_eval_episodes,
        log_dir=Path(args.out_dir) / f"sac_discrete_seed{args.seed}",
        extra_config=vars(args),
    )

    train(agent, env, rb, cfg, eval_envs={'return': eval_env})


if __name__ == '__main__':
    main()
