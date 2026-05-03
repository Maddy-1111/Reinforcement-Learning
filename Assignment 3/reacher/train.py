"""SAC training on dm_control reacher-easy under one of R_a / R_b / R_c.

Every 10K env steps we evaluate the same (deterministic) policy under all three
reward formulations, satisfying the assignment's cross-reward logging requirement.

Usage:
    python train.py --reward a --seed 1 --num-train-steps 1000000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow `python reacher/train.py` from the Assignment 3 folder.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sac_core import SACAgent, ReplayBuffer, TrainConfig, train, utils  # noqa: E402

from reacher_env import make_reacher  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--reward', choices=['a', 'b', 'c'], required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out-dir', default='runs')
    p.add_argument('--backend', default=None)
    p.add_argument('--device', default='cuda')

    p.add_argument('--num-train-steps', type=int, default=500_000)
    p.add_argument('--num-seed-steps', type=int, default=10_000)
    p.add_argument('--replay-buffer-capacity', type=int, default=1_000_000)
    p.add_argument('--max-episode-steps', type=int, default=1000)
    p.add_argument('--eval-frequency', type=int, default=10_000)
    p.add_argument('--num-eval-episodes', type=int, default=20)
    p.add_argument('--checkpoint-frequency', type=int, default=0)
    # Per-instructor middleground: cap each eval episode at 1000 steps. For Rc
    # this means a single timeout window; the env applies its -20 reset penalty
    # on the 1000th step, so eval returns are -k (goal in k<1000) or -1020
    # (timeout). Training-time Rc semantics (multi-window with penalties) are
    # unchanged.
    p.add_argument('--max-eval-episode-steps', type=int, default=1000)

    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--hidden-depth', type=int, default=2)
    p.add_argument('--discount', type=float, default=0.99)
    p.add_argument('--init-temperature', type=float, default=0.1)
    p.add_argument('--actor-lr', type=float, default=1e-4)
    p.add_argument('--critic-lr', type=float, default=1e-4)
    p.add_argument('--alpha-lr', type=float, default=1e-4)
    p.add_argument('--critic-tau', type=float, default=0.005)
    return p


def main():
    args = build_argparser().parse_args()

    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available())
        else 'cpu')

    utils.set_seed_everywhere(args.seed)

    env = make_reacher(reward_variant=args.reward, seed=args.seed,
                       max_episode_steps=args.max_episode_steps,
                       backend=args.backend)
    # cross-reward eval envs (fresh instances, different seed stream)
    eval_envs = {
        f'R_{v}': make_reacher(reward_variant=v, seed=args.seed + 10_000,
                                max_episode_steps=args.max_episode_steps,
                                backend=args.backend)
        for v in ('a', 'b', 'c')
    }

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [float(env.action_space.low.min()),
                    float(env.action_space.high.max())]

    agent = SACAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_range=action_range,
        device=device,
        discount=args.discount, init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        critic_tau=args.critic_tau, batch_size=args.batch_size,
        learnable_temperature=True,
        hidden_dim=args.hidden_dim, hidden_depth=args.hidden_depth,
        target_entropy=-action_dim,
    )

    rb = ReplayBuffer(obs_shape=env.observation_space.shape,
                      action_shape=env.action_space.shape,
                      capacity=args.replay_buffer_capacity, device=device)

    cfg = TrainConfig(
        num_train_steps=args.num_train_steps,
        num_seed_steps=args.num_seed_steps,
        eval_frequency=args.eval_frequency,
        num_eval_episodes=args.num_eval_episodes,
        checkpoint_frequency=args.checkpoint_frequency,
        log_dir=Path(args.out_dir) / f"sac_R{args.reward}_seed{args.seed}",
        extra_config=vars(args),
        max_eval_episode_steps=args.max_eval_episode_steps,
    )

    train(agent, env, rb, cfg, eval_envs=eval_envs)


if __name__ == '__main__':
    main()
