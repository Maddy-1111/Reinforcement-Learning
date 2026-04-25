"""SAC on modified Pendulum-v1 (align to theta_target).

Handles:
    - Q2.1.2  automated alpha, sweep theta_target in {0,-10,30,-60,90,-90,120,-150}
    - Q2.1.5a manual alpha_mnl for {-60, 90, 120, -150}
    - Q2.1.5b manual alpha_mnl vs auto under reward-scale {10x, 0.1x} for theta=90

Usage:
    # automated alpha (default)
    python train.py --theta 0 --seed 1
    python train.py --theta -150 --seed 1

    # manual alpha (learnable_temperature off, init_temperature = alpha_mnl)
    python train.py --theta 90 --alpha-mode manual --alpha 0.2 --seed 1

    # reward-scale for Q2.1.5b
    python train.py --theta 90 --alpha-mode manual --alpha 0.2 --reward-scale 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sac_core import SACAgent, ReplayBuffer, TrainConfig, train, utils  # noqa: E402

from pendulum_env import make_pendulum  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--theta', type=float, required=True,
                   help='Target angle theta_target in degrees.')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out-dir', default='runs')
    p.add_argument('--device', default='cuda')

    p.add_argument('--alpha-mode', choices=['auto', 'manual'], default='auto')
    p.add_argument('--alpha', type=float, default=0.2,
                   help='Fixed alpha when --alpha-mode manual.')
    p.add_argument('--reward-scale', type=float, default=1.0)

    p.add_argument('--num-train-steps', type=int, default=50_000)
    p.add_argument('--num-seed-steps', type=int, default=500)
    p.add_argument('--replay-buffer-capacity', type=int, default=50_000)
    p.add_argument('--max-episode-steps', type=int, default=1000)
    p.add_argument('--eval-frequency', type=int, default=10_000)
    p.add_argument('--num-eval-episodes', type=int, default=20)

    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--hidden-depth', type=int, default=2)
    p.add_argument('--discount', type=float, default=0.99)
    p.add_argument('--actor-lr', type=float, default=3e-4)
    p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--alpha-lr', type=float, default=3e-4)
    p.add_argument('--critic-tau', type=float, default=0.005)
    p.add_argument('--init-temperature', type=float, default=0.2,
                   help='(auto mode only) initial alpha before learning.')
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available())
        else 'cpu')
    utils.set_seed_everywhere(args.seed)

    env = make_pendulum(theta_target_deg=args.theta,
                        reward_scale=args.reward_scale,
                        max_episode_steps=args.max_episode_steps,
                        seed=args.seed)
    eval_env = make_pendulum(theta_target_deg=args.theta,
                             reward_scale=args.reward_scale,
                             max_episode_steps=args.max_episode_steps,
                             seed=args.seed + 10_000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [-1.0, 1.0]  # agent-space (actor outputs tanh in [-1,1])

    learnable = (args.alpha_mode == 'auto')
    init_temp = args.alpha if not learnable else args.init_temperature

    agent = SACAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_range=action_range,
        device=device,
        discount=args.discount, init_temperature=init_temp,
        alpha_lr=args.alpha_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        critic_tau=args.critic_tau, batch_size=args.batch_size,
        learnable_temperature=learnable,
        hidden_dim=args.hidden_dim, hidden_depth=args.hidden_depth,
        target_entropy=-action_dim,
    )

    rb = ReplayBuffer(obs_shape=env.observation_space.shape,
                      action_shape=env.action_space.shape,
                      capacity=args.replay_buffer_capacity, device=device)

    # Tag the run dir so that sweeps don't collide.
    tag = f"theta{int(args.theta)}_{args.alpha_mode}"
    if args.alpha_mode == 'manual':
        tag += f"_a{args.alpha:g}"
    if args.reward_scale != 1.0:
        tag += f"_rs{args.reward_scale:g}"
    log_dir = Path(args.out_dir) / f"sac_{tag}_seed{args.seed}"

    cfg = TrainConfig(
        num_train_steps=args.num_train_steps,
        num_seed_steps=args.num_seed_steps,
        eval_frequency=args.eval_frequency,
        num_eval_episodes=args.num_eval_episodes,
        log_dir=log_dir,
        extra_config=vars(args),
    )

    train(agent, env, rb, cfg, eval_envs={'return': eval_env})


if __name__ == '__main__':
    main()
