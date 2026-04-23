"""Continuous-SAC on LunarLander-v3 (continuous).

Covers:
    - Q2.2.1/2: vanilla auto-alpha training.
    - Q2.2.3:   hover-box variant. `--hover-bonus 200.0` plus
                `--swap-bonus-at-step N --swap-bonus-to -100.0` schedules the
                +200 -> -100 reward change mid-training. Two alpha modes:
                  (i) manual alpha=0.01     -> `--alpha-mode manual --alpha 0.01`
                  (ii) automated            -> `--alpha-mode auto`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sac_core import SACAgent, ReplayBuffer, TrainConfig, train, utils  # noqa: E402

from lander_env import make_continuous  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out-dir', default='runs')
    p.add_argument('--device', default='cuda')

    p.add_argument('--alpha-mode', choices=['auto', 'manual'], default='auto')
    p.add_argument('--alpha', type=float, default=0.01)

    p.add_argument('--hover-bonus', type=float, default=None,
                   help='If set, add this bonus once per episode when lander '
                        'enters the hover box (|x|<0.1, 0.4<|y|<0.6).')
    p.add_argument('--swap-bonus-at-step', type=int, default=None,
                   help='Step at which to change hover-bonus (Q2.2.3).')
    p.add_argument('--swap-bonus-to', type=float, default=-100.0)

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
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available())
        else 'cpu')
    utils.set_seed_everywhere(args.seed)

    env = make_continuous(hover_bonus=args.hover_bonus,
                          max_episode_steps=args.max_episode_steps,
                          seed=args.seed)
    eval_env = make_continuous(hover_bonus=args.hover_bonus,
                               max_episode_steps=args.max_episode_steps,
                               seed=args.seed + 10_000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [float(env.action_space.low.min()),
                    float(env.action_space.high.max())]

    learnable = (args.alpha_mode == 'auto')
    init_temp = args.alpha if not learnable else args.init_temperature

    agent = SACAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_range=action_range,
        device=device, discount=args.discount, init_temperature=init_temp,
        alpha_lr=args.alpha_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        critic_tau=args.critic_tau, batch_size=args.batch_size,
        learnable_temperature=learnable,
        hidden_dim=args.hidden_dim, hidden_depth=args.hidden_depth,
        target_entropy=-action_dim,
    )

    rb = ReplayBuffer(obs_shape=env.observation_space.shape,
                      action_shape=env.action_space.shape,
                      capacity=args.replay_buffer_capacity, device=device)

    # Reward schedule: if --swap-bonus-at-step, flip hover bonus mid-training.
    schedule = None
    if args.swap_bonus_at_step is not None:
        new_bonus = float(args.swap_bonus_to)
        schedule = [(int(args.swap_bonus_at_step),
                     lambda e, b=new_bonus: (e.set_hover_bonus(b),
                                             eval_env.set_hover_bonus(b))[0])]

    # Tag run directory
    tag_parts = [f"sac_{args.alpha_mode}"]
    if args.alpha_mode == 'manual':
        tag_parts.append(f"a{args.alpha:g}")
    if args.hover_bonus is not None:
        tag_parts.append(f"hov{args.hover_bonus:g}")
    if args.swap_bonus_at_step is not None:
        tag_parts.append(f"swap{args.swap_bonus_at_step}to{args.swap_bonus_to:g}")
    tag_parts.append(f"seed{args.seed}")
    log_dir = Path(args.out_dir) / '_'.join(tag_parts)

    cfg = TrainConfig(
        num_train_steps=args.num_train_steps,
        num_seed_steps=args.num_seed_steps,
        eval_frequency=args.eval_frequency,
        num_eval_episodes=args.num_eval_episodes,
        log_dir=log_dir, extra_config=vars(args),
    )

    train(agent, env, rb, cfg, eval_envs={'return': eval_env},
          reward_schedule=schedule)


if __name__ == '__main__':
    main()
