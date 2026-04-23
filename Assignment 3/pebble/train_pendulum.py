"""PEBBLE on the modified Pendulum-v1 (Bonus §3 Q1, Q2).

Q1:  theta_target in {0, -60, 90, 120, -150} with a simulated teacher.
     Compare PEBBLE vs SAC-with-ground-truth-reward.
Q2:  feedback-budget sweep.

Usage:
    python train_pendulum.py --theta 90 --feedback-budget 1000 --seed 1
    python train_pendulum.py --theta 90 --feedback-budget 200  --seed 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sac_core import SACAgent, ReplayBuffer, utils  # noqa: E402
from pendulum.pendulum_env import make_pendulum  # noqa: E402
from pebble import PebbleConfig, train_pebble, OracleTeacher  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--theta', type=float, required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out-dir', default='runs')
    p.add_argument('--device', default='cuda')

    # PEBBLE-specific
    p.add_argument('--feedback-budget', type=int, default=1000)
    p.add_argument('--feedback-frequency', type=int, default=5000)
    p.add_argument('--queries-per-session', type=int, default=20)
    p.add_argument('--candidate-pool-size', type=int, default=200)
    p.add_argument('--segment-length', type=int, default=50)
    p.add_argument('--selection', choices=['uniform', 'disagreement'],
                   default='disagreement')
    p.add_argument('--unsup-steps', type=int, default=9000)
    p.add_argument('--ensemble-size', type=int, default=3)
    p.add_argument('--reward-epochs', type=int, default=50)
    p.add_argument('--teacher-mistake-prob', type=float, default=0.0)

    # training common
    p.add_argument('--num-train-steps', type=int, default=200_000)
    p.add_argument('--num-seed-steps', type=int, default=1000)
    p.add_argument('--replay-buffer-capacity', type=int, default=1_000_000)
    p.add_argument('--max-episode-steps', type=int, default=1000)
    p.add_argument('--eval-frequency', type=int, default=10_000)
    p.add_argument('--num-eval-episodes', type=int, default=20)

    # SAC
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--hidden-depth', type=int, default=2)
    p.add_argument('--actor-lr', type=float, default=3e-4)
    p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--alpha-lr', type=float, default=3e-4)
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(
        args.device if (args.device == 'cpu' or torch.cuda.is_available())
        else 'cpu')
    utils.set_seed_everywhere(args.seed)

    env = make_pendulum(theta_target_deg=args.theta, seed=args.seed,
                        max_episode_steps=args.max_episode_steps)
    eval_env = make_pendulum(theta_target_deg=args.theta,
                             seed=args.seed + 10_000,
                             max_episode_steps=args.max_episode_steps)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_range=[-1.0, 1.0],
        device=device,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, hidden_depth=args.hidden_depth,
        learnable_temperature=True, target_entropy=-action_dim,
    )

    rb = ReplayBuffer(obs_shape=env.observation_space.shape,
                      action_shape=env.action_space.shape,
                      capacity=args.replay_buffer_capacity, device=device)

    teacher = OracleTeacher(mistake_prob=args.teacher_mistake_prob,
                            seed=args.seed)

    log_dir = Path(args.out_dir) / (
        f"pebble_theta{int(args.theta)}_fb{args.feedback_budget}"
        f"_{args.selection}_seed{args.seed}")

    cfg = PebbleConfig(
        num_train_steps=args.num_train_steps,
        unsup_steps=args.unsup_steps,
        num_seed_steps=args.num_seed_steps,
        total_feedback_budget=args.feedback_budget,
        feedback_frequency=args.feedback_frequency,
        queries_per_session=args.queries_per_session,
        candidate_pool_size=args.candidate_pool_size,
        segment_length=args.segment_length,
        selection=args.selection,
        ensemble_size=args.ensemble_size,
        reward_epochs=args.reward_epochs,
        eval_frequency=args.eval_frequency,
        num_eval_episodes=args.num_eval_episodes,
        log_dir=log_dir,
        extra_config=vars(args),
    )

    train_pebble(agent, env, eval_env, rb, teacher, cfg)


if __name__ == '__main__':
    main()
