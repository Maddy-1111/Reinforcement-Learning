"""Q2.3.3(a): Evaluate a trained SAC-R{i} policy for 500 episodes of 5000 steps.

Per-episode metrics:
    - steps_to_goal: first timestep at which `in_target` is True (np.inf if never).
    - steps_in_target: total timesteps with `in_target` True.

Saves a CSV `final_eval.csv` in the run directory and prints mean/CI.

Usage:
    python eval_final_policy.py --run-dir runs/sac_Rb_seed1 --num-episodes 500 \
        --episode-length 5000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from sac_core import SACAgent, utils  # noqa: E402

from reacher_env import make_reacher  # noqa: E402


def rollout_once(agent, env, max_steps):
    obs = env.reset()
    agent.reset()
    steps_to_goal = None
    steps_in_target = 0
    for t in range(max_steps):
        with utils.eval_mode(agent.actor, agent.critic):
            action = agent.act(obs, sample=False)
        obs, _r, done, info = env.step(action)
        if info.get('in_target'):
            if steps_to_goal is None:
                steps_to_goal = t + 1
            steps_in_target += 1
        if done:
            break
    if steps_to_goal is None:
        steps_to_goal = float('inf')
    return steps_to_goal, steps_in_target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True,
                   help='Path to runs/sac_R{a,b,c}_seed{n}')
    p.add_argument('--num-episodes', type=int, default=500)
    p.add_argument('--episode-length', type=int, default=5000)
    p.add_argument('--eval-reward', default=None,
                   help='Which reward variant to use for the env (a|b|c). '
                        'Defaults to the one the policy was trained on.')
    p.add_argument('--backend', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=12345)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / 'config.json'
    with open(cfg_path) as f:
        run_cfg = json.load(f)

    reward = args.eval_reward or run_cfg['reward']
    device = torch.device(args.device if (args.device == 'cpu'
                                          or torch.cuda.is_available())
                          else 'cpu')

    env = make_reacher(reward_variant=reward, seed=args.seed,
                       max_episode_steps=args.episode_length,
                       backend=args.backend)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [float(env.action_space.low.min()),
                    float(env.action_space.high.max())]

    agent = SACAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_range=action_range,
        device=device,
        hidden_dim=run_cfg.get('hidden_dim', 1024),
        hidden_depth=run_cfg.get('hidden_depth', 2),
        target_entropy=-action_dim,
    )
    agent.load(run_dir / 'model_final.pt', map_location=device)

    utils.set_seed_everywhere(args.seed)

    out_path = run_dir / 'final_eval.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'steps_to_goal', 'steps_in_target'])
        all_stg, all_sit = [], []
        for ep in range(args.num_episodes):
            stg, sit = rollout_once(agent, env, args.episode_length)
            w.writerow([ep, stg, sit])
            all_stg.append(stg)
            all_sit.append(sit)

    finite_stg = [x for x in all_stg if np.isfinite(x)]
    print(f"Wrote {out_path}")
    print(f"steps_to_goal:   mean={np.mean(finite_stg) if finite_stg else np.nan:.1f} "
          f"(n_reached={len(finite_stg)}/{len(all_stg)})")
    print(f"steps_in_target: mean={np.mean(all_sit):.1f}  "
          f"std={np.std(all_sit):.1f}")


if __name__ == '__main__':
    main()
