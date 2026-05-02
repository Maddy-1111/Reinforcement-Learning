"""Generic SAC training loop shared by Pendulum / LunarLander / Reacher.

Supports:
    - continuous or discrete agents (duck-typed via agent.act / agent.update)
    - K-step random-action seed phase (10K by default, per assignment)
    - periodic offline evaluation (20 episodes every 10K steps, per assignment)
    - cross-reward evaluation: pass `eval_envs={'R_a': env_a, 'R_b': env_b, ...}`
      to evaluate the same policy in multiple envs per eval tick
    - reward-schedule hook (LL hover-box swap +200 -> -100): pass
      `reward_schedule=[(step, callable_that_mutates_env), ...]`
    - generic per-step callback for future extensions (PEBBLE: preference
      collection, reward-model updates, replay relabeling)
    - CSV logging of eval returns and per-episode training diagnostics

Everything is driven by a plain object (no hydra / no omegaconf).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import utils


def evaluate_policy(agent, env, num_episodes=20):
    returns = []
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset()
        done = False
        ep_return = 0.0
        while not done:
            # actor-only eval_mode is enough; continuous agent exposes
            # `actor`/`critic`, discrete exposes the same.
            if hasattr(agent, 'actor') and hasattr(agent, 'critic'):
                with utils.eval_mode(agent.actor, agent.critic):
                    action = agent.act(obs, sample=False)
            else:
                action = agent.act(obs, sample=False)
            obs, reward, done, _ = env.step(action)
            ep_return += reward
        returns.append(ep_return)
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_in_envs(agent, eval_envs: Dict[str, Any], num_episodes=20):
    """Evaluate same policy in a dict of {label: env}. Returns dict of stats."""
    out = {}
    for label, env in eval_envs.items():
        mean, std = evaluate_policy(agent, env, num_episodes=num_episodes)
        out[label] = {'mean': mean, 'std': std}
    return out


class TrainConfig:
    """Plain container (simpler than dataclass to keep Python-version agnostic)."""

    def __init__(self,
                 num_train_steps=1_000_000,
                 num_seed_steps=10_000,
                 eval_frequency=10_000,
                 num_eval_episodes=20,
                 checkpoint_frequency=0,
                 log_dir='runs/tmp',
                 extra_config=None):
        self.num_train_steps = int(num_train_steps)
        self.num_seed_steps = int(num_seed_steps)
        self.eval_frequency = int(eval_frequency)
        self.num_eval_episodes = int(num_eval_episodes)
        self.checkpoint_frequency = int(checkpoint_frequency)
        self.log_dir = Path(log_dir)
        self.extra_config = extra_config or {}


def train(agent,
          env,
          replay_buffer,
          cfg: TrainConfig,
          *,
          eval_envs: Optional[Dict[str, Any]] = None,
          sample_random_action: Optional[Callable[[], Any]] = None,
          reward_schedule: Optional[Sequence[Tuple[int, Callable]]] = None,
          step_callback: Optional[Callable] = None,
          on_eval: Optional[Callable] = None,
          verbose: bool = True) -> Path:
    """Main training loop.

    Args:
        agent: must expose `.act(obs, sample=bool) -> action`, `.reset()`,
               `.update(replay_buffer, step) -> dict`, `.save(path)`.
        env: gym-style env with `.reset() -> obs`, `.step(a) -> (obs, r, done, info)`,
             `.action_space.sample()`, `.observation_space.shape`,
             and an integer `_max_episode_steps` attribute if time-limit bootstrap
             is desired. `info` may include `'timeout'` / `'terminated'` flags.
        replay_buffer: sac_core.ReplayBuffer instance.
        cfg: TrainConfig.
        eval_envs: optional {label: env} for cross-reward evaluation. If None,
                   we evaluate on the training env.
        sample_random_action: callable returning a random action (during seed
                   phase). If None, uses `env.action_space.sample()`.
        reward_schedule: list of (step, fn(env)) to invoke at given step; used
                   e.g. for LL hover-box reward swap mid-training.
        step_callback: called after every env step with
                   `callback(step, transition, agent, replay_buffer)` where
                   `transition = (obs, action, reward, next_obs, done, info)`.
                   Used by PEBBLE to collect preference queries.
        on_eval: called after every eval tick with `on_eval(step, eval_stats)`.
        verbose: print one-line status per eval tick.

    Returns:
        Path to the log directory.
    """
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # dump config for reproducibility
    try:
        with open(log_dir / 'config.json', 'w') as f:
            json.dump(cfg.extra_config, f, indent=2, default=str)
    except Exception:
        pass

    # Eval env default: training env itself
    if eval_envs is None:
        eval_envs = {'return': env}

    # CSVs
    progress_fields = ['step'] + sum(
        [[f'{k}_mean', f'{k}_std'] for k in eval_envs], [])
    progress_fields += ['alpha', 'wall_time']
    train_fields = ['step', 'episode', 'episode_reward', 'episode_len',
                    'critic_loss', 'actor_loss', 'entropy', 'alpha']

    prog_f = open(log_dir / 'progress.csv', 'w', newline='')
    trn_f = open(log_dir / 'train_log.csv', 'w', newline='')
    prog_w = csv.DictWriter(prog_f, fieldnames=progress_fields)
    trn_w = csv.DictWriter(trn_f, fieldnames=train_fields)
    prog_w.writeheader()
    trn_w.writeheader()

    if sample_random_action is None:
        sample_random_action = lambda: env.action_space.sample()

    # schedule: sorted by step
    schedule = sorted(reward_schedule or [], key=lambda x: x[0])
    sched_idx = 0

    start_time = time.time()
    step = 0
    episode = 0
    episode_reward = 0.0
    episode_len = 0
    last_metrics: Dict[str, float] = {}

    obs = env.reset()
    agent.reset()

    def _log_eval(at_step: int):
        stats = evaluate_in_envs(agent, eval_envs,
                                 num_episodes=cfg.num_eval_episodes)
        row: Dict[str, Any] = {'step': at_step}
        for k, v in stats.items():
            row[f'{k}_mean'] = v['mean']
            row[f'{k}_std'] = v['std']
        row['alpha'] = float(getattr(agent, 'alpha',
                                     np.nan).item() if hasattr(
            getattr(agent, 'alpha', None), 'item') else
                             float('nan'))
        row['wall_time'] = time.time() - start_time
        prog_w.writerow(row)
        prog_f.flush()

        if on_eval is not None:
            on_eval(at_step, stats)

        if verbose:
            summary = ' '.join(
                f"{k}={v['mean']:+.2f}" for k, v in stats.items())
            print(f"[step {at_step:>8d}] {summary} "
                  f"alpha={row['alpha']:.3f} "
                  f"t={row['wall_time']:.0f}s")

    try:
        # Pre-training evaluation at step=0 so plots start from the
        # untrained-policy baseline (per assignment Figure 1 layout).
        _log_eval(0)

        while step < cfg.num_train_steps:
            # reward-schedule: fire hooks that have come due
            while sched_idx < len(schedule) and schedule[sched_idx][0] <= step:
                _, fn = schedule[sched_idx]
                fn(env)
                sched_idx += 1

            # action selection
            if step < cfg.num_seed_steps:
                action = sample_random_action()
            else:
                if hasattr(agent, 'actor') and hasattr(agent, 'critic'):
                    with utils.eval_mode(agent.actor, agent.critic):
                        action = agent.act(obs, sample=True)
                else:
                    action = agent.act(obs, sample=True)

            # learning update
            if step >= cfg.num_seed_steps and len(replay_buffer) >= 1:
                last_metrics = agent.update(replay_buffer, step)

            next_obs, reward, done, info = env.step(action)

            done_float = float(bool(done))
            timeout = bool(info.get('timeout', False))
            terminated = bool(info.get('terminated', not timeout and bool(done)))
            # If episode ended purely due to time limit, keep bootstrap.
            done_no_max = 0.0 if (timeout and not terminated) else done_float

            replay_buffer.add(obs, action, reward, next_obs, done_float,
                              done_no_max)

            if step_callback is not None:
                step_callback(step,
                              (obs, action, reward, next_obs, done_float, info),
                              agent, replay_buffer)

            obs = next_obs
            episode_reward += reward
            episode_len += 1
            step += 1

            # evaluation
            if step % cfg.eval_frequency == 0:
                _log_eval(step)

            # checkpointing
            if (cfg.checkpoint_frequency > 0
                    and step % cfg.checkpoint_frequency == 0
                    and hasattr(agent, 'save')):
                agent.save(log_dir / f'model_step{step}.pt')

            if done:
                episode += 1
                trn_w.writerow({
                    'step': step,
                    'episode': episode,
                    'episode_reward': episode_reward,
                    'episode_len': episode_len,
                    'critic_loss': last_metrics.get('critic_loss', ''),
                    'actor_loss': last_metrics.get('actor_loss', ''),
                    'entropy': last_metrics.get('entropy', ''),
                    'alpha': last_metrics.get('alpha', ''),
                })
                trn_f.flush()
                obs = env.reset()
                agent.reset()
                episode_reward = 0.0
                episode_len = 0
    finally:
        prog_f.close()
        trn_f.close()
        if hasattr(agent, 'save'):
            agent.save(log_dir / 'model_final.pt')
    return log_dir
