"""PEBBLE: unsupervised pre-training + preference-based reward learning + SAC.

Reference: Lee, Smith, Abbeel. "PEBBLE: Feedback-Efficient Interactive
Reinforcement Learning via Relabeling Experience and Unsupervised
Pre-training." ICML 2021. arXiv:2106.05091.

High-level loop
---------------
1. Unsupervised pre-training: for `unsup_steps` env steps, SAC trains on a
   state-entropy intrinsic reward (k-NN based, APT-style). No teacher queries.
2. Preference-based training:
   for each env step:
       - collect transition, store in traj_buffer (with gt_reward) and in the
         SAC replay_buffer (reward will be overwritten by learned R_psi soon)
       - every `feedback_frequency` steps (and before doing `num_queries`
         queries have been spent):
           - sample `num_queries` candidate segment pairs
           - pick the `num_queries` best pairs by selection strategy
           - label via simulated teacher (ground-truth reward)
           - add to preference buffer
           - train reward-model ensemble for `reward_epochs` epochs
           - relabel replay_buffer rewards with ensemble mean
       - SAC update step as usual (on learned-reward transitions)
       - periodic offline evaluation against ground-truth env reward
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from sac_core import utils
from sac_core.training_loop import evaluate_in_envs

from .preference_buffer import PreferenceBuffer
from .reward_model import RewardModelEnsemble
from .trajectory_buffer import TrajectoryBuffer


# ---------------------------------------------------------------------------
# k-NN state-entropy intrinsic reward
# ---------------------------------------------------------------------------

class StateEntropyIntrinsicReward:
    """Running k-NN state-entropy estimator (APT-style).

    Keeps a rolling buffer of recent states. At each call, intrinsic reward
    r_int(s) = log( 1/k * sum_i ||s - s_{nn_i}||_2 + eps ), normalized by
    running std of the raw (pre-log) distances.
    """

    def __init__(self, obs_dim, k=5, buffer_size=10_000):
        self.obs_dim = int(obs_dim)
        self.k = int(k)
        self.buffer_size = int(buffer_size)
        self.buf = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.idx = 0
        self.full = False
        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 1e-4

    def add(self, obs):
        self.buf[self.idx] = obs
        self.idx = (self.idx + 1) % self.buffer_size
        self.full = self.full or self.idx == 0

    def reward(self, obs):
        n = self.buffer_size if self.full else self.idx
        if n < self.k + 1:
            return 0.0
        sub = self.buf[:n]
        d = np.linalg.norm(sub - obs[None, :], axis=1)
        # take k smallest *non-zero* (exclude self if in buffer)
        if d.size > self.k:
            nn = np.partition(d, self.k)[:self.k]
        else:
            nn = d
        mean_d = float(np.mean(nn) + 1e-8)
        r = np.log(mean_d)
        # running normalization
        self._count += 1.0
        delta = r - self._running_mean
        self._running_mean += delta / self._count
        self._running_var += delta * (r - self._running_mean)
        std = (self._running_var / self._count) ** 0.5 + 1e-6
        return float((r - self._running_mean) / std)


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def pick_pairs_uniform(pairs, num):
    if len(pairs) <= num:
        return pairs
    idxs = np.random.choice(len(pairs), size=num, replace=False)
    return [pairs[i] for i in idxs]


def pick_pairs_disagreement(pairs, num, reward_model: RewardModelEnsemble):
    """Pick pairs with highest ensemble disagreement on P[seg1 > seg0]."""
    if len(pairs) <= num:
        return pairs
    device = reward_model.device
    disagreements = []
    for (s0o, s0a, _s0g, s1o, s1a, _s1g) in pairs:
        with torch.no_grad():
            seg0 = torch.as_tensor(s0o, device=device)
            seg0a = torch.as_tensor(s0a, device=device)
            seg1 = torch.as_tensor(s1o, device=device)
            seg1a = torch.as_tensor(s1a, device=device)
            R0_per = reward_model.predict_segment_return(seg0, seg0a)
            R1_per = reward_model.predict_segment_return(seg1, seg1a)
            probs = torch.sigmoid(R1_per - R0_per)  # (N,)
        disagreements.append(float(probs.std().item()))
    order = np.argsort(disagreements)[::-1][:num]
    return [pairs[i] for i in order]


# ---------------------------------------------------------------------------
# Reward-model-backed "reward_fn" used to relabel the replay buffer
# ---------------------------------------------------------------------------

def make_relabel_fn(reward_model: RewardModelEnsemble):
    def _fn(obs, action, _next_obs):
        r = reward_model.mean(
            torch.as_tensor(obs, device=reward_model.device).unsqueeze(0),
            torch.as_tensor(action, device=reward_model.device).unsqueeze(0))
        return float(r.item())
    return _fn


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class PebbleConfig:
    def __init__(self,
                 num_train_steps=500_000,
                 unsup_steps=9_000,
                 num_seed_steps=1_000,
                 # feedback schedule
                 total_feedback_budget=1_000,
                 feedback_frequency=5_000,
                 queries_per_session=20,
                 candidate_pool_size=200,
                 segment_length=50,
                 selection='disagreement',  # 'uniform' | 'disagreement'
                 # reward model
                 ensemble_size=3,
                 reward_hidden_dim=256,
                 reward_hidden_depth=3,
                 reward_lr=3e-4,
                 reward_epochs=50,
                 reward_batch_size=64,
                 # eval
                 eval_frequency=10_000,
                 num_eval_episodes=20,
                 log_dir='runs/pebble',
                 # teacher
                 teacher_tie_threshold=0.0,
                 teacher_mistake_prob=0.0,
                 # intrinsic reward
                 intrinsic_k=5,
                 intrinsic_buffer=10_000,
                 extra_config=None):
        self.num_train_steps = int(num_train_steps)
        self.unsup_steps = int(unsup_steps)
        self.num_seed_steps = int(num_seed_steps)
        self.total_feedback_budget = int(total_feedback_budget)
        self.feedback_frequency = int(feedback_frequency)
        self.queries_per_session = int(queries_per_session)
        self.candidate_pool_size = int(candidate_pool_size)
        self.segment_length = int(segment_length)
        self.selection = selection
        self.ensemble_size = int(ensemble_size)
        self.reward_hidden_dim = int(reward_hidden_dim)
        self.reward_hidden_depth = int(reward_hidden_depth)
        self.reward_lr = float(reward_lr)
        self.reward_epochs = int(reward_epochs)
        self.reward_batch_size = int(reward_batch_size)
        self.eval_frequency = int(eval_frequency)
        self.num_eval_episodes = int(num_eval_episodes)
        self.log_dir = Path(log_dir)
        self.teacher_tie_threshold = float(teacher_tie_threshold)
        self.teacher_mistake_prob = float(teacher_mistake_prob)
        self.intrinsic_k = int(intrinsic_k)
        self.intrinsic_buffer = int(intrinsic_buffer)
        self.extra_config = extra_config or {}


def train_pebble(agent, env, eval_env, replay_buffer, teacher,
                 cfg: PebbleConfig,
                 sample_random_action=None,
                 verbose=True):
    """Run the PEBBLE loop. Returns log directory."""
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(cfg.extra_config, f, indent=2, default=str)

    obs_dim = env.observation_space.shape[0]
    action_dim = (env.action_space.shape[0]
                  if hasattr(env.action_space, 'shape') and env.action_space.shape
                  else 1)

    reward_model = RewardModelEnsemble(
        obs_dim=obs_dim, action_dim=action_dim,
        ensemble_size=cfg.ensemble_size,
        hidden_dim=cfg.reward_hidden_dim,
        hidden_depth=cfg.reward_hidden_depth,
        device=agent.device, lr=cfg.reward_lr)

    pref_buf = PreferenceBuffer(
        capacity=max(cfg.total_feedback_budget, 1),
        segment_length=cfg.segment_length,
        obs_dim=obs_dim, action_dim=action_dim)

    traj_buf = TrajectoryBuffer(
        capacity=max(50_000, cfg.candidate_pool_size * cfg.segment_length * 2),
        obs_dim=obs_dim, action_dim=action_dim)

    intrinsic = StateEntropyIntrinsicReward(
        obs_dim=obs_dim, k=cfg.intrinsic_k, buffer_size=cfg.intrinsic_buffer)

    if sample_random_action is None:
        sample_random_action = lambda: env.action_space.sample()

    # CSV
    prog_fields = ['step', 'phase', 'return_mean', 'return_std',
                   'feedback_used', 'reward_loss', 'reward_acc',
                   'alpha', 'wall_time']
    prog_f = open(log_dir / 'progress.csv', 'w', newline='')
    prog_w = csv.DictWriter(prog_f, fieldnames=prog_fields)
    prog_w.writeheader()

    start_time = time.time()
    step = 0
    feedback_used = 0
    last_reward_metrics = {'reward_model_loss': float('nan'),
                           'reward_model_acc': float('nan')}
    obs = env.reset()
    agent.reset()

    def do_feedback_session():
        nonlocal feedback_used, last_reward_metrics
        if feedback_used >= cfg.total_feedback_budget:
            return
        budget_left = cfg.total_feedback_budget - feedback_used
        to_query = min(cfg.queries_per_session, budget_left)
        pool = traj_buf.sample_segment_pairs(cfg.candidate_pool_size,
                                             cfg.segment_length)
        if not pool:
            return
        if cfg.selection == 'disagreement' and len(pref_buf) > 0:
            picked = pick_pairs_disagreement(pool, to_query, reward_model)
        else:
            picked = pick_pairs_uniform(pool, to_query)
        for (s0o, s0a, s0g, s1o, s1a, s1g) in picked:
            label = teacher.label(s0g, s1g)
            pref_buf.add(s0o, s0a, s1o, s1a, label)
            feedback_used += 1
            if feedback_used >= cfg.total_feedback_budget:
                break
        # Train reward model
        metrics_accum = []
        for _ in range(cfg.reward_epochs):
            batches = pref_buf.sample_batches(cfg.reward_batch_size)
            m = reward_model.update_one_epoch(batches)
            if m:
                metrics_accum.append(m)
        if metrics_accum:
            last_reward_metrics = {
                k: float(np.mean([m[k] for m in metrics_accum]))
                for k in metrics_accum[0]
            }
        # Relabel stored transitions in replay with learned reward model
        if len(replay_buffer) > 0:
            replay_buffer.relabel_rewards(make_relabel_fn(reward_model))

    try:
        while step < cfg.num_train_steps:
            phase = 'unsup' if step < cfg.unsup_steps else 'preference'

            # action selection
            if step < cfg.num_seed_steps:
                action = sample_random_action()
            else:
                if hasattr(agent, 'actor') and hasattr(agent, 'critic'):
                    with utils.eval_mode(agent.actor, agent.critic):
                        action = agent.act(obs, sample=True)
                else:
                    action = agent.act(obs, sample=True)

            next_obs, gt_reward, done, info = env.step(action)

            # decide what reward to store in the SAC replay
            if phase == 'unsup':
                intrinsic.add(obs)
                stored_reward = intrinsic.reward(next_obs)
            else:
                # predicted reward from the current ensemble mean
                if len(pref_buf) > 0:
                    with torch.no_grad():
                        stored_reward = float(reward_model.mean(
                            torch.as_tensor(obs, device=agent.device).unsqueeze(0),
                            torch.as_tensor(np.atleast_1d(action).astype(np.float32),
                                            device=agent.device).unsqueeze(0)
                        ).item())
                else:
                    # no preferences yet — fall back to 0 (agent hasn't seen any
                    # preference feedback so the reward model is untrained)
                    stored_reward = 0.0

            done_float = float(bool(done))
            timeout = bool(info.get('timeout', False))
            terminated = bool(info.get('terminated', not timeout and bool(done)))
            done_no_max = 0.0 if (timeout and not terminated) else done_float

            replay_buffer.add(obs, np.atleast_1d(action).astype(np.float32),
                              stored_reward, next_obs, done_float, done_no_max)
            traj_buf.add(obs, action, gt_reward, done)

            obs = next_obs
            step += 1

            # SAC update
            if step >= cfg.num_seed_steps and len(replay_buffer) > 0:
                agent.update(replay_buffer, step)

            # Preference collection / reward-model retraining
            if (phase == 'preference'
                    and step % cfg.feedback_frequency == 0
                    and feedback_used < cfg.total_feedback_budget):
                do_feedback_session()

            # Kick off the first feedback session at the unsup -> pref boundary
            if (step == cfg.unsup_steps
                    and feedback_used < cfg.total_feedback_budget):
                do_feedback_session()

            # Evaluation (always against ground-truth-reward env)
            if step % cfg.eval_frequency == 0:
                stats = evaluate_in_envs(
                    agent, {'return': eval_env},
                    num_episodes=cfg.num_eval_episodes)
                row = {
                    'step': step, 'phase': phase,
                    'return_mean': stats['return']['mean'],
                    'return_std': stats['return']['std'],
                    'feedback_used': feedback_used,
                    'reward_loss': last_reward_metrics['reward_model_loss'],
                    'reward_acc': last_reward_metrics['reward_model_acc'],
                    'alpha': (agent.alpha.item()
                              if hasattr(agent, 'alpha') else float('nan')),
                    'wall_time': time.time() - start_time,
                }
                prog_w.writerow(row)
                prog_f.flush()
                if verbose:
                    print(f"[step {step:>8d}] {phase} "
                          f"ret={row['return_mean']:+.2f} "
                          f"fb={feedback_used}/{cfg.total_feedback_budget} "
                          f"rmloss={row['reward_loss']:.3f} "
                          f"rmacc={row['reward_acc']:.3f} "
                          f"alpha={row['alpha']:.3f} "
                          f"t={row['wall_time']:.0f}s")

            if done:
                obs = env.reset()
                agent.reset()
    finally:
        prog_f.close()
        if hasattr(agent, 'save'):
            agent.save(log_dir / 'model_final.pt')
        torch.save(
            {f'member_{i}': m.state_dict()
             for i, m in enumerate(reward_model.models)},
            log_dir / 'reward_model.pt')
    return log_dir
