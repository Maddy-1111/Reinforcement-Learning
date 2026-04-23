"""Rolling buffer of recent (obs, action, ground_truth_reward) tuples.

Used to sample segments for preference queries. Separate from the SAC replay
buffer so that:
    - we can sample contiguous segments by time (not random transitions),
    - we retain the ground-truth reward for teacher labeling even after the
      SAC replay buffer has been relabeled with the learned reward.
"""
from __future__ import annotations

import numpy as np


class TrajectoryBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.gt_reward = np.zeros(self.capacity, dtype=np.float32)
        # mask[t] = 1 if t is a valid "start of segment" (i.e. no episode boundary
        # in the next H steps). We'll check at sampling time instead.
        self.ep_boundary = np.zeros(self.capacity, dtype=bool)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, gt_reward, done):
        i = self.idx
        self.obs[i] = obs
        if np.ndim(action) == 0:
            self.act[i, 0] = action
        else:
            self.act[i, :len(action)] = action
        self.gt_reward[i] = float(gt_reward)
        self.ep_boundary[i] = bool(done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_segment_pairs(self, num_pairs, segment_length):
        """Return a list of (seg0_obs, seg0_act, seg0_gt, seg1_obs, seg1_act, seg1_gt).

        Segments are contiguous chunks of length H that don't cross episode
        boundaries (we check ep_boundary within the chunk, excluding the last
        step which may legitimately be done).
        """
        n = len(self)
        if n < segment_length + 1:
            return []
        # valid start indices (t such that [t, t+H) lies entirely in buffer
        # and contains no ep boundary except possibly at t+H-1)
        max_start = n - segment_length
        pairs = []
        attempts = 0
        while len(pairs) < num_pairs and attempts < num_pairs * 20:
            attempts += 1
            t0 = np.random.randint(0, max_start)
            t1 = np.random.randint(0, max_start)
            if t0 == t1:
                continue
            # Reject if an episode boundary lies strictly inside either segment
            # (not at the final step — allowed).
            if self.ep_boundary[t0:t0 + segment_length - 1].any():
                continue
            if self.ep_boundary[t1:t1 + segment_length - 1].any():
                continue
            pairs.append((
                self.obs[t0:t0 + segment_length].copy(),
                self.act[t0:t0 + segment_length].copy(),
                self.gt_reward[t0:t0 + segment_length].copy(),
                self.obs[t1:t1 + segment_length].copy(),
                self.act[t1:t1 + segment_length].copy(),
                self.gt_reward[t1:t1 + segment_length].copy(),
            ))
        return pairs
