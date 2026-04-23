"""Storage for preference queries: pairs of segments plus teacher label."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


class PreferenceBuffer:
    """Stores up to `capacity` preference queries.

    A query is (seg0_obs, seg0_act, seg1_obs, seg1_act, label), where
    segments have fixed length H. Label y in {0, 0.5, 1}.
    """

    def __init__(self, capacity, segment_length, obs_dim, action_dim):
        self.capacity = int(capacity)
        self.segment_length = int(segment_length)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.seg0_obs = np.zeros(
            (self.capacity, self.segment_length, self.obs_dim), dtype=np.float32)
        self.seg0_act = np.zeros(
            (self.capacity, self.segment_length, self.action_dim), dtype=np.float32)
        self.seg1_obs = np.zeros_like(self.seg0_obs)
        self.seg1_act = np.zeros_like(self.seg0_act)
        self.labels = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, seg0_obs, seg0_act, seg1_obs, seg1_act, label):
        i = self.idx
        self.seg0_obs[i] = seg0_obs
        self.seg0_act[i] = seg0_act
        self.seg1_obs[i] = seg1_obs
        self.seg1_act[i] = seg1_act
        self.labels[i] = float(label)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_batches(self, batch_size) -> List[Tuple]:
        n = len(self)
        if n == 0:
            return []
        batches = []
        order = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idxs = order[start:start + batch_size]
            batches.append((
                torch.as_tensor(self.seg0_obs[idxs]),
                torch.as_tensor(self.seg0_act[idxs]),
                torch.as_tensor(self.seg1_obs[idxs]),
                torch.as_tensor(self.seg1_act[idxs]),
                torch.as_tensor(self.labels[idxs]),
            ))
        return batches
