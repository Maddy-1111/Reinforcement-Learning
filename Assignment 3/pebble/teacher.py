"""Simulated teachers for PEBBLE (ground-truth reward oracle).

Implements the 'oracle' teacher from Lee et al. 2021: prefers the segment with
higher ground-truth return; emits 0.5 (equally preferred) if returns are within
`tie_threshold`. Optional stochastic label flipping (`mistake_prob`) mimics a
noisy teacher.
"""
from __future__ import annotations

import numpy as np


class OracleTeacher:
    def __init__(self, tie_threshold=0.0, mistake_prob=0.0, seed=0):
        self.tie_threshold = float(tie_threshold)
        self.mistake_prob = float(mistake_prob)
        self._rng = np.random.default_rng(seed)

    def label(self, seg0_gt, seg1_gt):
        """Return soft label y in {0, 0.5, 1}: 1 if seg1 preferred."""
        R0 = float(np.sum(seg0_gt))
        R1 = float(np.sum(seg1_gt))
        diff = R1 - R0
        if abs(diff) <= self.tie_threshold:
            return 0.5
        y = 1.0 if diff > 0 else 0.0
        if self.mistake_prob > 0 and self._rng.random() < self.mistake_prob:
            y = 1.0 - y
        return y
