import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils


class DoubleQCritic(nn.Module):
    """Critic network implementing clipped double Q-learning (SAC)."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        # concatenate state and action
        obs_action = torch.cat([obs, action], dim=-1)

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        # ensure shape consistency (batch, 1)
        q1 = q1.view(-1, 1)
        q2 = q2.view(-1, 1)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def min_q(self, obs, action):
        """Convenience function for SAC target computation."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            if isinstance(m1, nn.Linear):
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)