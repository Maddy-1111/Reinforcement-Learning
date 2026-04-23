"""Categorical policy for discrete SAC."""
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F

from . import utils


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = utils.mlp(obs_dim, hidden_dim, num_actions, hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, obs):
        logits = self.trunk(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        dist = pyd.Categorical(logits=logits)
        return dist, probs, log_probs
