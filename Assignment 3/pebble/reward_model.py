"""Reward-model ensemble for PEBBLE.

Each ensemble member is a small MLP r_psi(s, a) -> R. The training signal is
the Bradley-Terry preference loss over pairs of segments:

    P[sigma1 > sigma0] = sigma( sum_t r_psi(s1_t, a1_t) - sum_t r_psi(s0_t, a0_t) )

with a soft label y in {0, 0.5, 1}. Loss is cross-entropy.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _mlp(input_dim, hidden_dim, hidden_depth, output_dim=1):
    if hidden_depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
    for _ in range(hidden_depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class RewardModelEnsemble(nn.Module):
    """Ensemble of N reward models sharing the same architecture."""

    def __init__(self, obs_dim, action_dim, ensemble_size=3,
                 hidden_dim=256, hidden_depth=3, device='cpu',
                 lr=3e-4, weight_decay=0.0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.ensemble_size = int(ensemble_size)
        self.device = torch.device(device)

        self.models = nn.ModuleList([
            _mlp(self.obs_dim + self.action_dim, hidden_dim, hidden_depth)
            for _ in range(self.ensemble_size)
        ]).to(self.device)

        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
            for m in self.models
        ]

    def _cat(self, obs, action):
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        return torch.cat([obs, action.float()], dim=-1)

    def per_member(self, obs, action):
        """Return (N, B) stacked per-member predictions."""
        x = self._cat(obs, action)
        outs = [m(x).squeeze(-1) for m in self.models]
        return torch.stack(outs, dim=0)

    def mean(self, obs, action):
        """Ensemble-mean reward prediction."""
        with torch.no_grad():
            preds = self.per_member(
                torch.as_tensor(obs, dtype=torch.float32, device=self.device),
                torch.as_tensor(action, dtype=torch.float32, device=self.device))
            return preds.mean(dim=0)

    def disagreement(self, obs, action):
        """Std across ensemble members (for disagreement-based sampling)."""
        with torch.no_grad():
            preds = self.per_member(
                torch.as_tensor(obs, dtype=torch.float32, device=self.device),
                torch.as_tensor(action, dtype=torch.float32, device=self.device))
            return preds.std(dim=0)

    def predict_segment_return(self, seg_obs, seg_actions, member_idx=None):
        """Predicted return of a segment: sum_t r_psi(s_t, a_t).

        seg_obs:     (H, obs_dim) tensor
        seg_actions: (H, action_dim) tensor
        Returns (N,) per-member returns if member_idx is None, else scalar.
        """
        x = self._cat(seg_obs, seg_actions)  # (H, obs+act)
        if member_idx is None:
            # (N, H)
            outs = torch.stack([m(x).squeeze(-1) for m in self.models], dim=0)
            return outs.sum(dim=-1)
        return self.models[member_idx](x).squeeze(-1).sum()

    def update_one_epoch(self, preference_batches):
        """Train each ensemble member on one pass of preference mini-batches.

        `preference_batches` is a list of (seg0_obs, seg0_act, seg1_obs, seg1_act, label).
        Each ensemble member gets its own bootstrap sample of the list.
        """
        losses = []
        accs = []
        for m_idx, model in enumerate(self.models):
            opt = self.optimizers[m_idx]
            # bootstrap sample
            N = len(preference_batches)
            if N == 0:
                continue
            idxs = np.random.randint(0, N, size=N)
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            for bi in idxs:
                s0o, s0a, s1o, s1a, y = preference_batches[bi]
                s0o = s0o.to(self.device)
                s0a = s0a.to(self.device)
                s1o = s1o.to(self.device)
                s1a = s1a.to(self.device)
                y = y.to(self.device)

                # segment returns per pair in the batch: (B,)
                # s0o shape (B, H, obs), reshape to (B*H, obs)
                B, H, _ = s0o.shape
                r0 = model(self._cat(s0o.reshape(B * H, -1),
                                     s0a.reshape(B * H, -1))).squeeze(-1)
                r1 = model(self._cat(s1o.reshape(B * H, -1),
                                     s1a.reshape(B * H, -1))).squeeze(-1)
                R0 = r0.view(B, H).sum(dim=-1)
                R1 = r1.view(B, H).sum(dim=-1)
                logits = torch.stack([R0, R1], dim=-1)  # (B, 2)
                # label in {0 = seg0 preferred, 1 = seg1 preferred, 0.5 = tie}
                # Cross-entropy against soft label
                log_probs = F.log_softmax(logits, dim=-1)
                soft = torch.stack([1.0 - y, y], dim=-1)
                loss = -(soft * log_probs).sum(dim=-1).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += float(loss.item()) * B
                pred = logits.argmax(dim=-1)
                hard_y = (y > 0.5).long()
                total_correct += int((pred == hard_y).sum().item())
                total_count += B

            if total_count:
                losses.append(total_loss / total_count)
                accs.append(total_correct / total_count)

        return {
            'reward_model_loss': float(np.mean(losses)) if losses else 0.0,
            'reward_model_acc': float(np.mean(accs)) if accs else 0.0,
        }
