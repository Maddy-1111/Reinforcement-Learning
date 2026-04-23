"""Discrete SAC (Christodoulou 2019, arXiv:1910.07207).

Key differences vs. continuous SAC:
    - Policy is Categorical. No reparameterization needed: we take the exact
      expectation E_{a ~ π}[.] by summing over actions.
    - Critic outputs Q(s, .) for all |A| actions; we index by taken action for
      the TD target and sum over pi(a|s) for actor / target-V.
    - Default target entropy is `entropy_ratio * log(|A|)` (ratio=0.98 is the
      original paper's suggestion).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import utils
from .actor_discrete import CategoricalActor
from .critic_discrete import DiscreteDoubleQCritic


class DiscreteSACAgent:
    def __init__(self,
                 obs_dim,
                 num_actions,
                 device,
                 discount=0.99,
                 init_temperature=0.1,
                 alpha_lr=1e-4,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 actor_betas=(0.9, 0.999),
                 critic_betas=(0.9, 0.999),
                 alpha_betas=(0.9, 0.999),
                 actor_update_frequency=1,
                 critic_tau=0.005,
                 critic_target_update_frequency=2,
                 batch_size=256,
                 learnable_temperature=True,
                 target_entropy_ratio=0.98,
                 target_entropy=None,
                 hidden_dim=256,
                 hidden_depth=2):
        self.num_actions = int(num_actions)
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.actor = CategoricalActor(obs_dim, num_actions, hidden_dim,
                                      hidden_depth).to(self.device)
        self.critic = DiscreteDoubleQCritic(obs_dim, num_actions, hidden_dim,
                                            hidden_depth).to(self.device)
        self.critic_target = DiscreteDoubleQCritic(
            obs_dim, num_actions, hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature),
                                      device=self.device,
                                      dtype=torch.float32,
                                      requires_grad=True)
        if target_entropy is None:
            target_entropy = target_entropy_ratio * np.log(num_actions)
        self.target_entropy = float(target_entropy)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr, betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def reset(self):
        pass

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.unsqueeze(0)
        with torch.no_grad():
            dist, probs, _ = self.actor(obs)
            if sample:
                a = dist.sample()
            else:
                a = probs.argmax(dim=-1)
        return int(a.item())

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            _, next_probs, next_log_probs = self.actor(next_obs)
            tQ1, tQ2 = self.critic_target(next_obs)
            tQ = torch.min(tQ1, tQ2)
            # V(s') = E_a[ Q(s',a) - alpha * log pi(a|s') ]
            next_V = (next_probs * (tQ - self.alpha.detach() * next_log_probs)
                      ).sum(dim=-1, keepdim=True)
            target_Q = reward + not_done * self.discount * next_V

        q1_all, q2_all = self.critic(obs)
        action_idx = action.long().view(-1, 1)
        q1 = q1_all.gather(1, action_idx)
        q2 = q2_all.gather(1, action_idx)
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def update_actor_and_alpha(self, obs):
        _, probs, log_probs = self.actor(obs)
        with torch.no_grad():
            q1, q2 = self.critic(obs)
            q = torch.min(q1, q2)
        # actor loss: E_a[ alpha * log pi - Q ]
        actor_loss = (probs * (self.alpha.detach() * log_probs - q)
                      ).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # entropy under current policy
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        alpha_loss_val = None
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (entropy.detach() - self.target_entropy))
            alpha_loss = alpha_loss.mean() if alpha_loss.ndim else alpha_loss
            # Note the sign: we want entropy -> target_entropy.
            # d L_alpha / d alpha = (entropy - target_entropy)  =>  alpha shrinks
            # when entropy > target. Flip to match standard form:
            alpha_loss = -(self.log_alpha *
                           (entropy - self.target_entropy).detach())
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        return actor_loss.item(), alpha_loss_val, entropy.item()

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = \
            replay_buffer.sample(self.batch_size)

        metrics = {'batch_reward': reward.mean().item()}
        metrics['critic_loss'] = self.update_critic(
            obs, action, reward, next_obs, not_done_no_max)

        if step % self.actor_update_frequency == 0:
            a_loss, alpha_loss, ent = self.update_actor_and_alpha(obs)
            metrics['actor_loss'] = a_loss
            metrics['entropy'] = ent
            metrics['alpha'] = self.alpha.item()
            if alpha_loss is not None:
                metrics['alpha_loss'] = alpha_loss

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
        return metrics

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt['log_alpha'].to(self.device))
