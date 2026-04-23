"""SAC agent with automated temperature tuning (target entropy).

Adapted from https://github.com/denisyarats/pytorch_sac (sac.py), stripped of
the hydra / custom logger dependencies so it can be driven by a plain script.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import utils
from .actor import DiagGaussianActor
from .critic import DoubleQCritic


class SACAgent:
    """SAC with automated temperature tuning via target entropy."""

    def __init__(self,
                 obs_dim,
                 action_dim,
                 action_range,
                 device,
                 discount=0.99,
                 init_temperature=0.1,
                 alpha_lr=1e-4,
                 alpha_betas=(0.9, 0.999),
                 actor_lr=1e-4,
                 actor_betas=(0.9, 0.999),
                 actor_update_frequency=1,
                 critic_lr=1e-4,
                 critic_betas=(0.9, 0.999),
                 critic_tau=0.005,
                 critic_target_update_frequency=2,
                 batch_size=1024,
                 learnable_temperature=True,
                 hidden_dim=1024,
                 hidden_depth=2,
                 log_std_bounds=(-5, 2),
                 target_entropy=None):
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = DoubleQCritic(obs_dim, action_dim, hidden_dim,
                                    hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim, hidden_dim,
                                           hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(obs_dim, action_dim, hidden_dim,
                                       hidden_depth,
                                       list(log_std_bounds)).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature),
                                      device=self.device,
                                      dtype=torch.float32)
        self.log_alpha.requires_grad = True
        # default target entropy = -|A| (as in pytorch_sac)
        self.target_entropy = (-action_dim
                               if target_entropy is None else target_entropy)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)
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
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss_val = None
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        return actor_loss.item(), alpha_loss_val, (-log_prob.mean()).item()

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = \
            replay_buffer.sample(self.batch_size)

        metrics = {'batch_reward': reward.mean().item()}
        metrics['critic_loss'] = self.update_critic(obs, action, reward,
                                                    next_obs, not_done_no_max)

        if step % self.actor_update_frequency == 0:
            actor_loss, alpha_loss, entropy = self.update_actor_and_alpha(obs)
            metrics['actor_loss'] = actor_loss
            metrics['entropy'] = entropy
            metrics['alpha'] = self.alpha.item()
            if alpha_loss is not None:
                metrics['alpha_loss'] = alpha_loss

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
        return metrics

    def save(self, path):
        torch.save(
            {
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
