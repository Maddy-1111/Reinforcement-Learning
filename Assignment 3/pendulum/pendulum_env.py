"""Pendulum-v1 with a modified reward that targets an arbitrary angle.

The default Pendulum-v1 reward is
    r = -(theta_norm^2 + 0.1 * theta_dot^2 + 0.001 * action^2)
where theta_norm wraps theta to (-pi, pi] so upright (theta=0) is the target.

For PA3 §2.1, the target is an arbitrary angle theta_target (in degrees):
    r = -(wrap(theta - theta_target)^2 + 0.1 * theta_dot^2 + 0.001 * action^2)

Observation, action space, and dynamics are unchanged from Pendulum-v1.
Action space is [-2, 2]; the wrapper rescales agent output from [-1, 1] to
[-2, 2] so the squashed-Normal actor in sac_core plugs in directly.

Episodes are fixed length, truncated at 1000 steps (per PDF).

Supports:
    - theta_target in degrees
    - reward_scale multiplier (for Q5b: 10x, 0.1x)
"""
from __future__ import annotations

import math

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("gymnasium is required for Pendulum-v1") from e


def _wrap_to_pi(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi


class PendulumTargetEnv:
    """Gym-style wrapper (reset returns obs; step returns (obs, r, done, info))."""

    def __init__(self, theta_target_deg=0.0, reward_scale=1.0,
                 max_episode_steps=1000, seed=0):
        self.theta_target = math.radians(float(theta_target_deg))
        self.reward_scale = float(reward_scale)
        self._max_episode_steps = int(max_episode_steps)
        self._seed = int(seed)

        self._env = gym.make('Pendulum-v1')
        self._env.reset(seed=self._seed)

        # Action: agent outputs in [-1, 1], we rescale to the env's native
        # [-2, 2] before stepping.
        self._native_low = float(self._env.action_space.low[0])
        self._native_high = float(self._env.action_space.high[0])
        assert math.isclose(self._native_high, -self._native_low)

        self.action_dim = int(self._env.action_space.shape[0])
        self.obs_dim = int(self._env.observation_space.shape[0])

        self.action_space = _Box(low=-1.0, high=1.0, shape=(self.action_dim,))
        self.observation_space = _Box(low=-np.inf, high=np.inf,
                                       shape=(self.obs_dim,))
        self._steps = 0

    def seed(self, s):
        self._seed = int(s)
        self._env.reset(seed=self._seed)
        return [self._seed]

    def reset(self):
        obs, _ = self._env.reset()
        self._steps = 0
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        native_action = action * self._native_high  # [-1,1] -> [-2,2]
        obs, _default_r, terminated, truncated, info = self._env.step(native_action)
        obs = np.asarray(obs, dtype=np.float32)
      
        # 1. Get current angle
        cos_th, sin_th, th_dot = float(obs[0]), float(obs[1]), float(obs[2])
        theta = math.atan2(sin_th, cos_th)

        # Max = 1.0 (on target), Min = -1.0 (opposite side)
        reward_pos = math.cos(theta - self.theta_target) 

        # 3. Damping: Prevent jittering once at the target
        reward_vel = -0.1 * (th_dot ** 2)

        # 4. Final Reward (keeping your scale multiplier support)
        reward = (reward_pos + reward_vel) * self.reward_scale

        self._steps += 1
        timeout = self._steps >= self._max_episode_steps
        done = bool(terminated) or timeout

        info = dict(info or {})
        info['timeout'] = bool(timeout)
        info['terminated'] = bool(terminated)
        info['theta'] = theta
        info['theta_err'] = diff
        return obs, float(reward), done, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


class _Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy()
        self.shape = tuple(shape)
        self.dtype = dtype
        self._rng = np.random.default_rng()

    def sample(self):
        return self._rng.uniform(self.low, self.high).astype(self.dtype)


def make_pendulum(theta_target_deg=0.0, reward_scale=1.0,
                  max_episode_steps=1000, seed=0):
    return PendulumTargetEnv(theta_target_deg=theta_target_deg,
                             reward_scale=reward_scale,
                             max_episode_steps=max_episode_steps,
                             seed=seed)
