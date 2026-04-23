"""LunarLander-v3 wrapper (continuous and discrete) with optional hover-box reward.

Continuous: action space [-1, 1]^2 — plugs into SACAgent directly.
Discrete:   4 discrete actions — plugs into DiscreteSACAgent directly.

For Q2.2.3 the wrapper supports a hover-box bonus:
    - If |x| < 0.1 and 0.4 < |y| < 0.6, and bonus not yet awarded this episode,
      add `hover_bonus` to reward (default +200; flip to -100 mid-training).
    - Only awarded once per episode.
    - Flip the bonus mid-training by calling `env.set_hover_bonus(-100.0)` from
      the training loop's reward-schedule hook.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("gymnasium is required for LunarLander-v3") from e


HOVER_BOX = {'x_abs': 0.1, 'y_lo': 0.4, 'y_hi': 0.6}


class LunarLanderEnv:
    def __init__(self, continuous=True, hover_bonus=None,
                 max_episode_steps=1000, seed=0):
        self._is_continuous = bool(continuous)
        self._env = gym.make('LunarLander-v3',
                             continuous=self._is_continuous,
                             max_episode_steps=max_episode_steps)
        self._env.reset(seed=int(seed))
        self._seed = int(seed)

        self.observation_space = _Box(
            low=-np.inf, high=np.inf,
            shape=self._env.observation_space.shape, dtype=np.float32)
        if self._is_continuous:
            self.action_space = _Box(
                low=self._env.action_space.low.astype(np.float32),
                high=self._env.action_space.high.astype(np.float32),
                shape=self._env.action_space.shape, dtype=np.float32)
            self.num_actions = None
        else:
            self.action_space = _Discrete(self._env.action_space.n)
            self.num_actions = int(self._env.action_space.n)

        self._max_episode_steps = int(max_episode_steps)
        self.hover_bonus = None if hover_bonus is None else float(hover_bonus)
        self._hover_awarded = False
        self._steps = 0

    # external hook used by the training loop's reward schedule
    def set_hover_bonus(self, new_bonus):
        self.hover_bonus = float(new_bonus)

    def seed(self, s):
        self._seed = int(s)
        self._env.reset(seed=self._seed)
        return [self._seed]

    def reset(self):
        obs, _ = self._env.reset()
        self._hover_awarded = False
        self._steps = 0
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        if self._is_continuous:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        else:
            action = int(action)

        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        # Hover-box bonus (once per episode)
        if self.hover_bonus is not None and not self._hover_awarded:
            x, y = float(obs[0]), float(obs[1])
            if (abs(x) < HOVER_BOX['x_abs']
                    and HOVER_BOX['y_lo'] < abs(y) < HOVER_BOX['y_hi']):
                reward = float(reward) + self.hover_bonus
                self._hover_awarded = True
                info = dict(info or {})
                info['hover_bonus_awarded'] = True

        self._steps += 1
        timeout = bool(truncated) or self._steps >= self._max_episode_steps
        done = bool(terminated) or timeout

        info = dict(info or {})
        info['timeout'] = bool(timeout)
        info['terminated'] = bool(terminated)
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


class _Discrete:
    def __init__(self, n):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64
        self._rng = np.random.default_rng()

    def sample(self):
        return int(self._rng.integers(0, self.n))


def make_continuous(hover_bonus=None, max_episode_steps=1000, seed=0):
    return LunarLanderEnv(continuous=True, hover_bonus=hover_bonus,
                          max_episode_steps=max_episode_steps, seed=seed)


def make_discrete(max_episode_steps=1000, seed=0):
    return LunarLanderEnv(continuous=False, hover_bonus=None,
                          max_episode_steps=max_episode_steps, seed=seed)
