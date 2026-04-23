"""Reacher-easy env wrapper with the three reward formulations from PA3 Fig. 3.

Rewards:
    R_a: +1 if finger is in target, else -(||x_goal - x_pos|| + ||action||^2).
         Fixed-length truncated episodes (T=1000).
    R_b: +1 if in target, else 0. Fixed-length truncated episodes (T=1000).
         This is the default dm_control reacher reward.
    R_c: -1 every step until episode termination (MountainCar-style).
         Episode terminates when finger reaches target with near-zero velocity.
         On timeout (T=1000) the env auto-resets and the agent incurs a penalty.

The observation returned is the default dm_control reacher observation
(concatenated to a 1-D float32 vector).

Two backends are supported automatically:
    1. dm_control (preferred): dm_control.suite.load('reacher', 'easy')
    2. shimmy: gymnasium.make('dm_control/reacher-easy-v0')

The implementation avoids any dependence on dmc2gym (unmaintained).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _try_dm_control():
    try:
        from dm_control import suite  # noqa: F401
        return 'dm_control'
    except Exception:
        return None


def _try_shimmy():
    try:
        import gymnasium  # noqa: F401
        import shimmy  # noqa: F401
        return 'shimmy'
    except Exception:
        return None


def detect_backend():
    for fn in (_try_dm_control, _try_shimmy):
        b = fn()
        if b is not None:
            return b
    raise ImportError(
        "Need either `dm_control` or `shimmy[dm-control]` installed. "
        "See reacher/README.md.")


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def _flatten_obs(obs_dict_or_array):
    if isinstance(obs_dict_or_array, dict):
        parts = []
        for k in sorted(obs_dict_or_array.keys()):
            parts.append(np.asarray(obs_dict_or_array[k]).ravel())
        return np.concatenate(parts).astype(np.float32)
    return np.asarray(obs_dict_or_array, dtype=np.float32).ravel()


def _physics_state(physics):
    """Extract finger pos, target pos, target radius, qvel from the physics."""
    # finger position (xy) — 'finger' geom in dm_control reacher.xml
    finger_xyz = physics.named.data.geom_xpos['finger']
    target_xyz = physics.named.data.geom_xpos['target']
    target_size = float(physics.named.model.geom_size['target', 0])
    finger_pos = np.asarray(finger_xyz[:2], dtype=np.float64)
    target_pos = np.asarray(target_xyz[:2], dtype=np.float64)
    qvel = np.asarray(physics.data.qvel, dtype=np.float64)
    return finger_pos, target_pos, target_size, qvel


def compute_reward(variant, in_target, dist_to_target, action, at_target_still):
    """Compute reward for the given variant.

    Args:
        variant: 'a', 'b', or 'c'.
        in_target: bool, finger inside the target disk.
        dist_to_target: scalar, ||x_goal - x_pos||.
        action: np.ndarray, action vector.
        at_target_still: bool, in target AND ~zero velocity (for R_c term).
    """
    if variant == 'a':
        if in_target:
            return 1.0
        return float(-(dist_to_target + np.sum(np.square(action))))
    if variant == 'b':
        return 1.0 if in_target else 0.0
    if variant == 'c':
        # -1 every step until termination; per-step reward is always -1.
        # Termination handled by env wrapper.
        return -1.0
    raise ValueError(f"Unknown reward variant: {variant}")


# ---------------------------------------------------------------------------
# Env wrapper (gym-style API: reset/step returning (obs, reward, done, info))
# ---------------------------------------------------------------------------

class ReacherEnv:
    """dm_control reacher-easy with one of three reward formulations.

    API:
        obs = env.reset()
        obs, reward, done, info = env.step(action)
    `done=True` only if the task (R_c) reached the goal with near-zero velocity.
    Timeouts at T=1000 are handled by the training loop via `info['TimeLimit']`
    and `env._max_episode_steps` (set to 1000).
    """

    VELOCITY_THRESHOLD = 1e-2  # rad/s, considered "near zero"

    def __init__(self,
                 reward_variant='b',
                 max_episode_steps=1000,
                 seed=1,
                 backend=None):
        assert reward_variant in ('a', 'b', 'c'), reward_variant
        self.reward_variant = reward_variant
        self._max_episode_steps = int(max_episode_steps)
        self._seed = int(seed)
        self._backend = backend or detect_backend()
        self._build()

        # Spaces
        self.action_space = _Box(low=-1.0, high=1.0, shape=(self.action_dim,),
                                 dtype=np.float32)
        self.observation_space = _Box(low=-np.inf, high=np.inf,
                                       shape=(self.obs_dim,), dtype=np.float32)

        self._steps = 0

    # -- backend setup ----------------------------------------------------

    def _build(self):
        if self._backend == 'dm_control':
            from dm_control import suite
            self._env = suite.load(domain_name='reacher',
                                   task_name='easy',
                                   task_kwargs={'random': self._seed})
            self._is_dm = True
            # action spec
            act_spec = self._env.action_spec()
            self.action_dim = int(act_spec.shape[0])
            # obs dim (flatten default dict obs)
            ts = self._env.reset()
            self.obs_dim = int(_flatten_obs(ts.observation).shape[0])
        elif self._backend == 'shimmy':
            import gymnasium as gym
            self._env = gym.make('dm_control/reacher-easy-v0')
            self._env.reset(seed=self._seed)
            self._is_dm = False
            self.action_dim = int(self._env.action_space.shape[0])
            obs, _ = self._env.reset(seed=self._seed)
            self.obs_dim = int(_flatten_obs(obs).shape[0])
        else:
            raise RuntimeError(self._backend)

    # -- physics access (works for both backends) -------------------------

    def _physics(self):
        if self._is_dm:
            return self._env.physics
        # shimmy wraps dm_env -> env.unwrapped._env._env is the dm_env
        inner = self._env.unwrapped
        # try common attrs
        for attr in ('_env', 'env'):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
                break
        return inner.physics

    # -- API --------------------------------------------------------------

    def seed(self, seed):
        self._seed = int(seed)
        # rebuild to apply seed cleanly
        self._build()
        return [self._seed]

    def reset(self):
        if self._is_dm:
            ts = self._env.reset()
            obs = _flatten_obs(ts.observation)
        else:
            obs, _ = self._env.reset()
            obs = _flatten_obs(obs)
        self._steps = 0
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if self._is_dm:
            ts = self._env.step(action)
            next_obs = _flatten_obs(ts.observation)
        else:
            res = self._env.step(action)
            # gymnasium: obs, reward, terminated, truncated, info
            next_obs = _flatten_obs(res[0])

        phys = self._physics()
        finger, target, radius, qvel = _physics_state(phys)
        dist = float(np.linalg.norm(target - finger))
        in_target = dist < radius
        still = bool(np.max(np.abs(qvel)) < self.VELOCITY_THRESHOLD)
        at_goal_still = in_target and still

        reward = compute_reward(self.reward_variant, in_target, dist, action,
                                at_goal_still)

        self._steps += 1
        timeout = self._steps >= self._max_episode_steps

        if self.reward_variant == 'c':
            terminated = at_goal_still
        else:
            terminated = False

        done = bool(terminated or timeout)
        info = {
            'dist_to_target': dist,
            'in_target': in_target,
            'at_goal_still': at_goal_still,
            'timeout': timeout,
            'terminated': terminated,
        }
        return next_obs, float(reward), done, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


class _Box:
    """Minimal Box-like space (no gym dependency)."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy()
        self.shape = tuple(shape)
        self.dtype = dtype
        self._rng = np.random.default_rng()

    def sample(self):
        return self._rng.uniform(self.low, self.high).astype(self.dtype)


def make_reacher(reward_variant, seed=1, max_episode_steps=1000, backend=None):
    return ReacherEnv(reward_variant=reward_variant,
                      seed=seed,
                      max_episode_steps=max_episode_steps,
                      backend=backend)
