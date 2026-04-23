import numpy as np
import torch


class ReplayBuffer(object):
    """Transition buffer supporting continuous (float) or discrete (int) actions.

    Pass `action_shape=()` and `action_dtype=np.int64` for discrete action
    envs; pass `action_shape=(act_dim,)` with default float32 otherwise.
    """

    def __init__(self, obs_shape, action_shape, capacity, device,
                 action_dtype=np.float32):
        self.capacity = int(capacity)
        self.device = device

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((self.capacity, *action_shape),
                                dtype=action_dtype)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)

        self._action_dtype = action_dtype
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        # np.copyto doesn't accept a scalar dst (0-d slice); use direct
        # assignment which works for both scalar and vector actions.
        self.actions[self.idx] = action
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        # Actions: keep int dtype for discrete, float for continuous.
        if np.issubdtype(self._action_dtype, np.integer):
            actions = torch.as_tensor(self.actions[idxs], device=self.device).long()
        else:
            actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs],
                                    device=self.device).float()
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device).float()

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    # --- utilities for PEBBLE / reward relabeling ---------------------------

    def relabel_rewards(self, reward_fn):
        """Overwrite stored rewards using `reward_fn(obs, action, next_obs) -> r`.

        Used for PEBBLE (learned reward model) and for mid-training reward
        changes. Operates only on filled slots.
        """
        n = len(self)
        for i in range(n):
            r = reward_fn(self.obses[i], self.actions[i], self.next_obses[i])
            self.rewards[i, 0] = float(r)
