import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time
from multiprocessing import Pool, cpu_count
import json
import os

LOG_FILE = "results.jsonl"  # JSON lines (append-friendly)

# -------- Device (force CPU for multiprocessing) --------
device = torch.device("cpu")

ENV_NAME = "MountainCar-v0"

# -------- Env --------
def make_env(seed):
    env = gym.make(ENV_NAME)
    env.reset(seed=seed)
    env._max_episode_steps = 2000
    return env

# -------- Replay Buffer --------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)

        return (
            torch.from_numpy(np.array(s)).float().to(device),
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.from_numpy(np.array(s_next)).float().to(device),
            torch.tensor(d, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full PER)
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, s, a, r, s_next, done):
        # New transitions get max priority so they are guaranteed to be sampled at least once
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s_next, done))
        else:
            self.buffer[self.pos] = (s, a, r, s_next, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == len(self.priorities):
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        # Calculate probabilities: P(i) = p_i^alpha / sum(p_i^alpha)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance Sampling (IS) weights to correct bias
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        s, a, r, s_next, d = zip(*samples)
        return (
            torch.from_numpy(np.array(s)).float().to(device),
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.from_numpy(np.array(s_next)).float().to(device),
            torch.tensor(d, dtype=torch.float32, device=device),
            indices,
            weights
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Add small constant to avoid zero priority


# -------- Hyperparams --------
GAMMA = 0.99
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
BUFFER_SIZE = 10000

EPS_START = 0.75
EPS_END = 0.05
EPS_DECAY = 150000

TARGET_UPDATE = 2000
NUM_EPISODES = 1000
MAX_STEPS = 2000

RHO = 1
HIDDEN_SIZE = 32

ALPHA = 0.6  # PER prioritization exponent
BETA_START = 0.4  # PER importance sampling exponent start
BETA_INC = 100000

# -------- Model --------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_dim)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# -------- Action --------
def select_action(state, q_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return q_net(state).argmax(dim=1).item()
    
# -------- Logging --------

LOG_JSONL = "results.jsonl"

def save_result(seed, rewards):
    record = {
        "label": "madhav_Q4_a",
        "seed": seed,

        "hyperparams": {
            "rho": RHO,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "eps_decay": EPS_DECAY,
            "target_update": TARGET_UPDATE,
            "max_steps": MAX_STEPS,
            "num_episodes": NUM_EPISODES,
            "hidden_size": HIDDEN_SIZE,
            "alpha": ALPHA,
            "beta_start": BETA_START,
            "beta_inc": BETA_INC
        },

        "rewards": rewards
    }

    # -------- JSONL (append safe) --------
    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")

# -------- Training --------
def train_dqn(seed=0):
    start_time = time.time()

    env = make_env(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPS_START
    epsilon_decay = (EPS_START - EPS_END) / EPS_DECAY

    rewards = []
    total_steps = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(MAX_STEPS):
            total_steps += 1

            action = select_action(state, q_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > BATCH_SIZE:
                for _ in range(RHO):
                    s, a, r, s_next, d = replay_buffer.sample(BATCH_SIZE)

                    q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        max_next_q = target_net(s_next).max(1)[0]
                        target = r + GAMMA * max_next_q * (1 - d)

                    loss = nn.MSELoss()(q_values, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

            epsilon = max(EPS_END, epsilon - epsilon_decay)
            
            if done:
                break

        rewards.append(episode_reward)

        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Seed {seed}] Episode {episode}, Reward: {episode_reward}, "
                  f"Epsilon: {epsilon:.3f}, Time: {elapsed:.2f}s")

    env.close()

    total_time = time.time() - start_time
    print(f"[Seed {seed}] DONE in {total_time:.2f}s | Steps/sec: {total_steps/total_time:.2f}")
    save_result(seed, rewards)
    return rewards


# -------- Training --------
def train_per_dqn(seed=0):
    start_time = time.time()

    env = make_env(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA)

    beta_increment = (1.0 - BETA_START) / BETA_INC
    beta = BETA_START

    epsilon = EPS_START
    epsilon_decay = (EPS_START - EPS_END) / EPS_DECAY

    rewards = []
    total_steps = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(MAX_STEPS):
            total_steps += 1

            action = select_action(state, q_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > BATCH_SIZE:
                for _ in range(RHO):
                    s, a, r, s_next, d, indices, weights = replay_buffer.sample(BATCH_SIZE, beta)
                    q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        max_next_q = target_net(s_next).max(1)[0]
                        target = r + GAMMA * max_next_q * (1 - d)

                    td_errors = torch.abs(q_values - target).detach().cpu().numpy()

                    loss = (weights * nn.MSELoss(reduction='none')(q_values, target)).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    replay_buffer.update_priorities(indices, td_errors)
                beta = min(1.0, beta + beta_increment)
    
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

            epsilon = max(EPS_END, epsilon - epsilon_decay)
            
            if done:
                break

        rewards.append(episode_reward)

        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[Seed {seed}] Episode {episode}, Reward: {episode_reward}, "
                  f"Epsilon: {epsilon:.3f}, Time: {elapsed:.2f}s")

    env.close()

    total_time = time.time() - start_time
    print(f"[Seed {seed}] DONE in {total_time:.2f}s | Steps/sec: {total_steps/total_time:.2f}")
    save_result(seed, rewards)
    return rewards


# -------- Main --------
if __name__ == "__main__":
    start = time.time()

    num_workers = min(8, cpu_count())
    print(f"Using {num_workers} workers")

    with Pool(num_workers) as p:
        all_rewards = p.map(train_dqn, range(15))

    total_time = time.time() - start
    print(f"\nALL DONE in {total_time:.2f}s")