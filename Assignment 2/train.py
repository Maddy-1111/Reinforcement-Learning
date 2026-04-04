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

# -------- Model --------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# -------- Hyperparams --------
GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 128
BUFFER_SIZE = 20000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100000

TARGET_UPDATE = 2000
NUM_EPISODES = 1000
MAX_STEPS = 2000

# -------- Action --------
def select_action(state, q_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return q_net(state).argmax(dim=1).item()
    
# -------- Logging --------
def save_result(seed, rewards, total_steps, total_time):
    record = {
        "seed": seed,
        "rewards": rewards,
        "steps": total_steps,
        "time": total_time,
        "steps_per_sec": total_steps / total_time,

        # hyperparams
        "GAMMA": GAMMA,
        "LR": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "BUFFER_SIZE": BUFFER_SIZE,
        "EPS_START": EPS_START,
        "EPS_END": EPS_END,
        "EPS_DECAY": EPS_DECAY,
        "TARGET_UPDATE": TARGET_UPDATE,
        "NUM_EPISODES": NUM_EPISODES,
        "MAX_STEPS": MAX_STEPS
    }

    with open(LOG_FILE, "a") as f:
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

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPS_START
    epsilon_decay = (EPS_START - EPS_END) / EPS_DECAY

    rewards = []
    total_steps = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            total_steps += 1

            action = select_action(state, q_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > BATCH_SIZE:
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
    save_result(seed, rewards, total_steps, total_time)
    return rewards

# -------- Multiprocessing wrapper --------
def run(seed):
    return train_dqn(seed)

# -------- Main --------
if __name__ == "__main__":
    start = time.time()

    num_workers = min(8, cpu_count())
    print(f"Using {num_workers} workers")

    with Pool(num_workers) as p:
        all_rewards = p.map(run, range(15))

    total_time = time.time() - start
    print(f"\nALL DONE in {total_time:.2f}s")