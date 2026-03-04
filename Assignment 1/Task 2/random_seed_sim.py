import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm
import joblib

episodes = 3000

def make_bins(num_bins):
    low = np.array([-1, -1, -1, -1, -4 * np.pi, -9 * np.pi])
    high = np.array([1, 1, 1, 1, 4 * np.pi, 9 * np.pi])
    return [np.linspace(l, h, num_bins - 1) for l, h in zip(low, high)]

def discretize(obs, bins):
    return tuple(np.digitize(obs[i], bins[i]) for i in range(6))

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(3)
    return np.argmax(Q[state])

def rl_agent(params):
    """
    params: (alpha, epsilon)
    Returns: (alpha, epsilon, mean_final_reward)
    """
    global episodes
    alpha, epsilon, algo, seed = params
    
    # Task Hyperparameters
    gamma = 0.99
    max_steps = 500
    num_bins = 10
    eta = 0.0
    
    env = gym.make("Acrobot-v1")
    bins = make_bins(num_bins)
    Q = np.zeros((num_bins + 1,)*6 + (3,), dtype=np.float32)
    returns = []


    for ep in range(episodes):
        obs, _ = env.reset(seed=seed)
        state = discretize(obs, bins)

        action = epsilon_greedy(Q, state, epsilon)

        total_reward = 0

        for _ in range(max_steps):

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_next = discretize(obs_next, bins)
            action_next = epsilon_greedy(Q, state_next, epsilon)
    
            if algo == 'QLearning':
                Q[state][action] += alpha * (reward + gamma * np.max(Q[state_next]) - Q[state][action])
            elif algo == 'SARSA':
                Q[state][action] += alpha * (reward + gamma * Q[state_next][action_next] - Q[state][action])

            state = state_next
            action = action_next
            total_reward += reward

            if done:
                break

        returns.append(total_reward)
    
    env.close()
    return returns

if __name__ == "__main__":
    algo_fitted_params = {
        'SARSA': (0.2, 0.005),
        # 'QLearning': (0.05, 0.005),
        'QLearning': (0.2, 0.005),
    }


    for algo in algo_fitted_params.keys():
        print(f"\nRunning 10 seed simulations for fitted {algo}...")

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(rl_agent, (*algo_fitted_params[algo], algo, i)) for i in range(10)]
            
            all_data = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                res = future.result() # (alpha, eps, mean_r, returns, Q)

                all_data.append(res)
    
        joblib.dump(all_data, f'./outputs/random_seeds_{algo}.pkl')
        print("=" * 50)
    
    
