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
    alpha, epsilon, algo = params
    
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

        obs, _ = env.reset()
        state = discretize(obs, bins)

        action = epsilon_greedy(Q, state, epsilon)

        total_reward = 0

        for _ in range(max_steps):

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_next = discretize(obs_next, bins)
            action_next = epsilon_greedy(Q, state_next, epsilon)

            # Modified Reward (eta=0 -> reward = -1)
            c1, s1, c2, s2 = obs_next[:4]
            h = -c1 - (c1*c2 - s1*s2)

            reward = (eta*h)/2 + np.sign(-1 + eta*h) * ((2 - eta*h)/2)

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
    return (alpha, epsilon, np.mean(returns), returns, Q)

def rl_agent_offline(Q, episodes):
    max_steps = 500
    num_bins = 10
    
    env = gym.make("Acrobot-v1")
    bins = make_bins(num_bins)
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs, bins)
        action = np.argmax(Q[state])
        total_reward = 0

        for _ in range(max_steps):
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = discretize(obs_next, bins)
            action = np.argmax(Q[state])

            total_reward += reward
            if done: break
        
        returns.append(total_reward)
    
    env.close()
    return np.mean(returns), np.var(returns)

if __name__ == "__main__":
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    epsilons = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
    algos = ['SARSA', 'QLearning']
    param_grid = list(product(alphas, epsilons))

    for algo in algos:
        print(f"\nRunning Hyperparameter Sweep for {algo}...")
        print(f"Starting Sweep on {len(param_grid)} configurations...")

        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(rl_agent, (*p, algo)) for p in param_grid]
            
            all_data = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                res = future.result() # (alpha, eps, mean_r, returns, Q)
                
                offline_mean, offline_var = rl_agent_offline(res[4], 1000)
                
                all_data.append({
                    'alpha': res[0],
                    'epsilon': res[1],
                    # 'mean_reward': res[2],
                    'returns': res[3],
                    # 'Q_table': res[4],
                    'offline_mean': offline_mean,
                    'offline_var': offline_var
                })
    
        joblib.dump(all_data, f'./outputs/sweep_results_{algo}.pkl')
        print("=" * 50)
    
    
