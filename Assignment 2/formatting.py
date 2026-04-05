import json
import re

# File paths
CSV_FILE = "rho8.csv"
LOG_JSONL = "results.jsonl"

# Fixed Hyperparameters for this batch
hyperparams = {
    "rho": 8,
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "batch_size": 128,
    "buffer_size": 20000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay": 100000,
    "target_update": 2000,
    "max_steps": 2000,
    "num_episodes": 1000,
    "hidden_size": 64
}

def parse_csv_to_jsonl():
    all_seeds_data = []
    current_rewards = []
    
    # Regex to pull numbers from "Episode X, Reward: Y, Epsilon: Z"
    pattern = re.compile(r"Episode (\d+), Reward: ([-+]?\d*\.\d+|\d+)")

    with open(CSV_FILE, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                episode_num = int(match.group(1))
                reward = float(match.group(2))
                
                # If we hit Episode 0 and already have data, save the previous seed
                if episode_num == 0 and current_rewards:
                    all_seeds_data.append(current_rewards)
                    current_rewards = []
                
                current_rewards.append(reward)
        
        # Append the very last seed processed
        if current_rewards:
            all_seeds_data.append(current_rewards)

    # Append to JSONL
    with open(LOG_JSONL, "a") as f:
        for i, rewards in enumerate(all_seeds_data):
            record = {
                "label": "soumya_Q4_a",
                "seed": i,  # Assigning index as seed ID
                "hyperparams": hyperparams,
                "rewards": rewards
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Successfully appended {len(all_seeds_data)} seeds to {LOG_JSONL}")

if __name__ == "__main__":
    parse_csv_to_jsonl()