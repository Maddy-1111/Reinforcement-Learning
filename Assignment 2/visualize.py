import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import imageio
import cv2

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def draw_text(img, text, x, y, font_scale=0.5):
    """Draws small, full-black text."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 0, 0), 1, cv2.LINE_AA)

def run_2x2_grid(model_path):
    device = torch.device("cpu")
    seeds = [0, 19, 42, 123]
    envs = [gym.make("MountainCar-v0", render_mode="rgb_array") for _ in range(4)]
    
    model = QNetwork(2, 3, hidden_size=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    states = [env.reset(seed=seeds[i])[0] for i, env in enumerate(envs)]
    dones = [False] * 4
    rewards = [0] * 4
    frames = []

    print("Generating 2x2 grid video...")

    while not all(dones):
        combined_rows = []
        for i in range(4):
            if not dones[i]:
                st = torch.from_numpy(states[i]).float().unsqueeze(0)
                with torch.no_grad():
                    action = model(st).argmax(dim=1).item()
                states[i], r, term, trunc, _ = envs[i].step(action)
                rewards[i] += r
                dones[i] = term or trunc

            # Get frame, draw border, and text
            frame = envs[i].render().copy()
            # Draw border
            cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0,0,0), 5)
            # Draw Seed and Score
            draw_text(frame, f"Seed: {seeds[i]}", 10, 20)
            draw_text(frame, f"Score: {int(rewards[i])}", 10, 40)
            if seeds[i] == 19:
                draw_text(frame, "Intentionally chosen unlucky seed", 10, 60, font_scale=0.4)
            combined_rows.append(frame)

        # Create 2x2 grid
        row1 = np.concatenate((combined_rows[0], combined_rows[1]), axis=1)
        row2 = np.concatenate((combined_rows[2], combined_rows[3]), axis=1)
        grid = np.concatenate((row1, row2), axis=0)
        frames.append(grid)

    imageio.mimsave("grid_comparison.mp4", frames, fps=30)
    print("Video saved as grid_comparison.mp4")
    [e.close() for e in envs]

if __name__ == "__main__":
    run_2x2_grid("model_seed_0.pth")