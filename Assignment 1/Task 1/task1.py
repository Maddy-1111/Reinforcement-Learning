import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 5
GAMMA = 0.3
THETA = 1e-6  

lake = (0, 0)
fire = (4, 4)
smoke_cells = {(1, 2), (3, 2)}
boulders = {(2, 4), (3, 4)}

actions = ["N", "S", "E", "W", "H"]  

states = []
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        for water in [0, 1]:
            states.append((x, y, water))

V = {s: 0.0 for s in states}                # Value function initialized to 0

def is_terminal(state):
    x, y, water = state
    if (x, y) in boulders:
        return True
    if (x, y) == fire and water == 1:
        return True
    return False

def move(x, y, action):
    if action == "N":
        return x , y + 1
    if action == "S":
        return x , y - 1
    if action == "E":
        return x + 1, y 
    if action == "W":
        return x - 1, y 
    return x, y  # Hover

def in_grid(x, y):return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def get_transitions(state, action):
    """
    Returns list of (probability, next_state, reward)
    """
    x, y, water = state

    if is_terminal(state):
        return [(1.0, state, 0)]   
    
    transitions = []

    if action == "H":
        next_state = (x, y, water)
        reward = compute_reward(state, next_state)
        return [(1.0, next_state, reward)]            

    if (x, y) in smoke_cells:
        p_intended = 0.4
        p_perp = 0.1
        p_stay = 0.4
    else:
        p_intended = 0.7
        p_perp = 0.1
        p_stay = 0.1

    nx, ny = move(x, y, action)
    if not in_grid(nx, ny):
        nx, ny = x, y
    next_water = water
    if (nx, ny) == lake:
        next_water = 1
    next_state = (nx, ny, next_water)
    transitions.append((p_intended, next_state,compute_reward(state, next_state)))

    if action in ["N", "S"]:
        perp_actions = ["E", "W"]
    else:
        perp_actions = ["N", "S"]

    for a in perp_actions:
        nx, ny = move(x, y, a)
        if not in_grid(nx, ny):
            nx, ny = x, y
        next_water = water
        if (nx, ny) == lake:
            next_water = 1
        next_state = (nx, ny, next_water)
        transitions.append((p_perp, next_state,compute_reward(state, next_state)))

    next_state = (x, y, water)
    transitions.append((p_stay, next_state,compute_reward(state, next_state)))

    return transitions

def compute_reward(state, next_state):
    x, y, water = next_state
    if (x, y) in boulders:
        return -100
    if (x, y) == fire and water == 1:
        return 100
    reward = -1
    if (x, y) in smoke_cells:
        reward -= 90
    return reward

def value_iteration():
    iteration=0
    history = {}
    while True:
        delta = 0
        new_V = V.copy()
        iteration += 1 
        for s in states:
            if is_terminal(s):
                continue
            action_values = []
            for a in actions:
                total = 0
                transitions = get_transitions(s, a)
                for prob, next_state, reward in transitions:
                    total += prob * (reward + GAMMA * V[next_state])
                action_values.append(total)

            best_value = max(action_values)
            new_V[s] = best_value

            delta = max(delta, abs(V[s] - best_value))

        V.update(new_V)
        if iteration in [1, 2]:
            history[iteration] = V.copy()
        print(f"Iteration {iteration}, delta = {delta:.6f}")
        if delta < THETA:
            history["final"] = V.copy()
            break

    print("Value iteration converged.", iteration)
    return history

def extract_policy():
    policy = {}

    for s in states:
        if is_terminal(s):
            policy[s] = None
            continue
        best_action = None
        best_value = -float("inf")
        for a in actions:
            total = 0
            transitions = get_transitions(s, a)
            for prob, next_state, reward in transitions:
                total += prob * (reward + GAMMA * V[next_state])
            if total > best_value:
                best_value = total
                best_action = a
        policy[s] = best_action
    return policy

def get_value_grid(V_dict, water):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            grid[y, x] = V_dict[(x, y, water)]
    return grid

def plot_all_snapshots(history):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    titles = [
        "Iter 1 - No Water",
        "Iter 2 - No Water",
        "Final - No Water",
        "Iter 1 - Has Water",
        "Iter 2 - Has Water",
        "Final - Has Water",
    ]

    snapshots = [
        history[1], history[2], history["final"],
        history[1], history[2], history["final"]
    ]

    waters = [0, 0, 0, 1, 1, 1]

    arrow_symbol = {
        "N": "↑",
        "S": "↓",
        "E": "→",
        "W": "←",
        "H": "•"
    }

    global V

    for ax, title, V_dict, water in zip(axes.flatten(), titles, snapshots, waters):

        V = V_dict
        policy = extract_policy()

        values = get_value_grid(V_dict, water)
        ax.imshow(values, cmap="coolwarm", origin="lower")

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                s = (x, y, water)
                value_text = f"{values[y, x]:.1f}"
                if policy[s] is None:
                    arrow_text = ""
                else:
                    arrow_text = arrow_symbol[policy[s]]

                ax.text(x, y,f"{value_text}\n{arrow_text}",ha="center",va="center",fontsize=9)

        ax.set_title(title)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    history = value_iteration()
    plot_all_snapshots(history)


