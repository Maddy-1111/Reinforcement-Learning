import numpy as np

GRID_SIZE = 5
GAMMA = 0.95
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

# Value function initialized to 0
V = {s: 0.0 for s in states}

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
        reward -= 10
    return reward

def value_iteration():
    iteration=0
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
        print(f"Iteration {iteration}, delta = {delta:.6f}")
        if delta < THETA:
            break

    print("Value iteration converged.", iteration)
    return V

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

V = value_iteration()
policy = extract_policy()

print("Optimal value at start (0,0,0):", V[(0, 0, 0)])

import matplotlib.pyplot as plt
import numpy as np

# Convert value dictionary into 5x5 grid for given water phase
def get_value_grid(water):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            grid[y, x] = V[(x, y, water)]
    return grid


# Convert policy dictionary into arrow symbols
def get_policy_grid(water):
    grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            s = (x, y, water)

            if (x, y) in boulders:
                grid[x, y] = "B"
            elif (x, y) == lake:
                grid[x, y] = "L"
            elif (x, y) == fire:
                grid[x, y] = "F"
            elif s in policy and policy[s] is not None:
                action = policy[s]
                arrows = {
                    "N": "↑",
                    "S": "↓",
                    "E": "→",
                    "W": "←",
                    "H": "•"
                }
                grid[y, x] = arrows[action]
            else:
                grid[x, y] = ""

    return grid


def plot_phase(water, title):
    values = get_value_grid(water)
    policy_grid = get_policy_grid(water)

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(values, cmap="coolwarm", origin="lower")

    # Add numbers and arrows
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            text = f"{values[i,j]:.1f}\n{policy_grid[i,j]}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="black", fontsize=10)

    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.set_title(title)
    plt.colorbar(im)
    plt.show()


# ==========================
# Plot both phases
# ==========================

plot_phase(water=0, title="Phase 1: No Water (Navigate to Lake)")
plot_phase(water=1, title="Phase 2: Has Water (Navigate to Fire)")