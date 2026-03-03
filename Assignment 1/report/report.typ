#set page(paper: "a4", margin: (x: 2cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt) // Standard LaTeX look
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 18pt, weight: "bold")[Programming Assignment 1: Reinforcement Learning] \
  #text(size: 14pt)[DA6400: Jan-May 2026] \
  #v(1em)
  #grid(
    columns: (1fr, 1fr, 1fr),
    align: center,
    [Kevin Kinsey S \ EP23B027], [Madhav Tadepalli \ EE23B040], [Soumya Lodha \ EE23B140]
  )
  #v(1em)
  #line(length: 100%)
]

= Gridworld and Value Iteration

== MDP Formulation
We define the drone navigation problem as an MDP tuple $(S, A, P, R, gamma)$:

- *State Space ($S$):* Represented as $(x, y, w)$, where $x, y in {0, 1, 2, 3, 4}$ are grid coordinates and $w in {0, 1}$ indicates water status. Total $|S| = 50$.
- *Action Space ($A$):* $A = {"North", "South", "East", "West", "Hover"}$.
- *Terminal States:* - *Crash:* Boulders at {(2,4), (3,4)}.
    - *Success:* Fire zone at (4,4) if $w=1$.
- *Discount Factor ($gamma$):* Initially $0.95$.

== Transition and Reward Matrices
For cell (3,3) and action $a = "South"$:

*Transition Matrix ($P$):*
$ mat(
  0, 0, 0;
  0.1, 0.1, 0.1;
  0, 0.7, 0;
) $
The $0.7$ represents the intended move, $0.1$ for each perpendicular direction (East/West), and $0.1$ for staying in place due to motor breaks.

*Reward Matrix ($R$):*
$ mat(
  -100, -100, 100;
  -11, -1, -1;
  -1, -1, -1;
) $
Rewards include the $-1$ per-step penalty, $-10$ for hazardous smoke, and terminal values.

== Impact of Discount Factor ($gamma = 0.3$)
1. *Value Function & Policy:* Convergence is much faster (approx. 15 iterations) but the agent becomes extremely short-sighted. Values for distant states settle at $V approx frac(-1, 1 - 0.3) = -1.43$.
2. *Hazardous States:* Since $gamma$ is low, the immediate $-10$ penalty of smoke outweighs the future $+100$ goal reward. The drone exhibits "avoidance" behavior even when the goal is near.
3. *Hovering:* The agent prefers hovering near hazards because the risk of drifting into a boulder (-100) or smoke (-11) is too high relative to the heavily discounted future reward.
4. *Stuck States:* We observed loops in the top-left corner because the value gradient is too flat to guide the drone towards the lake/fire zone.

== High Penalty and Strong Wind
1. *Hazardous Avoidance:* With a $-90$ penalty, the drone takes significantly longer paths to maintain a safety buffer from smoke cells.
2. *Hovering Preference:* Hovering is preferred in the cell between a boulder and smoke, as any movement carries a high probability of a massive penalty.
3. *Strong Wind (40/25/35):* The increased perpendicular drift makes the center of the grid a "death trap." Most states near the middle now favor hovering or moving toward the boundaries to minimize the risk of being pushed into hazards.

= TD-based Control in Acrobot

== Environment and Discretization
The Acrobot-v1 state space is continuous.



== Algorithm Comparison
1. *SARSA vs. Q-Learning:* Q-Learning (off-policy) converges faster to the optimal $V^*$ but shows more "dips" during training. SARSA (on-policy) is more stable as it accounts for the actual exploration moves taken.
2. *Hyperparameters:* We found that a decaying $epsilon$ schedule ($1.0 arrow 0.1$) is necessary. A constant $epsilon$ leads to sub-optimal policies that never stop "chancing" bad moves.
3. *Binning Granularity:* Increasing bins to 20 improves the policy but exponentially increases the state space, leading to the "curse of dimensionality" and requiring significantly more episodes to visit all states.

== Modified Reward Analysis (Theoretical)
Given: $r = frac(eta h, 2) + "sign"(-1 + eta h) (frac(2 - eta h, 2))$ 

1. *Learning Speed:* This reward function is "dense" compared to the original sparse $-1$ reward. It should lead to faster initial learning as the agent receives immediate feedback on its height $h$.
2. *$eta = 0.5$ Case:* This scales the height feedback down. The agent might "dither" at lower heights because the reward signal is too weak to overcome the constant penalty of time.
3. *$eta in {1, 2, 5}$:* Higher $eta$ values make the agent aggressive. However, if $eta$ is too high, the agent might prioritize reaching a "safe" high altitude over actually swinging past the goal line. We recommend $eta in [1, 2]$ to balance height-seeking with goal-reaching.
4. *Reward Design Insights:* Dense rewards prevent "vanishing gradients" in RL but must be carefully shaped to ensure the agent doesn't find a "local optima" (e.g., just staying high without finishing).

#pagebreak()
= Appendix: Code Reproducibility
The `task1.py` file contains the full implementation of the Gridworld MDP and Value Iteration. Requirements are listed in `requirements.txt`.