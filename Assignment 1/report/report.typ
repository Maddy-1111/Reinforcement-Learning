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
- *Action Space ($A$):* $A = {"North", "South", "East", "West", "Hover"}$. |A| = 5
- *Terminal States:* - *Crash:* Boulders at {(2,4), (3,4)}.
    - *Success:* Fire zone at (4,4) if $w=1$.
- *Discount Factor ($gamma$):* Initially $0.95$.

#text(size: 14pt, weight: "bold")[Transition and Reward Matrices]

The $0.7$ represents the intended move, $0.1$ for each perpendicular direction, and $0.1$ for staying in place from state (3,3).

Transition matrices and intended direction respectively -
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  rows: auto,
  gutter: 5pt,           
  align: center + horizon,     
  [North],[South],[West],[East],
  $ mat(0, 0.7, 0; 0.1, 0.1, 0.1; 0, 0, 0) $,
  $ mat(0, 0, 0; 0.1, 0.1, 0.1; 0, 0.7, 0) $,
  $ mat(0, 0.1, 0; 0.7, 0.1, 0; 0, 0.1, 0) $,
  $ mat(0, 0.1, 0; 0, 0.1, 0.7; 0, 0.1, 0) $
)
For Hover, its just deterministic so 1 at (3,3) and 0 everywhere else.

*Reward Matrix ($R$):*
$ mat(
  -100, -100, 100;
  -11, -1, -1;
  -1, -1, -1;
) $
Rewards include the $-1$ per-step penalty, $-10$ for hazardous smoke, and terminal values.
== Value Iteration ($gamma = 0.95$)

We initialise all V (state value functions) as 0.

The  value iteration algorithm converges after 118 iterations.

Top 3 grids show the first, second and final iterations for the case of no water.

Bottom 3 grids are in the state with water towards the fire zone.

As represented by the heat gradient, the state value function is stronger near the terminal state.

The arrows represent the optimal policy (greedy with respect to the V - one step lookahead)
#image("Optimal_MDP.png", width: 100%,)
== Impact of Discount Factor ($gamma = 0.3$)
1. *Value Function & Policy:* 
- Convergence is much faster (approx. 15 iterations) but the agent becomes extremely short-sighted. 
- For states far from terminal reward, the +100 at the fire is heavily discounted and only immediate step costs matter. So the agent keeps receiving −1 indefinitely, the value function becomes: -1/(1-$gamma$)=-1.43 (for all states except a few clsoe to the terminal state)
- Far from the fire, the policy is less strongly oriented and the gradient toward the goal becomes flatter. We see only 3-4 states near the fire state with a strong value.

#image("MDP_0.3gamma.png", width: 100%,)
2. *Hazardous States:* 
- Since $gamma$ is low, the immediate $-10$ penalty of smoke outweighs the future $+100$ goal reward. The drone exhibits "avoidance" behavior even when the goal is near and takes longer routes.
- This also leads to the greedy policy preferring to go out of the grid or just hover around in some states near the hazards.
3. *Hovering:*
- Hovering yields a deterministic reward of −1 with no risk of entering hazardous or crash states. 
- If all movement actions carry a non-negligible probability of drifting into smoke (−10) or boulders (−100), the expected reward of moving may be worse than −1. This effect is more pronounced when γ = 0.3, since the agent heavily discounts future rewards and this leads to few hover states in the final value function grid.
4. *Stuck States:*
- We observed loops in the top-left corner because the value gradient is too flat to guide the drone towards the lake/fire zone.
- We also observe, some arrows going out of the grid. In general, the policy prioritizes avoiding immediate penalties rather than aggressively moving toward the terminal state.

== High Penalty and Strong Wind
1. *Hazardous Avoidance:* 
- With a $-90$ penalty, there is a clear “negative basin” or depression around hazardous regions. The negative values are much stronger and in a wider region compared to -10 penalty case.
- It also takes more than double the iterations to converge compared to $gamma$=0.95.
- The optimal policy remains globally goal-directed but deviates strongly near hazardous regions, preferring longer and safer paths instead of the shortest route. 
- In tight regions between smoke and boulders, hovering may become optimal.
- Similar to $gamma$ = 0.3 case, some loops and out of the grid arrows are observed in the policy.
#image("MDP_90penalty.png", width: 100%,)
2. *Hovering Preference:* 
- Hovering is preferred in the cell between a boulder and smoke.
- Since entering smoke yields −91 and boulder -100, even a small probability drastically lowers the expected reward of moving. Hovering gives a deterministic −1 with no risk, so it can be optimal in tight regions near hazards.
3. *Longer Path Preference:* 
- If the shortest path passes near smoke, the expected penalty due to stochastic drift can outweigh the benefit of fewer steps. Because γ = 0.95, the agent values long-term return and thus prefers a longer but safer path that avoids large expected penalties.
4. *Strong Wind:* 
- With stronger wind, the probability of unintended sideways movement increases to 50%, greatly amplifying the risk of entering hazardous cells. 
- Since the hazard penalty is −90, even a 25% chance of drifting into smoke leads to a very large expected penalty. 
- As a result, the effective influence of hazardous cells expands to a much larger region of the grid. 
- The optimal policy becomes highly conservative, with more states choosing to hover or move outward at boundaries to reduce risk. Even near the +100 terminal state, some actions avoid direct movement due to high stochastic risk. 
- Overall, the policy is dominated by strong hazard avoidance rather than shortest-path goal seeking.

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