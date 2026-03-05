#set page(paper: "a4", margin: (x: 2cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt) // Standard LaTeX look
#set heading(numbering: "1.1")
#set page(numbering: "1")

#show figure.caption: it => [
  #text(size: 0.8em)[#it]
]

#text(size: 18pt, weight: "bold")[Programming Assignment 1 (BONUS q answered)] \
#text(size: 14pt)[DA6400: Jan-May 2026]
#line(length: 100%)

= Team Details
#align(center)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    align: center,
    [Kevin Kinsey S \ EP23B027], [Madhav Tadepalli \ EE23B040], [Soumya Lodha \ EE23B140]
  )
]

== Work Split
The work split was even across both tasks, with each team member contributing to both the MDP formulation and the TD-based control implementation. To minimize code conflicts, we did all the work while sitting together ideating and coding. Naturally we did some small splits over the cells in the notebook, but since it was all to a specific task so it was easy to merge and review each other's code. Hence, we don't have a specific breakdown of who did what, but we all contributed to the entire codebase and report writing collaboratively.

== Links
Drive: sdfngkljsbkhg

GitHub: ndsbfgkfjg

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
For Hover, its just deterministic so 1 at (3,3) and 0 everywhere else:
$ mat(0, 0, 0; 0, 1, 0; 0, 0, 0) $

*Reward Matrix ($R$):*
$ mat(
  -100, -100, 100;
  -11, -1, -1;
  -1, -1, -1;
) $
Rewards include the $-1$ per-step penalty, $-10$ for hazardous smoke, and $0$ terminal values (which happen to not be present in the grid since its centred at $(3, 3)$).
== Value Iteration ($gamma = 0.95$)

- We initialise all V (state value functions) as 0.
- The  value iteration algorithm converges after 118 iterations.
- Top 3 grids show the first, second and final iterations for the case of no water.
- Bottom 3 grids are in the state with water towards the fire zone.
- As represented by the heat gradient, the state value function is stronger near the terminal state.
- The arrows represent the optimal policy (greedy with respect to the V - one step lookahead)
#figure(image("Optimal_MDP.png", width: 100%,), caption: [$gamma$ = 0.95 Optimal Value Function and Policy])
== Impact of Discount Factor ($gamma = 0.3$)
a) *Value Function & Policy:* 
- Convergence is much faster (approx. 15 iterations) but the agent becomes extremely short-sighted. 
- For states far from terminal reward, the +100 at the fire is heavily discounted and only immediate step costs matter. So the agent keeps receiving −1 indefinitely, the value function becomes: -1/(1-$gamma$)=-1.43 (for all states except a few clsoe to the terminal state)
- Far from the fire, the policy is less strongly oriented and the gradient toward the goal becomes flatter. We see only 3-4 states near the fire state with a strong value.

#figure(image("MDP_0.3gamma.png", width: 100%,), caption: [$gamma$ = 0.3 Optimal Value Function and Policy])

b) *Hazardous States:* 
- Since $gamma$ is low, the immediate $-10$ penalty of smoke outweighs the future $+100$ goal reward. The drone exhibits "avoidance" behavior even when the goal is near and takes longer routes.
- This also leads to the greedy policy preferring to go out of the grid or just hover around in some states near the hazards.

c) *Hovering:*
- Hovering yields a deterministic reward of −1 with no risk of entering hazardous or crash states. 
- If all movement actions carry a non-negligible probability of drifting into smoke (−10) or boulders (−100), the expected reward of moving may be worse than −1. This effect is more pronounced when γ = 0.3, since the agent heavily discounts future rewards and this leads to few hover states in the final value function grid.

d) *Stuck States:*
- We observed loops in the top-left corner because the value gradient is too flat to guide the drone towards the lake/fire zone.
- We also observe, some arrows going out of the grid. In general, the policy prioritizes avoiding immediate penalties rather than aggressively moving toward the terminal state.

== High Penalty and Strong Wind
a) *Hazardous Avoidance:* 
- With a $-90$ penalty, there is a clear “negative basin” or depression around hazardous regions. The negative values are much stronger and in a wider region compared to -10 penalty case.
- It also takes more than double the iterations to converge compared to $gamma$=0.95.
- The optimal policy remains globally goal-directed but deviates strongly near hazardous regions, preferring longer and safer paths instead of the shortest route. 
- In tight regions between smoke and boulders, hovering may become optimal.
- Similar to $gamma$ = 0.3 case, some loops and out of the grid arrows are observed in the policy.
#figure(image("MDP_90penalty.png", width: 100%,), caption: [$gamma$ = 0.95, Penalty = -90])

b) *Hovering Preference:* 
- Hovering is preferred in the cell between a boulder and smoke.
- Since entering smoke yields −91 and boulder -100, even a small probability drastically lowers the expected reward of moving. Hovering gives a deterministic −1 with no risk, so it can be optimal in tight regions near hazards.

c) *Longer Path Preference:* 
- If the shortest path passes near smoke, the expected penalty due to stochastic drift can outweigh the benefit of fewer steps. Because γ = 0.95, the agent values long-term return and thus prefers a longer but safer path that avoids large expected penalties.

d) *Strong Wind:* 
- With stronger wind, the probability of unintended sideways movement increases to 50%, greatly amplifying the risk of entering hazardous cells. 
- Since the hazard penalty is −90, even a 25% chance of drifting into smoke leads to a very large expected penalty. 
- As a result, the effective influence of hazardous cells expands to a much larger region of the grid. 
- The optimal policy becomes highly conservative, with more states choosing to hover or move outward at boundaries to reduce risk. Even near the +100 terminal state, some actions avoid direct movement due to high stochastic risk. 
- Overall, the policy is dominated by strong hazard avoidance rather than shortest-path goal seeking.

#figure(image("MDP_highdrift.png", width: 100%,), caption: [Strong Wind Effect on Optimal Value Function and Policy])

= TD-based Control in Acrobot

== Code for SARSA and Q-Learning
The `Task 2.ipynb` file contains the full implementation of both SARSA and Q-Learning algorithms for the Acrobot environment.
The code is structured to allow easy modification of hyperparameters and reward functions.
It also has an extra set of cells in the end to visualise the algorithms (through the lib);

Additonally the `random_seed_sim.py` file contains the code for running multiple simulations with different random seeds to analyze the stability and variance of the learned policies.
And the `hyperparam_sweep.py` file contains the code for performing a grid search over hyperparameters (alpha, epsilon) to find the best performing configurations for both algorithms.

We've also included `keyboard_acrobot.py` which allows manual control of the Acrobot environment using keyboard inputs; this code was utilised for fun to verify our understanding of the environment and compare with visualisations presented in the notebook.

== Algorithm Comparison | SARSA vs. Q-Learning
=== *Hyperparameter Sweep*
We performed a grid search over learning rates (alpha from $[0.01, 0.05, 0.1, 0.2, 0.5]$) and exploration rates (epsilon from $[0.005, 0.01, 0.05, 0.1, 0.2, 0.3]$) for both algorithms.

==== SARSA
#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    figure(image("SARSA_hyperparam_plot.png", width: 100%), caption: "SARSA Hyperparameter Sweep - Return vs Episode"),
    figure(image("SARSA_hyperparam_hm.png", width: 100%), caption: "SARSA Hyperparameter Sweep - Heatmap"),
  )
]

The top three hyperparameter configurations ($alpha, epsilon$) resulting in the highest mean returns across the sweep are:

#align(center)[
  #table(
      columns: (auto, auto, auto),
      inset: 7pt,
      fill: (x, y) => if y == 0 { luma(230) } else { white },
      align: center + horizon,
      [*Configuration ($alpha, epsilon$)*], [*Mean Return*], [*Return Variance*],
      [(0.2, 0.005)], [-253.30], [6502.18],
      [(0.1, 0.005)], [-268.74],[8227.71],
      [(0.05, 0.2)], [-279.53],[7087.36],
)
]

From the plots, its generally seen that a lower epsilon (more exploitation) generally leads to better returns, and moderate alpha values (0.05 to 0.2) perform best. Higher epsilon values lead to poor performance due to excessive randomness in action selection.

With regards to the variance, we see that the best performing configurations also have relatively lower variance compared to other configurations, which suggests that they are more stable across different runs.

And the alpha values that are too high (e.g., 0.5) lead to higher variance and worse performance, likely due to the agent making large updates to the Q-values which can destabilize learning.

==== Q-Learning
#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    figure(image("QL_hyperparam_plot.png", width: 100%), caption: "Q-Learning Hyperparameter Sweep - Return vs Episode"),
    figure(image("QL_hyperparam_hm.png", width: 100%), caption: "Q-Learning Hyperparameter Sweep - Heatmap"),
  )
]

The top three hyperparameter configurations ($alpha, epsilon$) resulting in the highest mean returns across the sweep are:

#align(center)[
  #table(
      columns: (auto, auto, auto),
      inset: 7pt,
      fill: (x, y) => if y == 0 { luma(230) } else { white },
      align: center + horizon,
      [*Configuration ($alpha, epsilon$)*], [*Mean Return*], [*Return Variance*],
      [(0.05, 0.005)], [-291.82], [8950.55],
      [(0.2, 0.005)], [-295.19],[6672.54],
      [(0.05, 0.01)], [-296.07],[9044.44],
)
]
As seen in the plots, Q-Learning also shows similar trends in terms of hyperparameter sensitivity, with lower epsilon values generally leading to better returns. However, Q-Learning has slightly better mean returns in the top configurations.

The alpha values that are too high (e.g., 0.5) also lead to higher variance and worse performance in Q-Learning, similar to SARSA.

==== SARSA vs. Q-Learning
- Both algorithms show similar trends in terms of hyperparameter sensitivity, with lower $epsilon$ (more exploitation) generally leading to better returns, and moderate $alpha$ values (0.05 to 0.2) performing best.
- Q-Learning has slightly better mean returns in the top configurations, but also higher variance compared to SARSA, which is expected since Q-Learning is an off-policy method and can be more unstable.
- The heatmaps show that both algorithms are sensitive to the choice of $epsilon$, with very high exploration rates ($epsilon > 0.1$) leading to poor performance due to excessive randomness in action selection.

Now, we try out the 10 random seed simulations for the best performing hyperparameters for both algorithms to analyze stability and variance in returns.

#figure(image("ten_seed_run_comp.png", width: 90%), caption: "Ten Seed Run Comparison")

As seen from the plot, both algorithms show significant variance across different random seeds, with some runs achieving much higher returns than others.

The results suggest that while the best hyperparameter configurations can lead to good performance on average, there is still a lot of variability in the learning process due to the stochastic nature of the environment and the algorithms.

== Epislon Decay Effect on Online & Offline Learning
- We implemented an epsilon decay schedule where $epsilon$ starts at 1.0 and decays exponentially to a minimum value over the course of training.
- This allows the agent to explore more in the early stages and gradually shift towards exploitation as it learns.
- The plot below compares the returns with and without epsilon decay for both algorithms.
#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    figure(image("SARSA_decay.png", width: 100%), caption: "SARSA Returns with Epsilon Decay"),
    figure(image("QL_decay.png", width: 100%), caption: "Q-Learning Returns with Epsilon Decay"),
  )
]
- Seen from the graph point of view, epsilon decay results look quite similar to the best fixed epsilon runs. (from viewpoint of the EMA smoothed returns)
- However, the raw returns show that the decay runs have much higher variance in the early stages due to high exploration, but eventually converge to similar performance as the best fixed epsilon runs. This suggests that while epsilon decay can help with exploration, it may not always lead to better final performance compared to well-tuned fixed epsilon values.

== Bin Size Effect on State Discretization
- We experimented with different bin sizes for discretizing the continuous state space of the Acrobot environment. The bin size determines how finely we partition the state space, which can impact the learning process.

#figure(image("bin_sweep.png", width: 80%), caption: "Bin Size Effect on SARSA Returns")

- From the graph we can clearly see 2 major trends:
1. *Smaller Bin Sizes (e.g., 5):* Lead to faster initial learning as the agent receives more granular feedback on its state. However, it can also lead to overfitting to specific state-action pairs and may struggle to generalize, resulting in higher variance in returns.
2. *Larger Bin Sizes (e.g., 20):* Provide a more generalized representation of the state space, which can help with stability and reduce variance. However, it may also lead to slower learning as the agent receives less specific feedback, especially in the early stages of training

#pagebreak()
== Modified Reward Analysis (Theoretical)
#v(1em)
Given: $r = frac(eta h, 2) + "sign"(-1 + eta h) (frac(2 - eta h, 2))$ 

#figure(image("mod_reward_f.png", width: 80%), caption: "Modified Reward Function for Different Eta Values")

1. *Learning Speed:* This reward function is "dense" compared to the original sparse $-1$ reward. It should lead to faster initial learning as the agent receives immediate feedback on its height $h$.
2. *$eta = 0.5$ Case:* This scales the height feedback down. The agent might "dither" at lower heights because the reward signal is too weak to overcome the constant penalty of time.
3. *$eta in {1, 5, 100}$:* Higher $eta$ values make the agent aggressive. However, if $eta$ is too high, the agent might prioritize reaching a "safe" high altitude over actually swinging past the goal line.
4. *Reward Design Insights:* Dense rewards prevent "vanishing gradients" in RL but must be carefully shaped to ensure the agent doesn't find a "local optima" (e.g., just staying high without finishing).

#line(length: 100%)

*Note:* This concludes the proper part of the assignment. The rest of the sections are to answer the BONUS questions and are not mandatory. Hence, it exceeds the page limit, but we have included it for completeness and to demonstrate our understanding of the concepts.
#pagebreak()

=== BONUS section: Implenentation of Modified Reward
We implemented the modified reward function in the the provided notebook itself.We've tried for different values of $eta$ and observed the general shape of the learning curves.

#figure(image("eta_sweep_sep.png", width: 80%), caption: "Modified Reward Function Effect on SARSA Returns")

Since, the values of $eta$ being different, the scale of returns also differ significantly. So we can only compare the general shape of the learning curves and not the absolute return values.
- For $eta = 0.5$, the learning curve is very flat and shows very slow improvement, which aligns with our theoretical analysis that the reward signal is too weak.
- For $eta = 1$, the learning curve shows steady improvement, indicating that the agent is able to learn effectively with this reward shaping.
- For $eta = 5$ and $eta = 100$, the learning curves show rapid initial improvement but then plateau, suggesting that the agent may be getting stuck in a local optimum of just trying to stay at a high altitude rather than learning to swing past the goal line.

To compare the learning curves more clearly, we run the resultant Q Table for each $eta$ value under a offline evaluation mode (greedy policy) and observed the returns.

#align(center)[
  #table(
      columns: (auto, auto, auto),
      inset: 7pt,
      fill: (x, y) => if y == 0 { luma(230) } else { white },
      align: center + horizon,
      [*$eta$*], [*Mean Return*], [*Return Variance*],
      [0.5], [-348.6580 ], [7159.8590],
      [1], [-318.4385],[9361.0912],
      [5], [-250.6370],[6834.4422],
      [100], [-308.9750],[7815.7484],
)
]

As seen from the table, $eta = 5$ has the best mean return. It seems than $eta = 5$ provides a strong enough reward signal to encourage the agent to learn effectively, while not being so high that it leads to suboptimal behavior of just trying to stay at a high altitude. Hence, it seems like higher $eta$ is beneficial up to a certain point, but excessively high values can lead to unintended consequences in the learning process as observed in the $eta = 100$ case.

== BONUS\#2: Visualisation
=== "Lil' Fun" Section
At the end of our notebook, we added a fun section where we visualise the learned policies of both SARSA and Q-Learning on the Acrobot environment. We used the `visualize_policy` function to create animations of the Acrobot swinging up and balancing based on the learned Q-values. Running it in the actually cell shows a video of the Acrobot performing the task according to the learned policy. This was a great way to qualitatively assess the behavior of the agent and see how well it has learned to control the Acrobot.

=== Keyboard Control Interface
We implemented a simple keyboard control interface for the Acrobot environment using the `keyboard_acrobot.py` file. This allows us to manually control the two joints of the Acrobot using arrow keys and observe the resulting motion in the environment.
- The left and right arrow keys control the first joint, while the up and down arrow keys control the second joint.
- This manual control helped us gain a better intuitive understanding of the dynamics of the Acrobot and the challenges involved in learning to swing up.

#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    figure(image("policy_visual.png", width: 100%), caption: "SARSA Learned Policy Visualization"),
    figure(image("keyboard_visual.png", width: 72%), caption: "Keyboard Control Interface for Acrobot")
  )
]