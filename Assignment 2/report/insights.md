Q2 :

The return curves show that the agent gradually improves from consistently poor returns to near-optimal performance, indicating that DQN successfully learns a stable strategy for reaching the goal. During early training, the car fails to reach the hill and oscillates near the valley, resulting in long episodes with low returns. As learning progresses, the agent discovers a more efficient trajectory and the variance across seeds decreases, showing convergence to a consistent policy.

The optimal behavior learned by the car is not to directly accelerate toward the target. Instead, the car first moves away from the goal (to the left), climbs the opposite hill, and then accelerates forward to build sufficient momentum to reach the right hill. If the accumulated momentum is still insufficient, the car oscillates back and forth—moving slightly forward, then backward again—until it gains enough velocity to reach the top. This behavior emerges because the engine power alone is insufficient to climb the hill directly; the agent must exploit gravitational potential energy by swinging to build speed.

Thus, the learned policy resembles a momentum-building swing strategy rather than greedy movement toward the goal. This is the optimal solution for the MountainCar dynamics, and visualizing the environment clearly shows the car repeatedly going backward before finally accelerating forward to reach the target. This matches the expected optimal behavior for the environment described in the assignment.

Q3 :

a) plots

b) 
From the plots, the truncation length strongly affects learning:
(a) Comparison of truncation = 200, 1000, 2000

Truncation = 200
Long flat region at −200 for ~300 episodes → agent never reaches goal.
Learning starts very late and improves slowly.
High variance and poor final performance.
Clearly insufficient horizon to discover the momentum strategy.

Truncation = 1000
Learning starts immediately.
Rapid improvement from very low returns.
However, the curve shows noticeable dips and higher variance even later in training.
Policy improves but remains less stable.

Truncation = 2000
Smooth and consistent improvement.
Much smaller variance after convergence
Stabilizes around the best return and stays there
Most stable and best-performing policy.

Increasing truncation length improves final performance,exploration and stability but may slow learning slightly due to longer unsuccessful episodes.

c)
Artificial truncation is introduced to:

prevent extremely long episodes when the agent fails
keep training computationally bounded
allow consistent comparison across algorithms
avoid replay buffer being dominated by very long failures

Without truncation, early training episodes could last extremely long.

d)
Truncation is harmful when too small because:

MountainCar requires multiple oscillations
short horizon prevents momentum buildup
agent repeatedly experiences failure
learning becomes delayed and unstable

From the plots:

200 → clearly harmful (late learning, poor convergence)
1000 → learns but unstable (visible dips and variance)
2000 → smooth convergence and stable performance

Therefore, 2000 timesteps is the most appropriate truncation length for fair evaluation.
It allows the agent enough time to build momentum while also producing stable learning and consistent final performance.

Q4:

a)
i. attach the two plots

ii. From the learning curves, all variants (ρ = 1, 2, 4, 8) eventually converge to similar final performance, but their learning speed and stability differ. Lower replay factors (ρ = 1, 2) improve faster initially, while higher replay factors (ρ = 4, 8) learn more slowly and show slightly larger variability early on.

ρ = 1 → fastest learning and best final performance
ρ = 2 → slightly slower, similar convergence
ρ = 4 → slower learning, similar final return
ρ = 8 → slowest learning and slightly worse stability

The final performance comparison plot shows that ρ = 1 achieves the best mean return, followed by ρ = 4, while ρ = 2 and ρ = 8 perform slightly worse. However, the 95% confidence intervals overlap substantially across all ρ values, indicating that the differences in final performance are not statistically significant. Thus, replay factor mainly affects learning dynamics rather than the final achievable policy.

iii. Increasing the replay factor means performing more gradient updates per environment step using the same replay buffer. This has two competing effects:

Positive effect-

Improves data reuse
Reduces sample complexity
Can extract more information from each transition

Negative effect-

Updates become less correlated with fresh data
Risk of overfitting to stale replay buffer samples
Slower adaptation to new policy distribution

Thus, increasing ρ does not improve final policy performance, but generally reduces learning speed. Sample efficiency in terms of environment interactions may improve, but computational cost increases.
