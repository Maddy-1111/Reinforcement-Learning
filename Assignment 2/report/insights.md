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

iv. 
Advantageous when:
-Environment interaction is expensive
-Simulation is slow or real-world data collection is costly
-Computation is cheap relative to data

Example: robotics training on real hardware
Collecting a single trajectory requires physical movement and time. Using larger ρ allows the agent to reuse collected data more extensively and learn with fewer real-world interactions.

Disadvantageous when:
-Environment interaction is cheap (simulation)
-Real-time learning is required
-Compute budget is limited
-Overfitting to stale data harms adaptation

Example: online recommendation system
User behavior changes quickly. High ρ causes the model to over-train on old interactions, reducing adaptability to new user preferences.

v. Planning is any computational process that uses a model (transition probabilities and rewards) of the environment to improve policies - you don't need real environment interaction, just use data from some simulations to estimate how the environment would behave.

Here, in the case that replay Factor>1, we get only one new update, but we continue sampling rho(more than 1) times each step - meaning we are reusing older transitions to learn about current environment. 
thus DQN with ρ > 1 resembles planning because multiple updates are performed using stored transitions without additional environment interaction. Although no explicit model is learned, the replay buffer serves as an implicit sample-based model of the environment. Hence, increasing ρ increases the amount of computation-based policy improvement, similar to planning methods.

b)
i. The performance distributions for all replay factors appears unimodal,indicating that across seeds the algorithm typically converges to one consistent level of performance rather than distinct good/bad modes.

The distributions are approximately bell-shaped, but not perfectly Gaussian. They show slight skewness, particularly for higher replay factors (ρ = 4 and ρ = 8), which have longer tails toward lower performance values. This suggests that larger replay factors occasionally produce worse runs, increasing variability. In contrast, ρ = 1 is more tightly concentrated, indicating more consistent performance across seeds.

Intuitively, these distributions convey that increasing ρ does not significantly change the typical performance, but it reduces reliability. Lower ρ values produce more stable and predictable learning, whereas higher ρ values occasionally lead to poorer convergence due to heavier reuse of replay data.

ii. 
-Although the mean performances of all ρ values are similar, the distributions show that ρ = 1 is more tightly concentrated, meaning it is more reliable across seeds. Higher replay factors (especially ρ = 4, 8) have wider spreads, indicating a higher chance of poorer runs even if the average looks similar.

-The left tails for larger ρ values extend further toward lower returns. This suggests that higher replay factors occasionally lead to significantly worse outcomes, which is not obvious from the mean learning curves alone.

-The peaks of all distributions lie close together, showing that increasing ρ does not shift the distribution significantly to better performance. Instead, it mainly increases variance. This suggests additional replay updates do not improve the final solution quality.

-The distribution for ρ = 2 is slightly left-skewed compared to others. This suggests that a small increase in replay updates introduces instability without providing sufficient averaging benefits: with ρ = 1 updates closely track fresh data and remain stable, while larger replay factors (ρ = 4, 8) perform many updates that average out noise; however, ρ = 2 lies in between—introducing bias from replayed data but not enough updates to stabilize learning—leading to a mild degradation in typical performance.

iii.
1. Safety-critical robotics (e.g. autonomous cars)
Two algorithms may have similar mean returns, but distributions can show that one occasionally fails catastrophically. A learning curve would hide this, while the distribution reveals rare but dangerous failures, which is crucial for deployment.

2. Sparse-reward environments (e.g., maze navigation)
Some runs may discover the goal early while others never do. The mean curve averages these together and looks smooth, but the distribution becomes multimodal (success vs failure), revealing that the algorithm is unreliable.

c)
i. plot
ii. From the tolerance interval plot, ρ = 1 and ρ = 2 have relatively tighter bands, especially after convergence, indicating more consistent performance across runs. In contrast, ρ = 4 shows the widest tolerance interval during learning, suggesting high variability, while ρ = 8 also remains wider as well. After convergence, all intervals shrink but higher ρ values still show slightly larger spread.

This provides different information from part (a). While the mean curves suggested similar performance across ρ values, tolerance intervals reveal how much individual runs can deviate. Even when means overlap, higher ρ values show larger variability, indicating less consistent behavior.
iii. 
ρ = 1 → narrowest tolerance interval → most reliable and stable learning, best worst case guarantee
ρ = 2 → slightly wider but still stable, tolerable worst-case performance
ρ = 4 → widest interval → least reliable, large variability, poorest worst case outcomes
ρ = 8 → moderate variability, less stable than ρ = 1 but better than ρ = 4, still poor worst case outcomes

Thus, smaller replay factors are more robust, while larger ρ values increase risk of poor runs
iv.
Confidence intervals (CI):Describe uncertainty in the mean,does not reflect variability of individual runs , a 95% CI means that if we repeat the experiment many times, 95% of the computed intervals would contain the true mean.

Tolerance intervals (TI):A tolerance interval gives a range that contains a specified proportion of individual runs with a certain confidence. For example, an (α = 0.05, β = 0.9) tolerance interval means we are 95% confident that 90% of all runs lie within this interval. It reflects spread and variability of performance, including worst-case behavior.

When useful
CI → comparing average algorithm performance
TI → evaluating reliability, variability, deployment safety

When misleading
CI misleading when variability is large (mean hides failures)
TI misleading if only mean performance matters

As number of runs → infinity
Confidence interval → converges to true mean (width → 0)
Tolerance interval → converges to true performance distribution bounds (non-zero width reflecting intrinsic variability)
if algorithm is completely deterministic than Tolerance would go to 0.

d)
i. From the sensitivity plots, DQN with ρ = 4 is less sensitive to batch size than ρ = 1.
For ρ = 1,batch size 64 gives the best result, while other batch sizes increase variance. This indicates that ρ = 1 depends strongly on choosing an appropriate batch size.

In contrast, for ρ = 4, performance remains relatively similar across batch sizes (64–512), with smaller differences in mean AUC and reduced variation. This suggests that increasing replay factor makes learning less sensitive to the mini-batch size, since multiple updates per step average gradient noise.

When keeping total samples per step fixed (ρ × batch size constant), it is wiser to use a smaller ρ with a larger batch size. The results show that ρ = 1 with moderate/large batch sizes achieves better mean performance and lower variance than ρ = 4 with smaller batches. Larger batch sizes provide more stable gradient estimates, whereas increasing ρ reuses the same replay data multiple times, which does not improve performance and can introduce bias. Using smaller ρ with a larger batch size is also computationally more efficient.

ii. 
DQN with ρ = 4 is more sensitive to the target network refresh rate than ρ = 1. For both methods, very frequent updates (small refresh rate: 500–1000) lead to fast and stable learning. However, when the refresh rate becomes too large (4000–8000), performance degrades and variance increases, especially for ρ = 4, indicating strong instability due to stale targets.

When the refresh rate is too low (very frequent updates), learning is stable because the target closely tracks the online network. When the refresh rate is too high (infrequent updates), the target becomes outdated, leading to inaccurate TD targets and unstable learning; this effect is much stronger for ρ = 4 since multiple updates amplify errors from the stale target.

From the plots, ρ > 1 appears more stable than ρ = 1 for frequent updates.

When the target network is updated frequently, the TD targets change quickly and stay close to the current Q-network.

With ρ = 4, the agent performs multiple updates using this fresh and consistent target, so each update reinforces similar gradients. This averages noise and reduces oscillations, leading to smoother learning.

With ρ = 1, only one update is made per step. Since the target changes frequently, each update is based on slightly different targets, causing more fluctuation in Q-values and visible dips in performance.

iii.
In the robotics setting where collecting trajectories is expensive, increasing ρ is attractive because it allows more learning from limited real-world data. However, the hyperparameter sensitivity results show that higher ρ is also more sensitive to choices such as target update rate and batch size, and can become unstable if these are not tuned carefully. This implies that while larger ρ improves data efficiency, it reduces robustness, making deployment riskier on real hardware where failures are costly.

In contrast, smaller ρ  is more robust across hyperparameters, though it may require more environment interaction. For real-world deployment, this suggests a trade-off: larger ρ is useful when data collection is expensive, but it requires careful tuning and monitoring to avoid instability; smaller ρ is safer and more reliable when robustness is more important than minimizing interactions.

Thus, the key lesson for deployment is that higher replay factors improve sample efficiency but reduce reliability, whereas lower replay factors provide more stable and robust performance, which may be preferable in safety-critical real-world systems.

5.
Using Prioritized Experience Replay (PER) generally diminishes the benefit of increasing the replay factor ρ. With PER, transitions with high TD-error are sampled more frequently, so each update is already more informative. Increasing ρ then repeatedly samples the same high-priority transitions, reducing diversity and causing overfitting to a small subset of experiences.

From the plot, all replay factors (ρ = 1, 2, 4) achieve similar final performance, and increasing ρ does not improve convergence. In fact, ρ = 1 learns fastest and most smoothly, while ρ = 2 and especially ρ = 4 show larger oscillations and higher variance throughout training. This indicates that with PER, higher replay factors introduce instability rather than improving sample efficiency.

Repeatedly updating on prioritized samples amplifies noise and bias, leading to diminishing or negative returns from increasing ρ. Therefore, PER reduces the advantage of large replay factors, and smaller ρ values become preferable.