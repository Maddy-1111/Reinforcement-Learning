# 2.1 Pendulum

1. Design and write appropriate reward function

- An effective reward function for the modified Pendulum environment is based on the cosine of the angular error between the current angle ( \theta ) and the target angle ( \theta_{target} ). It can be defined as:

[r_t = \cos(\theta_t - \theta_{target})]

This reward is maximized when the pendulum is exactly at the target angle (i.e., ( \theta_t = \theta_{target} ), giving reward = 1), and decreases smoothly as the deviation increases. The cosine formulation naturally captures the circular nature of angles and avoids discontinuities at ( \pm 180^\circ ).

To improve control behavior, a small penalty on angular velocity and control effort can be added:

[r_t = \cos(\theta_t - \theta_{target}) - 0.1*\dot{\theta}_t^2]

where ( \dot{\theta}_t ) is angular velocity 

This reward encourages the agent to:

1. Reach the desired angle (via cosine term),
2. Stabilize at that position (penalizing velocity),

Thus, the cosine-based reward is smooth, bounded, and well-suited for learning stable target alignment behavior.

2. The code

3. From the plot, the learning speed and final performance vary noticeably across target angles.

**Learning speed:**

* Targets like (theta = -150, 10, 120, 0) learn the fastest. They show rapid improvement and reach high returns early (≈40K steps).
* **Moderate angles** such as 30 and -60 learn more gradually, with steady but slower improvement.
* **Harder targets** like 90 and -90 initially struggle (sharp drop around 20K steps), indicating exploration difficulty before recovering.

**Final performance (quality of behavior):**

* **Best performance** is achieved by ( theta = -150 ), which consistently attains the highest returns (~950+), indicating very stable alignment.
* ( theta = -10) and (0) also achieve high returns (~800), showing good convergence to the target.
* ( 120) reaches moderately high performance (~750) but with less improvement over time.
* ( 30) achieves moderate returns (~550–600), indicating partial success.
* **Lowest performance** is seen for ( -60), ( 90 ), and ( -90) (~300–400), suggesting difficulty in stabilizing at these angles.

**Overall interpretation:**
Learning is faster and more stable for targets closer to the natural upright/downward equilibrium or those easier to reach via pendulum dynamics. Targets such as ±90 are harder because they correspond to dynamically unstable points where gravity induces maximum deviation, requiring continuous and precise torque to counteract it. This leads to high sensitivity to errors, amplification of small deviations, sparse and difficult exploration trajectories, and noisy learning signals—collectively making both control and learning significantly more challenging.

4. Optimal behavior consists of two aspects:

1. Swing-up phase (energy shaping):
The agent first injects energy to move the pendulum toward the target angle, often using oscillatory motion to build sufficient momentum.

2. Approximate stabilization / limit-cycle behavior:
For closer to stable points like theta = 0, -10,1-150 it stabilises and stays at some slight deviation from the target.
For most larger angles, instead of perfectly holding the target angle, the agent often:
- oscillates around the target, or
- settles at a nearby easier-to-maintain angle (e.g., ~140° for a 120° target)

This happens because for difficult targets, maintaining the exact angle requires continuous precise torque, so the agent prefers stable or quasi-stable behaviors (oscillations or shifted equilibria) that still yield reasonably high reward.

This is reflected in the following visualizations of the system for theta target = -10 and 90 degrees - https://drive.google.com/drive/folders/1ux6DAtJJqaZ-ZaceHcT3RbjIQQfjiK6e?usp=sharing

5. 
(a) — Manual vs. automated α

We swept manual α values for each θ_target ∈ {−60, 90, 120, −150}. The candidate values were chosen by inspecting the final 5–6 values of the auto-tuned α from the corresponding Q2.1.2 runs (the values α settles to in the last 10–20K env steps), and then refining around them. The final αmnl values for every angle are:

θ_target	αmnl
−60     	0.01
90	        0.015
120	        0.005
−150	    0.005

Difficulty of tuning: Tuning α manually was moderately difficult: the optimal αmnl differed across target angles so a single value did not transfer. Looking at the auto-α trajectories was a useful prior.

Manual vs. auto comparison: Across θ ∈ {90, 120, −150}, the manual αmnl (dashed lines) achieves comparable or higher final return than auto (solid lines) and noticeably tighter confidence bands — i.e., better final convergence and lower variance across seeds. The trade-off is that manual tuning required a per-θ sweep to find these values, whereas auto-α produced reasonable performance out of the box for all four targets.

The exception is θ = −60: here auto-α slightly outperforms the best αmnl from our sweep. This is consistent with auto-α not stabilising during training for this target — the entropy requirement appears to shift over the course of learning, so any fixed α from the sweep is suboptimal for at least part of training, while the auto adjustment can adapt. This itself illustrates a limitation of manual tuning: when the optimal entropy temperature is non-stationary, a single scalar cannot match the adaptive scheme.

(b)

For scaling = 0.1, both methods reach the same asymptotic return (~+30, equivalent to the +300 from c=1). Manual is slightly tighter; auto shows a small mid-training dip (~step 30k) but recovers. 
For scaling = 10, manual converges to ≈+3200 (equivalent to +320 unscaled) by step 10k and stays there with moderate variance. Auto α breaks down: returns crash to ≈ −1100 around step 30k before partially recovering by step 50k. The variance band is huge throughout training.

SAC objective J = E[Σ (r − α · log π(a|s))]

Fixed (manual) α: the entropy term is unchanged, but the reward signal grows/shrinks by c. For c = 10, the reward dominates a fixed α = 0.015, so the policy quickly becomes near-deterministic on a good action — same behaviour as the c = 1 case but reached faster. For c = 0.1, the entropy term is now relatively larger, so the policy stays a bit more exploratory but still converges.

Automated α: the dual update on α is driven by the current entropy of the policy relative to the target entropy. The actor and critic gradients, however, scale linearly with c. So with c = 10 the critic Q-values and actor losses are 10× larger, but the α-loss is unchanged in magnitude — the optimiser pushes the policy toward determinism faster than α can compensate, the entropy collapses, learning becomes unstable, α has to ramp up by 10× to catch up, and we get the large mid-training dip and high variance that the plot shows. For c = 0.1 the opposite happens but it's gentler, because the policy is already near-deterministic and a slightly larger relative entropy term doesn't hurt.

Manual α adapts well to both scales because αmnl was already in a robust regime. Auto α adapts well to c = 0.1 but is fragile to c = 10 -  auto-α is not invariant to reward scale, and is one motivation for using a fixed manually-tuned α when the reward magnitude is known and stable.

# BONUS

1. 
Structure - 
- Unsupervised pre-training for the first 9 000 env steps: SAC trains on a k-NN state-entropy intrinsic reward (k = 5, rolling buffer of 10 000 states) — no teacher queries.
- Preference-based training (steps 9 000 → 50 000): every 5 000 env steps a feedback session runs.
- Reward learning. Each ensemble member is an MLP r_ψ(s, a) (256-hidden × 3-deep, Tanh) trained with the Bradley–Terry preference loss for 50 epochs over the preference buffer at every session.
- Budget. We used a total feedback budget of 1000 preference queries (≈ 50 sessions × 20 queries). 

Comparison vs. SAC trained on the ground-truth reward. 

θ = −60: Both methods plateau near +200 by step 40 K. PEBBLE is slightly lower at convergence (≈ +170) than SAC(≈ +260) but the bands overlap throughout
θ = 90: SAC reaches ≈ +320 by step 10 K but exhibits a large mid-training dip. PEBBLE rises more gradually but reaches the same ≈ +320 plateau by step 30 K, with tighter variance.
θ = 120: SAC reaches ≈ +700 by step 10 K; PEBBLE catches up by step 40–50 K. Final returns are within ≈ 50 of each other (~+760 vs ~+710).
θ = −150: SAC reaches its asymptote (~+960) by step 10 K; PEBBLE catches up by step 30 K. Final performance is identical.
θ = 0: PEBBLE converges to same value as SAC (~600) but with more variance and much slower (SAC converges after 30k steps only)

Learning efficiency: SAC is uniformly more sample-efficient: it has access to per-step reward signal from step 0, whereas PEBBLE spends steps 0–9 K on unsupervised pre-training (no task signal at all) and another ~10 K steps building up enough preference labels for the reward model to be accurate. The visible "lag" in the orange curve (typically 10–20 K env steps behind SAC) is the cost of replacing the reward function with 1 000 preference labels.

Final performance: Despite the slower start, PEBBLE matches SAC-GT's final return at every target except a small shortfall at θ = 120 and a within-noise gap at θ = −60. This is the key empirical takeaway: with only 1000 binary preference queries — and no access to the analytical reward function — PEBBLE recovers a reward representation that yields essentially the same converged policy as training directly on the true reward.

Conclusion: PEBBLE is less sample-efficient than ground-truth SAC (≈ 10–20 K-step lag, attributable to unsupervised pre-training and the warm-up of the reward model), but recovers comparable final performance across all target angles. This validates preference-based reward learning as a viable substitute when designing or specifying the reward is hard or impossible — at the cost of a modest amount of additional environment interaction.

2. 