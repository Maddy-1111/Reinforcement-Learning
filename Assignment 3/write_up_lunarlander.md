# 2.2 Lunar Lander

## Q2.2.1 — Continuous SAC with Automated Temperature Tuning

Implemented the desired SAC for the 
**Results (see `lander_continuous.png`, `lander_alpha_continuous.png`):** Training over 500K steps across 15 seeds shows the agent converging to a mean return of **~240** (near the gymnasium solve threshold of 200). The 95% CI narrows significantly after 300K steps, indicating consistent convergence across seeds. α drops rapidly from its initialisation of 0.2 to ~0.07 within the first 50K steps, then continues to decay slowly to ~0.05 by 500K steps, reflecting the agent's increasing confidence as it learns a near-optimal policy.

---

## Q2.2.2 — Agent Behavior Across Training Stages

Based on the learning curve and temperature evolution:

**Initial stage (0 – 50K steps, return ≈ −185 to 0):**
The agent behaves near-randomly, crashing into the ground or flying off-screen within seconds. The high entropy (α ≈ 0.2 → 0.08) drives wide exploration. α drops very sharply here as the agent quickly discovers that some actions (firing engines) are consistently better than others, reducing uncertainty.

**Intermediate stage (50K – 200K steps, return ≈ 0 to 190):**
The agent learns to slow its descent and roughly aim for the landing zone but struggles with precise lateral control and gentle touchdown. **Local optima are visible** in this region: some seeds plateau around return 75–100 for tens of thousands of steps (visible as individual gray traces in the plot) before escaping and improving further. This corresponds to a hovering/soft-falling policy that avoids crashes but does not earn the large landing bonus (+100 per leg contact). The wide 95% CI in this region confirms cross-seed variability.

**Final stage (300K+ steps, return ≈ 220–250):**
The agent consistently achieves soft landings between the flags, firing main and side engines precisely to minimise horizontal drift and vertical velocity at touchdown. Behavior is stable and repeatable. α has settled to ~0.05, indicating a near-deterministic but slightly stochastic policy that still explores small perturbations. The 95% CI narrows, showing most seeds have converged to similar performance.

---

## Q2.2.3 — Hover-Box Reward Variant

### Environment modification
An additional +200 reward is given once per episode when the lander enters the hover-box (|x| < 0.1, 0.4 < |y| < 0.6). At step 250K this bonus is flipped to −100.

### (a) Quantitative and qualitative comparison

**See `lander_hover.png` and `lander_alpha_hover.png`.**

**Before reward swap (0 – 250K steps):**
Both variants learn the hover-box bonus quickly. Fixed α=0.01 reaches a slightly higher peak return (~380) versus auto α (~320) by the swap point. This is because the very small fixed α suppresses exploration, causing the agent to exploit the +200 hover bonus aggressively and greedily — it learns a policy that reliably enters the hover-box each episode. Auto α, maintaining higher entropy (~0.10), explores more broadly and converges to a somewhat lower but more robust policy.

**After reward swap (250K – 500K steps):**
Both variants suffer an immediate sharp drop in return (~−50 to +80) as the hover behavior, now penalised at −100, hurts performance. However, the two variants behave differently in recovery:

- **Fixed α=0.01** recovers more slowly, settling at ~172 by 500K steps. Its low fixed entropy means the policy is tightly committed to the old hover behavior; with little incentive to explore alternatives, it struggles to unlearn the penalised strategy. The CI is also wider, suggesting inconsistent adaptation across seeds.

- **Auto α** recovers to ~191 by 500K steps with a slightly tighter CI. The α evolution plot shows that auto α does **not** spike after the swap — it continues its noisy downward trend around ~0.07–0.10. However, its α is already an order of magnitude higher than the fixed value of 0.01 throughout training, meaning the auto α policy is inherently less deterministic and retains more stochasticity even at convergence. This residual entropy makes it marginally easier to shift away from the penalised hover behavior, though the practical difference in final return is small (~191 vs ~172).

### (b) Implications for the maximum entropy formulation

This experiment offers a nuanced view of the maximum entropy formulation. The results show that the two SAC versions behave **surprisingly similarly** after the reward swap — both suffer a sharp drop and recover to roughly the same final return (~172 vs ~191). This is itself an interesting finding: the entropy bonus does not automatically "rescue" the agent from a non-stationary reward.

The key structural difference is that auto α settles at a significantly higher entropy level (~0.07–0.10) compared to manual α (0.01) throughout training. This has two consequences:

1. **Before the swap:** The fixed α=0.01 agent is more exploitative — it commits harder to the hover behavior and reaches a higher peak return (~380 vs ~320). The maximum entropy formulation with auto α trades off some peak performance for a more spread-out, less committed policy.

2. **After the swap:** The higher residual entropy of auto α means it never fully committed to the hover strategy, so unlearning it is marginally easier. However, because both agents still recover to similar final values, this advantage is modest in practice.

The broader lesson about the maximum entropy formulation is that it provides a **soft commitment** to learned behaviors rather than a hard one. This is a form of implicit regularisation — the agent always maintains some probability mass on suboptimal actions, which prevents catastrophic over-specialisation. However, it is not a silver bullet for reward non-stationarity: if the reward changes drastically and the agent has had enough time to converge, even a max-entropy policy can become heavily committed to the wrong behavior. The auto-tuned α helps primarily in that it calibrates the degree of commitment to what the current reward signal justifies, rather than fixing it arbitrarily. The ~19-point gap in final return (~191 vs ~172) reflects this marginal but measurable benefit of adaptive entropy regulation.

---

## Q2.2.4 — Discrete Action Version

### (a) Discrete SAC formulation

Standard SAC is derived for continuous actions using the reparameterisation trick on a squashed-Gaussian policy, which is not applicable to discrete actions. Christodoulou (2019) adapts SAC for discrete actions by:

1. **Policy**: A categorical distribution over actions, parameterised by a softmax over a neural network's logits. No reparameterisation is needed.
2. **Critic**: Outputs Q(s, a) for all actions simultaneously (a vector of size |A|), rather than taking (s, a) as input.
3. **Actor loss**: Since the categorical distribution is not reparametrisable, the policy gradient is computed as an exact expectation over all actions (feasible when |A| is small): `J(π) = E_s[ Σ_a π(a|s) · (α log π(a|s) − Q(s,a)) ]`
4. **Temperature update**: Uses `H_target = −ratio × log(|A|)` as the target entropy (ratio=0.98 here), keeping entropy close to the maximum possible for the action space.

This formulation recovers all the desirable properties of SAC (entropy regularisation, double-Q for overestimation reduction, soft policy improvement) while being applicable to environments with a finite action set.

### (b) Discrete SAC results

**See `lander_discrete_vs_dqn.png`.** Discrete SAC trains on the 4-action discrete LunarLander. Over 500K steps and 13 seeds, it achieves a mean final return of **~17** (95% CI: roughly −40 to +75), indicating only partial learning and high variance across seeds. Several seeds fail to converge entirely (visible as the flat gray traces near 0). This suggests the discrete-SAC hyperparameters (particularly the target entropy ratio and learning rates) may require more tuning for this environment, or that 500K steps is insufficient for consistent convergence.

### (c) DQN comparison

DQN with a double-Q trick achieves a mean final return of **~245** at 500K steps — nearly **14× higher** than discrete-SAC — and with much lower variance (95% CI: roughly 225–265). DQN's learning curve rises more steeply from ~100K steps onward, while discrete-SAC's mean barely rises above 0 until ~250K steps.

| Algorithm | Return at 500K (mean ± 95% CI) | Approx. steps to return > 150 |
|---|---|---|
| DQN | 244.8 ± 10 | ~250K |
| Discrete-SAC | 17.0 ± 58 | not reliably reached |

### (d) Algorithm preference for discrete actions

**In terms of performance and sample efficiency:** DQN is clearly preferable here. It achieves near-optimal landing performance in ~250K steps versus discrete-SAC's inconsistent progress over 500K steps. This is likely because DQN's ε-greedy exploration with a simple greedy policy is well-matched to LunarLander's relatively low-dimensional discrete action space, while discrete-SAC's entropy regularisation — designed primarily for continuous control where exploration is harder — may over-regularise in this setting.

**In terms of ease of implementation:** DQN is also simpler. It requires one Q-network, one target network, and one loss. Discrete-SAC requires separate actor and critic networks, a temperature parameter with its own optimizer, and careful matching of the target entropy to the action space size. Debugging is more involved.

**Overall:** For environments with small discrete action spaces where the main challenge is credit assignment rather than exploration (as in LunarLander), DQN (or its variants — Double DQN, Dueling DQN) is the preferred choice due to its simplicity, stability, and strong empirical performance. Discrete-SAC would become more attractive in settings requiring principled exploration (e.g., large discrete action spaces, sparse rewards, or multi-task scenarios) where the entropy regularisation provides a meaningful benefit.
