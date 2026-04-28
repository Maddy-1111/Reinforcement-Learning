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

* Targets like (theta = -150, 10, 0) learn the fastest. They show rapid improvement and reach high returns early (≈30K steps).
* **Moderate angles** such as 30 and -60 learn more gradually, with steady but slower improvement.
* **Harder targets** like 90 and -90 initially struggle (sharp drop around 20K steps), indicating exploration difficulty before recovering.
* 120 shows relatively stable but slower learning compared to easier angles.

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
