# DA6400 PA3 — Soft Actor-Critic

Implementation of SAC (continuous + discrete) and experiments on the three
environments prescribed in the PDF. The reference code is from 
[denisyarats/pytorch_sac](https://github.com/denisyarats/pytorch_sac).
The actual working code lives in `sac_core/` plus the three per-env folders below.

## Layout

```
Assignment 3/
├── sac_core/               # shared SAC primitives (env-agnostic)
│   ├── actor.py            # squashed-Normal policy (continuous)
│   ├── actor_discrete.py   # Categorical policy (discrete)
│   ├── critic.py           # double-Q, takes (s,a) — continuous
│   ├── critic_discrete.py  # double-Q, outputs R^|A| — discrete
│   ├── sac.py              # SACAgent (continuous; auto+manual alpha)
│   ├── sac_discrete.py     # DiscreteSACAgent
│   ├── replay_buffer.py    # float-action + int-action buffer, supports reward relabeling
│   ├── training_loop.py    # generic train() with eval/schedule/step hooks
│   └── utils.py
├── pendulum/               # §2.1
│   ├── pendulum_env.py     # Pendulum-v1 with theta_target + reward scale
│   ├── train.py            # auto / manual alpha via --alpha-mode
│   └── plot.py
├── lunarlander/            # §2.2
│   ├── lander_env.py       # v3 continuous/discrete + hover-box reward
│   ├── train_continuous.py # Q2.2.1–3 (supports mid-training reward swap)
│   ├── train_discrete.py   # Q2.2.4(b)
│   ├── train_dqn.py        # Q2.2.4(c) — skeleton; plug in PA2 DQN
│   └── plot.py
├── reacher/                # §2.3
│   ├── reacher_env.py      # dm_control reacher-easy with R_a / R_b / R_c
│   ├── train.py            # cross-reward eval every 10K steps
│   ├── eval_final_policy.py  # Q2.3.3(a) — 500×5000-step final-policy eval
│   └── plot_results.py
├── pytorch_sac/            # reference repo, untouched
├── requirements.txt
└── README.md
```

## Install

```
pip install -r requirements.txt
```

Reacher needs `dm_control`. LunarLander needs `box2d-py` (your `rl_venv`
already has `swig`/`box2d` working).

## Reproducing per section

> All experiments should be run for 15 seeds and plotted with mean ± CI
> (per PDF). The snippets below show one seed each; loop in your shell /
> job launcher over `--seed` for the 15-seed sweep.

### §2.1 Pendulum

Q2.1.2 — auto alpha sweep over theta_target:
```
for th in 0 -10 30 -60 90 -90 120 -150; do
  python pendulum/train.py --theta $th --seed 1 --num-train-steps 200000
done
python pendulum/plot.py --runs-dir pendulum/runs --alpha-mode auto \\
    --thetas 0 -10 30 -60 90 -90 120 -150 --seeds 1 --out pendulum_auto.png
```

Q2.1.5(a) — manual α (`α_mnl`) for {-60, 90, 120, -150}:
```
python pendulum/train.py --theta 90 --alpha-mode manual --alpha 0.2 --seed 1
python pendulum/plot.py --compare-modes --theta 90 --manual-alpha 0.2 \\
    --seeds 1 --out pendulum_auto_vs_manual_90.png
```

Q2.1.5(b) — reward scaling at θ=90:
```
for rs in 0.1 1.0 10.0; do
  python pendulum/train.py --theta 90 --alpha-mode manual --alpha 0.2 \\
      --reward-scale $rs --seed 1
  python pendulum/train.py --theta 90 --alpha-mode auto \\
      --reward-scale $rs --seed 1
done
python pendulum/plot.py --reward-scale-sweep --theta 90 --manual-alpha 0.2 \\
    --scales 0.1 1.0 10.0 --seeds 1 --out pendulum_rs_sweep.png
```

### §2.2 LunarLander

Q2.2.1–2 — continuous vanilla:
```
python lunarlander/train_continuous.py --seed 1 --num-train-steps 500000
```

Q2.2.3 — hover-box reward swap (+200 → −100 at step 250K):
```
# version (i): manual alpha = 0.01
python lunarlander/train_continuous.py --alpha-mode manual --alpha 0.01 \\
    --hover-bonus 200 --swap-bonus-at-step 250000 --swap-bonus-to -100 \\
    --num-train-steps 500000 --seed 1
# version (ii): auto alpha
python lunarlander/train_continuous.py --alpha-mode auto \\
    --hover-bonus 200 --swap-bonus-at-step 250000 --swap-bonus-to -100 \\
    --num-train-steps 500000 --seed 1
python lunarlander/plot.py --mode hover --runs-dir lunarlander/runs \\
    --swap-step 250000 --seeds 1 --out lander_hover.png
```

Q2.2.4(b) — discrete-SAC:
```
python lunarlander/train_discrete.py --seed 1 --num-train-steps 500000
```

Q2.2.4(c) — DQN vs discrete-SAC: plug the PA2 DQN into
`lunarlander/train_dqn.py` (see that file's note), write `progress.csv` in
the same schema, then:
```
python lunarlander/plot.py --mode discrete-vs-dqn --runs-dir lunarlander/runs \\
    --seeds 1 --out lander_discrete_vs_dqn.png
```

### §2.3 Reacher

Q2.3.1 — SAC-R{a,b,c}, cross-reward evaluation every 10K steps:
```
python reacher/train.py --reward a --seed 1 --num-train-steps 1000000
python reacher/train.py --reward b --seed 1 --num-train-steps 1000000
python reacher/train.py --reward c --seed 1 --num-train-steps 1000000
python reacher/plot_results.py --runs-dir reacher/runs --seeds 1
```

Q2.3.3(a) — final-policy 500×5000-step evaluation (steps_to_goal /
steps_in_target):
```
python reacher/eval_final_policy.py --run-dir reacher/runs/sac_Rb_seed1 \\
    --num-episodes 500 --episode-length 5000
```

## §3 PEBBLE (bonus) — implemented

Lives in [`pebble/`](pebble/). Reuses `sac_core/` unchanged.

Modules:
- `reward_model.py` — ensemble MLP reward model with Bradley-Terry loss.
- `preference_buffer.py` — stores (segment0, segment1, label) tuples.
- `trajectory_buffer.py` — rolling log of recent transitions with ground-truth
  rewards, used to sample segment pairs for preference queries.
- `teacher.py` — `OracleTeacher` (compares ground-truth segment returns;
  supports optional mistake probability).
- `pebble_trainer.py` — end-to-end loop: unsupervised pre-training with k-NN
  state-entropy intrinsic reward, preference query sessions, reward-model
  updates, full replay-buffer relabeling, SAC updates, periodic eval against
  ground-truth return.
- `train_pendulum.py` — Bonus Q1, Q2 entry point.
- `train_reacher.py` — Bonus Q3 entry point (three teachers, one per reward
  formulation R_a / R_b / R_c).

Bonus Q1 — PEBBLE on Pendulum:
```
for th in 0 -60 90 120 -150; do
  python pebble/train_pendulum.py --theta $th --feedback-budget 1000 --seed 1
done
```
Compare against `pendulum/train.py --theta ... --alpha-mode auto`
(ground-truth SAC) using `pendulum/plot.py`.

Bonus Q2 — feedback-budget sweep:
```
for fb in 100 500 1000 2000; do
  python pebble/train_pendulum.py --theta 90 --feedback-budget $fb --seed 1
done
```

Bonus Q3 — PEBBLE on Reacher, three teacher variants:
```
python pebble/train_reacher.py --teacher a --seed 1
python pebble/train_reacher.py --teacher b --seed 1
python pebble/train_reacher.py --teacher c --seed 1
```

Key PEBBLE hyperparameters (exposed as CLI flags):
- `--feedback-budget`: total teacher queries allowed over the run.
- `--feedback-frequency`: env steps between query sessions.
- `--queries-per-session`: queries drawn each session (until budget is spent).
- `--segment-length`: timesteps per segment.
- `--selection uniform|disagreement`: pair selection strategy. Disagreement
  picks pairs where ensemble members most disagree on P[σ¹≻σ⁰].
- `--unsup-steps`: env steps of unsupervised pre-training with the k-NN
  intrinsic reward before any teacher query.
- `--ensemble-size`, `--reward-epochs`: reward-model hyperparameters.
- `--teacher-mistake-prob`: optional label noise (0 = oracle).

## Notes

- **15 seeds.** Every script exposes `--seed`. Wrap in a shell loop (or slurm
  array) for the 15-seed sweep the PDF requires.
- **Eval cadence.** Every script logs mean ± std of 20 offline deterministic
  episodes every 10K env steps, matching the PDF's requirement.
- **x-axis is env timesteps** (not episodes), per the PDF.
- **10K-step random-action seed phase** is on by default (`--num-seed-steps
  10000`).
- **Action ranges.** Pendulum's native `[-2, 2]` is handled by
  `pendulum_env.py` (the agent outputs in `[-1, 1]` via tanh, wrapper
  rescales). LunarLander continuous is already `[-1, 1]^2`. Reacher is
  `[-1, 1]^2`.
- **Timeout bootstrap.** The training loop masks `done` on timeouts so the
  Bellman target keeps bootstrapping (`done_no_max = 0` when timeout and not
  terminated). Envs set `info['timeout']` / `info['terminated']` accordingly.
