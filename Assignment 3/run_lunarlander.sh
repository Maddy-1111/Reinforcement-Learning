#!/usr/bin/env bash
# run_lunarlander.sh — all 75 LunarLander jobs (5 experiments × 15 seeds)
#
# Run from Assignment 3/ :
#   bash run_lunarlander.sh
#
# Estimated wall time on M2 (4 parallel workers): ~10 hours
# Logs: lunarlander/logs/<job>.log   Progress CSVs: lunarlander/runs/

set -eo pipefail

PYTHON=".venv/bin/python"
MAX_JOBS=4
LOGS="lunarlander/logs"
mkdir -p "$LOGS"

# ── simple parallel job queue ──────────────────────────────────────────────────
declare -a _PIDS=()
declare -a _NAMES=()

_reap() {
    local -a live_pids=() live_names=()
    local i
    for i in "${!_PIDS[@]}"; do
        if kill -0 "${_PIDS[$i]}" 2>/dev/null; then
            live_pids+=("${_PIDS[$i]}")
            live_names+=("${_NAMES[$i]}")
        fi
    done
    _PIDS=("${live_pids[@]:-}")
    _NAMES=("${live_names[@]:-}")
}

_throttle() {
    while true; do
        _reap
        [ "${#_PIDS[@]}" -lt "$MAX_JOBS" ] && break
        sleep 3
    done
}

launch() {
    local name="$1"; shift
    _throttle
    printf '[START] %s\n' "$name"
    "$PYTHON" "$@" >"$LOGS/${name}.log" 2>&1 \
        && printf '[DONE]  %s\n' "$name" \
        || printf '[FAIL]  %s  (see %s)\n' "$name" "$LOGS/${name}.log" &
    _PIDS+=($!)
    _NAMES+=("$name")
}

START_TIME=$(date +%s)
echo "======================================================"
echo " LunarLander overnight run  —  $(date)"
echo " MAX_JOBS=$MAX_JOBS   logs → $LOGS"
echo "======================================================"

# ── Q2.2.1 — continuous SAC, auto temperature ─────────────────────────────────
echo ""
echo "--- Q2.2.1 : continuous SAC (auto α) ---"
for seed in $(seq 1 15); do
    launch "cont_auto_s${seed}" \
        lunarlander/train_continuous.py \
        --alpha-mode auto \
        --seed "$seed" --num-train-steps 500000 \
        --device cpu --out-dir lunarlander/runs
done

# ── Q2.2.3 (i) — manual α=0.01, hover +200 → −100 ───────────────────────────
echo ""
echo "--- Q2.2.3(i) : hover-swap, manual α=0.01 ---"
for seed in $(seq 1 15); do
    launch "hover_manual_s${seed}" \
        lunarlander/train_continuous.py \
        --alpha-mode manual --alpha 0.01 \
        --hover-bonus 200 --swap-bonus-at-step 250000 --swap-bonus-to -100 \
        --seed "$seed" --num-train-steps 500000 \
        --device cpu --out-dir lunarlander/runs
done

# ── Q2.2.3 (ii) — auto α, hover +200 → −100 ─────────────────────────────────
echo ""
echo "--- Q2.2.3(ii) : hover-swap, auto α ---"
for seed in $(seq 1 15); do
    launch "hover_auto_s${seed}" \
        lunarlander/train_continuous.py \
        --alpha-mode auto \
        --hover-bonus 200 --swap-bonus-at-step 250000 --swap-bonus-to -100 \
        --seed "$seed" --num-train-steps 500000 \
        --device cpu --out-dir lunarlander/runs
done

# ── Q2.2.4(b) — discrete SAC ─────────────────────────────────────────────────
echo ""
echo "--- Q2.2.4(b) : discrete SAC ---"
for seed in $(seq 1 15); do
    launch "disc_sac_s${seed}" \
        lunarlander/train_discrete.py \
        --seed "$seed" --num-train-steps 500000 \
        --device cpu --out-dir lunarlander/runs
done

# ── Q2.2.4(c) — DQN ──────────────────────────────────────────────────────────
echo ""
echo "--- Q2.2.4(c) : DQN ---"
for seed in $(seq 1 15); do
    launch "dqn_s${seed}" \
        lunarlander/train_dqn.py \
        --seed "$seed" --num-train-steps 500000 \
        --device cpu --out-dir lunarlander/runs
done

# ── wait for remaining jobs ────────────────────────────────────────────────────
echo ""
echo "All 75 jobs queued — waiting for last batch to finish..."
wait

ELAPSED=$(( $(date +%s) - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "======================================================"
echo " Finished — $(date)"
printf ' Total wall time: %dh %02dm\n' "$HOURS" "$MINS"
echo "======================================================"
echo ""
echo "Results are in lunarlander/runs/"
echo "Run plots with:"
echo "  .venv/bin/python lunarlander/plot.py --mode continuous \\"
echo "      --runs-dir lunarlander/runs --seeds \$(seq -s' ' 1 15) \\"
echo "      --out lander_continuous.png"
echo ""
echo "  .venv/bin/python lunarlander/plot.py --mode hover \\"
echo "      --runs-dir lunarlander/runs --swap-step 250000 \\"
echo "      --seeds \$(seq -s' ' 1 15) --out lander_hover.png"
echo ""
echo "  .venv/bin/python lunarlander/plot.py --mode discrete-vs-dqn \\"
echo "      --runs-dir lunarlander/runs --seeds \$(seq -s' ' 1 15) \\"
echo "      --out lander_discrete_vs_dqn.png"
