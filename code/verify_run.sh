#!/usr/bin/env bash
# Verification re-runs. Writes all JSONs to results_verify/.
# Each step prefixes a START/DONE line so a Monitor can track progress.
set -u
cd "$(dirname "$0")"
export DSABRE_OUT_DIR="$PWD/results_verify"
mkdir -p "$DSABRE_OUT_DIR"
LOG="$DSABRE_OUT_DIR/verify_run.log"
: > "$LOG"

run() {
    local tag="$1"; shift
    local t0=$(date +%s)
    echo "[START $tag] $(date -Iseconds)" | tee -a "$LOG"
    if "$@" >>"$LOG" 2>&1; then
        local dt=$(( $(date +%s) - t0 ))
        echo "[DONE  $tag] ${dt}s" | tee -a "$LOG"
    else
        local rc=$?
        echo "[FAIL  $tag] rc=$rc" | tee -a "$LOG"
    fi
}

PY=python3
DMAPS_PY=/opt/anaconda3/envs/dmaps/bin/python

# --- Wave 1: fast suites (~15 min)
run bench_25q              $PY benchmark.py --suite 25
run bench_36q              $PY benchmark.py --suite 36
run bench_8L_25            $PY bench_bgrid_8links.py --suite 25
run bench_8L_36            $PY bench_bgrid_8links.py --suite 36
run bench_pytket_layout    $PY bench_pytket_layout.py
run fill_sweep_dse         $PY run_fill_sweep.py dse
run fill_sweep_dmaps       $DMAPS_PY run_fill_sweep.py dmaps
run ablate_node_decay_all  $PY ablate_node_decay.py
run regen_ablation_corners $PY regen_ablation_corners.py

# --- Wave 2: medium (~30 min)
run bench_64q              $PY benchmark.py --suite 64
run dmaps_bench_25_36_64   $DMAPS_PY run_dmaps_bench.py 25 36 64

# --- Wave 3: heavy (~2 h, 360q dominates)
run bench_100q             $PY bench_large.py --suite 100
run bench_200q             $PY bench_large.py --suite 200
run bench_pytket_large     $PY bench_pytket_large.py
run bench_360q             $PY bench_large.py --suite 360

echo "[ALL DONE] $(date -Iseconds)" | tee -a "$LOG"
