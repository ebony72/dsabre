"""
probe_multiplier_abort.py — why does the 64-qubit Multiplier abort on the
3x3-grid-of-3x3-cores architecture (64q_c33), and does relaxing the router's
iteration/backup limits fix it?

`ablate_common.run_protocol` discards `router.route()`'s `failure_log`, so the
completion run (ablate_occupancy_complete64.py) only ever recorded
`aborted: true` with no reason.  This script calls `router.route()` directly,
per seed and per fbf pass, and prints:
  - which of the four abort conditions fired (ITERATION_LIMIT,
    NO_ACTIONS_NO_FALLBACK, DEADLOCK_NO_RECOVERY, DEADLOCK_BACKUP_EXHAUSTED)
  - iteration count and gates remaining at the abort point
  - core occupancy at the abort point (is every core simultaneously full?)

Then, for the seed/pass that aborts, it re-runs at 5x the default
max_iterations and max_backup_attempts to see whether the limits are the
proximate cause (a near-miss that finishes given more budget) or whether the
router is genuinely stuck (relaxing limits burns time without progress).

Usage:  python3 code/probe_multiplier_abort.py [--relax 5]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ablate_common import SUITES, core_occupancy, default_layouts, load_circuit
from dsabre_ext import dSABRE_BurstExt

SUITE = "64q_c33"
CIRCUIT = "multiplier"


def route_one(router, dag, l2p, label):
    t0 = time.time()
    m, final_l2p = router.route(dag, l2p)
    dt = time.time() - t0
    status = "ABORT" if m["aborted"] else "OK"
    print(f"    [{label}] {status}  iters-in-log={m['failure_log']}  "
          f"eprs={m['eprs']} ls={m['ls']}  backups={m['backup_activations']}  "
          f"t={dt:.1f}s", flush=True)
    return m, final_l2p, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relax", type=float, default=5.0,
                    help="multiplier on max_iterations and max_backup_attempts "
                         "for the relaxed re-run")
    ap.add_argument("--seed", type=int, default=0,
                    help="which of the 3 default-layout seeds to probe")
    args = ap.parse_args()

    s = SUITES[SUITE]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]
    print(f"architecture: {s['arch_name']}", flush=True)
    print(f"default hw:   max_iterations={hw.max_iterations}  "
          f"deadlock_limit={hw.deadlock_limit}  "
          f"max_backup_attempts={hw.max_backup_attempts}", flush=True)

    qc, dag, rev_dag, ncx = load_circuit(SUITE, CIRCUIT)
    print(f"circuit: {CIRCUIT}  CX={ncx}", flush=True)

    layouts = default_layouts(qc, dag, arch, n, seed=0, n_seeds=3)
    l2p = layouts[args.seed]
    occ0 = core_occupancy(l2p, arch)
    print(f"seed {args.seed} initial per-core occupancy: {occ0}  "
          f"(core capacity = {len(arch.core_qubits(0))} each)", flush=True)

    # ── Pass 1: default limits, forward pass only (this is what aborted in
    # the completion run — the fbf protocol's first 'f' step) ────────────────
    print("\n── default limits, forward pass ──", flush=True)
    router = dSABRE_BurstExt(arch, hw)
    m1, l2p_after, dt1 = route_one(router, dag, l2p, "default/forward")

    if not m1["aborted"]:
        print("Forward pass completed under default limits; the abort must "
              "come from a later seed or the reversed-DAG pass. Re-run with "
              "--seed to check the others, or inspect fbf as a whole.",
              flush=True)
        return

    remaining = m1["failure_log"][-1][2] if m1["failure_log"] else "?"
    total_gates = sum(1 for _ in dag.two_qubit_ops()) + sum(
        1 for n in dag.op_nodes() if len(n.qargs) < 2)
    print(f"\n  gates remaining at abort: {remaining} of ~{total_gates} "
          f"total DAG nodes", flush=True)

    # ── Relaxed re-run: same seed, same forward pass, wider limits ──────────
    relax = args.relax
    relaxed_hw = replace(hw,
                         max_iterations=int(hw.max_iterations * relax),
                         max_backup_attempts=int(hw.max_backup_attempts * relax))
    print(f"\n── relaxed limits ({relax}x), forward pass: "
          f"max_iterations={relaxed_hw.max_iterations}  "
          f"max_backup_attempts={relaxed_hw.max_backup_attempts} ──", flush=True)
    router_relaxed = dSABRE_BurstExt(arch, relaxed_hw)
    m2, l2p_after2, dt2 = route_one(router_relaxed, dag, l2p, "relaxed/forward")

    print("\n── verdict ──", flush=True)
    reason = m1["failure_log"][-1][0] if m1["failure_log"] else "UNKNOWN"
    print(f"  abort reason (default limits): {reason}", flush=True)
    if not m2["aborted"]:
        print(f"  relaxing limits {relax}x FIXES it: {m2['eprs']} EPR, "
              f"{m2['ls']} SWAP, {dt2:.0f}s (vs {dt1:.0f}s to abort)", flush=True)
    else:
        reason2 = m2["failure_log"][-1][0] if m2["failure_log"] else "UNKNOWN"
        remaining2 = m2["failure_log"][-1][2] if m2["failure_log"] else "?"
        print(f"  relaxing limits {relax}x does NOT fix it: still aborts "
              f"({reason2}, {remaining2} gates remaining, {dt2:.0f}s spent)",
              flush=True)
        if reason == "NO_ACTIONS_NO_FALLBACK":
            print("  NO_ACTIONS_NO_FALLBACK is not a budget limit — it means "
                  "no legal candidate AND no fallback SWAP existed at that "
                  "state.  No iteration/backup budget can fix a state with no "
                  "legal move; that needs a different routing decision "
                  "earlier, or more free capacity in the architecture.",
                  flush=True)


if __name__ == "__main__":
    main()
