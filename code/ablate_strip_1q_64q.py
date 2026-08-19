"""ablate_strip_1q_64q.py — what happens if single-qubit gates are removed
before routing?

Single-qubit gates never constrain routing: they are always executable, and the
router drains them from the front layer for free (`metrics["1q_gates"]`).  They
are nonetheless 38-75% of the op nodes in the 64q suite, and they are *not*
inert to the heuristic:

  * `_bfs_ext` (the production inter-core extended set) advances `layer_depth`
    per BFS layer over ALL op nodes.  A 2q gate sitting behind three 1q gates on
    its wire lands at depth 4 instead of depth 1, so `cfg.lookahead_decay`
    discounts it far more heavily -- and, since E is capped at
    `lookahead_size`, a diluted layer sequence changes which gates enter E at
    all.  `_get_local_extended` restricts E to a core, so the intra-core scorer
    inherits the same shift.
  * `_indeg`, `_remaining`, the per-iteration `front_layer()` drain scan and the
    per-pass `deepcopy(dag)` all scale with the total node count, so 1q gates
    cost compile time whether or not they change a decision.

Three arms, all dSE (`dSABRE_BFSExt`, the production router), all under
benchmark.py's protocol (3 SabreLayout seeds, fwd -> bwd -> fwd, best EPR):

  full          the shipped pipeline: layout from the full circuit, route the
                full DAG.  Must reproduce results/results_64q.json's dSE column.
  strip_route   IDENTICAL layouts to `full`, but the router (and the reversed
                DAG of pass 2) sees only the 2q skeleton.  Isolates the effect
                on the routing heuristic alone.
  strip_all     the realistic preprocessing pipeline: SabreLayout is also run on
                the stripped circuit.  SabreLayout is not invariant to the strip
                (layouts differ on 5 of the 9 circuits), so this arm carries a
                layout effect on top of the routing effect.

The EPR/SWAP schedule from a stripped run stays valid for the full circuit: 1q
gates are re-inserted afterwards on their own wires, and no routing decision can
be blocked by one.  Only `metrics["1q_gates"]` becomes meaningless (it is 0).

Reading the output
------------------
The protocol-level number is a min over three layout seeds, and the per-seed
spread on this suite is wide (qnn: 495 and 641 EPR from the same stripped DAG
on two different layouts).  A min over three therefore turns a modest shift into
a coin flip.  `full` and `strip_route` run the SAME layouts, so `per_seed_eprs`
is a set of paired observations -- that is the comparison that answers whether
stripping systematically helps.  `--seed-groups` widens it.

Usage:
  python3 ablate_strip_1q_64q.py                 # all three arms, 9 circuits
  python3 ablate_strip_1q_64q.py --circuits ae qft
  python3 ablate_strip_1q_64q.py --seed-groups 3 --arms full strip_route
"""

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import replace
from math import prod

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import run_sabre_passes, sabre_locked_boundary_layout
from circuit_paths import circuits_path

# ── Suite (mirrors benchmark.py's "64q") ───────────────────────────────────────
CIRCUIT_DIR = circuits_path("qasm_64")
SUFFIX      = "_nativegates_ibm_qiskit_opt3_64.qasm"
ARCH        = build_h_grid_architecture(r=2, s=3, m=4)
HW          = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                             max_iterations=20000)

# qasm_64/ is shared with other projects and has picked up files that are not
# part of the published 9-circuit suite -- whitelist, don't glob-all.
CANONICAL = {"ae", "ghz", "graphstate", "qft", "qnn", "random",
             "qpeexact", "qaoa", "multiplier"}

# Published CX counts (Table VI).  A mismatch means the shared circuit
# directory was regenerated with the wrong MQT Bench version; abort rather than
# report numbers that cannot be compared against anything.
EXPECTED_CX = {"ae": 1962, "ghz": 63, "graphstate": 64, "qft": 1966,
               "qnn": 8126, "random": 1627,
               "qpeexact": 2139, "qaoa": 3920, "multiplier": 13040}

ARMS = ["full", "strip_route", "strip_all"]

_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
OUT_PATH = os.path.join(_RESULTS_DIR, "results_strip_1q_64q.json")


def load_qasm(path):
    """benchmark.py's loader, verbatim: strip measurements and barriers."""
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def strip_1q(dag):
    """Drop every 1q op node, keeping the 2q gates in place.

    Removing nodes from a copy of `dag` (rather than rebuilding a circuit from
    its 2q gates) keeps `dag.qubits` object-identical, so a layout dict built
    against the full DAG is directly usable against the stripped one -- which is
    what the `strip_route` arm needs.
    """
    out = circuit_to_dag(dag_to_circuit(dag))
    for n in list(out.op_nodes()):
        if len(n.qargs) < 2:
            out.remove_op_node(n)
    return out


def route_arm(router, dag, rev_dag, layouts):
    """benchmark.py's per-router protocol: every layout seed, keep the best EPR.

    `time_s` charges the whole protocol (all seeds), matching benchmark.py;
    `time_seed_s` is the winning seed's own fwd->bwd->fwd time.
    """
    best = None
    per_seed = []
    protocol_s = 0.0
    for layout in layouts:
        t0 = time.perf_counter()
        m = run_sabre_passes(router, dag, rev_dag, layout)
        protocol_s += time.perf_counter() - t0
        if m and not m.get("aborted"):
            per_seed.append(m["eprs"])
            if best is None or m["eprs"] < best["eprs"]:
                best = m
        else:
            per_seed.append(None)
    if best is None:
        return {"aborted": True, "time_s": round(protocol_s, 3),
                "per_seed_eprs": per_seed}
    return {
        "aborted": False,
        "eprs": best["eprs"], "ls": best["ls"],
        "teles": best["teles"], "catcomms": best["catcomms"],
        "one_q_drained": best["1q_gates"],
        "time_s": round(protocol_s, 3),
        "time_seed_s": round(best["compile_time"], 3),
        "per_seed_eprs": per_seed,
        "backup_activations": best.get("backup_activations", 0),
        "force_make_room": best.get("force_make_room", 0),
    }


def gmean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    return prod(vals) ** (1 / len(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuits", nargs="*", default=None,
                    help="subset of the suite to run (default: all 9)")
    ap.add_argument("--seed-groups", type=int, default=1,
                    help="SabreLayout seed groups; each contributes 3 layouts. "
                         "1 (default) reproduces benchmark.py's best-of-3; "
                         "higher widens the paired per-seed sample.")
    ap.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS,
                    help="which arms to run (default: all three)")
    ap.add_argument("--out", default=None, help="override the results JSON path")
    ap.add_argument("--decay", type=float, default=None,
                    help="override cfg.lookahead_decay (default 0.9).  The "
                         "measured effect of stripping is almost entirely a "
                         "shift in E's depths, hence in the decay**depth "
                         "weight -- `--decay 1.0` on the `full` arm removes "
                         "that weighting instead of removing gates, and tests "
                         "whether the two routes to it agree.")
    args = ap.parse_args()

    arms_wanted = [a for a in ARMS if a in args.arms]
    out_path = args.out or OUT_PATH
    seed_starts = [3 * g for g in range(args.seed_groups)]

    wanted = set(args.circuits) if args.circuits else CANONICAL
    unknown = wanted - CANONICAL
    if unknown:
        raise SystemExit(f"not in the 64q suite: {sorted(unknown)}")

    files = sorted(f for f in glob.glob(os.path.join(CIRCUIT_DIR, "*.qasm"))
                   if os.path.basename(f).replace(SUFFIX, "") in wanted)
    if len(files) != len(wanted):
        found = {os.path.basename(f).replace(SUFFIX, "") for f in files}
        raise SystemExit(f"missing circuits: {sorted(wanted - found)}")

    hw = HW if args.decay is None else replace(HW, lookahead_decay=args.decay)
    router = dSABRE_BurstExt(ARCH, hw)
    if args.decay is not None:
        print(f"  [lookahead_decay = {hw.lookahead_decay}]", flush=True)

    hdr = (f"{'circuit':<12} {'2q':>6} {'1q':>6}"
           f" {'full_epr':>9} {'sR_epr':>7} {'sA_epr':>7}"
           f" {'full_ls':>8} {'sR_ls':>7} {'sA_ls':>7}"
           f" {'full_t':>7} {'sR_t':>6} {'sA_t':>6}"
           f" {'sR/full':>8} {'sA/full':>8}")
    print("=" * len(hdr), flush=True)
    print("  64q suite -- effect of removing all 1q gates before routing (dSE)", flush=True)
    print("=" * len(hdr), flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    records = []
    for path in files:
        name = os.path.basename(path).replace(SUFFIX, "")
        qc, dag = load_qasm(path)
        n_2q = sum(1 for _ in dag.two_qubit_ops())
        if n_2q != EXPECTED_CX[name]:
            raise SystemExit(
                f"{name}: {n_2q} CX, published table says {EXPECTED_CX[name]}. "
                f"The shared circuit directory has been regenerated; refusing "
                f"to benchmark against numbers that no longer compare.")
        n_ops = len(dag.op_nodes())

        rev_dag  = circuit_to_dag(qc.reverse_ops())
        sdag     = strip_1q(dag)
        srev_dag = strip_1q(rev_dag)
        sqc      = dag_to_circuit(sdag)

        # `full` and `strip_route` share these layouts exactly; `strip_all` runs
        # SabreLayout on the stripped circuit instead.
        layouts_full, layouts_strp = [], []
        for sd in seed_starts:
            layouts_full += sabre_locked_boundary_layout(qc, dag, ARCH, seed=sd)
            layouts_strp += sabre_locked_boundary_layout(sqc, sdag, ARCH, seed=sd)
        layouts_equal = (len(layouts_full) == len(layouts_strp)
                         and all(a == b for a, b in zip(layouts_full, layouts_strp)))

        arm_input = {
            "full":        (dag,  rev_dag,  layouts_full),
            "strip_route": (sdag, srev_dag, layouts_full),
            "strip_all":   (sdag, srev_dag, layouts_strp),
        }
        arms = {a: route_arm(router, *arm_input[a]) for a in arms_wanted}

        def g(a, k):
            r = arms.get(a)
            return None if r is None or r.get("aborted") else r.get(k)

        def pct(a, b):
            if a is None or b is None or b == 0:
                return "    ---"
            return f"{100 * (a - b) / b:+.1f}%"

        def i(v): return str(v) if v is not None else "---"
        def f(v): return f"{v:.2f}" if v is not None else " ---"

        print(f"{name:<12} {n_2q:>6} {n_ops - n_2q:>6}"
              f" {i(g('full','eprs')):>9} {i(g('strip_route','eprs')):>7} {i(g('strip_all','eprs')):>7}"
              f" {i(g('full','ls')):>8} {i(g('strip_route','ls')):>7} {i(g('strip_all','ls')):>7}"
              f" {f(g('full','time_s')):>7} {f(g('strip_route','time_s')):>6} {f(g('strip_all','time_s')):>6}"
              f" {pct(g('strip_route','eprs'), g('full','eprs')):>8}"
              f" {pct(g('strip_all','eprs'), g('full','eprs')):>8}", flush=True)

        records.append(dict(circuit=name, qubits=qc.num_qubits, cx=n_2q,
                            ops=n_ops, one_q=n_ops - n_2q,
                            layouts_equal=layouts_equal, arms=arms))
        # Partial results survive a crash or a kill.
        n_layouts = 3 * args.seed_groups
        with open(out_path, "w") as fh:
            json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"),
                                     suite="64q",
                                     arch="H-grid 2x3 4x4 (96 qubits)",
                                     router="dSABRE_BFSExt (dSE)",
                                     protocol=f"{n_layouts} SabreLayout seeds, "
                                              f"fwd->bwd->fwd, best EPR",
                                     seed_groups=args.seed_groups,
                                     arms=arms_wanted),
                           results=records), fh, indent=2)

    print("-" * len(hdr), flush=True)
    line = f"{'gmean':<12} {'':>6} {'':>6}"
    for k in ("eprs", "ls", "time_s"):
        for a in ARMS:
            width = {"eprs": 9 if a == "full" else 7,
                     "ls": 8 if a == "full" else 7,
                     "time_s": 7 if a == "full" else 6}[k]
            if a not in arms_wanted:
                line += f" {'---':>{width}}"
                continue
            vals = [r["arms"][a].get(k) for r in records
                    if not r["arms"][a].get("aborted")]
            line += f" {gmean(vals):>{width}.2f}"
    base = gmean([r["arms"]["full"].get("eprs") for r in records
                  if not r["arms"]["full"].get("aborted")]) if "full" in arms_wanted else None
    for a in ARMS[1:]:
        if a not in arms_wanted or base is None:
            line += f" {'---':>8}"
            continue
        v = gmean([r["arms"][a].get("eprs") for r in records
                   if not r["arms"][a].get("aborted")])
        line += f" {100 * (v - base) / base:>+7.1f}%"
    print(line, flush=True)

    for a in arms_wanted:
        tot_e = sum(r["arms"][a]["eprs"] for r in records
                    if not r["arms"][a].get("aborted"))
        tot_t = sum(r["arms"][a]["time_s"] for r in records)
        ab = [r["circuit"] for r in records if r["arms"][a].get("aborted")]
        print(f"  {a:<12} total EPR {tot_e:>6}   total time {tot_t:>7.1f}s"
              f"   aborted: {ab or 'none'}", flush=True)

    # Paired per-seed view: `full` and `strip_route` ran identical layouts, so
    # seed i of one is directly comparable with seed i of the other.  This is
    # the sample that says whether stripping helps; the best-of-N row above is
    # a min over a wide spread and moves on one lucky seed.
    if {"full", "strip_route"} <= set(arms_wanted):
        w = l = t = 0
        ratios = []
        for r in records:
            for a, b in zip(r["arms"]["full"]["per_seed_eprs"],
                            r["arms"]["strip_route"]["per_seed_eprs"]):
                if a is None or b is None:
                    continue
                ratios.append(b / a)
                w, l, t = (w + 1, l, t) if b < a else (w, l + 1, t) if b > a else (w, l, t + 1)
        print(f"\n  paired per-seed (n={len(ratios)}): strip_route better on {w}, "
              f"worse on {l}, tied on {t}; "
              f"gmean ratio {gmean(ratios):.4f} ({100 * (gmean(ratios) - 1):+.2f}% EPR)",
              flush=True)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
