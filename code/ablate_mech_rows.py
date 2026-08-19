"""tab:mech, rows 1-2: disable capacity penalty / extended-set lookahead.

Row 3 ("Topological extended set, not BFS") is benchmark.py's own dS column
-- General_dSABRE_Router at default config -- so it needs no separate driver;
see results_{suite}.json's "dS" entries.

This script reuses benchmark.py's own helpers directly (SUITES, load_qasm,
CANONICAL_CIRCUITS) so the protocol is identical to what produced tab:main
and the "Full" baseline: SabreLayout corners-removed x3 seeds, fwd->bwd->fwd,
best of pass1/pass3, dSE (dSABRE_BurstExt) only. Only cap_penalty or
weight_extended is changed from the suite's own HardwareConfig.
"""
import os, sys, glob, json, time
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit.converters import circuit_to_dag

import benchmark as B
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

_OUT = os.path.join(_HERE, "results", "results_mech_ablation.json")


def run_row(suite_name: str, hw_override):
    s = B.SUITES[suite_name]
    arch = s["arch"]
    hw = replace(s["hw"], **hw_override)
    router = dSABRE_BurstExt(arch, hw)

    qasm_files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
    canon = B.CANONICAL_CIRCUITS.get(suite_name)
    if canon:
        qasm_files = [f for f in qasm_files
                     if os.path.basename(f).replace(s["suffix"], "") in canon]

    rows = []
    for qf in qasm_files:
        cname = os.path.basename(qf).replace(s["suffix"], "")
        qc, dag = B.load_qasm(qf)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        sl_layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        best_m = None
        t0 = time.perf_counter()
        for layout in sl_layouts:
            m = run_sabre_passes(router, dag, rev_dag, layout)
            if m and not m.get("aborted"):
                if best_m is None or m["eprs"] < best_m["eprs"]:
                    best_m = m
        el = time.perf_counter() - t0
        aborted = best_m is None
        rows.append(dict(circuit=cname, cx=n_cx,
                         eprs=(best_m["eprs"] if best_m else None),
                         ls=(best_m["ls"] if best_m else None),
                         aborted=aborted, secs=round(el, 2)))
        print(f"    {cname:11} cx={n_cx:6d} eprs={rows[-1]['eprs']} "
              f"aborted={aborted} ({el:.1f}s)", flush=True)
    return rows


def gmean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return float("nan")
    import math
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


CONFIGS = {
    "no_cap_penalty":   dict(cap_penalty=0.0),
    "no_lookahead":     dict(weight_extended=0.0),
}

if __name__ == "__main__":
    suites = sys.argv[1:] or ["25q", "64q"]
    out = {}
    if os.path.exists(_OUT):
        try:
            out = json.load(open(_OUT))
        except Exception:
            out = {}
    for suite in suites:
        out.setdefault(suite, {})
        for label, override in CONFIGS.items():
            print(f"== {suite} / {label} ({override}) ==", flush=True)
            rows = run_row(suite, override)
            g = gmean([r["eprs"] for r in rows])
            n_abort = sum(1 for r in rows if r["aborted"])
            out[suite][label] = dict(rows=rows, gmean=g, n_aborted=n_abort)
            print(f"  -> gmean={g:.1f}  aborted={n_abort}/{len(rows)}\n",
                  flush=True)
            with open(_OUT, "w") as f:
                json.dump(out, f, indent=2)
    print(f"Saved -> {_OUT}", flush=True)
