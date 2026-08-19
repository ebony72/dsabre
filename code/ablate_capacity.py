"""Ablation of the capacity mechanism, separating its two jobs.

Since 2026-08-09 capacity is handled in two places that the submitted design
collapsed into one, and `tab:mech`'s single "No capacity penalty" row can no
longer measure both:

  * a LEGALITY condition (`safe_mode=True`, `tier1_floor=2`): a teleport is
    legal only if its destination keeps a free slot, so no core reaches 0;
  * a SOFT SCORE TERM (`cap_penalty`), which grades preference among the
    moves legality already allows.

This driver runs the 2x2, plus the strict-floor variant, so the two can be
read apart:

  full         legality on,  penalty on    (the default; reproduces tab:main)
  no_soft      legality on,  penalty off   (what tab:mech's row now measures)
  no_legality  legality off, penalty on    (capacity as a score term only --
                                            the submitted design)
  neither      legality off, penalty off   (no capacity handling at all)
  strict       Tier 1 maintains the full reserve unaided (tier1_floor=None)

Protocol is benchmark.py's, via its own helpers, so "full" is directly
comparable to tab:main: SabreLayout corners-removed x3 seeds, fwd->bwd->fwd,
best of pass1/pass3, dSE only.

Aborts are the point of the exercise, not a nuisance: report them, and give a
matched geometric mean over the circuits every arm completes so the surviving
arms are not flattered by the ones that dropped out.
"""
import os, sys, glob, json, math, time
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit.converters import circuit_to_dag

import benchmark as B
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

_OUT = os.path.join(_HERE, "results", "results_ablate_capacity.json")

ARMS = {
    "full":        dict(safe_mode=True,  tier1_floor=2,    cap_penalty=15.0),
    "no_soft":     dict(safe_mode=True,  tier1_floor=2,    cap_penalty=0.0),
    "no_legality": dict(safe_mode=False,                   cap_penalty=15.0),
    "neither":     dict(safe_mode=False,                   cap_penalty=0.0),
    "strict":      dict(safe_mode=True,  tier1_floor=None, cap_penalty=15.0),
}


def gmean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def run_arm(suite_name: str, override: dict):
    s = B.SUITES[suite_name]
    arch = s["arch"]
    hw = replace(s["hw"], **override)
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
        best_m, n_layout_aborts = None, 0
        t0 = time.perf_counter()
        for layout in sl_layouts:
            try:
                m = run_sabre_passes(router, dag, rev_dag, layout)
            except Exception as e:                    # infeasible architecture
                print(f"    {cname}: RAISED {type(e).__name__}: {e}",
                      flush=True)
                m = None
            if m and not m.get("aborted"):
                if best_m is None or m["eprs"] < best_m["eprs"]:
                    best_m = m
            else:
                n_layout_aborts += 1
        el = time.perf_counter() - t0
        rows.append(dict(
            circuit=cname, cx=n_cx,
            eprs=(best_m["eprs"] if best_m else None),
            ls=(best_m["ls"] if best_m else None),
            aborted=best_m is None,
            layout_aborts=n_layout_aborts, n_layouts=len(sl_layouts),
            safe_routes=(best_m.get("safe_routes", 0) if best_m else None),
            force_make_room=(best_m.get("force_make_room", 0)
                             if best_m else None),
            secs=round(el, 2)))
        print(f"    {cname:11} cx={n_cx:6d} eprs={rows[-1]['eprs']} "
              f"abort={rows[-1]['aborted']} "
              f"({n_layout_aborts}/{len(sl_layouts)} layouts) ({el:.1f}s)",
              flush=True)
    return rows


if __name__ == "__main__":
    suites = sys.argv[1:] or ["25q", "64q"]
    out = json.load(open(_OUT)) if os.path.exists(_OUT) else {}
    for suite in suites:
        out.setdefault(suite, {})
        for label, override in ARMS.items():
            print(f"== {suite} / {label} ({override}) ==", flush=True)
            rows = run_arm(suite, override)
            out[suite][label] = dict(
                config=override, rows=rows,
                gmean_own=gmean([r["eprs"] for r in rows]),
                n_aborted=sum(1 for r in rows if r["aborted"]),
                n_circuits=len(rows))
            print(f"  -> gmean(own)={out[suite][label]['gmean_own']:.1f}  "
                  f"aborted={out[suite][label]['n_aborted']}/{len(rows)}\n",
                  flush=True)
            json.dump(out, open(_OUT, "w"), indent=2)

        # Matched comparison: only circuits every arm completed.
        arms = out[suite]
        common = [r["circuit"] for r in arms["full"]["rows"]]
        for label in arms:
            done = {r["circuit"] for r in arms[label]["rows"]
                    if not r["aborted"]}
            common = [c for c in common if c in done]
        print(f"-- {suite}: matched over {len(common)} circuits "
              f"completed by every arm --", flush=True)
        base = None
        for label in ARMS:
            by = {r["circuit"]: r["eprs"] for r in arms[label]["rows"]}
            g = gmean([by[c] for c in common])
            arms[label]["gmean_matched"] = g
            arms[label]["n_matched"] = len(common)
            if base is None:
                base = g
            delta = 100.0 * (g - base) / base
            arms[label]["delta_vs_full_pct"] = delta
            print(f"   {label:12} gmean={g:8.1f}  {delta:+7.1f}%  "
                  f"aborts={arms[label]['n_aborted']}", flush=True)
        json.dump(out, open(_OUT, "w"), indent=2)
    print(f"\nSaved -> {_OUT}", flush=True)
