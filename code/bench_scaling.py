r"""
bench_scaling.py — scalability on two decomposed axes, plus a granularity control.

The previous large-circuit series confounded two things: 100q->200q doubled the
core count while 360q instead grew the cores (2x3 of 9x9), so no single quantity
was held fixed and the paper had to concede "the three rows are not one series".
Worse, the banded QFTs interact only within a range of 19, so what actually sets
the inter-core gate fraction is *logical qubits per core* -- 17 at 100/200q
(56-57% of gates inter-core) against 60 at 360q (16.5%).  The old 360q row was
therefore an easier problem wearing a bigger number.

  --design b      core size fixed at 5x5, core count grows: 2x3 -> 3x4 -> 4x5.
                  Qubits per core stay 16.7/16.7/18.0, all below the band, so
                  the inter-core fraction is constant and the only variable is
                  the core graph: diameter 3 -> 5 -> 7, mean path 1.67 -> 3.00.
                  This is the axis nothing in the paper covers.

  --design gran   the same 200-qubit circuit on 6 big cores (2x3 of 7x7, 294
                  physical) instead of 12 small ones (3x4 of 5x5, 300 physical).
                  Near-identical qubit budget, opposite granularity: few big
                  cores or many small ones, with the circuit held fixed.

  --design regcz  the circulant regular-graph CZ family (gen_regcz_scaling.py)
                  on the design-b architectures: degree 4, range 2, so every
                  partner is within two ring positions and the inter-core
                  fraction is set by core size alone rather than drifting with
                  n as a random regular graph's would.  A second workload
                  alongside the banded QFTs, with degree under explicit
                  control.  dSE only -- dS is an ablation variant, and the
                  360q point alone costs hours.

max_iterations is 200,000, not the 20,000 used elsewhere.  Measured
requirements: 17,226 ops for regcz at 100 qubits and 26,537 for the design-b
360q QFT, so the usual cap would abort both -- silently, and looking exactly
like a routing failure.  Every row records the iteration outcome.

Usage:  python3 bench_scaling.py --design {b,gran,regcz}
Output: code/results/results_scaling_{design}.json
"""

import sys, os, json, time, argparse
from math import prod

sys.setrecursionlimit(200000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit.converters import circuit_to_dag

from architecture import build_h_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes, adaptive_corner_count
from benchmark import load_qasm
from bench_large import run_telesabre as _run_ts_large
from bench_80q import export_device_json

_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
DEVICE_DIR = os.path.expanduser("~/Documents/telesabre/devices")

# Recovery budgets derived per architecture at route() entry, not tuned per
# suite -- see benchmark.py's _HW and probe_derived_deadlock.py.
HW = HardwareConfig(deadlock_limit=None, max_backup_attempts=None,
                    max_iterations=None)

# TeleSABRE comes from bench_large.py, not benchmark.py.  The two drivers ship
# different config templates: benchmark.py spells the Hungarian layout
# parameter "init_layout_hun_free_qubit" where the documented key (and
# bench_large.py) is "init_layout_hun_min_free_qubit", so TeleSABRE silently
# falls back to its own default there.  It is not cosmetic -- on 64q QFT the
# two spellings give 410 and 312 EPR.  These are large-circuit rows, so they
# follow bench_large.py and the documented template.
def run_telesabre(path, dev):
    """bench_large's runner, with its 'ts_ls' swap key normalised to 'ls'."""
    r = _run_ts_large(path, dev, timeout=900)
    if r is not None and "ls" not in r:
        r["ls"] = r.pop("ts_ls", None)
    return r

_QFT = lambda n: os.path.expanduser(
    f"~/Documents/telesabre/circuits/qasm_{n}/"
    f"qft_nativegates_ibm_qiskit_opt3_{n}.qasm")
_RCZ = lambda n: os.path.join(_HERE, "circuits_regcz_scaling", f"regcz_circ_d4_n{n}.qasm")

# (label, logical qubits, (r, s, m), circuit path, expected CX)
DESIGNS = {
    "b": [("100q", 100, (2, 3, 5), _QFT(100),  3420),
          ("200q", 200, (3, 4, 5), _QFT(200),  7220),
          ("360q", 360, (4, 5, 5), _QFT(360), 13300)],
    "gran": [("200q-6x7x7",  200, (2, 3, 7), _QFT(200), 7220),
             ("200q-12x5x5", 200, (3, 4, 5), _QFT(200), 7220)],
    "regcz": [("100q", 100, (2, 3, 5), _RCZ(100),  3400),
              ("200q", 200, (3, 4, 5), _RCZ(200),  6800),
              ("360q", 360, (4, 5, 5), _RCZ(360), 12240)],
}
# dSE is \dSABRE{}.  dS (topological extended set) is run only where the two
# extended-set constructions are being compared -- Table VI, row 3 -- not as a
# second column everywhere, so these runs report dSE against TeleSABRE alone.
ROUTERS = {"b": ("dSE",), "gran": ("dSE",), "regcz": ("dSE",)}


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", choices=list(DESIGNS), required=True)
    a = ap.parse_args()
    rows = DESIGNS[a.design]
    out_path = os.path.join(_RESULTS_DIR, f"results_scaling_{a.design}.json")

    records = []
    for label, nq, (r, s, m), path, exp_cx in rows:
        if not os.path.exists(path):
            print(f"{label}: missing {path}", flush=True)
            continue
        arch = build_h_grid_architecture(r, s, m)
        import networkx as nx
        k = adaptive_corner_count(arch, nq)
        dev_name = f"H_grid_{r}_{s}_{m}_{m}"
        device = os.path.join(DEVICE_DIR, f"{dev_name}.json")
        export_device_json(arch, device, dev_name, r, s, m)

        qc, dag = load_qasm(path)
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        if exp_cx is not None and n_cx != exp_cx:
            raise SystemExit(f"{label}: {n_cx} CX, expected {exp_cx}; refusing "
                             f"to benchmark a different circuit")
        rev = circuit_to_dag(qc.reverse_ops())

        print(f"\n== {label}  {r}x{s} of {m}x{m}: {arch.num_cores} cores, "
              f"{len(arch.data_qubits)} physical, {len(arch.inter_core_links)} "
              f"links, core diam {nx.diameter(arch.core_graph)}, "
              f"fill {nq/len(arch.data_qubits):.2f}, k={k}, CX={n_cx}",
              flush=True)

        t0 = time.time()
        ts = run_telesabre(path, device)
        print(f"  {'TeleSABRE':<6} EPR={ts['eprs'] if ts else '---':>7}  "
              f"SWAP={ts['ls'] if ts else '---':>7}  ({time.time()-t0:.0f}s)",
              flush=True)

        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        rr = {}
        for key in ROUTERS[a.design]:
            cls = General_dSABRE_Router if key == "dS" else dSABRE_BurstExt
            router = cls(arch, HW)
            # `time_s` is the cost of the PROTOCOL -- every layout attempted,
            # summed, including ones that abort, because that time is spent
            # either way.  `time_seed_s` keeps the winning seed alone.  At 360q
            # the two differ by more than the factor of three the seed count
            # suggests: one layout aborts, so the total is 1843s against a
            # 743s best seed.  Recording only the winner understated dSABRE
            # threefold against pytket-dqc, whose times are search totals.
            t0, best, aborts = time.time(), None, 0
            for layout in layouts:
                mt = run_sabre_passes(router, dag, rev, layout)
                if mt and not mt.get("aborted"):
                    if best is None or mt["eprs"] < best["eprs"]:
                        best = mt
                else:
                    aborts += 1
            protocol_s = round(time.time() - t0, 1)
            rr[key] = (dict(eprs=best["eprs"], ls=best["ls"],
                            ops=best["eprs"] + best["ls"],
                            time_s=protocol_s,
                            time_seed_s=round(best["compile_time"], 1),
                            layout_aborts=aborts, aborted=False)
                       if best else dict(aborted=True, layout_aborts=aborts,
                                         time_s=protocol_s))
            d = rr[key]
            print(f"  {key:<6} EPR={d.get('eprs','ABORT'):>7}  "
                  f"SWAP={d.get('ls',0):>7}  ops={d.get('ops',0):>7}  "
                  f"({time.time()-t0:.0f}s, {aborts}/{len(layouts)} layouts aborted)",
                  flush=True)

        records.append(dict(design=a.design, label=label, qubits=nq,
                            arch=f"{r}x{s} of {m}x{m}", cores=arch.num_cores,
                            physical=len(arch.data_qubits),
                            links=len(arch.inter_core_links),
                            core_diameter=nx.diameter(arch.core_graph),
                            qubits_per_core=round(nq / arch.num_cores, 1),
                            reserved_corners=k, cx=n_cx, ts=ts, routers=rr))
        with open(out_path, "w") as f:              # save early and often
            json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"),
                                     design=a.design,
                                     max_iterations=HW.max_iterations,
                                     layout="SabreLayout per-core adaptive "
                                            "corner reservation, best of 3",
                                     pass_strategy="fwd -> bwd -> fwd"),
                           results=records), f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
