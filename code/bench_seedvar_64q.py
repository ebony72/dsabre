r"""
bench_seedvar_64q.py — per-seed 64q results, for the best-vs-median check.

The main tables report best-of-3 for both routers, each following its authors'
convention.  Section IV-A claims that the margin is not an artefact of that
choice, and backs it with a median-of-3 recomputation -- which needs every
seed's result, not just the winner.  benchmark.py keeps only the best, so this
driver records all three for both routers under the identical protocol:

  dSABRE    the three SabreLayout seeds of sabre_locked_boundary_layout, each
            run through fwd -> bwd (reversed DAG) -> fwd, best of pass 1/3
  TeleSABRE seeds 0-2, its own optimize_initial Hungarian layout

An earlier results/seed_variance_64q.json exists with no producer script in the
repo.  Its TeleSABRE column matches the corrected config, but its dSABRE column
does not match the current protocol at all (ae best 216 against the table's
242, ghz 16 against 9), so it is not used here.

Circuits run cheapest-first so partial results are usable early.

Output: code/results/results_seedvar_64q.json
"""

import sys, os, json, time
from math import prod
from statistics import median

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit.converters import circuit_to_dag

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from benchmark import load_qasm, TS_BIN, _ts_config

_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
CIRCUIT_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_64")
DEVICE = os.path.expanduser("~/Documents/telesabre/devices/H_grid_2_3_4_4.json")

ARCH = build_h_grid_architecture(2, 3, 4)
HW = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                    max_iterations=20000)

# cheapest first; the five-circuit TeleSABRE-matched set the paper cites is
# ae, ghz, graphstate, qft, qnn (random is the sixth, and TeleSABRE fails it)
CIRCUITS = ["ghz", "graphstate", "ae", "qft", "qpeexact", "random",
            "qaoa", "qnn", "multiplier"]
EXPECTED_CX = {"ae": 1962, "ghz": 63, "graphstate": 64, "qft": 1966,
               "qnn": 8126, "random": 1627, "qpeexact": 2139, "qaoa": 3920,
               "multiplier": 13040}


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def telesabre_per_seed(qasm):
    """EPR per seed, None where the seed does not converge."""
    import tempfile, subprocess
    out = []
    for seed in range(3):
        rpt = tempfile.mktemp(suffix=".json")
        cfg = _ts_config(seed, rpt, "seedvar")
        try:
            p = subprocess.run([TS_BIN, cfg, DEVICE, qasm], capture_output=True,
                               text=True, timeout=600)
        except subprocess.TimeoutExpired:
            os.unlink(cfg); out.append(None); continue
        td = tg = 0; ok = False
        for line in (p.stdout + p.stderr).splitlines():
            t = line.strip()
            if   "Teledata:" in t: td = int(t.split(":")[1])
            elif "Telegate:" in t: tg = int(t.split(":")[1])
            elif "Success: true" in t: ok = True
        os.unlink(cfg)
        if os.path.exists(rpt): os.unlink(rpt)
        out.append(td + tg if ok else None)
    return out


def main():
    out_path = os.path.join(_RESULTS_DIR, "results_seedvar_64q.json")
    records = []
    router = dSABRE_BurstExt(ARCH, HW)
    print(f"{'circuit':<12}{'TeleSABRE per seed':<26}{'dSABRE per seed':<26}",
          flush=True)
    for name in CIRCUITS:
        qf = os.path.join(CIRCUIT_DIR,
                          f"{name}_nativegates_ibm_qiskit_opt3_64.qasm")
        qc, dag = load_qasm(qf)
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        if EXPECTED_CX.get(name) not in (None, n_cx):
            raise SystemExit(f"{name}: {n_cx} CX, expected {EXPECTED_CX[name]}")
        rev = circuit_to_dag(qc.reverse_ops())
        t0 = time.time()

        ts = telesabre_per_seed(qf)
        ds = []
        for layout in sabre_locked_boundary_layout(qc, dag, ARCH, seed=0):
            m = run_sabre_passes(router, dag, rev, layout)
            ds.append(m["eprs"] if m and not m.get("aborted") else None)

        records.append(dict(circuit=name, cx=n_cx, ts_per_seed=ts,
                            dsabre_per_seed=ds))
        print(f"{name:<12}{str(ts):<26}{str(ds):<26}"
              f"({time.time()-t0:.0f}s)", flush=True)
        with open(out_path, "w") as f:
            json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"),
                                     arch="H-grid 2x3 of 4x4 (96 physical)",
                                     protocol="dSABRE: 3 SabreLayout seeds, "
                                              "fwd->bwd->fwd each; TeleSABRE: "
                                              "seeds 0-2"),
                           results=records), f, indent=2)

    # the paper's basis: circuits where TeleSABRE converges on every seed
    matched = [r for r in records
               if all(v is not None for v in r["ts_per_seed"])
               and all(v is not None for v in r["dsabre_per_seed"])
               and r["circuit"] in ("ae", "ghz", "graphstate", "qft", "qnn")]
    if matched:
        for stat, fn in (("best", min), ("median", median)):
            t = gmean([fn(r["ts_per_seed"]) for r in matched])
            d = gmean([fn(r["dsabre_per_seed"]) for r in matched])
            print(f"  {stat:<7} of 3, {len(matched)}-circuit matched set: "
                  f"TS {t:.1f}  dSABRE {d:.1f}  -> {100*(d-t)/t:+.1f}%",
                  flush=True)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
