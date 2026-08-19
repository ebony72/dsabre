"""
gran_ds_supplement.py -- add the dS (topological extended-set) column to
results_scaling_gran.json.

bench_scaling.py restricts ROUTERS["gran"] to dSE only (see its comment: dS
is run only where the two extended-set constructions are directly compared).
The granularity paragraph in dsabre.tex claims the 2.1x/1.5-point finding
holds "for both routers", which needs a dS number to stand behind it -- this
script supplies just that, on the same two architectures, without changing
bench_scaling.py's default (dSE-only) behaviour for every other design.

Usage: python3 gran_ds_supplement.py
"""
import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit.converters import circuit_to_dag
from architecture import build_h_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from layout import sabre_locked_boundary_layout, run_sabre_passes
import bench_scaling as bs

HW = HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=200000)
OUT = os.path.join(_HERE, "results", "results_scaling_gran.json")


def main():
    d = json.load(open(OUT))
    rows_by_label = {r[0]: r for r in bs.DESIGNS["gran"]}

    for rec in d["results"]:
        label = rec["label"]
        _, nq, (r, s, m), path, exp_cx = rows_by_label[label]
        arch = build_h_grid_architecture(r, s, m)
        qc, dag = bs.load_qasm(path)
        rev = circuit_to_dag(qc.reverse_ops())
        t0 = time.time()
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        router = General_dSABRE_Router(arch, HW)
        best = None
        aborts = 0
        protocol_s = 0.0
        for layout in layouts:
            ts = time.perf_counter()
            m_ = run_sabre_passes(router, dag, rev, layout)
            protocol_s += time.perf_counter() - ts
            if m_ is None or m_.get("aborted"):
                aborts += 1
                continue
            if best is None or m_["eprs"] < best["eprs"]:
                best = m_
        if best is None:
            print(f"{label}: dS ABORTED on all layouts", flush=True)
            rec["routers"]["dS"] = {"aborted": True}
        else:
            rec["routers"]["dS"] = dict(
                eprs=best["eprs"], ls=best["ls"], ops=best.get("ops"),
                layout_aborts=aborts, aborted=False,
                time_seed_s=round(best["compile_time"], 2),
                time_s=round(protocol_s, 1),
            )
            print(f"{label}: dS EPR={best['eprs']} SWAP={best['ls']} "
                  f"({time.time()-t0:.0f}s, {aborts}/3 aborted)", flush=True)
        with open(OUT, "w") as f:
            json.dump(d, f, indent=2)

    print(f"Saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
