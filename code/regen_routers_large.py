"""regen_routers_large.py — rerun only the dSABRE columns of the large suites.

The 100/200/360-qubit results files hold three tools: `ts` (TeleSABRE) and the
two dSABRE routers under `routers`.  Adopting the shared intra-core extended
set (2026-08-08) changed the dSABRE numbers and nothing else, so this rerun
touches `routers` and leaves `ts` exactly as recorded.  TeleSABRE is a
subprocess with 300-600 s per-seed timeouts on these suites and is
deterministic -- verified on the 25/36/64-qubit and heavy-hex suites, where a
full rerun reproduced every TeleSABRE count -- so rerunning it here would cost
hours and change nothing.  `pytket-dqc` needs no rerun at all: its results
files carry no dSABRE column, and the comparison tables join the two at
table-generation time.

Protocol is `bench_large.py`'s `run_dsabre` verbatim, including the metric
fields it records, so the merged records stay the shape `gen_tables.py`
expects.

Usage:  python3 regen_routers_large.py [--suite 100q] [--routers dSE]
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_heavy_hex_architecture
from bench_large import SUITES as _LARGE, _HW as _HW_LARGE
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from router import General_dSABRE_Router
from circuit_paths import circuits_path

ROUTERS = {"dS": General_dSABRE_Router, "dSE": dSABRE_BurstExt}

# bench_heavyhex.py's configuration.  It is included here because that driver
# refuses to overwrite a completed run (it wants --out-tag), and because it
# would rerun TeleSABRE, which the adoption did not change.
_HW_HH = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                        max_iterations=20000)
_C64 = circuits_path("qasm_64")
_SFX64 = "_nativegates_ibm_qiskit_opt3_64.qasm"

TARGETS = {
    "100q": dict(file="results_100q", hw=_HW_LARGE, **{
        k: _LARGE["100q"][k] for k in ("circuit_dir", "suffix", "arch")}),
    "200q": dict(file="results_200q", hw=_HW_LARGE, **{
        k: _LARGE["200q"][k] for k in ("circuit_dir", "suffix", "arch")}),
    "360q": dict(file="results_360q", hw=_HW_LARGE, **{
        k: _LARGE["360q"][k] for k in ("circuit_dir", "suffix", "arch")}),
    "hh-ring": dict(file="results_heavyhex", hw=_HW_HH, circuit_dir=_C64,
                    suffix=_SFX64, arch=build_heavy_hex_architecture(4, "ring")),
    "hh-star": dict(file="results_heavyhex_star", hw=_HW_HH, circuit_dir=_C64,
                    suffix=_SFX64,
                    arch=build_heavy_hex_architecture(4, "star")),
}


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def run_dsabre(router, qc, dag, rev_dag, arch, label):
    """`bench_large.run_dsabre`, verbatim apart from taking a built router."""
    sl_layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
    best = None
    for i, layout in enumerate(sl_layouts):
        t0 = time.perf_counter()
        m = run_sabre_passes(router, dag, rev_dag, layout)
        elapsed = time.perf_counter() - t0
        if m and not m.get("aborted"):
            print(f"    {label} sl_seed{i}: EPR={m['eprs']}, SWAP={m['ls']} "
                  f"({elapsed:.1f}s)", flush=True)
            if best is None or m["eprs"] < best["eprs"]:
                best = dict(eprs=m["eprs"], ls=m["ls"], layout=f"sl_seed{i}",
                            time_s=round(elapsed, 2), aborted=False,
                            backup_activations=m.get("backup_activations", 0),
                            force_make_room=m.get("force_make_room", 0),
                            # `backup_activations` counts only the deadlock
                            # path and so undercounts the guaranteed
                            # transaction; `safe_routes` is the gate count it
                            # actually retired.
                            safe_routes=m.get("safe_routes", 0),
                            safe_route_failed=m.get("safe_route_failed", 0),
                            relay_hops=m.get("relay_hops", 0))
        else:
            print(f"    {label} sl_seed{i}: ABORTED ({elapsed:.1f}s)", flush=True)
    return best if best is not None else dict(aborted=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="",
                    choices=["", "100q", "200q", "360q", "hh-ring", "hh-star"])
    ap.add_argument("--routers", default="dS,dSE")
    args = ap.parse_args()

    suites = [args.suite] if args.suite else list(TARGETS)
    keys = [k for k in args.routers.split(",") if k]

    for su in suites:
        s = TARGETS[su]
        path = os.path.join(_HERE, "results", f"{s['file']}.json")
        if not os.path.exists(path):
            print(f"{su}: {path} missing, skipping", flush=True)
            continue
        with open(path) as fh:
            doc = json.load(fh)
        arch = s["arch"]
        print(f"\n{'='*70}\n  {su}  (rerunning {keys}; ts left untouched)\n{'='*70}",
              flush=True)

        for rec in doc["results"]:
            cname = rec["circuit"]
            qasm = os.path.join(s["circuit_dir"], f"{cname}{s['suffix']}")
            if not os.path.exists(qasm):
                print(f"  {cname}: qasm missing, leaving record as is", flush=True)
                continue
            qc, dag = load_qasm(qasm)
            n_cx = sum(1 for _ in dag.two_qubit_ops())
            if n_cx != rec["cx"]:
                sys.exit(f"  {cname}: CX {n_cx} != recorded {rec['cx']} -- the "
                         f"circuit was regenerated, refusing to merge")
            rev_dag = circuit_to_dag(qc.reverse_ops())
            print(f"  {cname}  ({n_cx} CX)", flush=True)
            for k in keys:
                old = rec["routers"].get(k, {})
                new = run_dsabre(ROUTERS[k](arch, s["hw"]), qc, dag, rev_dag,
                                 arch, k)
                rec["routers"][k] = new
                o = "ABORT" if old.get("aborted") else old.get("eprs")
                n = "ABORT" if new.get("aborted") else new.get("eprs")
                print(f"    {k}: {o} -> {n}", flush=True)
            with open(path, "w") as fh:
                json.dump(doc, fh, indent=1)

        doc.setdefault("meta", {})["routers_regenerated"] = time.strftime("%Y-%m-%d")
        doc["meta"]["routers_note"] = (
            "dSABRE columns rerun after the shared intra-core extended set "
            "became the default (2026-08-08); ts is unchanged from the "
            "previous run")
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1)
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
