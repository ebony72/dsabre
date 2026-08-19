"""regen_routers_scaling.py — rerun only the dSABRE column of a scaling file.

`results_scaling_b.json` feeds the QFT-scalability panel of the paper's main
table (`gen_main_merged.py`), and holds TeleSABRE alongside dSABRE.  Adopting
the shared intra-core extended set changed only the latter, so this rerun
touches `routers` and leaves `ts` as recorded.

Protocol is `bench_scaling.py`'s, including its record fields (`ops`,
`layout_aborts`, protocol-total `time_s`).

Usage:  python3 regen_routers_scaling.py [--design b]
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_h_grid_architecture
from bench_scaling import DESIGNS, HW, ROUTERS
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from router import General_dSABRE_Router


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="b", choices=list(DESIGNS))
    args = ap.parse_args()

    path = os.path.join(_HERE, "results", f"results_scaling_{args.design}.json")
    doc = json.load(open(path))
    by_label = {r["label"]: r for r in doc["results"]}

    for label, nq, (r, s, m), qasm, cx in DESIGNS[args.design]:
        rec = by_label.get(label)
        if rec is None:
            print(f"{label}: not in the results file, skipping", flush=True)
            continue
        if not os.path.exists(qasm):
            print(f"{label}: {qasm} missing, leaving record as is", flush=True)
            continue
        qc, dag = load_qasm(qasm)
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        if n_cx != rec["cx"]:
            sys.exit(f"{label}: CX {n_cx} != recorded {rec['cx']}, refusing to merge")
        rev = circuit_to_dag(qc.reverse_ops())
        arch = build_h_grid_architecture(r, s, m)
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        print(f"\n{label}  ({nq} logical, {r}x{s} of {m}x{m}, {n_cx} CX)", flush=True)

        for key in ROUTERS[args.design]:
            cls = General_dSABRE_Router if key == "dS" else dSABRE_BurstExt
            router = cls(arch, HW)
            t0, best, aborts = time.time(), None, 0
            for layout in layouts:
                mt = run_sabre_passes(router, dag, rev, layout)
                if mt and not mt.get("aborted"):
                    if best is None or mt["eprs"] < best["eprs"]:
                        best = mt
                else:
                    aborts += 1
            protocol_s = round(time.time() - t0, 1)
            old = rec["routers"].get(key, {})
            rec["routers"][key] = (
                dict(eprs=best["eprs"], ls=best["ls"],
                     ops=best["eprs"] + best["ls"], time_s=protocol_s,
                     time_seed_s=round(best["compile_time"], 1),
                     layout_aborts=aborts, aborted=False)
                if best else dict(aborted=True, layout_aborts=aborts,
                                  time_s=protocol_s))
            n = rec["routers"][key]
            print(f"  {key}: EPR {old.get('eprs', 'ABORT')} -> "
                  f"{n.get('eprs', 'ABORT')}, SWAP {old.get('ls')} -> "
                  f"{n.get('ls')}, {protocol_s}s "
                  f"({aborts}/{len(layouts)} layouts aborted)", flush=True)
            json.dump(doc, open(path, "w"), indent=1)

    doc.setdefault("meta", {})["routers_regenerated"] = time.strftime("%Y-%m-%d")
    doc["meta"]["routers_note"] = ("dSABRE column rerun after the shared "
                                   "intra-core extended set became the default "
                                   "(2026-08-08); ts unchanged")
    json.dump(doc, open(path, "w"), indent=1)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
