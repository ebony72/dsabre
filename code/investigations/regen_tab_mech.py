"""regen_tab_mech.py — the scoring-term ablation of Table VI (tab:mech).

Four configurations, geometric-mean EPR over the 25q and 64q suites:

  Full                                  the default router
  No capacity penalty (c_pen = 0)       cap_penalty = 0
  No extended-set lookahead (w_e = 0)   weight_extended = 0
  Topological extended set (not BFS)    General_dSABRE_Router (dS)

`Full` and the topological row are read from results_{25,36,64}q.json rather
than rerun -- `benchmark.py` already produces both columns under exactly this
protocol -- so only the two disabled-term configurations are routed here.

Protocol is benchmark.py's: SabreLayout corners-removed best of 3 seeds,
run_sabre_passes (fwd -> bwd -> fwd), best layout by EPR.

Output: results/results_tab_mech.json
"""
import glob
import json
import os
import sys
import time
import warnings
from math import prod

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from benchmark import SUITES, CANONICAL_CIRCUITS
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

OUT = os.path.join(_HERE, "results", "results_tab_mech.json")

CONFIGS = {
    "no_cap_penalty": dict(cap_penalty=0.0),
    "no_lookahead": dict(weight_extended=0.0),
}


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def main():
    doc = {"meta": {"date": time.strftime("%Y-%m-%d"),
                    "note": "scoring-term ablation behind tab:mech; Full and "
                            "the topological row come from "
                            "results_{25,36,64}q.json"},
           "results": {}}

    for sname in ("25q", "64q"):
        s = SUITES[sname]
        arch, hw = s["arch"], s["hw"]
        files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
        canon = CANONICAL_CIRCUITS.get(sname)
        if canon:
            files = [f for f in files
                     if os.path.basename(f).replace(s["suffix"], "") in canon]

        # Full and dS from the canonical results file
        canon_doc = json.load(open(os.path.join(_HERE, "results",
                                                f"results_{sname}.json")))
        full = {r["circuit"]: r["routers"]["dSE"] for r in canon_doc["results"]}
        topo = {r["circuit"]: r["routers"].get("dS", {})
                for r in canon_doc["results"]}
        doc["results"].setdefault(sname, {})
        doc["results"][sname]["full"] = {
            c: (None if v.get("aborted") else v["eprs"]) for c, v in full.items()}
        doc["results"][sname]["topological"] = {
            c: (None if v.get("aborted") else v.get("eprs")) for c, v in topo.items()}

        print(f"\n{'='*66}\n  {sname}\n{'='*66}", flush=True)
        print(f"{'circuit':<12} {'full':>8} {'topo':>8} "
              + " ".join(f"{k:>16}" for k in CONFIGS), flush=True)

        for cfg_name, kw in CONFIGS.items():
            doc["results"][sname].setdefault(cfg_name, {})
        for f in files:
            cname = os.path.basename(f).replace(s["suffix"], "")
            qc, dag = load_qasm(f)
            rev_dag = circuit_to_dag(qc.reverse_ops())
            layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
            row = []
            for cfg_name, kw in CONFIGS.items():
                cfg = HardwareConfig(**{**hw.__dict__, **kw})
                router = dSABRE_BurstExt(arch, cfg)
                best, aborts = None, 0
                for layout in layouts:
                    m = run_sabre_passes(router, dag, rev_dag, layout)
                    if m and not m.get("aborted"):
                        if best is None or m["eprs"] < best["eprs"]:
                            best = m
                    else:
                        aborts += 1
                val = None if best is None else best["eprs"]
                doc["results"][sname][cfg_name][cname] = val
                doc["results"][sname][cfg_name].setdefault("_aborts", {})
                doc["results"][sname][cfg_name]["_aborts"][cname] = aborts
                row.append(f"{'ABORT' if val is None else val:>16}")
            print(f"{cname:<12} {str(doc['results'][sname]['full'][cname]):>8} "
                  f"{str(doc['results'][sname]['topological'][cname]):>8} "
                  + " ".join(row), flush=True)
            json.dump(doc, open(OUT, "w"), indent=1)

    print(f"\n{'='*66}\n  gmean EPR (tab:mech)\n{'='*66}", flush=True)
    print(f"{'configuration':<34} {'25q':>10} {'Δ%':>9} {'64q':>10} {'Δ%':>9}",
          flush=True)
    base = {su: gmean(list(doc["results"][su]["full"].values()))
            for su in ("25q", "64q")}
    rows = [("Full (baseline)", "full"),
            ("No capacity penalty (c_pen=0)", "no_cap_penalty"),
            ("No extended-set lookahead (w_e=0)", "no_lookahead"),
            ("Topological extended set (not BFS)", "topological")]
    for label, key in rows:
        cells = []
        for su in ("25q", "64q"):
            vals = [v for c, v in doc["results"][su][key].items()
                    if not c.startswith("_")]
            g = gmean(vals)
            n_ab = sum(1 for v in vals if v is None)
            d = "---" if key == "full" else f"{100*(g-base[su])/base[su]:+.1f}"
            cells.append(f"{g:>10.1f}" + (f"{d:>9}" if key != "full" else f"{'---':>9}")
                         + ("*" if n_ab else " "))
        print(f"{label:<34} " + " ".join(cells), flush=True)
    print("\n* one or more circuits aborted; the gmean is over the rest",
          flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
