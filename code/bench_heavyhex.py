"""
bench_heavyhex.py — architecture-independence check on a non-grid topology.

Runs the 64-qubit suite on a ring of four IBM 27-qubit heavy-hex cores
(108 physical qubits, 4 inter-core links) instead of the H-grid of 4x4 grid
cores used in the main evaluation.  Same circuits, same protocol, same
hyperparameters as benchmark.py — only the architecture changes.

The heavy-hex tile is irregular (degree 1-3, diameter 12 vs 6 for a 4x4 grid),
so this isolates how much of dSABRE's advantage depends on grid-shaped cores.
TeleSABRE runs on the same architecture via a generated device JSON.

Output: code/results/results_heavyhex.json
"""

import sys, os, json, glob, time
from math import prod

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import networkx as nx

from architecture import build_heavy_hex_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from benchmark import run_telesabre, load_qasm

_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

CIRCUIT_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_64")
SUFFIX      = "_nativegates_ibm_qiskit_opt3_64.qasm"
DEVICE_JSON = os.path.expanduser(
    "~/Documents/telesabre/devices/HeavyHex_ring4_27.json")

NUM_CORES = 4
HW = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                    max_iterations=20000)

# The six circuits common to every suite, spanning dense (QNN, Random),
# structured (AE, QFT) and sparse (GHZ, Graphstate) interaction graphs.  This
# is a topology-sensitivity check, not a headline suite; the 13k-CX multiplier
# would dominate wall time on a diameter-12 core without adding evidence.
CIRCUITS = ["ae", "ghz", "graphstate", "qft", "qnn", "random"]

# CX counts the main-text 64q table reports, used as a preflight check.  The
# qnn entry was replaced on 2026-07-09 by a 63-CX circuit and restored from its
# .bak on 2026-07-27; the check stays so a future regeneration cannot silently
# swap a circuit under us again.
EXPECTED_CX = {"ae": 1962, "ghz": 63, "graphstate": 64, "qft": 1966,
               "qnn": 8126, "random": 1627}
OVERRIDES = {}   # circuit directory is canonical again


def export_device_json(arch, path: str, name: str) -> str:
    """Write the architecture in TeleSABRE's device-JSON schema.

    node_positions is presentational only; we lay each tile out with a seeded
    spring embedding and translate the tiles around a circle so the exported
    device renders sensibly.
    """
    qpc = len(arch.core_qubits(0))
    tile = nx.Graph()
    tile.add_edges_from((u % qpc, v % qpc) for u, v in arch.intra[0].edges())
    pos = nx.spring_layout(tile, seed=7, scale=3.0)

    import math
    positions = [None] * len(arch.data_qubits)
    for c in range(arch.num_cores):
        ang = 2 * math.pi * c / arch.num_cores
        cx, cy = 9.0 * math.cos(ang), 9.0 * math.sin(ang)
        for n in arch.core_qubits(c):
            x, y = pos[n % qpc]
            positions[n] = [round(float(x + cx), 4), round(float(y + cy), 4)]

    dev = {"device": {
        "name": name,
        "num_cores": arch.num_cores,
        "num_qubits": len(arch.data_qubits),
        "intra_core_edges": sorted([int(u), int(v)] for u, v in
                                   (e for g in arch.intra.values() for e in g.edges())),
        "inter_core_edges": [[int(u), int(v)] for u, v in arch.inter_core_links],
        "node_positions": positions,
    }}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dev, f, indent=1)
    return path


def gmean(lst):
    lst = [x for x in lst if x is not None and x > 0]
    return prod(lst) ** (1 / len(lst)) if lst else float("nan")


def main():
    arch = build_heavy_hex_architecture(NUM_CORES)
    export_device_json(arch, DEVICE_JSON, "HeavyHex_ring4_27")
    print(f"device → {DEVICE_JSON}", flush=True)
    print(f"cores={arch.num_cores} physical={len(arch.data_qubits)} "
          f"links={len(arch.inter_core_links)} "
          f"core_graph={sorted(arch.core_graph.edges())}", flush=True)

    routers = {"dS": General_dSABRE_Router(arch, HW),
               "dSE": dSABRE_BurstExt(arch, HW)}

    hdr = (f"{'circuit':<12}  {'q':>3}  {'cx':>5}  {'TS_epr':>7}  {'TS_t':>6}"
           f"  {'dS_epr':>7}  {'dS_ls':>7}  {'dSE_epr':>8}  {'dSE_ls':>7}"
           f"  {'dS/TS%':>8}  {'dSE/TS%':>8}")
    print("\n" + "=" * len(hdr), flush=True)
    print("  64q circuits on heavy-hex ring (4 x 27q, 108 physical)", flush=True)
    print("=" * len(hdr), flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    records = []
    out_path = os.path.join(_RESULTS_DIR, "results_heavyhex.json")

    for qf in sorted(glob.glob(os.path.join(CIRCUIT_DIR, "*.qasm"))):
        cname = os.path.basename(qf).replace(SUFFIX, "")
        if cname not in CIRCUITS:
            continue
        t0 = time.time()
        if cname in OVERRIDES:
            print(f"  [{cname}] using {os.path.basename(OVERRIDES[cname])} "
                  f"(directory copy has the wrong CX count)", flush=True)
            qf = OVERRIDES[cname]
        qc, dag = load_qasm(qf)
        from qiskit.converters import circuit_to_dag
        rev_dag = circuit_to_dag(qc.reverse_ops())
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        if EXPECTED_CX.get(cname) not in (None, n_cx):
            raise SystemExit(
                f"circuit mismatch: {cname} has {n_cx} CX gates, the 64q table "
                f"reports {EXPECTED_CX[cname]}. Refusing to benchmark a "
                f"different circuit under the same name.")

        ts = run_telesabre(qf, DEVICE_JSON)
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)

        rr = {}
        for k, router in routers.items():
            best = None
            for layout in layouts:
                m = run_sabre_passes(router, dag, rev_dag, layout)
                if m and not m.get("aborted"):
                    if best is None or m["eprs"] < best["eprs"]:
                        best = m
            rr[k] = (dict(eprs=best["eprs"], ls=best["ls"],
                          time_s=round(best["compile_time"], 3), aborted=False)
                     if best else {"aborted": True})

        def _i(v): return str(v) if v is not None else "---"
        def _p(a, b):
            if a is None or b is None or b == 0: return "    ---"
            return f"{100*(a-b)/b:+.1f}%"

        ts_epr = ts["eprs"] if ts else None
        ts_t = f"{ts['time_s']:.2f}" if ts else "---"
        row = (f"{cname:<12}  {qc.num_qubits:>3}  {n_cx:>5}  {_i(ts_epr):>7}"
               f"  {ts_t:>6}")
        for k in ("dS", "dSE"):
            r = rr[k]
            row += (f"  {_i(r.get('eprs')):>7}  {_i(r.get('ls')):>7}" if k == "dS"
                    else f"  {_i(r.get('eprs')):>8}  {_i(r.get('ls')):>7}")
        for k in ("dS", "dSE"):
            row += f"  {_p(rr[k].get('eprs'), ts_epr):>8}"
        print(row + f"  ({time.time()-t0:.0f}s)", flush=True)

        records.append(dict(suite="heavyhex", circuit=cname,
                            qubits=qc.num_qubits, cx=n_cx, ts=ts, routers=rr))

        with open(out_path, "w") as f:                 # save early and often
            json.dump(dict(meta=dict(
                date=time.strftime("%Y-%m-%d"),
                suite="heavyhex",
                arch=f"ring of {NUM_CORES} IBM 27q heavy-hex cores "
                     f"({len(arch.data_qubits)} physical, "
                     f"{len(arch.inter_core_links)} links)",
                layout="SabreLayout per-core adaptive corner reservation, best of 3",
                pass_strategy="fwd -> bwd (reversed DAG) -> fwd; best of pass1/pass3",
            ), results=records), f, indent=2)

    print("-" * len(hdr), flush=True)
    ts_e = [r["ts"]["eprs"] for r in records if r["ts"]]
    line = f"{'gmean':<12}  {'':>3}  {'':>5}  {gmean(ts_e):>7.1f}  {'':>6}"
    for k in ("dS", "dSE"):
        e = [r["routers"][k].get("eprs") for r in records
             if not r["routers"][k].get("aborted")]
        l = [r["routers"][k].get("ls") for r in records
             if not r["routers"][k].get("aborted")]
        line += (f"  {gmean(e):>7.1f}  {gmean(l):>7.1f}" if k == "dS"
                 else f"  {gmean(e):>8.1f}  {gmean(l):>7.1f}")
    print(line, flush=True)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
