"""
ablate_node_decay.py — ablation: node decay ON vs OFF on 25q and 64q suites.

Runs dSE (dSABRE_BurstExt) with enable_node_decay=True and =False on every
circuit in the chosen suite, using the same layout seeds and fwd/bwd/fwd SABRE
passes as the main benchmark.  Reports EPR and local-SWAP counts plus a gmean
summary.  Usage: python3 ablate_node_decay.py [--suite 25q|64q]
"""
import sys, os, math, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

SUITES = {
    "25q": dict(
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
        suffix="_nativegates_ibm_qiskit_opt3_25.qasm",
        arch=build_b_grid_architecture(r=2, s=2, m=4),
        hw_kwargs={},
    ),
    "36q": dict(
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_36"),
        suffix="_nativegates_ibm_qiskit_opt3_36.qasm",
        arch=build_b_grid_architecture(r=2, s=2, m=4),
        hw_kwargs={},
    ),
    "64q": dict(
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
        suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
        arch=build_h_grid_architecture(r=2, s=3, m=4),
        hw_kwargs=dict(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000),
    ),
}

_pm = PassManager([RemoveBarriers()])


def load_circuit(path):
    qc = _pm.run(QuantumCircuit.from_qasm_file(path))
    dag = circuit_to_dag(qc)
    rev_dag = circuit_to_dag(qc.reverse_ops())
    return qc, dag, rev_dag


def best_over_layouts(router, qc, dag, rev_dag, arch):
    layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
    best = None
    for layout in layouts:
        m = run_sabre_passes(router, dag, rev_dag, layout)
        if m is None:
            continue
        if best is None or m["eprs"] < best["eprs"]:
            best = m
    return best


def gmean(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def bench_suite(suite_name, cfg):
    arch       = cfg["arch"]
    hw_kwargs  = cfg["hw_kwargs"]
    cdir       = cfg["circuit_dir"]
    suffix     = cfg["suffix"]

    hw_on  = HardwareConfig(enable_node_decay=True,  **hw_kwargs)
    hw_off = HardwareConfig(enable_node_decay=False, **hw_kwargs)

    qasm_files = sorted(f for f in os.listdir(cdir) if f.endswith(suffix))

    print(f"\n{'─'*76}", flush=True)
    print(f"  Node-decay ablation — {suite_name} (dSE, fwd/bwd/fwd, 3 layout seeds)", flush=True)
    print(f"{'─'*76}", flush=True)
    hdr = f"  {'Circuit':<14}  {'EPR(on)':>8}  {'EPR(off)':>8}  {'ΔEPR%':>7}  {'LS(on)':>7}  {'LS(off)':>7}"
    print(hdr, flush=True)
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}", flush=True)

    epr_on_all, epr_off_all = [], []
    rows = []

    for fname in qasm_files:
        circ = fname.replace(suffix, "")
        qc, dag, rev_dag = load_circuit(os.path.join(cdir, fname))

        r_on  = dSABRE_BurstExt(arch, hw_on)
        r_off = dSABRE_BurstExt(arch, hw_off)

        t0 = time.perf_counter()
        m_on  = best_over_layouts(r_on,  qc, dag, rev_dag, arch)
        m_off = best_over_layouts(r_off, qc, dag, rev_dag, arch)
        elapsed = time.perf_counter() - t0

        if m_on is None or m_off is None:
            status_on  = "ABORT" if m_on  is None else str(m_on["eprs"])
            status_off = "ABORT" if m_off is None else str(m_off["eprs"])
            print(f"  {circ:<14}  {status_on:>8}  {status_off:>8}", flush=True)
            rows.append(dict(circuit=circ, on=None if m_on is None else dict(eprs=m_on["eprs"], ls=m_on["ls"]),
                             off=None if m_off is None else dict(eprs=m_off["eprs"], ls=m_off["ls"])))
            continue

        epr_on, epr_off = m_on["eprs"], m_off["eprs"]
        ls_on,  ls_off  = m_on["ls"],   m_off["ls"]
        rows.append(dict(circuit=circ, on=dict(eprs=epr_on, ls=ls_on), off=dict(eprs=epr_off, ls=ls_off), time_s=round(elapsed,2)))
        pct = (epr_off - epr_on) / max(epr_on, 1) * 100
        sign = "+" if pct >= 0 else ""
        print(f"  {circ:<14}  {epr_on:>8}  {epr_off:>8}  {sign}{pct:>6.1f}%  {ls_on:>7}  {ls_off:>7}  ({elapsed:.1f}s)", flush=True)
        epr_on_all.append(epr_on)
        epr_off_all.append(epr_off)

    gm_on  = gmean(epr_on_all)
    gm_off = gmean(epr_off_all)
    gm_pct = (gm_off - gm_on) / gm_on * 100 if gm_on > 0 else float("nan")
    sign = "+" if gm_pct >= 0 else ""
    print(f"{'─'*76}", flush=True)
    print(f"  {'gmean':<14}  {gm_on:>8.1f}  {gm_off:>8.1f}  {sign}{gm_pct:>6.1f}%", flush=True)
    print(f"\n  Positive ΔEPR% = decay=ON wins (off costs more EPRs).", flush=True)
    print(flush=True)

    out_dir = os.environ.get("DSABRE_OUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"ablate_node_decay_{suite_name}.json")
    with open(out, "w") as f:
        json.dump({"suite": suite_name, "rows": rows,
                   "gmean": {"on": gm_on, "off": gm_off, "pct": gm_pct}}, f, indent=2)
    print(f"  → saved {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["25q", "36q", "64q"], default=None,
                   help="Which suite to run (default: all)")
    args = p.parse_args()
    suites = [args.suite] if args.suite else ["25q", "36q", "64q"]
    for s in suites:
        bench_suite(s, SUITES[s])
