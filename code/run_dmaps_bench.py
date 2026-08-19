"""
Benchmark DMapS (Luo et al., 2025) on the dSABRE 25q / 36q / 64q suites.

DMapS combines:
    DMapS-M: two-stage qubit mapping (KaHyPar partitioning + parallel SABRE)
    DMapS-R: SABRE-derived routing that prioritises local SWAPs and uses
             cat-comm (1 EPR / remote CX) + TP-comm (2 EPR / remote SWAP)

This adapter feeds our B-grid (25q/36q) and H-grid (64q) architectures to
DMapS, then derives:
    epr         = remote CX (1 EPR each, Cat-Comm)
                + 2 * remote SWAP (TP-Comm)
    local_swap  = intra-chip SWAP count
    overall     = local_swap + 10 * remote_cx + 20 * remote_swap  (DMapS weights)
    time        = wall-clock of cpa_map_car_rout (excluding device parsing)

DMapS pins qiskit==0.39.2 / pytket==1.40, which conflicts with the main repo
qiskit; this script MUST be run with the `dmaps` conda env:

    /opt/anaconda3/envs/dmaps/bin/python code/run_dmaps_bench.py 25

Output:  code/results/results_dmaps_bench.json
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Make DMapS importable
# DMapS lives beside this repo under pyzoo/other/; the old ~/Documents/DMapS
# path is kept as a fallback.  This MUST precede the dsabre code directory on
# sys.path: dsabre's own router.py otherwise shadows DMapS's `router` package
# and the import below fails with "'router' is not a package".
_DMAPS_SRC = next(
    (p for p in (
        os.path.expanduser("~/Documents/pyzoo/other/DMapS/src"),
        os.path.expanduser("~/Documents/DMapS/src"),
    ) if os.path.isdir(p)),
    None,
)
if _DMAPS_SRC is None:
    sys.exit("[dmaps] DMapS checkout not found; expected pyzoo/other/DMapS/src")
sys.path.insert(0, _DMAPS_SRC)

# Quiet third-party noise
import logging
from circuit_paths import circuits_path
logging.disable(logging.WARNING)


CIRCUITS_ROOT = circuits_path()
RESULTS_PATH = os.path.join(
    os.environ.get("DSABRE_OUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    "results_dmaps_bench.json")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

NUM_SEEDS = 5   # best of N (DMapS has no seed knob; just rerun) — matches run_pytket_dqc_bench.py
QUBITS_PER_CHIP = 16   # 4x4 chips for all our suites


# ── Device JSON generators ───────────────────────────────────────────────────
def _chip_block(chip_qubits: List[int], couplings: List[Tuple[int, int]]) -> Dict:
    """Build the per-chip JSON block expected by DMapS's QuHardwareInfoReader."""
    qubit_names = [f"Q{q}" for q in chip_qubits]
    return {
        "use calibration file": False,
        "qubits": qubit_names,
        "fidelity": {n: {"x/2": 1.0} for n in qubit_names},
        "couplings": [
            {"qubit pair": [f"Q{u}", f"Q{v}"], "fidelity": 1.0}
            for u, v in couplings
        ],
    }


def _arch_to_dmaps_json(arch, out_path: str) -> Dict[int, int]:
    """
    Serialise a dSABRE DistributedArchitecture into a DMapS device JSON.

    DMapS assigns global qubit indices in the order qubits appear inside each
    chip block (chip 0 first, then chip 1, ...).  We renumber so that global
    index = chip_offset + local_index, giving a clean (idx // chip_size) -> chip
    lookup later.

    Returns the qubit->chip lookup keyed by DMapS's global index.
    """
    chip_ids = sorted(arch.intra.keys())
    qubit_chip: Dict[int, int] = {}
    cfg = {"has multiple chips": True, "chips": {}, "remote connection": []}

    # Renumber qubits per chip so DMapS's global idx matches our convention.
    old_to_new: Dict[int, int] = {}
    new_idx = 0
    for cidx, chip_id in enumerate(chip_ids):
        g = arch.intra[chip_id]
        for q in sorted(g.nodes()):
            old_to_new[q] = new_idx
            qubit_chip[new_idx] = cidx
            new_idx += 1

    for cidx, chip_id in enumerate(chip_ids):
        g = arch.intra[chip_id]
        chip_qubits_new = sorted(old_to_new[q] for q in g.nodes())
        couplings_new = [
            (old_to_new[u], old_to_new[v]) for u, v in g.edges()
        ]
        cfg["chips"][f"chip {cidx}"] = _chip_block(chip_qubits_new, couplings_new)

    # Inter-core links → DMapS "remote connection" entries.
    for u, v in arch.inter_core_links:
        cu, cv = arch.core_of(u), arch.core_of(v)
        cidx_u = chip_ids.index(cu)
        cidx_v = chip_ids.index(cv)
        cfg["remote connection"].append({
            f"chip {cidx_u}": f"Q{old_to_new[u]}",
            f"chip {cidx_v}": f"Q{old_to_new[v]}",
            "fidelity": 0.85,
        })

    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return qubit_chip


# ── Architecture builders ────────────────────────────────────────────────────
def build_arch_for_suite(suite: str):
    """Return a dSABRE DistributedArchitecture for one of our suites."""
    from architecture import build_b_grid_architecture, build_h_grid_architecture
    if suite in ("25", "36"):
        return build_b_grid_architecture(2, 2, 4)   # 64 phys, 4 chips
    if suite == "64":
        return build_h_grid_architecture(2, 3, 4)   # 96 phys, 6 chips
    raise ValueError(f"Unknown suite: {suite}")


# ── Metric extraction ────────────────────────────────────────────────────────
def derive_metrics(mapped_circ, qubit_chip: Dict[int, int]) -> Dict[str, int]:
    """
    Classify each 2-qubit op in the DMapS-routed circuit:
        cross-chip CX   -> 1 EPR  (Cat-Comm)
        cross-chip SWAP -> 2 EPR  (TP-Comm)
        intra-chip SWAP -> +1 local SWAP
    """
    remote_cx = remote_swap = local_swap = local_cx = 0
    find = mapped_circ.find_bit
    for instr, qargs, _ in mapped_circ.data:
        name = instr.name.lower()
        if name not in ("cx", "cnot", "swap"):
            continue
        if len(qargs) != 2:
            continue
        a = find(qargs[0]).index
        b = find(qargs[1]).index
        same_chip = qubit_chip[a] == qubit_chip[b]
        if name == "swap":
            if same_chip:
                local_swap += 1
            else:
                remote_swap += 1
        else:  # cx
            if same_chip:
                local_cx += 1
            else:
                remote_cx += 1
    epr = remote_cx + 2 * remote_swap
    overall = local_swap + 10 * remote_cx + 20 * remote_swap
    return {
        "epr": epr,
        "local_swap": local_swap,
        "remote_cx": remote_cx,
        "remote_swap": remote_swap,
        "local_cx": local_cx,
        "overall": overall,
    }


# ── Per-circuit runner ───────────────────────────────────────────────────────
@contextlib.contextmanager
def _silence_fd():
    """Redirect file descriptors 1 and 2 to /dev/null (catches C-level prints)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in (devnull, saved_out, saved_err):
            os.close(fd)


def run_one(qasm_path: str, device_json: str, qubit_chip: Dict[int, int],
            num_seeds: int = NUM_SEEDS) -> Dict:
    """Run DMapS num_seeds times, keep the result with min overall overhead."""
    from router.multi_mode_routing import MultiModeRouting

    best = None
    elapsed_total = 0.0
    seed_log: List[Dict] = []
    for seed in range(num_seeds):
        dqc = MultiModeRouting()
        t0 = time.perf_counter()
        try:
            with _silence_fd():
                _, _, mapped = dqc.cpa_map_car_rout(
                    origin_qasm_fn=Path(qasm_path),
                    chip_info_fn=Path(device_json),
                    chip_type="zcz",
                    intra_chip_all2all=False,
                )
            ok = True
        except Exception as e:
            ok = False
            err = repr(e)
        dt = time.perf_counter() - t0
        elapsed_total += dt
        if not ok:
            if best is None:
                best = {"epr": -1, "local_swap": -1, "remote_cx": -1,
                        "remote_swap": -1, "local_cx": -1, "overall": -1,
                        "error": err}
            continue
        m = derive_metrics(mapped, qubit_chip)
        m["seed"] = seed
        m["seed_time"] = dt
        seed_log.append({k: m[k] for k in ("seed", "epr", "local_swap",
                                            "remote_cx", "remote_swap",
                                            "overall", "seed_time")})
        if best is None or m["overall"] < best["overall"]:
            best = m
    best["total_time"] = elapsed_total
    best["seeds"] = seed_log
    return best


# ── Suite runner ─────────────────────────────────────────────────────────────
SUITES = {
    "25": {"dir": f"{CIRCUITS_ROOT}/qasm_25",
           "suffix": "_nativegates_ibm_qiskit_opt3_25.qasm"},
    "36": {"dir": f"{CIRCUITS_ROOT}/qasm_36",
           "suffix": "_nativegates_ibm_qiskit_opt3_36.qasm"},
    "64": {"dir": f"{CIRCUITS_ROOT}/qasm_64",
           "suffix": "_nativegates_ibm_qiskit_opt3_64.qasm"},
}


def run_suite(suite: str) -> List[Dict]:
    cfg = SUITES[suite]
    qasms = sorted(glob.glob(os.path.join(cfg["dir"], "*.qasm")))
    if not qasms:
        print(f"[dmaps] no qasm in {cfg['dir']}", flush=True)
        return []

    arch = build_arch_for_suite(suite)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        device_json = tf.name
    qubit_chip = _arch_to_dmaps_json(arch, device_json)

    print(f"\n[dmaps] suite={suite}q  device=({len(arch.intra)} chips, "
          f"{len(qubit_chip)} phys qubits)", flush=True)
    hdr = f"  {'circuit':<14}{'EPR':>6}{'lSWAP':>7}{'rCX':>5}{'rSWAP':>7}{'overall':>9}{'time':>8}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    rows = []
    for qf in qasms:
        cname = os.path.basename(qf).replace(cfg["suffix"], "")
        res = run_one(qf, device_json, qubit_chip)
        res["circuit"] = cname
        res["suite"] = suite
        rows.append(res)
        if res["epr"] < 0:
            print(f"  {cname:<14}  FAIL: {res.get('error','?')[:60]}", flush=True)
        else:
            print(f"  {cname:<14}{res['epr']:>6}{res['local_swap']:>7}"
                  f"{res['remote_cx']:>5}{res['remote_swap']:>7}"
                  f"{res['overall']:>9}{res['total_time']:>7.1f}s", flush=True)

    os.unlink(device_json)
    return rows


def main() -> None:
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(SUITES)
    bad = [k for k in keys if k not in SUITES]
    if bad:
        print(f"[dmaps] unknown suite(s): {bad}; pick from {list(SUITES)}")
        sys.exit(1)

    all_rows: Dict[str, List[Dict]] = {}
    for k in keys:
        all_rows[k] = run_suite(k)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n[dmaps] wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
