"""
probe_ts_time.py — TeleSABRE per-seed compile time on the 64-qubit suite,
measured on the same machine and in the same conditions as
`probe_compile_time.py` measures dSABRE.

Why re-measure.  The per-circuit times recorded in results_64q.json were taken
during a full benchmark sweep that also ran TeleSABRE and the dS router, and
they come out 1.07x-2.74x higher than a fresh isolated measurement of the same
work -- inconsistent scaling, so machine load rather than a difference in what
is being timed.  A table comparing three compilers should not mix an isolated
measurement of one against a loaded measurement of another.

Also records whether the three seeds agree, since TeleSABRE's determinism is
what makes best-of-3 unnecessary for it and therefore sets the seed multiplier
used in the appendix timing table.

Output: code/results/results_ts_time_64q.json (written incrementally)

Usage:  python3 code/probe_ts_time.py
"""

from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import subprocess
import tempfile

from benchmark import TS_BIN, _ts_config
from circuit_paths import circuits_path

CIRCUIT_DIR = circuits_path("qasm_64")
SUFFIX = "_nativegates_ibm_qiskit_opt3_64.qasm"
TS_DEV = os.path.expanduser("~/Documents/telesabre/devices/H_grid_2_3_4_4.json")
OUT = os.path.join(_HERE, "results", "results_ts_time_64q.json")

CIRCUITS = ["ghz", "graphstate", "qpeexact", "ae", "qft", "random",
            "qaoa", "qnn", "multiplier"]
N_SEEDS = 3
TIMEOUT = 300


def run_one(qasm, seed):
    rpt = tempfile.mktemp(suffix=".json")
    cfg = _ts_config(seed, rpt, "timing")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([TS_BIN, cfg, TS_DEV, qasm],
                              capture_output=True, text=True, timeout=TIMEOUT)
        elapsed = time.perf_counter() - t0
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        elapsed, out = time.perf_counter() - t0, ""
    finally:
        for p in (cfg, rpt):
            if os.path.exists(p):
                os.unlink(p)
    td = tg = 0
    ok = False
    for line in out.splitlines():
        s = line.strip()
        if "Teledata:" in s:
            td = int(s.split(":")[1])
        elif "Telegate:" in s:
            tg = int(s.split(":")[1])
        elif "Success: true" in s:
            ok = True
    return round(elapsed, 2), (td + tg if ok else None)


def main():
    rows = []
    print(f"TeleSABRE per-seed compile time, 64q suite "
          f"({N_SEEDS} seeds, {TIMEOUT}s timeout)\n", flush=True)
    for name in CIRCUITS:
        qasm = os.path.join(CIRCUIT_DIR, name + SUFFIX)
        times, eprs = [], []
        for seed in range(N_SEEDS):
            t, e = run_one(qasm, seed)
            times.append(t)
            eprs.append(e)
        agree = len({e for e in eprs if e is not None}) <= 1
        rows.append(dict(circuit=name, per_seed_s=times, total_s=round(sum(times), 2),
                         best_seed_s=min(times), eprs=eprs, seeds_agree=agree))
        print(f"  {name:12} seeds={times}  total={sum(times):8.2f}s"
              f"  EPR={eprs}  agree={agree}", flush=True)
        with open(OUT, "w") as f:
            json.dump(dict(meta=dict(
                date="2026-07-31", suite="64q",
                arch="H-grid 2x3 of 4x4 (96 physical)",
                binary=TS_BIN, device=TS_DEV, seeds=N_SEEDS, timeout_s=TIMEOUT,
                note="measured in isolation, matching probe_compile_time.py's "
                     "conditions; seeds_agree records whether best-of-3 buys "
                     "TeleSABRE anything",
            ), results=rows), f, indent=1)
    print(f"\nDone -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
