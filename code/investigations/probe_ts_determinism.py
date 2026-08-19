"""probe_ts_determinism.py -- is TeleSABRE really seed-independent?

Item 6 of the 0811 review.  The paper claims TeleSABRE is deterministic under
`optimize_initial` with the Hungarian layout, and backs the claim on the 64q
suite only.  Best-of-3 then buys dSABRE something and TeleSABRE nothing, so
the asymmetry has to be bounded -- but only where the determinism is checked.
This runs seeds 0/1/2 on every suite the paper reports and prints, per
circuit, whether the three EPR counts coincide.

Usage:  python3 probe_ts_determinism.py [--suite 25q ...]
"""

import os as _os, sys as _sys
# This script lives in code/investigations/; the implementation, results/ and
# circuit families it uses are one level up, in code/.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import sys, os, json, subprocess, tempfile, time, argparse
from circuit_paths import circuits_path

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

TS_BIN = os.path.expanduser("~/Documents/telesabre/telesabre")
DEV = os.path.expanduser("~/Documents/telesabre/devices")
CIRC = circuits_path()
OUT = os.path.join(_HERE, "results", "results_ts_determinism.json")

SUITES = {
    "25q": (f"{CIRC}/qasm_25", "_nativegates_ibm_qiskit_opt3_25.qasm",
            f"{DEV}/B_grid_2_2_4_4.json", 300,
            ["ae", "ghz", "graphstate", "qft", "qnn", "random"]),
    "36q": (f"{CIRC}/qasm_36", "_nativegates_ibm_qiskit_opt3_36.qasm",
            f"{DEV}/B_grid_2_2_4_4.json", 300,
            ["bv", "dj", "qaoa", "qpeexact", "vqe_su2", "wstate"]),
    "64q": (f"{CIRC}/qasm_64", "_nativegates_ibm_qiskit_opt3_64.qasm",
            f"{DEV}/H_grid_2_3_4_4.json", 300,
            ["ae", "ghz", "graphstate", "qft", "qnn", "random",
             "qpeexact", "qaoa", "multiplier"]),
    "100q": (f"{CIRC}/qasm_100", "_nativegates_ibm_qiskit_opt3_100.qasm",
             f"{DEV}/H_grid_2_3_5_5.json", 600, ["qft", "qpeexact"]),
    "200q": (f"{CIRC}/qasm_200", "_nativegates_ibm_qiskit_opt3_200.qasm",
             f"{DEV}/H_grid_3_4_5_5.json", 600, ["qft", "qpeexact"]),
    "360q": (f"{CIRC}/qasm_360", "_nativegates_ibm_qiskit_opt3_360.qasm",
             f"{DEV}/H_grid_4_5_5_5.json", 600, ["qft", "qpeexact"]),
}


def ts_config(seed, rpt):
    cfg = {"config": {
        "name": "determinism", "seed": seed,
        "energy_type": "extended-set",
        "usage_penalties_reset_interval": 5,
        "optimize_initial": True, "initial_layout_type": "hungarian",
        "teleport_bonus": 100, "telegate_bonus": 100, "safety_valve_iters": 100,
        "extended_set_size": 20, "extended_set_factor": 0.05,
        "inter_core_edge_weight": 2, "full_core_penalty": 10,
        "max_solving_deadlock_iterations": 1000,
        "gate_usage_penalty": 0.0, "swap_usage_penalty": 0.002,
        "teledata_usage_penaly": 0.005, "telegate_usage_penalty": 0.005,
        "init_layout_hun_min_free_gate": 5, "init_layout_hun_min_free_qubit": 4,
        "enable_passing_core_emptying_teleport_possibility": False,
        "max_iterations": 200000,
        "save_report": True, "report_filename": rpt,
        "required_successes": 1, "max_attempts": 10,
    }}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f); f.close()
    return f.name


def run_one(qasm, dev, seed, timeout):
    rpt = tempfile.mktemp(suffix=".json")
    cfg = ts_config(seed, rpt)
    try:
        proc = subprocess.run([TS_BIN, cfg, dev, qasm],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.unlink(cfg)
        return None
    td = tg = ls = 0
    ok = False
    for line in (proc.stdout + proc.stderr).splitlines():
        s = line.strip()
        if "Teledata:" in s: td = int(s.split(":")[1])
        elif "Telegate:" in s: tg = int(s.split(":")[1])
        elif "Swaps:" in s: ls = int(s.split(":")[1])
        elif "Success: true" in s: ok = True
    os.unlink(cfg)
    if os.path.exists(rpt): os.unlink(rpt)
    return dict(eprs=td + tg, teledata=td, telegate=tg, ls=ls) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append", choices=list(SUITES))
    args = ap.parse_args()
    out = []
    for suite in (args.suite or list(SUITES)):
        cdir, suffix, dev, timeout, circuits = SUITES[suite]
        print(f"\n=== {suite} ===", flush=True)
        for c in circuits:
            qasm = os.path.join(cdir, c + suffix)
            if not os.path.exists(qasm):
                print(f"  {c:<12} [missing]", flush=True)
                continue
            seeds = [run_one(qasm, dev, s, timeout) for s in (0, 1, 2)]
            eprs = [s["eprs"] if s else None for s in seeds]
            lss = [s["ls"] if s else None for s in seeds]
            ok = [e for e in eprs if e is not None]
            verdict = ("all fail" if not ok else
                       "IDENTICAL" if len(set(eprs)) == 1 else
                       f"VARIES (spread {max(ok) - min(ok)})")
            print(f"  {c:<12} EPR={str(eprs):<24} SWAP={str(lss):<26} {verdict}",
                  flush=True)
            out.append(dict(suite=suite, circuit=c, eprs=eprs, ls=lss,
                            verdict=verdict))
            with open(OUT, "w") as f:
                json.dump({"meta": {"date": time.strftime("%Y-%m-%d"),
                                    "what": "TeleSABRE seeds 0/1/2 under "
                                            "optimize_initial + hungarian"},
                           "results": out}, f, indent=1)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
