"""Record TeleSABRE's teledata/telegate split on the protocol the main tables use.

Distinct from probe_telesabre_actions.py, which streams a *single* attempt to
observe decision behaviour.  Here we run the benchmark configuration
(max_attempts=10, required_successes=1) and read main.c's final summary, so the
EPR counts reported below are exactly the ones behind Table IV: EPR = teledata
+ telegate, one pair per operation.

Usage:  python3 probe_telesabre_telegate.py 64 H_grid_2_3_4_4 ae,ghz,...
"""
import json, os, re, sys, subprocess
from circuit_paths import circuits_path

TS   = os.path.expanduser("~/Documents/telesabre/telesabre")
DEV  = os.path.expanduser("~/Documents/telesabre/devices/%s.json")
CIRC = circuits_path("qasm_%d/"
                          "%s_nativegates_ibm_qiskit_opt3_%d.qasm")
RE_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run(circuit, device, seed):
    cfg = {"config": {
        "name": "telegate", "seed": seed, "energy_type": "extended-set",
        "usage_penalties_reset_interval": 5, "optimize_initial": True,
        "initial_layout_type": "hungarian", "teleport_bonus": 100,
        "telegate_bonus": 100, "safety_valve_iters": 100, "extended_set_size": 20,
        "extended_set_factor": 0.05, "inter_core_edge_weight": 2,
        "full_core_penalty": 10, "max_solving_deadlock_iterations": 1000,
        "gate_usage_penalty": 0.0, "swap_usage_penalty": 0.002,
        "teledata_usage_penaly": 0.005, "telegate_usage_penalty": 0.005,
        "init_layout_hun_min_free_gate": 5, "init_layout_hun_min_free_qubit": 4,
        "enable_passing_core_emptying_teleport_possibility": False,
        "max_iterations": 200000, "save_report": False,
        "required_successes": 1, "max_attempts": 10}}
    p = "/tmp/_ts_telegate_cfg.json"
    json.dump(cfg, open(p, "w"))
    o = RE_ANSI.sub("", subprocess.run([TS, p, device, circuit],
                                       capture_output=True, text=True).stdout)
    if "No successful runs" in o:
        return {"success": False, "attempts_shown": len(re.findall(r"Solution has", o))}
    g = lambda k: int(re.search(rf"{k}:\s*(\d+)", o).group(1))
    return {"success": "Success: true" in o, "teledata": g("Teledata"),
            "telegate": g("Telegate"), "swaps": g("Swaps"),
            "deadlocks": g("Deadlocks"),
            "attempts_shown": len(re.findall(r"Solution has", o))}


if __name__ == "__main__":
    n, dev, names = int(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
    seeds = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["1"])]
    rows, TD = {}, 0
    TG = TOT = 0
    print(f"{'circuit':12s} {'EPR':>6s} {'teledata':>9s} {'telegate':>9s} {'tg %':>6s}  best-of seeds")
    for nm in names:
        c = CIRC % (n, nm, n)
        if not os.path.exists(c):
            print(f"  skip {nm}"); continue
        best = None
        for s in seeds:
            r = run(c, DEV % dev, s)
            if r.get("success") and (best is None or
                                     r["teledata"] + r["telegate"] < best["teledata"] + best["telegate"]):
                best = r
        if best is None:
            rows[nm] = {"success": False}
            print(f"{nm:12s} {'--':>6s} {'--':>9s} {'--':>9s} {'--':>6s}  DOES NOT CONVERGE")
            continue
        epr = best["teledata"] + best["telegate"]
        rows[nm] = {"success": True, "epr": epr, "teledata": best["teledata"],
                    "telegate": best["telegate"], "swaps": best["swaps"],
                    "deadlocks": best["deadlocks"]}
        TD += best["teledata"]; TG += best["telegate"]; TOT += epr
        print(f"{nm:12s} {epr:6d} {best['teledata']:9d} {best['telegate']:9d} "
              f"{100.0*best['telegate']/epr:5.1f}%", flush=True)
    print(f"{'TOTAL':12s} {TOT:6d} {TD:9d} {TG:9d} {100.0*TG/TOT if TOT else 0:5.1f}%")
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                        f"telesabre_telegate_{n}q.json")
    json.dump({"suite": n, "device": dev, "seeds": seeds, "per_circuit": rows,
               "totals": {"epr": TOT, "teledata": TD, "telegate": TG}},
              open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")
