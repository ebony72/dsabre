r"""
bench_sharedmap.py — the shared-initial-mapping control.

Re-runs dSABRE from TeleSABRE's own `optimize_initial` (Hungarian) layout
instead of its SabreLayout one, so the two routers start from an identical
allocation and only the routing loop differs.  If the margin survives, the gain
is in the routing loop rather than the layout front end -- which is the claim
Section IV-D makes.

Regenerated 2026-07-29.  The earlier figures were produced with a TeleSABRE
config whose Hungarian parameter was misspelled (`init_layout_hun_free_qubit`
for `init_layout_hun_min_free_qubit`), so the allocator ran at its default
rather than the intended setting -- and this experiment consumes that
allocator's output directly, so it moved with the fix.

Output: code/results/results_sharedmap.json
"""

import sys, os, json, glob, time, tempfile, subprocess
from math import prod

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit.converters import circuit_to_dag

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import run_sabre_passes
from benchmark import load_qasm, TS_BIN, _ts_config, NUM_TS_SEEDS
from bench_large import p2v_to_layout

_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")

SUITES = {
    "25q": dict(arch=build_b_grid_architecture(2, 2, 4),
                dev="B_grid_2_2_4_4.json", d="qasm_25", sfx="_25",
                circuits=["ae", "ghz", "graphstate", "qft", "qnn", "random"],
                hw=HardwareConfig()),
    "64q": dict(arch=build_h_grid_architecture(2, 3, 4),
                dev="H_grid_2_3_4_4.json", d="qasm_64", sfx="_64",
                circuits=["ae", "ghz", "graphstate", "qft", "qnn", "random",
                          "qpeexact", "qaoa", "multiplier"],
                hw=HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                                  max_iterations=20000)),
}


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def telesabre_with_layout(qasm, dev):
    """Best-EPR TeleSABRE run, returning its cost and its initial layout."""
    best = None
    for seed in range(NUM_TS_SEEDS):
        rpt = tempfile.mktemp(suffix=".json")
        cfg = _ts_config(seed, rpt, "sharedmap")
        try:
            p = subprocess.run([TS_BIN, cfg, dev, qasm], capture_output=True,
                               text=True, timeout=600)
        except subprocess.TimeoutExpired:
            os.unlink(cfg); continue
        td = tg = ls = 0; ok = False
        for line in (p.stdout + p.stderr).splitlines():
            t = line.strip()
            if   "Teledata:" in t: td = int(t.split(":")[1])
            elif "Telegate:" in t: tg = int(t.split(":")[1])
            elif "Swaps:"    in t: ls = int(t.split(":")[1])
            elif "Success: true" in t: ok = True
        p2v = None
        if os.path.exists(rpt):
            try:
                rep = json.load(open(rpt))
                if rep.get("iterations"):
                    p2v = rep["iterations"][0]["phys_to_virt"]
            except Exception:
                pass
            os.unlink(rpt)
        os.unlink(cfg)
        if ok and p2v is not None and (best is None or td + tg < best["eprs"]):
            best = dict(eprs=td + tg, ls=ls, seed=seed, p2v=p2v)
    return best


def main():
    out, records = os.path.join(_RESULTS_DIR, "results_sharedmap.json"), []
    for label, s in SUITES.items():
        arch, hw = s["arch"], s["hw"]
        dev = os.path.expanduser(f"~/Documents/telesabre/devices/{s['dev']}")
        cdir = os.path.expanduser(f"~/Documents/telesabre/circuits/{s['d']}")
        router = dSABRE_BurstExt(arch, hw)
        print(f"\n== {label} (shared TeleSABRE layout) ==", flush=True)
        ts_e, ds_e = [], []
        for name in s["circuits"]:
            qf = os.path.join(cdir, f"{name}_nativegates_ibm_qiskit_opt3"
                                    f"{s['sfx']}.qasm")
            if not os.path.exists(qf):
                print(f"  {name:<12} missing", flush=True); continue
            t0 = time.time()
            qc, dag = load_qasm(qf)
            ts = telesabre_with_layout(qf, dev)
            rec = dict(suite=label, circuit=name,
                       cx=sum(1 for _ in dag.two_qubit_ops()))
            if ts is None:
                rec.update(ts=None, dsabre=None)
                print(f"  {name:<12} TeleSABRE did not converge", flush=True)
            else:
                layout = p2v_to_layout(ts["p2v"], dag)
                rev = circuit_to_dag(qc.reverse_ops())
                m = run_sabre_passes(router, dag, rev, layout)
                d = (dict(eprs=m["eprs"], ls=m["ls"]) if m and not m.get("aborted")
                     else dict(aborted=True))
                rec.update(ts=dict(eprs=ts["eprs"], ls=ts["ls"], seed=ts["seed"]),
                           dsabre=d)
                if d.get("eprs"):
                    ts_e.append(ts["eprs"]); ds_e.append(d["eprs"])
                print(f"  {name:<12} TS={ts['eprs']:>6}  dSABRE={d.get('eprs','ABORT'):>6}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            records.append(rec)
            json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d")),
                           results=records), open(out, "w"), indent=2)
        if ts_e:
            print(f"  gmean over {len(ts_e)}: TS {gmean(ts_e):.1f}  "
                  f"dSABRE {gmean(ds_e):.1f}  -> "
                  f"{100*(gmean(ds_e)-gmean(ts_e))/gmean(ts_e):+.1f}%", flush=True)
    print(f"\nSaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
