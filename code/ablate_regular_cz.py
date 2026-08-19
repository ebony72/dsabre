r"""
ablate_regular_cz.py — mechanism ablation swept over interaction-graph degree.

The main-text ablation (regen_ablation_corners.py) turns each mechanism off on
a fixed six-circuit suite and reports one number per mechanism.  That answers
"how much does this term contribute?" but not "on what kind of circuit does it
contribute?", which is the question a reviewer asked: some mechanisms looked
decisive and others marginal, with no account of why.

Here the same knobs are swept over the structured family of gen_regular_cz.py,
in which interaction-graph degree d is the only thing that varies -- gate
volume is held near 2000 CX at every degree.  Each mechanism's contribution can
then be read as a function of how much inter-core traffic the circuit forces,
which is what the mechanisms are meant to respond to.

Architecture, layout protocol and hyperparameters are the 64q main-results
ones, so the `full` row is comparable with the 64q block of Table IV.
TeleSABRE is run per circuit as an absolute anchor, not as an ablation row.

Usage
-----
  python3 ablate_regular_cz.py --config full      # one config -> its own shard
  python3 ablate_regular_cz.py --telesabre        # the anchor -> its own shard
  python3 ablate_regular_cz.py --merge            # combine shards, print table
  python3 ablate_regular_cz.py                    # everything, serially

The six configurations are independent, so the intended run is one process per
config in parallel followed by --merge; serially it is several hours.

Output: code/results/results_regcz_{config}.json  (shards)
        code/results/results_regcz_ablation.json  (merged)
"""

import sys, os, json, time, argparse
from math import prod
from dataclasses import replace

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit.converters import circuit_to_dag

from architecture import build_h_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from benchmark import run_telesabre, load_qasm

_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

MANIFEST    = os.path.join(_HERE, "circuits_regcz", "manifest.json")
DEVICE_JSON = os.path.expanduser(
    "~/Documents/telesabre/devices/H_grid_2_3_4_4.json")

ARCH = build_h_grid_architecture(r=2, s=3, m=4)      # 64q main-results device
# Matches benchmark.py's _HW_LARGE, the config behind the 64q main table.
BASE = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                      max_iterations=20000)

# (label, router class, config).  One knob off per row, everything else at its
# default.  "no_relief" removed: congestion relief was deleted from the
# router (see CHANGES_FROM_SUBMITTED.md) -- "full" below now means relief-off
# always, so it no longer matches the pre-removal tab:regcz baseline this
# script used to reproduce.  "no_hopgain" removed: hop-gain was deleted from
# the router on the same basis -- the term no longer exists to ablate.
CONFIGS = [
    ("full",         dSABRE_BurstExt,       BASE),
    ("no_cap",       dSABRE_BurstExt,       replace(BASE, cap_penalty=0.0)),
    ("no_lookahead", dSABRE_BurstExt,       replace(BASE, weight_extended=0.0)),
    ("topo_ext",     General_dSABRE_Router, BASE),
]
CONFIG_BY_NAME = {c[0]: c for c in CONFIGS}

_META = dict(arch="H-grid 2x3 of 4x4 (96 physical)",
             family="regular-graph CZ ansatz, gen_regular_cz.py",
             layout="SabreLayout per-core adaptive corner reservation, best of 3",
             pass_strategy="fwd -> bwd (reversed DAG) -> fwd; best of pass1/pass3")


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def shard(name):
    return os.path.join(_RESULTS_DIR, f"results_regcz_{name}.json")


def load_manifest():
    circuits = json.load(open(MANIFEST))["circuits"]
    if not circuits:
        raise SystemExit("empty manifest; run gen_regular_cz.py first")
    return circuits


def prepare(circuits):
    """Load each circuit and its layouts once; layouts are config-independent."""
    out = []
    for c in circuits:
        qc, dag = load_qasm(os.path.join(_HERE, c["path"]))
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        if n_cx != c["cx"]:
            raise SystemExit(f"{c['name']}: manifest says {c['cx']} CX, file has "
                             f"{n_cx}; regenerate the family")
        out.append(dict(meta=c, qc=qc, dag=dag,
                        rev_dag=circuit_to_dag(qc.reverse_ops()),
                        layouts=sabre_locked_boundary_layout(qc, dag, ARCH,
                                                             seed=0)))
    return out


def run_config(name):
    label, cls, cfg = CONFIG_BY_NAME[name]
    prepared = prepare(load_manifest())
    router = cls(ARCH, cfg)
    print(f"[{label}] {len(prepared)} circuits on {_META['arch']}", flush=True)

    records = []
    for p in prepared:
        t0, best = time.time(), None
        for layout in p["layouts"]:
            m = run_sabre_passes(router, p["dag"], p["rev_dag"], layout)
            if m and not m.get("aborted"):
                if best is None or m["eprs"] < best["eprs"]:
                    best = m
        rec = dict(config=label, circuit=p["meta"]["name"],
                   degree=p["meta"]["degree"], seed=p["meta"]["seed"],
                   cx=p["meta"]["cx"],
                   **(dict(eprs=best["eprs"], ls=best["ls"],
                           time_s=round(best["compile_time"], 3), aborted=False)
                      if best else dict(aborted=True)))
        records.append(rec)
        print(f"  {p['meta']['name']:<20} d={rec['degree']}  "
              f"EPR={rec.get('eprs', 'ABORT'):>6}  SWAP={rec.get('ls', 0):>6}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        with open(shard(label), "w") as f:          # save early and often
            json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"),
                                     config=label, **_META),
                           results=records), f, indent=2)
    print(f"[{label}] done → {shard(label)}", flush=True)


def run_telesabre_anchor():
    rows = []
    for c in load_manifest():
        t0 = time.time()
        ts = run_telesabre(os.path.join(_HERE, c["path"]), DEVICE_JSON)
        rows.append(dict(circuit=c["name"], degree=c["degree"], seed=c["seed"],
                         cx=c["cx"], ts=ts))
        print(f"  {c['name']:<20} EPR={ts['eprs'] if ts else '---':>6}  "
              f"SWAP={ts['ls'] if ts else '---':>6}  ({time.time()-t0:.0f}s)",
              flush=True)
        with open(shard("telesabre"), "w") as f:
            json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"), **_META),
                           telesabre=rows), f, indent=2)
    print(f"[telesabre] done → {shard('telesabre')}", flush=True)


def merge():
    records, ts_rows, missing = [], [], []
    for label, _, _ in CONFIGS:
        if os.path.exists(shard(label)):
            records += json.load(open(shard(label)))["results"]
        else:
            missing.append(label)
    if os.path.exists(shard("telesabre")):
        ts_rows = json.load(open(shard("telesabre")))["telesabre"]
    if missing:
        print(f"warning: no shard for {', '.join(missing)}", flush=True)

    degrees = sorted({r["degree"] for r in records})
    out = os.path.join(_RESULTS_DIR, "results_regcz_ablation.json")
    with open(out, "w") as f:
        json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"), **_META),
                       telesabre=ts_rows, results=records), f, indent=2)

    hdr = f"{'config':<14}" + "".join(f"   d={d:<2} gmEPR    Δ%" for d in degrees)
    print("\n" + "=" * len(hdr), flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    full = {}
    for label, _, _ in CONFIGS:
        rs = [r for r in records if r["config"] == label]
        if not rs:
            continue
        row = f"{label:<14}"
        for d in degrees:
            sel = [r for r in rs if r["degree"] == d]
            g = gmean([r.get("eprs") for r in sel if not r["aborted"]])
            n_ab = sum(1 for r in sel if r["aborted"])
            if label == "full":
                full[d] = g
                row += f"  {g:>10.1f}   ---"
            else:
                row += f"  {g:>10.1f} {100*(g-full[d])/full[d]:+5.1f}"
            row += f"{'(%dab)' % n_ab if n_ab else '     '}"
        print(row, flush=True)

    if ts_rows:
        row = f"{'TeleSABRE':<14}"
        for d in degrees:
            sel = [r for r in ts_rows if r["degree"] == d]
            g = gmean([r["ts"]["eprs"] for r in sel if r["ts"]])
            n_ab = sum(1 for r in sel if not r["ts"])
            row += f"  {g:>10.1f} {100*(g-full[d])/full[d]:+5.1f}" \
                   f"{'(%dab)' % n_ab if n_ab else '     '}"
        print("-" * len(hdr), flush=True)
        print(row, flush=True)
    print(f"\nSaved → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=[c[0] for c in CONFIGS])
    ap.add_argument("--telesabre", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge()
    elif args.telesabre:
        run_telesabre_anchor()
    elif args.config:
        run_config(args.config)
    else:
        run_telesabre_anchor()
        for label, _, _ in CONFIGS:
            run_config(label)
        merge()


if __name__ == "__main__":
    main()
