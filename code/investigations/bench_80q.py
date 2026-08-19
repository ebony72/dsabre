r"""
bench_80q.py — the 80-qubit suite.

Originally built to settle whether the hop-gain reward earns its keep on a
wider core graph than the rest of the paper's suites: the architecture is a
2x5 H-grid of the same 4x4 cores used at 64q -- 10 cores, 160 physical
qubits, core diameter 5, mean core path 2.33 -- and the 80-qubit suite has
four circuits above 6,000 CX (qaoa, qnn, multiplier, random) against the
64-qubit suite's one.

hop-gain was ultimately removed from the router entirely (see
CHANGES_FROM_SUBMITTED.md): a post-relief-removal re-test across every suite
in this repo -- including a `no_hopgain` config run here, ae -20.4%,
qft -6.6%, qnn -1.4%, ghz/graphstate tied, but qpeexact +17.6% and
qaoa +6.6% the other way, netting -0.8% gmean -- found no scale or circuit
family on which it reliably helped once congestion relief was gone.  The
suite is kept because it is the strongest dSABRE-vs-TeleSABRE margin in the
evaluation (-51.7% gmean over the seven circuits both routers complete).

Usage
-----
  python3 bench_80q.py --config dsabre
  python3 bench_80q.py --telesabre
  python3 bench_80q.py --merge

Output: code/results/results_80q_{dsabre,telesabre}.json  (shards)
        code/results/results_80q.json                     (merged)
"""

import sys, os, json, time, argparse, math

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

import networkx as nx
from qiskit.converters import circuit_to_dag

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from benchmark import run_telesabre, load_qasm
from circuit_paths import circuits_path

_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

CIRCUIT_DIR = circuits_path("qasm_80")
SUFFIX      = "_nativegates_ibm_qiskit_opt3_80.qasm"
DEVICE_JSON = os.path.expanduser(
    "~/Documents/telesabre/devices/H_grid_2_5_4_4.json")

ARCH = build_h_grid_architecture(r=2, s=5, m=4)   # 10 cores, 160 physical
HW   = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                      max_iterations=20000)

CIRCUITS = ["ae", "ghz", "graphstate", "qft", "qnn", "random",
            "qpeexact", "qaoa", "multiplier"]

# The directory copy of qnn_80 is the v2.2.2 ZFeatureMap form -- 79 CX, a
# linear chain.  The rest of the qnn family across every suite is the v1.1.0
# ZZFeatureMap form, so use the deep reconstruction instead, exactly as
# bench_heavyhex.py did for qnn_64 while its directory copy was wrong.
OVERRIDES = {
    "qnn": os.path.join(CIRCUIT_DIR,
                        "qnn_nativegates_ibm_qiskit_opt3_80_DEEP.qasm"),
}

# Measured 2026-07-28.  A driver that silently benchmarks a regenerated circuit
# under a published name is the failure mode this repo has already hit twice.
EXPECTED_CX = {"ae": 2780, "ghz": 79, "graphstate": 80, "qft": 2660,
               "qnn": 12718, "random": 23381, "qpeexact": 2779,
               "qaoa": 6096, "multiplier": 20378}

CONFIGS = {"dsabre": HW}


def export_device_json(arch, path, name, r, s, m):
    """Write the H-grid in TeleSABRE's device-JSON schema (grid positions)."""
    pos = [None] * len(arch.data_qubits)
    for cr in range(r):
        for cs in range(s):
            base = (cr * s + cs) * m * m
            for lr in range(m):
                for ls in range(m):
                    pos[base + lr * m + ls] = [float(cs * (m + 2) + ls),
                                               float(-(cr * (m + 2) + lr))]
    dev = {"device": {
        "name": name,
        "num_cores": arch.num_cores,
        "num_qubits": len(arch.data_qubits),
        "intra_core_edges": sorted([int(u), int(v)] for u, v in
                                   (e for g in arch.intra.values()
                                    for e in g.edges())),
        "inter_core_edges": [[int(u), int(v)] for u, v in arch.inter_core_links],
        "node_positions": pos,
    }}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        # The devices directory is shared with other projects.  Never clobber a
        # file that is already there: verify the graph matches and leave it be.
        cur = json.load(open(path))["device"]
        same = (sorted(map(tuple, cur["inter_core_edges"]))
                == sorted(map(tuple, dev["device"]["inter_core_edges"]))
                and sorted(map(tuple, cur["intra_core_edges"]))
                == sorted(map(tuple, dev["device"]["intra_core_edges"])))
        if not same:
            raise SystemExit(
                f"{path} exists with a different topology; refusing to "
                f"overwrite a shared device file. Write a new name instead.")
        return path
    with open(path, "w") as f:
        json.dump(dev, f, indent=1)
    return path


def circuit_path(name):
    return OVERRIDES.get(name, os.path.join(CIRCUIT_DIR, name + SUFFIX))


def load_all():
    out = []
    for name in CIRCUITS:
        qc, dag = load_qasm(circuit_path(name))
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        if EXPECTED_CX.get(name) not in (None, n_cx):
            raise SystemExit(
                f"circuit mismatch: {name} has {n_cx} CX, expected "
                f"{EXPECTED_CX[name]}. Refusing to benchmark a different "
                f"circuit under the same name.")
        out.append(dict(name=name, qc=qc, dag=dag, cx=n_cx,
                        rev_dag=circuit_to_dag(qc.reverse_ops())))
    return out


def shard(tag):
    return os.path.join(_RESULTS_DIR, f"results_80q_{tag}.json")


_META = lambda: dict(
    date=time.strftime("%Y-%m-%d"),
    arch=f"H-grid 2x5 of 4x4 ({len(ARCH.data_qubits)} physical, "
         f"{ARCH.num_cores} cores, {len(ARCH.inter_core_links)} links, "
         f"core diameter {nx.diameter(ARCH.core_graph)})",
    layout="SabreLayout per-core adaptive corner reservation, best of 3",
    pass_strategy="fwd -> bwd (reversed DAG) -> fwd; best of pass1/pass3")


def run(tag):
    cfg = CONFIGS[tag]
    router = dSABRE_BurstExt(ARCH, cfg)
    print(f"[hop_gain {tag}] {_META()['arch']}", flush=True)
    records = []
    for c in load_all():
        t0, best = time.time(), None
        layouts = sabre_locked_boundary_layout(c["qc"], c["dag"], ARCH, seed=0)
        for layout in layouts:
            m = run_sabre_passes(router, c["dag"], c["rev_dag"], layout)
            if m and not m.get("aborted"):
                if best is None or m["eprs"] < best["eprs"]:
                    best = m
        rec = dict(config=tag, circuit=c["name"], cx=c["cx"])
        if best:
            teles = best.get("teles") or 0
            rec.update(eprs=best["eprs"], ls=best["ls"], teles=teles,
                       hops_per_tele=round(best["eprs"] / teles, 3) if teles else None,
                       time_s=round(best["compile_time"], 3), aborted=False)
        else:
            rec.update(aborted=True)
        records.append(rec)
        print(f"  {c['name']:<12} CX={c['cx']:>6}  EPR={rec.get('eprs','ABORT'):>6}"
              f"  SWAP={rec.get('ls',0):>6}  tele={rec.get('teles',0):>5}"
              f"  hops/tele={rec.get('hops_per_tele')}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        with open(shard(tag), "w") as f:            # save early and often
            json.dump(dict(meta=dict(config=tag, **_META()),
                           results=records), f, indent=2)
    print(f"[hop_gain {tag}] done → {shard(tag)}", flush=True)


def run_ts():
    export_device_json(ARCH, DEVICE_JSON, "H_grid_2_5_4_4", 2, 5, 4)
    print(f"device → {DEVICE_JSON}", flush=True)
    rows = []
    for c in load_all():
        t0 = time.time()
        ts = run_telesabre(circuit_path(c["name"]), DEVICE_JSON)
        rows.append(dict(circuit=c["name"], cx=c["cx"], ts=ts))
        print(f"  {c['name']:<12} EPR={ts['eprs'] if ts else '---':>6}  "
              f"SWAP={ts['ls'] if ts else '---':>6}  ({time.time()-t0:.0f}s)",
              flush=True)
        with open(shard("telesabre"), "w") as f:
            json.dump(dict(meta=_META(), telesabre=rows), f, indent=2)
    print(f"[telesabre] done → {shard('telesabre')}", flush=True)


def merge():
    ds = {r["circuit"]: r for r in
          json.load(open(shard("dsabre")))["results"]} if os.path.exists(shard("dsabre")) else {}
    ts = {r["circuit"]: r["ts"] for r in
          json.load(open(shard("telesabre")))["telesabre"]} if os.path.exists(shard("telesabre")) else {}

    merged, pairs = [], []
    print(f"\n{'circuit':<12}{'CX':>7}{'dSABRE':>9}{'SWAP':>8}"
          f"{'TS EPR':>9}{'delta%':>9}", flush=True)
    print("-" * 54, flush=True)
    for name in CIRCUITS:
        a = ds.get(name)
        ea = a.get("eprs") if a and not a.get("aborted") else None
        t  = ts.get(name)
        et = t["eprs"] if t else None
        d = f"{100*(ea-et)/et:+.1f}" if ea and et else "---"
        if ea and et:
            pairs.append(ea / et)
        print(f"{name:<12}{(a or {}).get('cx',0):>7}"
              f"{ea if ea is not None else '---':>9}"
              f"{(a or {}).get('ls','---'):>8}"
              f"{et if et is not None else '---':>9}{d:>9}", flush=True)
        merged.append(dict(circuit=name, dsabre=a, telesabre=t))

    if pairs:
        g = math.prod(pairs) ** (1 / len(pairs))
        print("-" * 54, flush=True)
        print(f"{'gmean':<12}{'':>7}{'':>9}{'':>8}{'':>9}{100*(g-1):+8.1f}%"
              f"   (over {len(pairs)} circuits)", flush=True)

    out = os.path.join(_RESULTS_DIR, "results_80q.json")
    with open(out, "w") as f:
        json.dump(dict(meta=_META(), results=merged), f, indent=2)
    print(f"\nSaved -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--telesabre", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:        merge()
    elif a.telesabre:  run_ts()
    elif a.config:     run(a.config)
    else:
        run_ts()
        for tag in CONFIGS:
            run(tag)
        merge()


if __name__ == "__main__":
    main()
