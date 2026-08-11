r"""Run DMapS on suite circuits absent from results_dmaps_bench.json, and merge.

`run_dmaps_bench.py` globs a whole suite and rewrites the results file from
scratch.  This runs only the circuits that are missing -- which is what you
want after adding circuits to a suite, since a DMapS pass over the 64q suite
costs ~30 minutes -- and merges them in without touching existing rows.

    /opt/anaconda3/envs/dmaps/bin/python code/run_dmaps_missing.py 64
    /opt/anaconda3/envs/dmaps/bin/python code/run_dmaps_missing.py 64 qaoa multiplier

With no circuit names, every circuit of the suite that has no row in the
results file is run.  Rows already present are never re-run or overwritten;
delete them first if you want them refreshed.

Suite membership comes from CANONICAL below where an entry exists, not from
the directory glob: `qasm_64/` is shared with other projects and also holds
`vqe_su2` and `wstate`, which belong to the 36q family and are not part of
the paper's nine-circuit 64q suite (they are cphm's -- see
`cphm/code/ls_supplement_64q.py` -- so leave the files alone and filter
here).  Named circuits are always run as given.

IMPORTANT -- the `if __name__ == "__main__"` guard below is load-bearing, not
boilerplate.  DMapS uses multiprocessing with spawn, so each child re-imports
this module; without the guard the child re-executes the benchmark at import
time, dies with "An attempt has been made to start a new process before the
current process has finished its bootstrapping phase", and writes its own
failure rows (epr=-1) over the parent's results.  The parent still prints
plausible-looking numbers while its DMapS workers are dying, so the corruption
is silent in the log and visible only in the merged JSON.  Measured
2026-08-11: with the guard the three 64q circuits took 27 minutes and gave
qpeexact 408 EPR; without it, 5 hours and 452.
"""
from __future__ import annotations

import glob
import json
import multiprocessing
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import run_dmaps_bench as R  # noqa: E402  (resolves DMapS on sys.path first)

# The published suite, by name.  Same whitelist as `ablate_strip_1q_64q.py`'s
# CANONICAL, restated rather than imported so this script pulls in no router
# dependencies.  A suite absent here falls back to the directory glob.
CANONICAL = {
    "64": {"ae", "ghz", "graphstate", "qft", "qnn", "random",
           "qpeexact", "qaoa", "multiplier"},
}


def missing_circuits(suite: str, results_path: str) -> list[str]:
    cfg = R.SUITES[suite]
    on_disk = [
        os.path.basename(p).replace(cfg["suffix"], "")
        for p in sorted(glob.glob(os.path.join(cfg["dir"], f"*{cfg['suffix']}")))
    ]
    canonical = CANONICAL.get(suite)
    if canonical is not None:
        skipped = [c for c in on_disk if c not in canonical]
        if skipped:
            print(f"[dmaps] suite={suite}q: ignoring off-suite circuits in "
                  f"{os.path.basename(cfg['dir'])}: {skipped}", flush=True)
        on_disk = [c for c in on_disk if c in canonical]
    have = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            have = {r["circuit"] for r in json.load(f).get(suite, [])}
    return [c for c in on_disk if c not in have]


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in R.SUITES:
        sys.exit(f"usage: run_dmaps_missing.py <{'|'.join(R.SUITES)}> [circuit ...]")
    suite = sys.argv[1]
    out = R.RESULTS_PATH

    wanted = sys.argv[2:] or missing_circuits(suite, out)
    if not wanted:
        print(f"[dmaps] suite={suite}q: nothing missing", flush=True)
        return

    cfg = R.SUITES[suite]
    arch = R.build_arch_for_suite(suite)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        device_json = tf.name
    qubit_chip = R._arch_to_dmaps_json(arch, device_json)

    print(f"[dmaps] suite={suite}q  {len(arch.intra)} chips, {len(qubit_chip)} phys "
          f"qubits; circuits={wanted}", flush=True)

    rows = []
    for cname in wanted:
        qf = os.path.join(cfg["dir"], f"{cname}{cfg['suffix']}")
        if not os.path.exists(qf):
            print(f"  {cname:<12} NO QASM at {qf}", flush=True)
            continue
        print(f"  {cname:<12} running ({R.NUM_SEEDS} seeds)...", flush=True)
        res = R.run_one(qf, device_json, qubit_chip)
        res["circuit"] = cname
        res["suite"] = suite
        rows.append(res)
        if res["epr"] < 0:
            print(f"  {cname:<12} FAIL: {res.get('error', '?')[:120]}", flush=True)
        else:
            print(f"  {cname:<12} EPR={res['epr']} lSWAP={res['local_swap']} "
                  f"rCX={res['remote_cx']} rSWAP={res['remote_swap']} "
                  f"overall={res['overall']} {res['total_time']:.1f}s", flush=True)

    os.unlink(device_json)

    data = {}
    if os.path.exists(out):
        with open(out) as f:
            data = json.load(f)
    have = {r["circuit"] for r in data.get(suite, [])}
    new = [r for r in rows if r["circuit"] not in have and r["epr"] >= 0]
    dropped = [r["circuit"] for r in rows if r["epr"] < 0]
    data[suite] = data.get(suite, []) + new
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[dmaps] merged {len(new)} rows into {out}"
          + (f"; dropped failures {dropped}" if dropped else ""), flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
