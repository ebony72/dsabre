r"""
migrate_time_keys.py — one-off migration of the `time_s` key in results JSONs.

Until 2026-07-31 three drivers recorded `time_s` as the compile time of the
*winning* seed, not of the best-of-N protocol that produced the reported count.
That understated dSABRE roughly threefold against pytket-dqc, whose published
timings are whole-search totals.  The drivers are fixed; this migrates the data
already on disk so the key means one thing everywhere.

    time_s        cost of the WHOLE protocol (every seed, summed)
    time_seed_s   cost of the single seed whose result is reported

For each affected row this script moves the legacy value into `time_seed_s`
and then either sets `time_s` to the true protocol total, where that total was
printed to the run log and is recorded in TOTALS below, or REMOVES `time_s`
entirely, where the total was never recorded.  Removing it is deliberate: a
consumer that reads `time_s` expecting the new meaning should get nothing
rather than a number that is silently a third of the truth.

NOT every file is affected, and a blanket rename would corrupt the ones that
are already right:

  * ablate_*.py route through `ablate_common.best_over_layouts`, which already
    accumulates `compile_time` across all candidate layouts and says so.  Every
    results_ablate_* file is therefore already a protocol total.
  * bench_pytket_fair.py times its whole best-of-5 search, so every
    results_pytket_fair* / pyfair_* file is already a protocol total.
  * results_compile_time_64q.json and results_ts_time_64q.json use their own
    explicit keys (total_s, per_seed_s, best_seed_s) and are untouched.

Only files listed in LEGACY are modified.  The list is explicit rather than
pattern-matched, because the distinction is a property of the driver that
wrote the file and cannot be inferred from its contents.

Usage:
  python3 code/migrate_time_keys.py            # dry run, prints what it would do
  python3 code/migrate_time_keys.py --apply    # write, after backing up
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")

# Files whose `time_s` carries the legacy best-seed meaning, by the driver that
# wrote them: benchmark.py, bench_large.py, bench_scaling.py, bench_80q.py,
# bench_extend_64q.py, ablate_regular_cz.py, ablate_hop_gain_360q*.py.
LEGACY = [
    "results_25q.json", "results_36q.json", "results_64q.json",
    "results_25q_8links.json", "results_36q_8links.json",
    "results_extend_64q.json",
    "results_100q.json", "results_100q.partial.json",
    "results_200q.json", "results_360q.json",
    "results_heavyhex.json", "results_heavyhex_star.json",
    "results_scaling_b.json", "results_scaling_gran.json",
    "results_scaling_regcz.json",
    "results_80q.json", "results_80q_dsabre.json", "results_80q_telesabre.json",
    "results_regcz_full.json", "results_regcz_no_cap.json",
    "results_regcz_no_hopgain.json", "results_regcz_no_lookahead.json",
    "results_regcz_no_relief.json", "results_regcz_topo_ext.json",
    "results_regcz_ablation.json", "results_regcz_telesabre.json",
    "results_fill_sweep_dse.json",
    "ablate_hop_gain_360q.json", "ablate_hop_gain_360q_qpe.json",
]

# Protocol totals recovered from the run logs that printed them
# (results/regen_final/scal_*.log).  Keyed by file -> row label -> router key.
TOTALS = {
    "results_scaling_b.json": {
        "100q": {"ts": 24.0, "dS": 83.0, "dSE": 80.0},
        "200q": {"ts": 479.0, "dS": 558.0, "dSE": 469.0},
        "360q": {"ts": 1411.0, "dS": 3969.0, "dSE": 1843.0},
    },
    "results_scaling_gran.json": {
        "200q-6x7x7": {"ts": 117.0, "dSE": 436.0},
        "200q-12x5x5": {"ts": 309.0, "dSE": 388.0},
    },
    "results_scaling_regcz.json": {
        "100q": {"ts": 73.0, "dSE": 343.0},
        "200q": {"ts": 645.0, "dSE": 1310.0},
        "360q": {"ts": 2715.0, "dSE": 5049.0},
    },
}

NOTE = ("time_s migrated 2026-07-31: the legacy value (compile time of the "
        "winning seed) moved to time_seed_s.  time_s now means the whole-"
        "protocol cost and is present only where the run log recorded it; "
        "absent means the total was never measured for that row.")


def migrate_node(node, total=None):
    """Move time_s -> time_seed_s in one dict; set or drop time_s."""
    if "time_s" not in node or "time_seed_s" in node:
        return 0
    node["time_seed_s"] = node.pop("time_s")
    if total is not None:
        node["time_s"] = total
    return 1


def walk(node, fname, row_label=None, router=None):
    """Migrate every timing dict below `node`, tracking the scaling row label."""
    n = 0
    if isinstance(node, dict):
        label = node.get("label", row_label)
        totals = TOTALS.get(fname, {}).get(label, {}) if label else {}
        if "time_s" in node and not isinstance(node["time_s"], (dict, list)):
            n += migrate_node(node, totals.get(router))
        for k, v in node.items():
            if k in ("routers",) and isinstance(v, dict):
                for rk, rv in v.items():
                    n += walk(rv, fname, label, rk)
            elif k == "ts" and isinstance(v, dict):
                n += walk(v, fname, label, "ts")
            elif isinstance(v, (dict, list)):
                n += walk(v, fname, label, router)
    elif isinstance(node, list):
        for v in node:
            n += walk(v, fname, row_label, router)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the files (default is a dry run)")
    args = ap.parse_args()

    backup = os.path.join(RESULTS, f"_pre_time_migration_{int(time.time())}")
    if args.apply:
        os.makedirs(backup, exist_ok=True)

    total_rows = 0
    for fname in LEGACY:
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            print(f"  {fname:42} MISSING -- skipped")
            continue
        with open(path) as f:
            doc = json.load(f)
        n = walk(doc, fname)
        if isinstance(doc, dict):
            doc.setdefault("meta", {})
            if isinstance(doc["meta"], dict):
                doc["meta"]["time_key_migration"] = NOTE
        known = sum(len(v) for v in TOTALS.get(fname, {}).values())
        print(f"  {fname:42} {n:4} rows migrated"
              f"{f', {known} with recovered totals' if known else ''}")
        total_rows += n
        if args.apply and n:
            shutil.copy2(path, os.path.join(backup, fname.replace("/", "_")))
            with open(path, "w") as f:
                json.dump(doc, f, indent=1)

    print(f"\n{total_rows} rows across {len(LEGACY)} files")
    if args.apply:
        print(f"originals backed up to {backup}")
    else:
        print("dry run -- nothing written; re-run with --apply")


if __name__ == "__main__":
    main()
