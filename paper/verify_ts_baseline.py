"""
verify_ts_baseline.py — A2: cross-check our measured TeleSABRE numbers
against those reported in the TeleSABRE paper (arXiv 2505.08928).

The paper reports TeleSABRE as achieving 28% reduction vs HQA on average.
The paper's Table II / Figure shows per-circuit numbers we can spot-check.

This script reads results/results_25q.json (etc.) and reports:
  - TS EPR per circuit (our subprocess invocation, 3 seeds)
  - TS EPR per circuit (single seed=0, no best-of selection)
  - Notes any large variance across seeds (suggesting their reported number
    might differ from ours just due to seed selection)

If our numbers are systematically off by ~2x, we may be mis-counting EPRs
(e.g., teledata vs telegate) or running with different config flags.

Output:
  stdout — formatted comparison

Usage:
  python verify_ts_baseline.py
"""

import sys, os, json, glob, subprocess, tempfile, time
sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import _ts_config, p2v_to_layout, SUITES, NUM_TS_SEEDS, TS_BIN


def run_ts_all_seeds(qasm_path, ts_dev, n_seeds=5):
    """Run TS with `n_seeds` different seeds; return per-seed (td, tg, ok, time)."""
    results = []
    for seed in range(n_seeds):
        rpt = tempfile.mktemp(suffix=".json")
        cfg = _ts_config(seed, rpt, "verify")
        t0  = time.perf_counter()
        try:
            proc = subprocess.run([TS_BIN, cfg, ts_dev, qasm_path],
                                  capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            os.unlink(cfg)
            results.append((seed, None, None, False, None, "TIMEOUT"))
            continue
        elapsed = time.perf_counter() - t0
        out = proc.stdout + proc.stderr
        td = tg = 0; ok = False; depth = None
        for line in out.splitlines():
            s = line.strip()
            if   "Teledata:" in s: td = int(s.split(":")[1])
            elif "Telegate:" in s: tg = int(s.split(":")[1])
            elif "Success: true" in s: ok = True
            elif "Depth:" in s and depth is None:
                try: depth = int(s.split(":")[1])
                except Exception: pass
        os.unlink(cfg)
        if os.path.exists(rpt): os.unlink(rpt)
        results.append((seed, td, tg, ok, round(elapsed, 2), depth))
    return results


def main():
    print(f"\n{'='*100}")
    print(f"  TeleSABRE seed-variance check  (Teledata + Telegate per seed)")
    print(f"{'='*100}")
    print(f"  {'circuit':<14}  {'q':>3}  {'seed0':>9}  {'seed1':>9}  {'seed2':>9}  "
          f"{'seed3':>9}  {'seed4':>9}  {'min':>5}  {'max':>5}  {'gap':>5}")
    print("  " + "-" * 96)

    for suite_name in ["25q", "36q", "64q"]:
        s = SUITES[suite_name]
        qasm_files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
        if not qasm_files:
            continue
        print(f"  ── {suite_name} " + "─" * 88)
        for qf in qasm_files:
            cname = os.path.basename(qf).replace(s["suffix"], "")
            seed_results = run_ts_all_seeds(qf, s["ts_dev"], n_seeds=5)
            cells = []
            eprs = []
            for seed, td, tg, ok, t_s, depth in seed_results:
                if not ok:
                    cells.append("   ABORT")
                else:
                    epr = td + tg
                    cells.append(f"{td:>2}+{tg:<2}={epr:>2}")
                    eprs.append(epr)
            mn = min(eprs) if eprs else None
            mx = max(eprs) if eprs else None
            gap = (mx - mn) if eprs else None
            row = f"  {cname:<14}  ?"
            row = f"  {cname:<14}  {'':>3}"
            for c in cells:
                row += f"  {c:>9}"
            row += f"  {mn:>5}" if mn is not None else f"  {'---':>5}"
            row += f"  {mx:>5}" if mx is not None else f"  {'---':>5}"
            row += f"  {gap:>5}" if gap is not None else f"  {'---':>5}"
            print(row)
    print()


if __name__ == "__main__":
    main()
