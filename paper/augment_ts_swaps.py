"""
augment_ts_swaps.py — extract TeleSABRE local-SWAP counts and patch the
existing results JSONs.

The original benchmark only records TS_EPR (= teledata + telegate); it
ignores the "Swaps: N" line in TS's output.  This script re-runs TS on
each circuit (single seed, picking the seed that gave the best EPR
already in the JSON), captures Swaps, and writes back into
results/results_{25q,36q,64q}.json under ts["ts_ls"].

Quick: ~30 seconds per suite on a warm machine.
"""

import sys, os, json, glob, subprocess, tempfile, time
sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import _ts_config, SUITES, TS_BIN


def run_ts_for_swaps(qasm_path, ts_dev, seed):
    rpt = tempfile.mktemp(suffix=".json")
    cfg = _ts_config(seed, rpt, "swap_extract")
    try:
        proc = subprocess.run([TS_BIN, cfg, ts_dev, qasm_path],
                              capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(cfg)
        if os.path.exists(rpt): os.unlink(rpt)
    out = proc.stdout + proc.stderr
    td = tg = sw = None; ok = False
    for line in out.splitlines():
        s = line.strip()
        if   "Teledata:" in s: td = int(s.split(":")[1])
        elif "Telegate:" in s: tg = int(s.split(":")[1])
        elif s.startswith("Swaps:") or s.lstrip().startswith("Swaps:"):
            try: sw = int(s.split("Swaps:")[1].strip())
            except Exception: pass
        elif "Success: true" in s: ok = True
    if ok and sw is not None and td is not None:
        return td, tg or 0, sw
    return None


def main():
    for suite in ["25q", "36q", "64q"]:
        s = SUITES[suite]
        path = f"/Users/sanjiangli/Documents/pyzoo/dsabre/paper/results/results_{suite}.json"
        with open(path) as f:
            data = json.load(f)
        print(f"\n── {suite} ──")
        for rec in data["results"]:
            ts = rec.get("ts")
            if ts is None:
                print(f"  {rec['circuit']:<14}  TS aborted, skipping")
                continue
            qf = os.path.join(s["circuit_dir"], rec["circuit"] + s["suffix"])
            r = run_ts_for_swaps(qf, s["ts_dev"], ts["seed"])
            if r is None:
                print(f"  {rec['circuit']:<14}  re-run failed, skipping")
                continue
            td, tg, sw = r
            old_eprs = ts["eprs"]
            new_eprs = td + tg
            ts["ts_ls"] = sw
            if new_eprs != old_eprs:
                print(f"  {rec['circuit']:<14}  EPR mismatch! old={old_eprs}, new={new_eprs}; LS={sw}")
            else:
                print(f"  {rec['circuit']:<14}  EPR={old_eprs}, TS_LS={sw}  ✓")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  → patched {path}")


if __name__ == "__main__":
    main()
