"""
analyze_mechanisms.py — summarize which router mechanisms actually fire.

Reads results/results_{25q,36q,64q}.json (instrumented) and reports per-circuit
counts of:
  relief_candidates    : proactive-relief teleport candidates generated
  relief_picks         : iterations where the chosen teleport came from relief
  backup_activations   : deadlock-recovery checkpoint rewinds
  force_make_room      : forced teleports to free a saturated core

Decides which mechanisms are load-bearing contributions vs defensive cushions
for the paper's §3 framing.
"""

import json, os, glob

ROUTERS = ["dS", "dSE"]
SUITES  = ["25q", "36q", "64q"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load(suite):
    p = os.path.join(RESULTS_DIR, f"results_{suite}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    print(f"\n{'='*88}")
    print(f"  Mechanism activation per circuit  (R-cand / R-pick / Backup / Force-room)")
    print(f"{'='*88}")
    print(f"  {'circuit':<14}  {'router':<5}  {'EPRs':>5}  {'R-cand':>7}  {'R-pick':>7}  {'Backup':>7}  {'F-room':>7}")
    print("  " + "-" * 76)

    grand = {k: {"R-cand": 0, "R-pick": 0, "Backup": 0, "F-room": 0, "n": 0} for k in ROUTERS}

    for suite in SUITES:
        data = load(suite)
        if data is None:
            continue
        print(f"  ── {suite} " + "─" * 70)
        for rec in data["results"]:
            cname = rec["circuit"]
            for k in ROUTERS:
                r = rec["routers"].get(k, {})
                if r.get("aborted") or r.get("eprs") is None:
                    continue
                rc = r.get("relief_candidates", 0)
                rp = r.get("relief_picks", 0)
                ba = r.get("backup_activations", 0)
                fr = r.get("force_make_room", 0)
                marker = "·"
                if rp > 0 or ba > 0 or fr > 0:
                    marker = "●"
                print(f"  {cname:<14}  {k:<5}  {r['eprs']:>5}  "
                      f"{rc:>7}  {rp:>7}  {ba:>7}  {fr:>7}  {marker}")
                grand[k]["R-cand"] += rc
                grand[k]["R-pick"] += rp
                grand[k]["Backup"] += ba
                grand[k]["F-room"] += fr
                grand[k]["n"]      += 1

    print(f"\n{'='*88}")
    print(f"  Grand totals across all suites/circuits")
    print(f"{'='*88}")
    print(f"  {'router':<6}  {'n':>4}  {'R-cand':>10}  {'R-pick':>10}  {'Backup':>10}  {'F-room':>10}")
    print("  " + "-" * 60)
    for k in ROUTERS:
        g = grand[k]
        print(f"  {k:<6}  {g['n']:>4}  {g['R-cand']:>10}  {g['R-pick']:>10}  "
              f"{g['Backup']:>10}  {g['F-room']:>10}")

    print(f"\n  Verdict on each mechanism:")
    for k in ROUTERS:
        g = grand[k]
        msg = []
        msg.append(f"R-cand={'fires' if g['R-cand']>0 else 'NEVER'}")
        msg.append(f"R-pick={'fires' if g['R-pick']>0 else 'NEVER'}")
        msg.append(f"Backup={'fires' if g['Backup']>0 else 'NEVER'}")
        msg.append(f"F-room={'fires' if g['F-room']>0 else 'NEVER'}")
        print(f"    {k}: " + ", ".join(msg))


if __name__ == "__main__":
    main()
