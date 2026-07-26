#!/bin/bash
# Wait for the pytket-dqc fair sweep to finish, then regenerate the merged
# main-results table body from the measured physical-model numbers and print a
# summary of what changed.
cd "$(dirname "$0")" || exit 1

while pgrep -f "bench_pytket_fair" > /dev/null; do sleep 30; done
echo "SWEEP COMPLETE"

python3 - <<'PY'
import json
d = json.load(open("results/results_pytket_fair.json"))
print(f"{'suite':<6}{'circuit':<12}{'published':>10}{'physical':>10}{'ports':>7}{'fits':>7}")
for r in d["results"]:
    print(f"{r['suite']:<6}{r['circuit']:<12}"
          f"{str(r['A_published']['ebits']):>10}"
          f"{str(r['C_physical']['ebits']):>10}"
          f"{str(r.get('A_required_ebit_mem')):>7}"
          f"{str(r.get('C_fits_port_bound')):>7}")
PY

echo
echo "=== regenerated table body ==="
python3 gen_main_merged.py | tail -45
