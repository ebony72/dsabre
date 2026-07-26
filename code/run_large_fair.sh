#!/bin/bash
# Run the physical-model pytket-dqc sweep on the large scalability suites,
# one after another so they do not contend for CPU.  Each writes its own log
# and its own copy of the results JSON, since the sweep's resume logic keys off
# a single file.
cd "$(dirname "$0")" || exit 1

while pgrep -f "bench_pytket_fair" > /dev/null; do sleep 30; done
cp -f results/results_pytket_fair.json results/results_pytket_fair_100q.json 2>/dev/null
echo "QUEUE: 100q recorded"

for suite in 200q 360q; do
  rm -f results/results_pytket_fair.json
  echo "QUEUE: starting $suite"
  python3 bench_pytket_fair.py --suite "$suite" --budget 900 \
      > "results/pyfair_${suite}.log" 2>&1
  echo "QUEUE: $suite exit=$?"
  cp -f results/results_pytket_fair.json "results/results_pytket_fair_${suite}.json" 2>/dev/null
done

echo "QUEUE: all large suites done"
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("results/results_pytket_fair_*q.json")):
    for r in json.load(open(f))["results"]:
        if r["suite"] in ("100q", "200q", "360q"):
            print(f"{r['suite']:<6}{r['circuit']:<10}"
                  f"pub={r['A_published']['ebits']}  "
                  f"phys={r['C_physical']['ebits']}  "
                  f"ports={r.get('A_required_ebit_mem')}  "
                  f"fits={r.get('C_fits_port_bound')}")
PY
