#!/bin/bash
# Chain the revision regeneration jobs: wait for the main suite run to finish,
# then re-run the mechanism ablation and the heavy-hex architecture check.
# Each stage logs to results/ so a partial run is still usable.
cd "$(dirname "$0")" || exit 1

MAIN_PID="$1"
if [ -n "$MAIN_PID" ]; then
  echo "QUEUE: waiting for main benchmark (pid $MAIN_PID)"
  while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 20; done
  echo "QUEUE: main benchmark finished"
fi

echo "QUEUE: starting mechanism ablation"
python3 regen_ablation_corners.py > results/regen_ablation.log 2>&1
echo "QUEUE: ablation exit=$?"

echo "QUEUE: starting heavy-hex architecture check"
python3 bench_heavyhex.py > results/regen_heavyhex.log 2>&1
echo "QUEUE: heavyhex exit=$?"

echo "QUEUE: all revision jobs complete"
