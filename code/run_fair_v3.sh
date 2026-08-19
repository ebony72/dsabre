#!/bin/bash
# The corrected pytket-dqc sweep: full-core A_published, non-starved
# C_physical (the even_split -> full_capacity fix of 2026-08-13), still with
# provenance recording and persisted distributions (option C of the
# 2026-08-12 consistency pass).  v1 = submitted/original.  v2 = provenance
# added but network-capacity bug present (see
# results/STALE_badnetwork_2026-08-13/README.md).  v3 = both fixed.
#
# One suite at a time -- two sweeps in parallel would contend for CPU and
# corrupt the compile-time column, which tab:timing reads from these same
# runs.  25q and 36q first (fast, and 25q's executability count is wanted
# soonest); 64q last (slow: ~4h, dominated by circuits where
# CoverEmbeddingSteinerDetached exhausts its budget every seed).
cd "$(dirname "$0")" || exit 1

for suite in 25q 36q 64q; do
  echo "QUEUE: starting $suite  $(date)"
  python3 bench_pytket_fair.py --suite "$suite" --budget 900 \
      --out "results_pytket_fair_v3_${suite}.json" \
      > "results/pyfair_v3_${suite}.log" 2>&1
  echo "QUEUE: $suite exit=$?  $(date)"
done
echo "QUEUE: done  $(date)"
