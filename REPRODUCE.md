# Reproducing the results in the paper

Every table and figure in the manuscript and in the online appendices, with the
command that produces it and the result file it reads. Commands are run from
`code/` unless stated otherwise.

## Environment

```
Python 3.13, Qiskit 2.3, NetworkX 3.5
pytket-dqc 0.0.1, pytket 2.16, KaHyPar 1.3.5   (cross-model comparison only)
```

```bash
pip install -r requirements.txt
```

Measurements in the paper ran single-threaded on an Apple M2, 16 GB,
macOS 26.5. `TeleSABRE` is a separate compiled C binary; see
[TeleSABRE baseline](#telesabre-baseline) below.

The drivers find their circuits through `code/circuit_paths.py`, which looks in
`$DSABRE_CIRCUITS`, then `benchmark_circuits/` in this repository, then
`~/Documents/telesabre/circuits`. A fresh clone needs no configuration — the
second root is the curated set that every published table was computed from.
Run `python circuit_paths.py` to print the resolved search path.

## Main paper

| Table / figure | Command | Result file |
|---|---|---|
| Table III, 25q/36q/64q suites | `python benchmark.py --suite all` | `results/results_{25q,36q,64q}.json` |
| Table III, scalability rows (100/200/360q) | `python bench_large.py` | `results/results_{100,200,360}q.json` |
| Table III, heavy-hex ring | `python bench_heavyhex.py --topology ring` | `results/results_heavyhex.json` |
| Table III, heavy-hex star | `python bench_heavyhex.py --topology star` | `results/results_heavyhex_star.json` |
| Table IV, mechanism ablation | `python ablate_mech_rows.py` | `results/results_mech_ablation.json` |
| Table IV, capacity rows | `python ablate_capacity.py` | `results/results_ablate_capacity.json` |
| Table IV, shared-layout control | `python bench_sharedmap.py` | `results/results_sharedmap.json` |
| Table V, recovery counts | `python benchmark.py --suite all` (Tier-2 counters) | `results/results_{25q,36q,64q}.json` |
| Table VI, `pytket-dqc` | `python bench_pytket_fair.py --suite all --budget 900` | `results/results_pytket_fair_v3_{25q,36q,64q}.json` |
| Figure 4 / Table II, running example | hand-constructed from the 2x3 H-grid; no driver | --- |

## Regenerating the paper's tables

The result JSONs are the source of truth; the LaTeX table bodies in
`code/tables/` are generated from them.

```bash
python gen_main_merged.py     # Table III  -> tables/main_merged.tex
python gen_fair_table.py      # Table VI   -> tables/fair.tex
python gen_corenet_table.py   # Appendix C.III
python gen_dmax_table.py      # Appendix D.II
```

`gen_main_merged.py` reproduces every number in Table III exactly; the only
differences from the manuscript are cosmetic label spellings (`vqe\_su2`
against VQE-SU2, `gmean (8)` against Gmean (8 common)).

## Online appendices

| Table | Command | Result file |
|---|---|---|
| A.I, capacity-mechanism arms | `python ablate_capacity.py` | `results/results_ablate_capacity.json` |
| B.I, complexity validation | `python bench_large.py` (timing columns) | `results/results_scaling_b.json` |
| C.I, cost-model sensitivity | reweights Table III counts; re-route check: `python probe_cost_ratio_sweep.py` | `results/results_cost_ratio_sweep.json` |
| C.II, 25-qubit ablation | `python ablate_mech_rows.py --suite 25q` | `results/results_mech_ablation.json` |
| C.III, non-grid core networks | `python bench_heavyhex.py --topology {ring,star}` then `python gen_corenet_table.py` | `results/results_heavyhex*.json` |
| C.IV, what one teledata pair buys | `python ablate_telegate.py` then `python analyze_telegate.py` | `results/results_telegate_64q.json` |
| C.V, compile time | `python benchmark.py --suite 64` (timing columns) | `results/results_64q.json` |
| C.VI, benchmark provenance | `python gen_seed_table.py` | `results/results_64q.json` |
| C.VII, per-seed counts | `python gen_seed_table.py` | `results/results_{25q,36q,64q}.json` |
| D.I, `pytket-dqc` capacity audit | `python bench_pytket_fair.py --suite all --budget 900` | `results/results_pytket_fair_v3_*.json` |
| D.II, entanglement lifetime | `python bench_dmax_lifetime.py` then `python gen_dmax_table.py` | `results/results_dmax_64q.json` |
| D.III, DMapS head-to-head | `python run_dmaps_bench.py` | `results/results_dmaps_bench.json` |

`results_scaling_b.json` (B.I) is the earlier 2026-08-06 run of the same
configuration, kept for its timing series only. The EPR counts in Table III
come from `results_{100,200,360}q.json`, not from it.

## TeleSABRE baseline

`TeleSABRE` is built from its own repository
(<https://github.com/Haimrich/telesabre>) and invoked positionally:

```bash
telesabre <config.json> <device.json> <circuit.qasm>
```

The benchmark drivers write the config JSON themselves. Two settings matter
for reproduction:

* `init_layout_hun_min_free_gate: 5` and
  `init_layout_hun_min_free_qubit: 4` — the values in `TeleSABRE`'s own
  shipped `configs/default.json`. The submitted version of this paper
  misspelled the second key, so the tool silently fell back to its
  compiled-in default of 3; the revision passes it correctly.
* `enable_passing_core_emptying_teleport_possibility: false` — also
  `TeleSABRE`'s shipped default.

Device files (`B_grid_2_2_4_4.json`, `H_grid_2_3_4_4.json`,
`H_grid_{2_3_5_5,4_3_5_5}.json`, `HeavyHex_{ring4,star4}_27.json`) come from
that repository's `devices/` directory.

## Verification

```bash
python verify_router.py        # router vs. the frozen pre-optimisation baseline
python verify_architecture.py  # composed phys_dist vs. a dense all-pairs table
bash verify_run.sh             # routed circuits are structurally equivalent to the inputs
```

`verify_router.py` pins `local_ext_mode="taint", safe_mode=False`, the mode the
frozen baseline implements, and reports `ALL IDENTICAL` when the incremental
implementation reproduces it exactly.

## Benchmark circuits

`benchmark_circuits/qasm_{25,36,64,100,200,360}/` hold the routed suites as
**MQT Bench v1.1.0** downloads, transpiled to the IBM native basis
`{CX, Rz, sqrt(X), X}` at Qiskit optimisation level 3, with measurements and
barriers stripped. Do not regenerate them with a newer MQT Bench: v2.x emits
structurally different circuits under the same names (`qnn` at 64 qubits is
8126 CX in v1.1.0 and 63 CX in v2.2.2), which would silently change every
published number.
