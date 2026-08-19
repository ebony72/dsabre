# investigations/

Supporting scripts that are not part of the reproduction path. Everything the
paper's tables are built from lives one level up, in `code/`; this folder holds
the work that led there.

Nothing in `code/` imports anything here — the dependency only runs the other
way — so this folder can be ignored entirely if you just want to reproduce the
paper. Start from [`../../REPRODUCE.md`](../../REPRODUCE.md) instead.

## What is here

| Group | Files | What they are |
|---|---|---|
| `probe_*.py` | 19 | One-question experiments. Several are the evidence behind [`../../CHANGES_FROM_SUBMITTED.md`](../../CHANGES_FROM_SUBMITTED.md) — for example `probe_relief_exact_reproduction.py` and `probe_relief_isolate_node_decay.py`, which reproduce the submitted congestion-relief number and then isolate `node_decay` as its whole cause. |
| `ablate_*.py` | 14 | Ablation arms that did not make the published table: occupancy skew, corner-mask variants, free-slot floors, priority eviction, monolithic layout, protocol variants. |
| `bench_*.py` | 7 | Superseded or exploratory benchmark drivers — an 80-qubit suite, an 8-link B-grid, seed-variance and extended-64q runs, and two earlier `pytket-dqc` configurations. |
| `regen_*.py` | 6 | Regeneration drivers used to refresh a table after a router change. |
| `gen_*.py` | 5 | Circuit generators. **These write into the shared MQT Bench tree by design** and are kept for provenance, not for re-running — see the warning in [`../../benchmark_circuits/README.md`](../../benchmark_circuits/README.md). |
| `analyze_*.py`, `run_*.py`, `sweep_params.py`, `migrate_time_keys.py`, `verify_telegate.py`, `check_full_core_layouts.py`, `gran_ds_supplement.py` | 9 | Analysis, parameter sweeps, one-off maintenance, and a layout audit across every architecture in the paper. |

## Running one

They are run the same way as the core drivers, from either directory:

```bash
python investigations/probe_relief_exact_reproduction.py
```

Each script puts `code/` on `sys.path` and resolves `_HERE` to `code/`, so
`results/`, `circuits_regcz/` and the implementation modules are found exactly
as they were before this folder existed. Result JSONs are still written to
`code/results/`, not to a `results/` directory here.

Circuits are located through `../circuit_paths.py`, which prefers the
repository's own `benchmark_circuits/`; set `DSABRE_CIRCUITS` to point
elsewhere.
