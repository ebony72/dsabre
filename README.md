# dSABRE

A SABRE-style router for **multi-core distributed quantum computers**
(DQCs).  Given a quantum circuit and a multi-core architecture, dSABRE
inserts intra-core SWAPs and inter-core teleports to make every
two-qubit gate executable while minimising EPR-pair consumption — the
dominant cost on near-term modular hardware.

> **This is the revised version** (2026-08-19).  It differs from the version
> first submitted in May 2026 in both the algorithm and the measured numbers —
> see [`CHANGES_FROM_SUBMITTED.md`](CHANGES_FROM_SUBMITTED.md).  The router
> as first submitted is preserved verbatim in `code/_legacy_*_submitted.py` so
> those numbers remain reproducible.

This repository hosts the reference implementation, the benchmark
harnesses, the benchmark circuits, the per-circuit result JSONs, and the
compiled paper and online appendices.

## Reproducing the paper

**[`REPRODUCE.md`](REPRODUCE.md) is the entry point**: it lists every table
and figure in the manuscript and in the online appendices, with the command
that produces it and the result file it reads.

```bash
pip install -r code/requirements.txt
cd code
python benchmark.py --suite all      # Table III, 25q/36q/64q suites
python bench_large.py                # Table III, 100q/200q/360q scalability
python ablate_mech_rows.py           # Table IV, mechanism ablation
```

Measurements in the paper ran single-threaded on an Apple M2, 16 GB,
macOS 26.5, under Python 3.13 / Qiskit 2.3 / NetworkX 3.5.

## What's in the box

```
code/                  the implementation and the drivers behind the paper's tables
  router.py            General_dSABRE_Router (multi-core routing loop)
  dsabre_ext.py        dSABRE_BFSExt — BFS extended-set variant ("dSABRE" in the paper)
  architecture.py      DistributedArchitecture, B-grid and H-grid builders
  layout.py            locality_aware_layout + run_passes helpers
  config.py            HardwareConfig
  circuit_paths.py     where the benchmark circuits are found
  benchmark.py         main benchmark runner (25q / 36q / 64q suites)
  bench_large.py       100q / 200q / 360q scalability sweep (QFT, QPEexact)
  bench_heavyhex.py    heavy-hex ring and star core networks
  bench_pytket_fair.py pytket-dqc cross-model baseline
  ablate_*.py          the three ablations reported in Table IV and Appendix A
  gen_*_table.py       LaTeX table bodies, generated from results/
  verify_*.py          router / architecture / structural-equivalence verifiers
  _baseline_*.py       frozen pre-optimisation router — the verifiers diff against it
  _legacy_*_submitted.py   the router exactly as submitted (May 2026), frozen
  results/             per-experiment JSON results (matches the paper's tables)
  tables/              generated LaTeX table bodies
  investigations/      60 supporting scripts that are NOT on the reproduction
                       path: probes, unpublished ablation arms, superseded
                       drivers, circuit generators.  Nothing in code/ imports
                       them; see investigations/README.md

benchmark_circuits/    the 45 circuits every table is computed from, with a
                       README recording provenance and per-circuit CX counts

paper/
  dsabre.pdf                  compiled manuscript (12 pp.)
  appendices_standalone.pdf   online appendices referenced from the paper (7 pp.)
  build_tables_xlsx.py        helper: builds paper_tables.xlsx from results/
  paper_tables.xlsx           consolidated table data
```

The paper's LaTeX sources are intentionally not in this repository.

**Do not regenerate the benchmark circuits.** They are MQT Bench v1.1.0
downloads; v2.x emits structurally different circuits under the same names
(`qnn` at 64 qubits is 8126 CX in v1.1.0 and 63 CX in v2.2.2), which would
silently change every published number.  See
`benchmark_circuits/README.md`.

## Using the router programmatically

```python
from qiskit.converters import circuit_to_dag
from architecture import build_h_grid_architecture
from config import HardwareConfig
from layout import locality_aware_layout, run_passes
from dsabre_ext import dSABRE_BFSExt
import random

arch    = build_h_grid_architecture(rows=2, cols=3, m=4)
config  = HardwareConfig()
router  = dSABRE_BFSExt(arch, config)

dag     = circuit_to_dag(my_circuit)
layout  = locality_aware_layout(dag, arch, rng=random.Random(0))
routed, metrics = run_passes(router, dag, layout, n=3)   # fwd-bwd-fwd
print(metrics["eprs"], metrics["ls"])                    # EPR count, local SWAPs
```

`route()` accepts a Qiskit `DAGCircuit` (not a `QuantumCircuit` — call
`circuit_to_dag` first).  `locality_aware_layout` requires a
`random.Random` instance, not a bare seed.  A single call like this runs one
seed and one layout, so it will not match the paper's tables, which report
the best of three SabreLayout seeds each routed forward → backward →
forward; use `benchmark.py` for a comparable number.

## Headline numbers (revised version)

Across 21 MQT Bench circuits at 25 / 36 / 64 logical qubits, dSABRE reduces
geometric-mean EPR consumption against TeleSABRE by

| suite | 25q | 36q | 64q |
| --- | --- | --- | --- |
| Δ geometric-mean EPR | −50.8 % | −62.0 % | −49.1 % |

on the circuits both routers complete.  dSABRE completes all 45 evaluated
circuit–architecture instances against TeleSABRE's 40, and compiles a
360-qubit QFT on twenty cores in 48 s.

Against `pytket-dqc`, which counts e-bits under gate teleportation and an
unbounded entanglement lifetime, dSABRE uses 56.1 % and 13.8 % fewer pairs at
25 and 36 qubits and 15.3 % more at 64 — but 4/6, 2/6 and at least 6/9 of
those distributions cannot be materialised within the evaluated
communication-port capacity.  The paper reports that comparison as a
cost-model study rather than a like-for-like one.

## Verification

```bash
cd code
python verify_router.py        # incremental router vs. the frozen baseline
python verify_architecture.py  # composed phys_dist vs. a dense all-pairs table
bash   verify_run.sh           # routed circuits are structurally equivalent to the inputs
```

## Citation

```bibtex
@misc{li2026dsabre,
      title={dSABRE: A SABRE-Style Router for Multi-Core Distributed Quantum Computers},
      author={Sanjiang Li},
      year={2026},
      eprint={2605.21960},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2605.21960},
}
```

## License

Released for academic use.  Please cite the paper if you use the
router, the benchmark scripts, or the result JSONs in derivative work.

## Contact

Sanjiang Li — `sanjiang.li@uts.edu.au` — ORCID
[0000-0002-3332-2546](https://orcid.org/0000-0002-3332-2546)
Centre for Quantum Software and Information, University of Technology
Sydney.
