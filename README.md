# dSABRE

A SABRE-style router for **multi-core distributed quantum computers**
(DQCs).  Given a quantum circuit and a multi-core architecture, dSABRE
inserts intra-core SWAPs and inter-core teleports to make every
two-qubit gate executable while minimising EPR-pair consumption — the
dominant cost on near-term photonic-linked modular hardware.

This repository hosts the reference implementation, benchmark
harnesses, per-circuit result JSONs, the standalone appendices, and
the compiled paper (`paper/dsabre.pdf`).

## What's in the box

```
code/                  reference implementation + benchmark scripts
  router.py            General_dSABRE_Router (multi-core routing loop)
  dsabre_ext.py        dSABRE_BurstExt — BFS extended-set variant ("dSABRE" in the paper)
  architecture.py      DistributedArchitecture, B-grid and H-grid builders
  layout.py            locality_aware_layout + run_passes helpers
  config.py            HardwareConfig
  benchmark.py         main benchmark runner (25q / 36q / 64q suites)
  bench_large.py       100q / 200q / 360q QFT scalability sweep
  bench_pytket_*.py    pytket-dqc baselines (main + layout-controlled)
  ablate_*.py          ablation drivers (extended-set decay, hop gain, ...)
  verify.py            structural-equivalence verifier for routed output
  verify_diff.py       cross-run diff for verification
  results/             per-experiment JSON results (matches paper tables)
  results_verify/      re-run verification artefacts

paper/
  dsabre.pdf           compiled manuscript (submitted to IEEE TCAD)
  appendices_standalone.pdf   online appendices referenced from the paper
  build_tables_xlsx.py        helper: builds paper_tables.xlsx from results/
  paper_tables.xlsx           consolidated table data
```

The paper's LaTeX sources are intentionally not in this repository.

## Quick start

dSABRE is pure Python (Qiskit + NetworkX).

```bash
pip install qiskit networkx numpy
cd code
python benchmark.py --suite 25         # 25-qubit B-grid suite
python benchmark.py --suite 36         # 36-qubit B-grid suite
python benchmark.py --suite 64         # 64-qubit H-grid suite
python benchmark.py --suite all        # all three
```

Each run reads MQT-Bench circuits, sets up the matching architecture,
runs SabreLayout (best of 3 seeds) followed by dSABRE
forward → backward → forward routing, and writes
`results/results_<suite>.json`.

To reproduce the scalability sweep (QFT at 100, 200, 360 qubits):

```bash
python bench_large.py
```

To re-verify that the routed circuits are structurally equivalent to
the originals:

```bash
bash verify_run.sh
```

## Using the router programmatically

```python
from qiskit.converters import circuit_to_dag
from architecture import build_h_grid_architecture
from config import HardwareConfig
from layout import locality_aware_layout, run_passes
from dsabre_ext import dSABRE_BurstExt
import random

arch    = build_h_grid_architecture(rows=2, cols=3, m=4)
config  = HardwareConfig()
router  = dSABRE_BurstExt(arch, config)

dag     = circuit_to_dag(my_circuit)
layout  = locality_aware_layout(dag, arch, rng=random.Random(0))
routed, metrics = run_passes(router, dag, layout, n=3)   # fwd-bwd-fwd
print(metrics["eprs"], metrics["ls"])                    # EPR count, local SWAPs
```

`route()` accepts a Qiskit `DAGCircuit` (not a `QuantumCircuit` — call
`circuit_to_dag` first).  `locality_aware_layout` requires a
`random.Random` instance, not a bare seed.

## Headline numbers

Across 18 MQT-Bench circuits at 25 / 36 / 64 logical qubits, dSABRE
reduces geometric-mean EPR consumption by

| vs.                            | 25q   | 36q   | 64q   |
| ------------------------------ | ----- | ----- | ----- |
| TeleSABRE (best of 3 seeds)    | −41 % | −44 % | −44 % |
| pytket-dqc (best of 5 seeds)   | −68 % | −29 % | −16 % |

with parity on the QFT scalability sweep up to 360 qubits on a
486-physical-qubit H-grid.  Full per-circuit tables, the ablation
study, and the cost-ratio sensitivity sweep are in the paper.

## Citation

```bibtex
@article{li2026dsabre,
  author  = {Li, Sanjiang},
  title   = {{dSABRE}: A {SABRE}-Style Router for Multi-Core
             Distributed Quantum Computers},
  journal = {IEEE Transactions on Computer-Aided Design of
             Integrated Circuits and Systems (under review)},
  year    = {2026}
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
