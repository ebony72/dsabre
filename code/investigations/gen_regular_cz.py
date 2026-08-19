r"""
gen_regular_cz.py — a structured synthetic circuit family for the ablation.

Each circuit fixes one random d-regular interaction graph on n logical qubits
and applies L layers of "single-qubit rotations, then a CZ across every edge".
The CZ-over-a-random-regular-graph construction is the one pytket-dqc ships as
`RegularGraphHypergraphCircuit`, so the family has precedent in the DQC
compilation literature rather than being invented here; the interleaved
rotations give it the shape of a hardware-efficient ansatz and keep repeated
layers from cancelling (CZ*CZ = I).

Why this family and not uniformly random circuits.  The circuit property a
distributed router actually responds to is the locality of the *interaction
graph*: how many distinct partners each qubit has, and therefore how much
traffic must cross a core boundary whatever the layout.  Holding the graph
d-regular makes that property exactly d and leaves nothing else varying.
Uniformly random circuits vary it only incidentally, and are reported to
distort DQC partitioning benchmarks -- inflating cut costs and even reordering
strategy rankings (Gragera Garces, arXiv:2605.01974) -- which is the opposite
of what an ablation needs.

Layer count is set per degree to hold the CX count near TARGET_CX, so sweeping
d isolates interaction *structure* from gate *volume*: a d=8 circuit is not
simply a longer d=2 circuit.

Transpilation is optimization_level=1, not the opt3 the MQT Bench suites use.
At opt3 `Collect2qBlocks`/`ConsolidateBlocks` resynthesises every repeated
interaction between a pair into a single two-qubit unitary, collapsing an
L-layer circuit towards its bare interaction graph and destroying the depth the
lookahead mechanisms exist to exploit -- measured on a d=4 draft, 2048 CZ came
back as 858 CX.  opt1 translates to the native set and optimises single-qubit
runs without touching two-qubit structure, so CX count is exactly L*n*d/2.

Circuits land in code/circuits_regcz/ -- deliberately NOT in the shared
~/Documents/telesabre/circuits/ tree, which holds MQT Bench v1.1.0 downloads
that several projects compare published numbers against.

Output: code/circuits_regcz/regcz_d{d}_s{seed}_{n}.qasm  + manifest.json
"""

import json, os, sys, time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile

OUT_DIR = os.path.join(_HERE, "circuits_regcz")

N_QUBITS  = 64
DEGREES   = [2, 3, 4, 6, 8]
INSTANCES = 3            # interaction-graph seeds per degree
TARGET_CX = 2000         # ~= the 64q AE (1962) and QFT (1966) circuits
BASIS     = ["rz", "sx", "x", "cx"]
OPT_LEVEL = 1            # see module docstring: opt3 consolidates the layers away


def layers_for(degree: int, n: int = N_QUBITS, target: int = TARGET_CX) -> int:
    """Layer count holding total CZ count nearest to `target`."""
    return max(1, round(target / (n * degree // 2)))


def build(degree: int, seed: int, n: int = N_QUBITS) -> QuantumCircuit:
    """L ansatz layers over one fixed random d-regular interaction graph."""
    g = nx.random_regular_graph(degree, n, seed=seed)
    edges = sorted(g.edges())
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers_for(degree, n)):
        for q in range(n):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
        for u, v in edges:
            qc.cz(u, v)
    return qc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for d in DEGREES:
        for s in range(INSTANCES):
            t0 = time.time()
            g = nx.random_regular_graph(d, N_QUBITS, seed=s)
            qc = build(d, s)
            tqc = transpile(qc, basis_gates=BASIS, optimization_level=OPT_LEVEL,
                            seed_transpiler=s)
            n_cx = sum(1 for inst in tqc.data if inst.operation.name == "cx")
            name = f"regcz_d{d}_s{s}_{N_QUBITS}"
            path = os.path.join(OUT_DIR, f"{name}.qasm")
            from qiskit import qasm2
            qasm2.dump(tqc, path)
            rec = dict(name=name, degree=d, seed=s, qubits=N_QUBITS,
                       layers=layers_for(d), interaction_edges=g.number_of_edges(),
                       cx=n_cx, depth=tqc.depth(),
                       path=os.path.relpath(path, _HERE))
            manifest.append(rec)
            print(f"{name:<20} L={rec['layers']:>2}  edges={rec['interaction_edges']:>4}"
                  f"  CX={n_cx:>5}  depth={rec['depth']:>5}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"),
                                 construction="L layers of (1q rotations, then "
                                              "CZ over every edge of one fixed "
                                              "random d-regular graph)",
                                 basis=BASIS, opt_level=OPT_LEVEL,
                                 target_cx=TARGET_CX),
                       circuits=manifest), f, indent=2)
    print(f"\nwrote {len(manifest)} circuits to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
