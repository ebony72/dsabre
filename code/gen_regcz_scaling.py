r"""
gen_regcz_scaling.py — the regular-graph CZ family at scalability sizes.

Same shape as gen_regular_cz.py (a fixed d-regular interaction graph, L layers
of 1q rotations then a CZ across every edge), generated at 100/200/360 logical
qubits so it can run beside the banded QFTs on the scalability architectures.

The interaction graph is a **circulant**: qubit i is joined to i +/- 1 .. i +/- d/2
around a ring, so it is d-regular *and* bounded-range.  That is the property a
scaling series needs.  With a random d-regular graph the chance a partner shares
a core is (q_per_core - 1)/(n - 1), which decays with n, so the inherently-
inter-core gate fraction would climb 84% -> 95% along the series by construction
and the series would measure the graph's non-locality rather than the machine.
A circulant of range d/2 puts every partner within d/2 positions, so once
qubits are laid out in order the inter-core fraction is set by core size alone
and stays flat as n grows -- the same property that makes the banded QFTs a
usable series, but with degree under explicit control.

Degree and layer count.  L is chosen so the CX count tracks the banded QFTs
(3420 / 7220 / 13300 CX), making the two families comparable in gate volume:
CX = L*n*d/2, so L*d = 68 gives CX = 34n.  Default d=3, L=23.

Degree must be even for a circulant (offsets come in +/- pairs); d=4 gives
range 2.  The random-graph variant of this family is kept in gen_regular_cz.py,
where it is the right choice: the ablation sweeps degree at fixed n, so its
non-locality is a constant, not a confound.

Output: code/circuits_regcz_scaling/regcz_d{d}_n{n}.qasm  + manifest.json
"""

import json, os, sys, time, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile

from gen_regular_cz import BASIS, OPT_LEVEL


def circulant_edges(n: int, degree: int):
    """Ring circulant: i joined to i +/- 1 .. i +/- degree/2, bounded range."""
    if degree % 2:
        raise SystemExit(f"circulant degree must be even, got {degree}")
    return sorted({tuple(sorted((i, (i + off) % n)))
                   for i in range(n) for off in range(1, degree // 2 + 1)})


def matching_order(n: int, edges):
    """Group edges into parallel matchings (a proper edge colouring).

    Emitting a circulant's edges in sorted order builds a chain -- (0,1),(0,2),
    (1,2),(1,3)... each consecutive pair shares a qubit -- so the DAG critical
    path runs the length of the ring and the front layer never holds more than
    a couple of gates.  That is a serialisation artefact of the gate ordering,
    not a property of the ansatz.  Colouring the line graph recovers the
    parallelism a hardware-efficient layer actually has: degree-4 circulant ->
    4 or 5 matchings per layer.
    """
    import networkx as nx
    g = nx.Graph(); g.add_edges_from(edges)
    colour = nx.greedy_color(nx.line_graph(g), strategy="largest_first")
    classes = {}
    for e, c in colour.items():
        classes.setdefault(c, []).append(tuple(sorted(e)))
    return [e for c in sorted(classes) for e in sorted(classes[c])]


def build_circulant(degree: int, seed: int, n: int, layers: int):
    """`layers` ansatz layers over one circulant interaction graph."""
    edges = circulant_edges(n, degree)
    ordered = matching_order(n, edges)
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
        for u, v in ordered:
            qc.cz(u, v)
    return qc, edges

OUT_DIR = os.path.join(_HERE, "circuits_regcz_scaling")
SIZES   = [100, 200, 360]
DEGREE  = 4           # circulant offsets +/-1, +/-2
LAYERS  = 17          # L*d = 68 -> CX = 34n, tracking the banded QFTs
SEED    = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degree", type=int, default=DEGREE)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for n in a.sizes:
        path = os.path.join(OUT_DIR, f"regcz_circ_d{a.degree}_n{n}.qasm")
        if os.path.exists(path):
            print(f"n={n:<4} exists, refusing to overwrite", flush=True)
            continue
        t0 = time.time()
        qc, edges = build_circulant(a.degree, SEED, n, a.layers)
        tqc = transpile(qc, basis_gates=BASIS, optimization_level=OPT_LEVEL,
                        seed_transpiler=SEED)
        n_cx = sum(1 for i in tqc.data if i.operation.name == "cx")
        qasm2.dump(tqc, path)
        rec = dict(name=os.path.basename(path)[:-5], qubits=n, degree=a.degree,
                   layers=a.layers, interaction_edges=len(edges),
                   graph="circulant", range=a.degree // 2,
                   cx=n_cx, depth=tqc.depth(),
                   path=os.path.relpath(path, _HERE))
        manifest.append(rec)
        print(f"n={n:<4} d={a.degree} L={a.layers}  edges={rec['interaction_edges']:>5}"
              f"  CX={n_cx:>6}  depth={rec['depth']:>5}  ({time.time()-t0:.0f}s)",
              flush=True)

    mpath = os.path.join(OUT_DIR, "manifest.json")
    prev = json.load(open(mpath))["circuits"] if os.path.exists(mpath) else []
    names = {r["name"] for r in manifest}
    with open(mpath, "w") as f:
        json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"),
                                 construction="L layers of (1q rotations, then "
                                              "CZ over every edge of a ring "
                                              "circulant of degree d)",
                                 basis=BASIS, opt_level=OPT_LEVEL),
                       circuits=[r for r in prev if r["name"] not in names]
                                 + manifest), f, indent=2)
    print(f"\n-> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
