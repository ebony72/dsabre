r"""
gen_80q_suite.py — generate the 80-qubit benchmark suite (2026-07-28).

Why an 80-qubit suite exists.  Every architecture in the paper up to now has a
core graph of diameter 2 or 3 -- including the 360-qubit one, which scaled core
*size* (4x4 -> 9x9) while leaving the core graph a 2x3 grid.  Mean inter-core
path length is therefore 1.33-1.67 hops everywhere, so roughly half of all
teleports are single-hop and the hop-gain term, which rewards intermediate hops
along core_path(c1,c2), has no intermediate hop to reward.  That is why the
"its value grows with path length" prediction was never confirmed at scale: the
path length never grew.  80 logical qubits on a 2x5 H-grid gives 10 cores of
the same 4x4 shape used at 64q, diameter 5 and mean path 2.33, so the core
graph is the only thing that changes.

Provenance.  These are generated with the installed MQT Bench v2.2.2
algorithm-level constructors plus a plain qiskit basis transpile, exactly as
gen_new_64q_circuits.py does for qaoa_64 and multiplier_64.  No published
number exists at 80 qubits, so there is nothing for a v1.1.0/v2.2.2 mismatch to
contradict -- with one exception: v2.2.2's `qnn` is a ZFeatureMap (a product
state, leaving a linear CX chain) where v1.1.0's is a ZZFeatureMap that
entangles every pair.  The rest of the suite's qnn family is the deep ZZ form,
so qnn_80 is built with gen_deep_qnn.py's construction rather than v2.2.2's.

The script refuses to overwrite an existing file.

Output: ~/Documents/telesabre/circuits/qasm_80/{name}_nativegates_ibm_qiskit_opt3_80.qasm
"""

import os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import qasm2, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
from mqt.bench import BenchmarkLevel, get_benchmark

N = 80
BASIS = ["rz", "sx", "x", "cx"]
OUT_DIR = os.path.expanduser(f"~/Documents/telesabre/circuits/qasm_{N}")

# suite name -> MQT Bench v2.2.2 benchmark name ("random" is "randomcircuit"
# in v2).  qnn is handled separately, see the module docstring.
FAMILIES = {
    "ae": "ae", "ghz": "ghz", "graphstate": "graphstate", "qft": "qft",
    "random": "randomcircuit", "qpeexact": "qpeexact", "qaoa": "qaoa",
    "multiplier": "multiplier",
}

HEADER_MQT = (
    "// Generated for the 80q suite with MQT Bench v2.2.2\n"
    "// (v1-format nativegates_ibm_qiskit_opt3: algorithm-level circuit +\n"
    "//  qiskit transpile to basis rz/sx/x/cx, optimization_level=3,\n"
    "//  seed_transpiler=0).  Measurements and barriers stripped.\n"
)
HEADER_QNN = (
    "// Deep QNN (ZZ feature map) at 80 qubits, dsabre/code/gen_80q_suite.py\n"
    "// Construction: ZZFeatureMap(80, reps=2) o RealAmplitudes(80, reps=1),\n"
    "// parameters bound from numpy default_rng(0), then\n"
    "// transpile(basis=[rz,sx,x,cx], optimization_level=3, seed_transpiler=0).\n"
    "// Matches the v1.1.0 qnn form used at 25/64/100/200 qubits; v2.2.2's qnn\n"
    "// is a ZFeatureMap and is NOT equivalent.\n"
)


def strip(qc):
    qc = qc.remove_final_measurements(inplace=False)
    return PassManager([RemoveBarriers()]).run(qc)


def build_mqt(bench_name):
    alg = get_benchmark(bench_name, BenchmarkLevel.ALG, N)
    return strip(transpile(alg, basis_gates=BASIS, optimization_level=3,
                           seed_transpiler=0))


def build_qnn():
    from gen_deep_qnn import build          # same construction, same seed
    return build(N)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for suite_name in list(FAMILIES) + ["qnn"]:
        path = os.path.join(
            OUT_DIR, f"{suite_name}_nativegates_ibm_qiskit_opt3_{N}.qasm")
        if os.path.exists(path):
            print(f"{suite_name:<12} exists, refusing to overwrite", flush=True)
            continue
        t0 = time.time()
        try:
            if suite_name == "qnn":
                qc, header = build_qnn(), HEADER_QNN
            else:
                qc, header = build_mqt(FAMILIES[suite_name]), HEADER_MQT
        except Exception as e:                # one family failing must not
            print(f"{suite_name:<12} FAILED: {type(e).__name__}: {e}",
                  flush=True)                 # cost the rest of the suite
            continue
        bad = set(qc.count_ops()) - set(BASIS)
        if bad:
            print(f"{suite_name:<12} FAILED: unexpected ops {bad}", flush=True)
            continue
        with open(path, "w") as f:
            f.write(header + qasm2.dumps(qc) + "\n")
        print(f"{suite_name:<12} q={qc.num_qubits:>3}  "
              f"CX={qc.count_ops().get('cx', 0):>6}  depth={qc.depth():>6}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\n→ {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
