"""
gen_new_64q_circuits.py — generate qaoa_64 and multiplier_64 for the extended
64q suite (2026-07-20).

The original suite circuits are MQT Bench v1.1.0 downloads
(nativegates_ibm_qiskit_opt3 = basis {rz, sx, x, cx}, qiskit opt level 3,
no coupling map).  qpeexact_64 already exists in the circuits directory from
the same source.  This script generates the two missing circuits using the
installed MQT Bench v2.2.2 *algorithm-level* constructors (v2's own
native-gates pass manager is incompatible with the installed qiskit, and
the v1 semantics are a plain basis transpile anyway):

  - 'qaoa'       at 64 qubits (v2 defaults: p=2 layers, seed=10, dense
    random max-cut instance -- the same construction that produced the
    36q suite's qaoa_36 under v1); parameters bound by v2's generator
  - 'multiplier' at 64 qubits (16-bit operands, qiskit MultiplierGate)

Transpile: qiskit.transpile(basis_gates=[rz,sx,x,cx], optimization_level=3,
seed_transpiler=0).  Output: OPENQASM 2.0 files named
  {name}_nativegates_ibm_qiskit_opt3_64.qasm
in ~/Documents/telesabre/circuits/qasm_64/.
"""

import os

from qiskit import qasm2, transpile
from mqt.bench import BenchmarkLevel, get_benchmark

OUT_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_64")

HEADER = (
    "// Generated for the extended 64q suite with MQT Bench v2.2.2\n"
    "// (v1-format nativegates_ibm_qiskit_opt3: algorithm-level circuit +\n"
    "//  qiskit transpile to basis rz/sx/x/cx, optimization_level=3,\n"
    "//  seed_transpiler=0)\n"
)

for name in ("qaoa", "multiplier"):
    alg = get_benchmark(name, BenchmarkLevel.ALG, 64)
    qc = transpile(
        alg,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=3,
        seed_transpiler=0,
    )
    # suite convention (cf. ae_64): no measurements, no barriers -- the
    # TeleSABRE parser rejects them and every router strips them anyway
    qc = qc.remove_final_measurements(inplace=False)
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers
    qc = PassManager([RemoveBarriers()]).run(qc)
    ops = qc.count_ops()
    bad = set(ops) - {"rz", "sx", "x", "cx"}
    assert not bad, f"{name}: unexpected ops {bad}"
    path = os.path.join(OUT_DIR, f"{name}_nativegates_ibm_qiskit_opt3_64.qasm")
    with open(path, "w") as f:
        f.write(HEADER + qasm2.dumps(qc) + "\n")
    print(
        f"{name}: {qc.num_qubits} qubits, {ops.get('cx', 0)} cx -> {path}",
        flush=True,
    )
