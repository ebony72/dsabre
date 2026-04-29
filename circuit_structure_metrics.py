"""
circuit_structure_metrics.py
=============================
Measures two structural properties of a quantum circuit:

  1. Commutativity Intensity (CI)
  2. Burst Communication Intensity (BI)

Both are computed from the EigenClusterHypergraph.

Usage
-----
from circuit_structure_metrics import profile_circuit
from eigen_cluster_hypergraph import Gate
profile_circuit(circuit, num_qubits, layout, name="my_circuit")

Or from Qiskit:
from circuit_structure_metrics import profile_from_qiskit
profile_from_qiskit(qc, layout_dict)
"""

import math
from typing import Dict, List
from eigen_cluster_hypergraph import Gate, EigenClusterHypergraph


def _build_hg(circuit, num_qubits):
    hg = EigenClusterHypergraph()
    hg.build(circuit, num_qubits=num_qubits)
    return hg


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Commutativity Intensity
# ═══════════════════════════════════════════════════════════════════════════════

def commutativity_intensity(hg: EigenClusterHypergraph) -> Dict:
    """
    SCR  — SN Compression Ratio: 1 - SNs/gates. Range [0,1].
           0 = no commutativity; 1 = all gates merge into one SN per wire.
    CPR  — Commuting-pair Ratio: fraction of front+extended gate pairs that
           commute globally (via commutes_in_hg). Sampled for tractability.
    """
    stats   = hg.stats()
    n_gates = stats["total_gates"]
    n_sn    = stats["supernodes"]
    if n_gates == 0:
        return {"error": "empty circuit"}

    scr    = 1.0 - n_sn / n_gates
    max_sn = max((len(sn.gates) for sn in hg.nodes), default=0)

    qubit_gates: Dict[int,int] = {}
    qubit_sns:   Dict[int,int] = {}
    for sn in hg.nodes:
        q = sn.qubit
        qubit_sns[q]   = qubit_sns.get(q, 0) + 1
        qubit_gates[q] = qubit_gates.get(q, 0) + len(sn.gates)
    wire_comp = {q: round(1.0 - qubit_sns[q] / qubit_gates[q], 3)
                 for q in qubit_gates if qubit_gates[q] > 0}

    sample  = hg.get_front_layer() + hg.get_extended_layer(depth=2)
    sample  = list({g.gate_id: g for g in sample}.values())
    n_pairs = len(sample) * (len(sample) - 1) // 2
    cpr = (sum(1 for i, g1 in enumerate(sample)
               for g2 in sample[i+1:] if hg.commutes_in_hg(g1, g2))
           / n_pairs) if n_pairs > 0 else 0.0

    return {
        "sn_compression_ratio":       round(scr,    4),
        "avg_sn_size":                round(stats["avg_sn_size"], 3),
        "max_sn_size":                max_sn,
        "commuting_pair_ratio":       round(cpr,    4),
        "wire_compression_per_qubit": wire_comp,
        "n_gates":  n_gates,
        "n_snodes": n_sn,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Burst Communication Intensity
# ═══════════════════════════════════════════════════════════════════════════════

def burst_intensity(hg: EigenClusterHypergraph, layout: Dict[int,int]) -> Dict:
    """
    RGF  — Remote Gate Fraction: fraction of 2q gates crossing a core boundary.
    BC   — Burst Coverage: fraction of remote gates in a burst block of size ≥ 2.
    ABS  — Average Burst Size of non-trivial blocks (≥ 2 gates).
    MBS  — Max Burst Size.
    BE   — Burst Efficiency: total burst gate slots / num remote gates.
           BE > 1 means multi-core overlap; BE = 1 is ideal.
    TSE  — Teleport Savings Estimate: (sum of (|block|-1) for non-trivial blocks)
           / num_remote_gates.  TSE=0.5 → 50% fewer EPR pairs expected.
    """
    all_gates   = list(hg._gates.values())
    two_q_gates = [g for g in all_gates if len(g.qubits) >= 2]
    if not two_q_gates:
        return {"error": "no 2-qubit gates"}

    remote_gates = [g for g in two_q_gates if hg.is_remote(g, layout)]
    n_remote = len(remote_gates)
    n_2q     = len(two_q_gates)
    rgf      = n_remote / n_2q if n_2q else 0.0

    if n_remote == 0:
        return {"remote_gate_fraction": 0.0, "burst_coverage": 1.0,
                "avg_burst_size": 0.0, "max_burst_size": 0,
                "burst_efficiency": 0.0, "teleport_savings_estimate": 0.0,
                "n_remote_gates": 0, "n_burst_blocks": 0,
                "n_non_trivial_burst_blocks": 0}

    burst_blocks: List[List[Gate]] = []
    for q in range(hg.num_qubits):
        for core, gates in hg.get_all_burst_blocks(q, layout).items():
            if gates:
                burst_blocks.append(gates)

    remote_ids   = {g.gate_id for g in remote_gates}
    non_trivial  = [b for b in burst_blocks if len(b) >= 2]
    covered_ids  = {g.gate_id for b in non_trivial for g in b} & remote_ids
    bc           = len(covered_ids) / n_remote
    abs_         = (sum(len(b) for b in non_trivial) / len(non_trivial)
                    if non_trivial else 0.0)
    mbs          = max((len(b) for b in burst_blocks), default=0)
    be           = sum(len(b) for b in burst_blocks) / n_remote
    savings      = sum(len(b) - 1 for b in non_trivial)
    tse          = savings / n_remote

    return {
        "remote_gate_fraction":       round(rgf,  4),
        "burst_coverage":             round(bc,   4),
        "avg_burst_size":             round(abs_, 3),
        "max_burst_size":             mbs,
        "burst_efficiency":           round(be,   3),
        "teleport_savings_estimate":  round(tse,  4),
        "n_remote_gates":             n_remote,
        "n_burst_blocks":             len(burst_blocks),
        "n_non_trivial_burst_blocks": len(non_trivial),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Combined profiler
# ═══════════════════════════════════════════════════════════════════════════════

def profile_circuit(circuit: List[Gate], num_qubits: int,
                    layout: Dict[int, int],
                    circuit_name: str = "circuit") -> Dict:
    hg = _build_hg(circuit, num_qubits)
    ci = commutativity_intensity(hg)
    bi = burst_intensity(hg, layout)
    _print_profile(circuit_name, ci, bi)
    scr = ci.get("sn_compression_ratio", 0)
    bc_ = bi.get("burst_coverage", 0)
    tse = bi.get("teleport_savings_estimate", 0)
    score = (scr + bc_ + tse) / 3
    verdict = ("★★★ HIGH — BurstDSABRE expected to outperform clearly" if score > 0.5 else
               "★★  MEDIUM — gains circuit-dependent" if score > 0.3 else
               "★   LOW — little commutativity/burst structure; dSABRE similar")
    return {"commutativity": ci, "burst": bi,
            "combined_score": round(score, 3), "verdict": verdict}


def profile_from_qiskit(qc, layout: Dict, name: str = None) -> Dict:
    """
    Build profile directly from a Qiskit QuantumCircuit + physical layout dict.
    layout: {Qubit -> physical_qubit_index}
    """
    from qiskit.converters import circuit_to_dag
    dag = circuit_to_dag(qc)
    circuit = []
    for gid, node in enumerate(dag.topological_op_nodes()):
        qubits = [dag.find_bit(q).index for q in node.qargs]
        circuit.append(Gate(gid, node.op.name.lower(), qubits))
    int_layout = {dag.find_bit(q).index: v for q, v in layout.items()
                  if hasattr(q, "_register")}
    return profile_circuit(circuit, qc.num_qubits, int_layout,
                           name or qc.name or "circuit")


def _print_profile(name, ci, bi):
    SEP = "─" * 62
    print(f"\n{'═'*62}")
    print(f"  Circuit Profile: {name}")
    print(f"{'═'*62}")

    print(f"\n  COMMUTATIVITY INTENSITY")
    print(f"  {SEP}")
    print(f"  Gates / SuperNodes           : {ci['n_gates']} / {ci['n_snodes']}")
    print(f"  SN Compression Ratio  (SCR)  : {ci['sn_compression_ratio']:.4f}  "
          f"{'★ high' if ci['sn_compression_ratio'] > 0.3 else '○ low'}")
    print(f"  Avg / Max SN size            : {ci['avg_sn_size']:.2f} / {ci['max_sn_size']}")
    print(f"  Commuting-pair Ratio  (CPR)  : {ci['commuting_pair_ratio']:.4f}  "
          f"{'★ high' if ci['commuting_pair_ratio'] > 0.4 else '○ low'}")
    print(f"  Wire compression per qubit   :")
    for q, v in sorted(ci.get("wire_compression_per_qubit", {}).items()):
        bar = "█" * int(v * 20)
        print(f"    q{q:<3}: {v:.3f}  |{bar}")

    print(f"\n  BURST COMMUNICATION INTENSITY")
    print(f"  {SEP}")
    print(f"  Remote gate fraction  (RGF)  : {bi['remote_gate_fraction']:.4f}  "
          f"{'★ high' if bi['remote_gate_fraction'] > 0.3 else '○ low'}")
    print(f"  Burst coverage        (BC)   : {bi['burst_coverage']:.4f}  "
          f"{'★ high' if bi['burst_coverage'] > 0.5 else '○ low'}")
    print(f"  Avg / Max burst size         : {bi['avg_burst_size']:.2f} / {bi['max_burst_size']}")
    print(f"  Burst efficiency      (BE)   : {bi['burst_efficiency']:.3f}")
    print(f"  Teleport savings est. (TSE)  : {bi['teleport_savings_estimate']:.4f}  "
          f"({bi['teleport_savings_estimate']*100:.1f}% fewer EPR pairs)")
    print(f"  Remote / burst blocks        : {bi['n_remote_gates']} / "
          f"{bi['n_burst_blocks']}  "
          f"(non-trivial: {bi['n_non_trivial_burst_blocks']})")
