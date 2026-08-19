"""probe_local_ext_order.py — why the lexicographic closure and the global
topological scan can order E_c differently.

Stops at the first call whose ordered E_c differs and dumps both lists next to
each gate's position in `wdag.topological_op_nodes()`.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_b_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BFSExt
from dsabre_local_bfs import dSABRE_LocalKahnLex
from layout import sabre_locked_boundary_layout, run_sabre_passes
from circuit_paths import circuits_path


class Stop(Exception):
    pass


class OrderProbe(dSABRE_BFSExt):
    le_calls = 0
    le_visits = 0
    front_skip = dSABRE_LocalKahnLex.front_skip
    _qkey = dSABRE_LocalKahnLex._qkey

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        ext = dSABRE_BFSExt._get_local_extended(self, wdag, front, core_ids, l2p)
        lex = dSABRE_LocalKahnLex._get_local_extended(self, wdag, front, core_ids, l2p)

        def k(lst):
            return [(g._node_id, d) for g, d in lst]

        for c in core_ids:
            if k(ext[c]) != k(lex[c]):
                order = {n._node_id: i for i, n in
                         enumerate(wdag.topological_op_nodes())}
                idx = {}
                for q in l2p:
                    idx[q] = wdag.find_bit(q).index
                print(f"core {c}: scan {len(ext[c])} gates, lex {len(lex[c])}, "
                      f"L={self.config.lookahead_size}", flush=True)
                print(f"{'':<4} {'scan: nid':>10} {'topo#':>6} {'qargs':>12} "
                      f"{'dep':>4}   |  {'lex: nid':>10} {'topo#':>6} "
                      f"{'qargs':>12} {'dep':>4}", flush=True)
                for i in range(max(len(ext[c]), len(lex[c]))):
                    a = ext[c][i] if i < len(ext[c]) else None
                    b = lex[c][i] if i < len(lex[c]) else None

                    def show(x):
                        if x is None:
                            return f"{'-':>10} {'-':>6} {'-':>12} {'-':>4}"
                        g, d = x
                        qa = tuple(idx[q] for q in g.qargs)
                        return (f"{g._node_id:>10} {order.get(g._node_id,-1):>6} "
                                f"{str(qa):>12} {d:>4}")
                    print(f"{i:<4} {show(a)}   |  {show(b)}", flush=True)
                # what sits between them in the global order
                span = [g._node_id for g, _ in ext[c]] + [g._node_id for g, _ in lex[c]]
                lo, hi = min(order[s] for s in span), max(order[s] for s in span)
                print(f"\nglobal order positions {lo}..{hi}:", flush=True)
                for n in wdag.topological_op_nodes():
                    p = order[n._node_id]
                    if lo <= p <= hi:
                        qa = tuple(idx[q] for q in n.qargs)
                        tag = []
                        if n._node_id in {g._node_id for g, _ in ext[c]}:
                            tag.append("scan")
                        if n._node_id in {g._node_id for g, _ in lex[c]}:
                            tag.append("lex")
                        print(f"  {p:>5} nid={n._node_id:<6} {n.name:<6} {str(qa):<12} "
                              f"{','.join(tag)}", flush=True)
                raise Stop
        return ext


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "graphstate"
    d = circuits_path("qasm_25")
    path = os.path.join(d, f"{name}_nativegates_ibm_qiskit_opt3_25.qasm")
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    dag = circuit_to_dag(qc)
    rev = circuit_to_dag(qc.reverse_ops())
    arch = build_b_grid_architecture(r=2, s=2, m=4)
    layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
    r = OrderProbe(arch, HardwareConfig())
    try:
        run_sabre_passes(r, dag, rev, layouts[0])
        print("no ordering mismatch found", flush=True)
    except Stop:
        pass


if __name__ == "__main__":
    main()
