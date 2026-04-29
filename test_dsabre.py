"""
Tests for dsabre: architecture, hypergraph, and routing.

Run with:  pytest test_dsabre.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import networkx as nx

from architecture import (
    DistributedArchitecture,
    build_basic_grid_architecture,
    build_b_grid_architecture,
    build_h_grid_architecture,
)
from config import HardwareConfig
from eigen_cluster_hypergraph import Gate, SuperNode, EigenClusterHypergraph


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_hg(gates, num_qubits):
    hg = EigenClusterHypergraph()
    hg.build(gates, num_qubits)
    return hg


def _expected_ready(hg):
    """Brute-force expected ready set: all non-executed SNs whose pred is done."""
    return {
        sn for sn in hg.nodes
        if not sn.is_fully_executed
        and (sn.pred is None or sn.pred.is_fully_executed)
    }


# ─── Architecture ─────────────────────────────────────────────────────────────

class TestArchitecture:

    def test_qubit_count_basic_grid(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        assert len(arch.data_qubits) == 2 * 2 * 3 * 3
        assert arch.num_cores == 4

    def test_qubit_count_b_grid(self):
        arch = build_b_grid_architecture(2, 3, 4)
        assert len(arch.data_qubits) == 2 * 3 * 4 * 4
        assert arch.num_cores == 6

    def test_qubit_count_h_grid(self):
        arch = build_h_grid_architecture(2, 3, 4)
        assert len(arch.data_qubits) == 2 * 3 * 4 * 4

    def test_core_of_maps_every_qubit(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for p in arch.data_qubits:
            c = arch.core_of(p)
            assert 0 <= c < arch.num_cores
            assert p in arch.core_qubits(c)

    def test_core_graph_2x2(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        # 2×2 grid of cores: 4 nodes, 4 edges (each core adjacent to 2 others)
        assert arch.core_graph.number_of_nodes() == 4
        assert arch.core_graph.number_of_edges() == 4

    def test_inter_links_symmetry(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for c0 in range(arch.num_cores):
            for c1 in range(arch.num_cores):
                fwd = arch.inter_links_between(c0, c1)
                rev = arch.inter_links_between(c1, c0)
                assert len(fwd) == len(rev)
                assert sorted(fwd) == sorted((v, u) for u, v in rev)

    def test_comm_qubits_registered(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for p in arch.comm_qubits:
            c = arch.core_of(p)
            assert p in arch._core_comm_ports[c]

    # phys_dist correctness

    def test_phys_dist_self_zero(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for p in arch.data_qubits:
            assert arch.phys_dist[p][p] == 0

    def test_phys_dist_adjacent_equals_edge_weight(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for u, v, data in arch.Gr.edges(data=True):
            w = data.get("weight", 1)
            assert arch.phys_dist[u][v] == w, f"Edge ({u},{v}) weight mismatch"

    def test_phys_dist_intra_matches_intra_dist(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for core_id in range(arch.num_cores):
            for u in arch.core_qubits(core_id):
                for v in arch.core_qubits(core_id):
                    assert arch.phys_dist[u][v] == arch.intra_dist[core_id][u][v]

    def test_phys_dist_cross_core_at_least_inter_weight(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for u, v in arch.inter_core_links:
            # The inter-link itself has weight 10; distance must be >= 10
            assert arch.phys_dist[u][v] >= 10

    def test_phys_dist_symmetric(self):
        arch = build_basic_grid_architecture(2, 1, 3)  # simple 2-core line
        for u in arch.data_qubits:
            for v in arch.data_qubits:
                assert arch.phys_dist[u][v] == arch.phys_dist[v][u]

    def test_core_dist_symmetric(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for c0 in range(arch.num_cores):
            for c1 in range(arch.num_cores):
                assert arch.core_dist[c0][c1] == arch.core_dist[c1][c0]

    def test_single_core_no_inter_links(self):
        arch = build_basic_grid_architecture(1, 1, 4)
        assert arch.num_cores == 1
        assert len(arch.inter_core_links) == 0
        assert len(arch.comm_qubits) == 0

    def test_all_qubits_in_full_graph(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        for p in arch.data_qubits:
            assert p in arch.Gr.nodes


# ─── Gate ─────────────────────────────────────────────────────────────────────

class TestGate:

    # commutativity

    def test_commutes_disjoint_qubits(self):
        assert Gate(0, "cx", [0, 1]).commutes_with(Gate(1, "cx", [2, 3]))

    def test_cx_same_control_qubit_commute(self):
        # Both have {"0","1"} on q0 → commute
        assert Gate(0, "cx", [0, 1]).commutes_with(Gate(1, "cx", [0, 2]))

    def test_cx_same_target_qubit_commute(self):
        # Both have {"x+","x-"} on q2 → commute
        assert Gate(0, "cx", [0, 2]).commutes_with(Gate(1, "cx", [1, 2]))

    def test_cx_ctrl_vs_target_no_commute(self):
        # q1: target {"x+","x-"} in g0 vs control {"0","1"} in g1
        assert not Gate(0, "cx", [0, 1]).commutes_with(Gate(1, "cx", [1, 2]))

    def test_rz_rz_same_qubit_commute(self):
        assert Gate(0, "rz", [0], [0.5]).commutes_with(Gate(1, "rz", [0], [1.0]))

    def test_rx_rz_no_commute(self):
        # {"x+","x-"} vs {"0","1"} on q0
        assert not Gate(0, "rx", [0], [0.5]).commutes_with(Gate(1, "rz", [0], [1.0]))

    def test_unknown_gate_different_ids_no_commute(self):
        g1 = Gate(0, "mygate", [0, 1])
        g2 = Gate(1, "mygate", [0, 1])
        assert not g1.commutes_with(g2)

    def test_unknown_gate_same_object_commutes_with_itself(self):
        # A gate trivially commutes with itself (shared qubits have identical dirs)
        g = Gate(0, "mygate", [0, 1])
        assert g.commutes_with(g)

    # eigen directions

    def test_eigen_dir_cx_control(self):
        assert Gate(0, "cx", [0, 1]).eigen_dir_on(0) == frozenset({"0", "1"})

    def test_eigen_dir_cx_target(self):
        assert Gate(0, "cx", [0, 1]).eigen_dir_on(1) == frozenset({"x+", "x-"})

    def test_eigen_dir_rz(self):
        assert Gate(0, "rz", [0]).eigen_dir_on(0) == frozenset({"0", "1"})

    def test_eigen_dir_swap_both_qubits(self):
        g = Gate(0, "swap", [0, 1])
        assert g.eigen_dir_on(0) == frozenset({"sw"})
        assert g.eigen_dir_on(1) == frozenset({"sw"})

    # name property

    def test_name_default_is_uppercase_type(self):
        assert Gate(0, "cx", [0, 1]).name == "CX"

    def test_name_override_by_label(self):
        assert Gate(0, "cx", [0, 1], label="my_cx").name == "my_cx"


# ─── SuperNode ────────────────────────────────────────────────────────────────

class TestSuperNode:

    def test_mark_executed_tracks_individually(self):
        g0, g1 = Gate(0, "rz", [0]), Gate(1, "rz", [0])
        sn = SuperNode(0, 0, frozenset({"0", "1"}), gates=[g0, g1])
        assert not sn.is_fully_executed
        sn.mark_executed(0)
        assert not sn.is_fully_executed
        sn.mark_executed(1)
        assert sn.is_fully_executed

    def test_pending_gates_decreases(self):
        g0, g1 = Gate(0, "rz", [0]), Gate(1, "rz", [0])
        sn = SuperNode(0, 0, frozenset({"0", "1"}), gates=[g0, g1])
        assert set(sn.pending_gates) == {g0, g1}
        sn.mark_executed(0)
        assert sn.pending_gates == [g1]

    def test_is_ready_no_pred(self):
        assert SuperNode(0, 0, frozenset({"0", "1"})).is_ready

    def test_is_ready_unfinished_pred(self):
        pred = SuperNode(0, 0, frozenset({"0", "1"}), gates=[Gate(0, "rz", [0])])
        succ = SuperNode(1, 0, frozenset({"0", "1"}), pred=pred)
        assert not succ.is_ready

    def test_is_ready_after_pred_done(self):
        g = Gate(0, "rz", [0])
        pred = SuperNode(0, 0, frozenset({"0", "1"}), gates=[g])
        succ = SuperNode(1, 0, frozenset({"0", "1"}), pred=pred)
        pred.mark_executed(0)
        assert succ.is_ready

    def test_succ_link_set_by_new_node(self):
        hg = EigenClusterHypergraph()
        sn0 = hg._new_node(0, frozenset({"0", "1"}), None)
        sn1 = hg._new_node(0, frozenset({"x+", "x-"}), sn0)
        assert sn0.succ is sn1
        assert sn1.pred is sn0


# ─── EigenClusterHypergraph ───────────────────────────────────────────────────

class TestEigenClusterHypergraph:

    # build / structure

    def test_single_cx_one_sn_per_qubit(self):
        hg = _make_hg([Gate(0, "cx", [0, 1])], 2)
        assert len(hg.nodes) == 2

    def test_commuting_cx_merge_on_control(self):
        # CX(0,1) and CX(0,2): same ctrl dir on q0 → merge into one SN on q0
        hg = _make_hg([Gate(0, "cx", [0, 1]), Gate(1, "cx", [0, 2])], 3)
        q0_snodes = [sn for sn in hg.nodes if sn.qubit == 0]
        assert len(q0_snodes) == 1
        assert len(q0_snodes[0].gates) == 2

    def test_noncommuting_creates_new_sn_on_qubit(self):
        # CX(0,1) then CX(1,2): q1 is target then control → different dirs → 2 SNs
        hg = _make_hg([Gate(0, "cx", [0, 1]), Gate(1, "cx", [1, 2])], 3)
        q1_snodes = [sn for sn in hg.nodes if sn.qubit == 1]
        assert len(q1_snodes) == 2

    def test_alternating_dirs_chain(self):
        # rx → rz → rx on q0: 3 separate SNs
        gates = [Gate(0, "rx", [0]), Gate(1, "rz", [0]), Gate(2, "rx", [0])]
        hg = _make_hg(gates, 1)
        assert len(hg.nodes) == 3
        # Check chain linkage
        snodes = sorted(hg.nodes, key=lambda sn: sn.node_id)
        assert snodes[0].pred is None
        assert snodes[1].pred is snodes[0]
        assert snodes[2].pred is snodes[1]

    def test_hyperedges_registered_for_2q_only(self):
        g0 = Gate(0, "cx", [0, 1])
        g1 = Gate(1, "rz", [0])
        hg = _make_hg([g0, g1], 2)
        assert 0 in hg.hyperedges
        assert 1 not in hg.hyperedges

    # front layer

    def test_front_layer_initial_single_gate(self):
        hg = _make_hg([Gate(0, "cx", [0, 1])], 2)
        fl = hg.get_front_layer()
        assert len(fl) == 1 and fl[0].gate_id == 0

    def test_front_layer_blocked_gate_absent(self):
        # CX(0,1) followed by RX(0): RX is in a new SN blocked by the CX SN on q0
        hg = _make_hg([Gate(0, "cx", [0, 1]), Gate(1, "rx", [0])], 2)
        fl_ids = {g.gate_id for g in hg.get_front_layer()}
        assert 0 in fl_ids
        assert 1 not in fl_ids

    def test_front_layer_after_removal_exposes_successor(self):
        hg = _make_hg([Gate(0, "cx", [0, 1]), Gate(1, "rx", [0])], 2)
        hg.remove_gate(0)
        fl_ids = {g.gate_id for g in hg.get_front_layer()}
        assert 1 in fl_ids

    def test_two_independent_gates_both_in_front(self):
        # CX(0,1) and CX(2,3): no shared qubits → both in front layer
        hg = _make_hg([Gate(0, "cx", [0, 1]), Gate(1, "cx", [2, 3])], 4)
        fl_ids = {g.gate_id for g in hg.get_front_layer()}
        assert fl_ids == {0, 1}

    # remove_gate

    def test_remove_gate_idempotent(self):
        hg = _make_hg([Gate(0, "cx", [0, 1])], 2)
        hg.remove_gate(0)
        hg.remove_gate(0)   # must not raise or corrupt
        assert hg.is_done()

    def test_remove_gate_unknown_raises_key_error(self):
        hg = _make_hg([Gate(0, "cx", [0, 1])], 2)
        with pytest.raises(KeyError):
            hg.remove_gate(999)

    def test_is_done_after_all_gates_removed(self):
        gates = [Gate(0, "cx", [0, 1]), Gate(1, "rz", [0])]
        hg = _make_hg(gates, 2)
        assert not hg.is_done()
        hg.remove_gate(0)
        hg.remove_gate(1)
        assert hg.is_done()

    def test_num_pending_decreases(self):
        # Three CX(0,1) gates commute → same SN, all pending
        gates = [Gate(i, "cx", [0, 1]) for i in range(3)]
        hg = _make_hg(gates, 2)
        assert hg.num_pending() == 3
        hg.remove_gate(0)
        assert hg.num_pending() == 2
        hg.remove_gate(1)
        hg.remove_gate(2)
        assert hg.num_pending() == 0

    # incremental ready-set correctness

    def test_ready_set_matches_brute_force_through_chain(self):
        """_ready_snodes must equal brute-force after every remove_gate."""
        # Alternating dirs → strict chain: each gate blocked by previous
        gates = [
            Gate(0, "rx", [0]),
            Gate(1, "rz", [0]),
            Gate(2, "rx", [0]),
        ]
        hg = _make_hg(gates, 1)
        assert hg._ready_snodes == _expected_ready(hg)
        hg.remove_gate(0)
        assert hg._ready_snodes == _expected_ready(hg)
        hg.remove_gate(1)
        assert hg._ready_snodes == _expected_ready(hg)
        hg.remove_gate(2)
        assert hg._ready_snodes == _expected_ready(hg)

    def test_ready_set_matches_brute_force_two_wire_circuit(self):
        """_ready_snodes must equal brute-force for a two-qubit circuit."""
        gates = [
            Gate(0, "cx", [0, 1]),   # q0: ctrl {"0","1"}, q1: tgt {"x+","x-"}
            Gate(1, "cx", [1, 2]),   # q1: ctrl {"0","1"} → new SN on q1
            Gate(2, "rz", [0]),      # q0: {"0","1"} same as CX ctrl → merges into SN0
        ]
        hg = _make_hg(gates, 3)

        def check():
            assert hg._ready_snodes == _expected_ready(hg), \
                f"ready_snodes={hg._ready_snodes} != expected={_expected_ready(hg)}"

        check()
        hg.remove_gate(0); check()
        hg.remove_gate(2); check()
        hg.remove_gate(1); check()

    def test_qubit_to_ready_sn_advances_on_removal(self):
        # q1 wire: CX(0,1) target SN → then CX(1,2) ctrl SN (different dirs)
        g0 = Gate(0, "cx", [0, 1])
        g1 = Gate(1, "cx", [1, 2])
        hg = _make_hg([g0, g1], 3)

        q1_head = hg._qubit_to_ready_sn[1]
        assert q1_head is not None
        assert g0 in q1_head.gates   # head SN holds g0 on q1

        hg.remove_gate(0)            # retires q1's head SN
        q1_next = hg._qubit_to_ready_sn[1]
        assert q1_next is not None
        assert g1 in q1_next.gates   # successor SN holds g1 on q1

    def test_ready_sn_for_helper(self):
        hg = _make_hg([Gate(0, "cx", [0, 1])], 2)
        sn0 = hg._ready_sn_for(0)
        sn1 = hg._ready_sn_for(1)
        assert sn0 is not None
        assert sn1 is not None
        hg.remove_gate(0)
        assert hg._ready_sn_for(0) is None
        assert hg._ready_sn_for(1) is None

    def test_initial_ready_snodes_all_have_no_pred(self):
        hg = _make_hg(
            [Gate(0, "cx", [0, 1]), Gate(1, "cx", [1, 2]), Gate(2, "rz", [2])],
            3,
        )
        for sn in hg._ready_snodes:
            assert sn.pred is None

    # get_tele_gain_block

    def test_tele_gain_no_gates_returns_zero(self):
        arch = build_basic_grid_architecture(2, 1, 3)
        hg = _make_hg([], 2)
        count = hg.get_tele_gain_block(0, 10, 0, {0: 0, 1: 10}, arch)
        assert count == 0

    def test_tele_gain_beneficial_move_counted(self):
        """
        2-core architecture (2 rows × 1 col × 3×3 qubits).
        q0 at phys 0 (corner of core 0), q1 at phys 13 (center of core 1).
        Teleporting q0 to phys 10 (comm port in core 1) reduces distance to q1.
        The pending CX gate should be counted as beneficial.
        """
        arch = build_basic_grid_architecture(2, 1, 3)
        # Inter-link: phys 7 (bottom-mid of core 0) → phys 10 (top-mid of core 1)
        links = arch.inter_links_between(0, 1)
        assert links, "Architecture must have at least one inter-core link"
        _, p_dst = links[0]   # comm port destination in core 1

        p_q0 = arch.core_qubits(0)[0]   # top-left corner of core 0
        p_q1 = arch.core_qubits(1)[4]   # center of core 1

        d_old = arch.phys_dist[p_q0][p_q1]
        d_new = arch.intra_dist[1][p_dst][p_q1]
        if d_new >= d_old:
            pytest.skip("Layout doesn't produce a gain for this architecture config")

        hg = _make_hg([Gate(0, "cx", [0, 1])], 2)
        count = hg.get_tele_gain_block(0, p_dst, p_q0, {0: p_q0, 1: p_q1}, arch)
        assert count == 1

    def test_tele_gain_multiple_pending_on_same_wire(self):
        """Multiple pending CX gates on the same wire all counted when beneficial."""
        arch = build_basic_grid_architecture(2, 1, 3)
        links = arch.inter_links_between(0, 1)
        assert links
        _, p_dst = links[0]

        p_q0 = arch.core_qubits(0)[0]
        p_q1 = arch.core_qubits(1)[4]
        d_old = arch.phys_dist[p_q0][p_q1]
        d_new = arch.intra_dist[1][p_dst][p_q1]
        if d_new >= d_old:
            pytest.skip("Layout doesn't produce a gain for this architecture config")

        # 3 CX gates on same wire → same SN → all pending
        gates = [Gate(i, "cx", [0, 1]) for i in range(3)]
        hg = _make_hg(gates, 2)
        count = hg.get_tele_gain_block(0, p_dst, p_q0, {0: p_q0, 1: p_q1}, arch)
        assert count == 3

    def test_tele_gain_harmful_move_returns_zero(self):
        """Moving q0 further from q1 should yield count=0."""
        arch = build_basic_grid_architecture(2, 1, 3)
        # Put q0 at the comm-port itself (phys 10), q1 at core-1 center (phys 13)
        # Moving q0 BACK to phys 0 (core 0) would be harmful
        p_q0_near = arch.core_qubits(1)[1]  # already in core 1
        p_q1 = arch.core_qubits(1)[4]
        # "new_phys" is somewhere in core 0 — further from q1
        p_far = arch.core_qubits(0)[0]

        d_old = arch.intra_dist[1][p_q0_near][p_q1]
        d_new = arch.phys_dist[p_far][p_q1]
        if d_new <= d_old:
            pytest.skip("Layout doesn't produce a harmful move for this config")

        hg = _make_hg([Gate(0, "cx", [0, 1])], 2)
        count = hg.get_tele_gain_block(0, p_far, p_q0_near, {0: p_q0_near, 1: p_q1}, arch)
        assert count == 0


# ─── Integration tests (Qiskit required) ──────────────────────────────────────

try:
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from router import General_dSABRE_Router
    from burst_router import BurstDSABRE
    from main import _topology_aware_core_assignment
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

skip_no_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")


def _layout_sequential(dag, arch):
    """Map dag.qubits[i] → arch.data_qubits[i]."""
    return {lq: arch.data_qubits[i] for i, lq in enumerate(dag.qubits)}


def _simple_arch():
    return build_basic_grid_architecture(2, 2, 3)   # 4 cores × 9 qubits = 36 total


@skip_no_qiskit
class TestGeneralDSABRE:

    def test_adjacent_qubits_no_overhead(self):
        """CX on already-adjacent physical qubits needs zero swaps/teles."""
        arch = build_basic_grid_architecture(1, 1, 4)   # single 4×4 core
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        # qubits 0=(0,0) and 1=(0,1) are adjacent
        layout = {dag.qubits[0]: 0, dag.qubits[1]: 1}
        metrics, _ = General_dSABRE_Router(arch, HardwareConfig()).route(dag, layout)
        assert not metrics["aborted"]
        assert metrics["ls"] == 0
        assert metrics["teles"] == 0

    def test_cross_core_gate_uses_teleport(self):
        arch = _simple_arch()
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        # Place qubits in different cores
        layout = {dag.qubits[0]: arch.core_qubits(0)[4],
                  dag.qubits[1]: arch.core_qubits(1)[4]}
        metrics, _ = General_dSABRE_Router(arch, HardwareConfig()).route(dag, layout)
        assert not metrics["aborted"]
        assert metrics["teles"] >= 1
        assert metrics["eprs"] == metrics["teles"]

    def test_single_qubit_gates_counted(self):
        arch = build_basic_grid_architecture(1, 1, 4)
        qc = QuantumCircuit(1)
        qc.h(0); qc.rz(0.5, 0); qc.x(0)
        dag = circuit_to_dag(qc)
        metrics, _ = General_dSABRE_Router(arch, HardwareConfig()).route(
            dag, {dag.qubits[0]: 0}
        )
        assert not metrics["aborted"]
        assert metrics["1q_gates"] == 3
        assert metrics["teles"] == 0

    def test_cost_formula_ls_plus_teles(self):
        """cost == ls * cost_local_swap + teles * cost_teleport."""
        arch = _simple_arch()
        config = HardwareConfig()
        qc = QuantumCircuit(4)
        qc.cx(0, 2); qc.cx(1, 3); qc.cx(0, 3)
        dag = circuit_to_dag(qc)
        metrics, _ = General_dSABRE_Router(arch, config).route(dag, _layout_sequential(dag, arch))
        assert not metrics["aborted"]
        expected = (metrics["ls"] * config.cost_local_swap
                    + metrics["teles"] * config.cost_teleport)
        assert abs(metrics["cost"] - expected) < 1e-9

    def test_final_layout_is_injective(self):
        """All physical positions in the final layout must be distinct."""
        arch = _simple_arch()
        qc = QuantumCircuit(6)
        for i in range(5):
            qc.cx(i, i + 1)
        dag = circuit_to_dag(qc)
        _, final = General_dSABRE_Router(arch, HardwareConfig()).route(
            dag, _layout_sequential(dag, arch)
        )
        phys = list(final.values())
        assert len(phys) == len(set(phys))

    def test_final_layout_phys_in_architecture(self):
        arch = _simple_arch()
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.cx(i, (i + 1) % 4)
        dag = circuit_to_dag(qc)
        _, final = General_dSABRE_Router(arch, HardwareConfig()).route(
            dag, _layout_sequential(dag, arch)
        )
        for p in final.values():
            assert p in arch.phys_dist, f"Physical qubit {p} not in architecture"

    def test_multi_gate_circuit_completes(self):
        arch = _simple_arch()
        qc = QuantumCircuit(8)
        for i in range(7):
            qc.cx(i, i + 1)
        qc.cx(0, 7)
        dag = circuit_to_dag(qc)
        metrics, _ = General_dSABRE_Router(arch, HardwareConfig()).route(
            dag, _layout_sequential(dag, arch)
        )
        assert not metrics["aborted"]


@skip_no_qiskit
class TestBurstDSABRE:

    def test_cross_core_gate_uses_teleport(self):
        arch = _simple_arch()
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        layout = {dag.qubits[0]: arch.core_qubits(0)[4],
                  dag.qubits[1]: arch.core_qubits(1)[4]}
        metrics, _ = BurstDSABRE(arch, HardwareConfig()).route(dag, layout)
        assert not metrics["aborted"]
        assert metrics["teles"] >= 1

    def test_eprs_equals_teles(self):
        arch = _simple_arch()
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.cx(i, (i + 1) % 4)
        dag = circuit_to_dag(qc)
        metrics, _ = BurstDSABRE(arch, HardwareConfig()).route(
            dag, _layout_sequential(dag, arch)
        )
        assert not metrics["aborted"]
        assert metrics["eprs"] == metrics["teles"]

    def test_cost_formula_ls_plus_teles(self):
        arch = _simple_arch()
        config = HardwareConfig()
        qc = QuantumCircuit(4)
        qc.cx(0, 2); qc.cx(1, 3); qc.cx(0, 3)
        dag = circuit_to_dag(qc)
        metrics, _ = BurstDSABRE(arch, config).route(dag, _layout_sequential(dag, arch))
        assert not metrics["aborted"]
        expected = (metrics["ls"] * config.cost_local_swap
                    + metrics["teles"] * config.cost_teleport)
        assert abs(metrics["cost"] - expected) < 1e-9

    def test_burst_saves_zero_no_teleport(self):
        """CX on adjacent same-core qubits: no teleport fires, burst_saves stays 0."""
        arch = build_basic_grid_architecture(1, 1, 4)   # single 4×4 core
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        # qubits 0=(0,0) and 1=(0,1) are adjacent — executes immediately
        layout = {dag.qubits[0]: 0, dag.qubits[1]: 1}
        metrics, _ = BurstDSABRE(arch, HardwareConfig()).route(dag, layout)
        assert not metrics["aborted"]
        assert metrics["teles"] == 0
        assert metrics["burst_saves"] == 0

    def test_burst_saves_is_nonneg_int(self):
        """burst_saves must always be a non-negative integer."""
        arch = _simple_arch()
        qc = QuantumCircuit(4)
        for _ in range(4):
            qc.cx(0, 2)
            qc.cx(1, 3)
        dag = circuit_to_dag(qc)
        metrics, _ = BurstDSABRE(arch, HardwareConfig(), weight_burst=4.0).route(
            dag, _layout_sequential(dag, arch)
        )
        assert not metrics["aborted"]
        assert isinstance(metrics["burst_saves"], int)
        assert metrics["burst_saves"] >= 0

    def test_burst_saves_positive_for_repeated_cross_core_gates(self):
        """
        5 CX(0,1) gates with q0 and q1 in different cores.
        All 5 are in the same SuperNode (same ctrl/tgt dirs) so
        get_tele_gain_block returns 5, making burst_saves > 0 after the teleport.
        """
        arch = build_basic_grid_architecture(2, 1, 4)   # 2 cores, 4×4 each
        qc = QuantumCircuit(2)
        for _ in range(5):
            qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        layout = {dag.qubits[0]: arch.core_qubits(0)[0],
                  dag.qubits[1]: arch.core_qubits(1)[0]}
        metrics, _ = BurstDSABRE(arch, HardwareConfig(), weight_burst=4.0).route(
            dag, layout
        )
        assert not metrics["aborted"]
        assert metrics["burst_saves"] > 0

    def test_metrics_keys_complete(self):
        arch = _simple_arch()
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        metrics, _ = BurstDSABRE(arch, HardwareConfig()).route(
            dag, _layout_sequential(dag, arch)
        )
        for key in ("ls", "teles", "eprs", "burst_saves", "cost",
                    "1q_gates", "aborted", "compile_time",
                    "backup_activations", "failure_log"):
            assert key in metrics, f"Missing metric key: {key}"

    def test_final_layout_is_injective(self):
        arch = _simple_arch()
        qc = QuantumCircuit(6)
        for i in range(5):
            qc.cx(i, i + 1)
        dag = circuit_to_dag(qc)
        _, final = BurstDSABRE(arch, HardwareConfig()).route(
            dag, _layout_sequential(dag, arch)
        )
        phys = list(final.values())
        assert len(phys) == len(set(phys))

    def test_multi_gate_circuit_completes(self):
        arch = _simple_arch()
        qc = QuantumCircuit(8)
        for i in range(7):
            qc.cx(i, i + 1)
        qc.cx(0, 7)
        dag = circuit_to_dag(qc)
        metrics, _ = BurstDSABRE(arch, HardwareConfig()).route(
            dag, _layout_sequential(dag, arch)
        )
        assert not metrics["aborted"]

    def test_single_qubit_only_circuit(self):
        arch = build_basic_grid_architecture(1, 1, 4)
        qc = QuantumCircuit(2)
        qc.h(0); qc.rz(0.5, 1); qc.x(0)
        dag = circuit_to_dag(qc)
        metrics, _ = BurstDSABRE(arch, HardwareConfig()).route(
            dag, {dag.qubits[0]: 0, dag.qubits[1]: 1}
        )
        assert not metrics["aborted"]
        assert metrics["teles"] == 0
        assert metrics["1q_gates"] == 3


# ─── Topology-aware placement ─────────────────────────────────────────────────

class TestTopologyAwarePlacement:
    """Tests for _topology_aware_core_assignment (no Qiskit required)."""

    def _star_graph(self, n):
        """Star with hub=0, leaves=1..n-1, all edges weight 1."""
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        for i in range(1, n):
            G.add_edge(0, i, weight=1)
        return G

    def test_empty_returns_empty(self):
        arch = build_basic_grid_architecture(1, 1, 3)
        G = nx.Graph()
        assert _topology_aware_core_assignment([], G, arch, 0) == {}

    def test_valid_bijection(self):
        arch = build_basic_grid_architecture(1, 1, 4)  # single 4×4 core
        G = self._star_graph(5)
        mapping = _topology_aware_core_assignment(list(range(5)), G, arch, 0)
        assert len(mapping) == 5
        assert len(set(mapping.values())) == 5       # injective
        for p in mapping.values():
            assert p in arch.core_qubits(0)

    def test_all_physicals_in_correct_core(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        G = self._star_graph(4)
        for core_id in range(arch.num_cores):
            qubits = list(range(core_id * 4, core_id * 4 + 4))
            G2 = nx.Graph()
            for q in qubits:
                G2.add_node(q)
            G2.add_edge(qubits[0], qubits[1], weight=2)
            mapping = _topology_aware_core_assignment(qubits, G2, arch, core_id)
            for p in mapping.values():
                assert p in arch.core_qubits(core_id)

    def test_hub_gets_most_central_physical(self):
        """Hub of a star interaction graph should land on the most central physical qubit."""
        arch = build_basic_grid_architecture(1, 1, 3)  # 3×3 core, center is most central
        G = self._star_graph(5)   # hub = logical qubit 0

        mapping = _topology_aware_core_assignment(list(range(5)), G, arch, 0)

        dist = arch.intra_dist[0]
        phys = arch.core_qubits(0)
        most_central = min(phys, key=lambda p: sum(dist[p][q] for q in phys))
        assert mapping[0] == most_central

    def test_local_search_reduces_cost(self):
        """
        Two high-weight pairs (0-1, 2-3) and one weak cross-pair (0-3).
        Optimal: place pair 0-1 adjacent, pair 2-3 adjacent.
        Verify the returned cost ≤ any random permutation's expected cost.
        """
        arch = build_basic_grid_architecture(1, 1, 4)  # 4×4 core
        G = nx.Graph()
        for i in range(4):
            G.add_node(i)
        G.add_edge(0, 1, weight=100)
        G.add_edge(2, 3, weight=100)
        G.add_edge(0, 3, weight=1)

        mapping = _topology_aware_core_assignment(list(range(4)), G, arch, 0)

        dist = arch.intra_dist[0]
        cost = (100 * dist[mapping[0]][mapping[1]]
                + 100 * dist[mapping[2]][mapping[3]]
                + 1   * dist[mapping[0]][mapping[3]])

        # The two high-weight pairs must each be placed at distance 1 (adjacent).
        assert dist[mapping[0]][mapping[1]] == 1
        assert dist[mapping[2]][mapping[3]] == 1

    def test_no_interaction_edges_returns_valid_assignment(self):
        """When there are no edges, any injective assignment is acceptable."""
        arch = build_basic_grid_architecture(1, 1, 3)
        G = nx.Graph()
        for i in range(4):
            G.add_node(i)
        mapping = _topology_aware_core_assignment(list(range(4)), G, arch, 0)
        assert len(mapping) == 4
        assert len(set(mapping.values())) == 4

    def test_single_qubit_assignment(self):
        arch = build_basic_grid_architecture(1, 1, 3)
        G = nx.Graph()
        G.add_node(0)
        mapping = _topology_aware_core_assignment([0], G, arch, 0)
        assert len(mapping) == 1
        assert list(mapping.keys()) == [0]
        assert mapping[0] in arch.core_qubits(0)


# ─── Decay-weighted lookahead ─────────────────────────────────────────────────

@skip_no_qiskit
class TestDecayLookahead:

    def _two_core_arch(self):
        return build_basic_grid_architecture(2, 1, 4)   # 2 cores × 4×4

    def test_decay_one_equals_default(self):
        """_delta_front with decay=1.0 must match the implicit default."""
        arch = self._two_core_arch()
        router = General_dSABRE_Router(arch, HardwareConfig())
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        q0, q1 = dag.qubits
        p0 = arch.core_qubits(0)[0]
        p1 = arch.core_qubits(1)[4]
        links = arch.inter_links_between(0, 1)
        if not links:
            pytest.skip("No inter-core links")
        _, new_phys = links[0]
        l2p = {q0: p0, q1: p1}
        gates = list(dag.topological_op_nodes())
        assert router._delta_front(q0, new_phys, gates, l2p, decay=1.0) == \
               router._delta_front(q0, new_phys, gates, l2p)

    def test_decay_weights_gates_proportionally(self):
        """
        3 identical CX(q0,q1) gates, all with the same per-gate improvement d.
        decay=1.0 → delta = 3d
        decay=0.5 → delta = (1 + 0.5 + 0.25)*d = 1.75d
        """
        arch = self._two_core_arch()
        router = General_dSABRE_Router(arch, HardwareConfig())
        qc = QuantumCircuit(2)
        qc.cx(0, 1); qc.cx(0, 1); qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        q0, q1 = dag.qubits
        p0 = arch.core_qubits(0)[0]
        p1 = arch.core_qubits(1)[4]
        links = arch.inter_links_between(0, 1)
        if not links:
            pytest.skip("No inter-core links")
        _, new_phys = links[0]

        d = arch.phys_dist[p0][p1] - arch.intra_dist[1][new_phys][p1]
        if d <= 0:
            pytest.skip("This layout config does not yield a positive gain")

        l2p = {q0: p0, q1: p1}
        gates = list(dag.topological_op_nodes())

        delta_full = router._delta_front(q0, new_phys, gates, l2p, decay=1.0)
        delta_half = router._delta_front(q0, new_phys, gates, l2p, decay=0.5)

        assert abs(delta_full - 3 * d) < 1e-9
        assert abs(delta_half - 1.75 * d) < 1e-9

    def test_decay_reduces_influence_of_deep_gates(self):
        """With decay < 1, deep gates matter less: delta_decay < delta_nodecay."""
        arch = self._two_core_arch()
        router = General_dSABRE_Router(arch, HardwareConfig())
        qc = QuantumCircuit(2)
        for _ in range(5):
            qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        q0, q1 = dag.qubits
        p0 = arch.core_qubits(0)[0]
        p1 = arch.core_qubits(1)[4]
        links = arch.inter_links_between(0, 1)
        if not links:
            pytest.skip("No inter-core links")
        _, new_phys = links[0]
        d = arch.phys_dist[p0][p1] - arch.intra_dist[1][new_phys][p1]
        if d <= 0:
            pytest.skip("No positive gain in this config")
        l2p = {q0: p0, q1: p1}
        gates = list(dag.topological_op_nodes())
        delta_full = router._delta_front(q0, new_phys, gates, l2p, decay=1.0)
        delta_decay = router._delta_front(q0, new_phys, gates, l2p, decay=0.8)
        assert delta_decay < delta_full

    def test_routing_still_completes_with_decay(self):
        """End-to-end: routing with lookahead_decay=0.8 must complete without abort."""
        arch = _simple_arch()
        config = HardwareConfig(lookahead_decay=0.8)
        qc = QuantumCircuit(8)
        for i in range(7):
            qc.cx(i, i + 1)
        qc.cx(0, 7)
        dag = circuit_to_dag(qc)
        metrics, _ = General_dSABRE_Router(arch, config).route(
            dag, _layout_sequential(dag, arch)
        )
        assert not metrics["aborted"]

    def test_cost_formula_still_holds_with_decay(self):
        """cost = ls*cost_ls + teles*cost_tele regardless of lookahead_decay."""
        arch = _simple_arch()
        config = HardwareConfig(lookahead_decay=0.7)
        qc = QuantumCircuit(4)
        qc.cx(0, 2); qc.cx(1, 3); qc.cx(0, 3)
        dag = circuit_to_dag(qc)
        metrics, _ = General_dSABRE_Router(arch, config).route(
            dag, _layout_sequential(dag, arch)
        )
        assert not metrics["aborted"]
        expected = (metrics["ls"] * config.cost_local_swap
                    + metrics["teles"] * config.cost_teleport)
        assert abs(metrics["cost"] - expected) < 1e-9


# ─── Correctness gap: _local_swap_path graceful fallback ──────────────────────

@skip_no_qiskit
class TestLocalSwapPathFallback:

    def test_forbidden_does_not_crash_routing(self):
        """
        Route a cross-core gate on a small architecture.  The forbidden-set
        fallback is exercised internally when comm ports block the shortest
        intra-core path; routing must complete without RuntimeError.
        """
        arch = build_basic_grid_architecture(2, 1, 3)   # 2 cores × 3×3
        qc = QuantumCircuit(2)
        for _ in range(4):
            qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        layout = {dag.qubits[0]: arch.core_qubits(0)[0],
                  dag.qubits[1]: arch.core_qubits(1)[0]}
        metrics, _ = BurstDSABRE(arch, HardwareConfig()).route(dag, layout)
        assert not metrics["aborted"]

    def test_fallback_produces_valid_layout(self):
        arch = build_basic_grid_architecture(2, 1, 3)
        qc = QuantumCircuit(2)
        qc.cx(0, 1); qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        layout = {dag.qubits[0]: arch.core_qubits(0)[0],
                  dag.qubits[1]: arch.core_qubits(1)[8]}
        _, final = BurstDSABRE(arch, HardwareConfig()).route(dag, layout)
        # All mapped physical qubits must be in the architecture.
        for p in final.values():
            assert p in arch.phys_dist


# ─── Correctness gap: distance-dependent teleport cost ────────────────────────

@skip_no_qiskit
class TestDistanceDependentTeleportCost:

    def test_flat_model_unchanged(self):
        """cost_teleport_per_hop=0 (default) gives same result as before."""
        arch = build_basic_grid_architecture(2, 1, 4)
        config_flat = HardwareConfig(cost_teleport_per_hop=0.0)
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        layout = {dag.qubits[0]: arch.core_qubits(0)[0],
                  dag.qubits[1]: arch.core_qubits(1)[0]}
        m, _ = BurstDSABRE(arch, config_flat).route(dag, layout)
        assert not m["aborted"]
        expected_base = m["teles"] * config_flat.cost_teleport + m["ls"] * config_flat.cost_local_swap
        assert abs(m["cost"] - expected_base) < 1e-9

    def test_per_hop_raises_cost(self):
        """cost_teleport_per_hop > 0 must increase total cost vs flat model."""
        arch = build_basic_grid_architecture(2, 1, 4)
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        layout = {dag.qubits[0]: arch.core_qubits(0)[0],
                  dag.qubits[1]: arch.core_qubits(1)[0]}

        m_flat, _ = BurstDSABRE(arch, HardwareConfig(cost_teleport_per_hop=0.0)).route(
            dag, layout)
        m_hop, _  = BurstDSABRE(arch, HardwareConfig(cost_teleport_per_hop=5.0)).route(
            dag, layout)

        if m_flat["teles"] == 0 or m_hop["teles"] == 0:
            pytest.skip("No teleports fired for this layout")

        assert m_hop["cost"] > m_flat["cost"]

    def test_per_hop_cost_formula(self):
        """
        cost = ls*cost_ls + sum_over_teles(cost_teleport + cost_teleport_per_hop * hops)

        Since all teleports in a 2-core 1-hop architecture span exactly 1 hop:
          cost = ls*cost_ls + teles*(cost_teleport + cost_teleport_per_hop)
        """
        arch = build_basic_grid_architecture(2, 1, 4)
        config = HardwareConfig(cost_teleport=10.0, cost_teleport_per_hop=3.0)
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        layout = {dag.qubits[0]: arch.core_qubits(0)[0],
                  dag.qubits[1]: arch.core_qubits(1)[0]}
        m, _ = BurstDSABRE(arch, config).route(dag, layout)
        assert not m["aborted"]
        # 2-core arch: every teleport crosses exactly 1 hop
        expected = (m["ls"] * config.cost_local_swap
                    + m["teles"] * (config.cost_teleport + config.cost_teleport_per_hop * 1))
        assert abs(m["cost"] - expected) < 1e-9

    def test_per_hop_dsabre_router(self):
        """Same cost model applies in General_dSABRE_Router."""
        arch = build_basic_grid_architecture(2, 1, 4)
        config = HardwareConfig(cost_teleport=10.0, cost_teleport_per_hop=2.0)
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        layout = {dag.qubits[0]: arch.core_qubits(0)[0],
                  dag.qubits[1]: arch.core_qubits(1)[0]}
        m, _ = General_dSABRE_Router(arch, config).route(dag, layout)
        assert not m["aborted"]
        expected = (m["ls"] * config.cost_local_swap
                    + m["teles"] * (config.cost_teleport + config.cost_teleport_per_hop * 1))
        assert abs(m["cost"] - expected) < 1e-9


# ─── Profiling: memory_report ────────────────────────────────────────────────

class TestMemoryReport:

    def test_returns_required_keys(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        r = arch.memory_report()
        for key in ("num_qubits", "num_cores", "phys_dist_entries",
                    "phys_dist_bytes_est", "intra_dist_entries",
                    "intra_dist_bytes_est", "total_bytes_est"):
            assert key in r, f"Missing key: {key}"

    def test_phys_dist_entries_equals_n_squared(self):
        arch = build_basic_grid_architecture(2, 1, 3)
        # comm_qubits are a subset of data_qubits; total nodes = len(data_qubits)
        n = len(arch.data_qubits)
        r = arch.memory_report()
        assert r["phys_dist_entries"] == n * n

    def test_intra_dist_entries_per_core_equals_m_squared_squared(self):
        m = 3
        arch = build_basic_grid_architecture(2, 2, m)
        r = arch.memory_report()
        # Each core has m*m nodes, so m^4 intra-dist entries total per core
        assert r["intra_dist_entries"] == arch.num_cores * (m * m) ** 2

    def test_bytes_are_positive(self):
        arch = build_basic_grid_architecture(1, 1, 3)
        r = arch.memory_report()
        assert r["phys_dist_bytes_est"] > 0
        assert r["total_bytes_est"] > 0

    def test_total_bytes_at_least_phys_dist_bytes(self):
        arch = build_basic_grid_architecture(2, 2, 3)
        r = arch.memory_report()
        assert r["total_bytes_est"] >= r["phys_dist_bytes_est"]

    def test_larger_arch_has_more_entries(self):
        small = build_basic_grid_architecture(2, 2, 3).memory_report()
        large = build_basic_grid_architecture(2, 2, 4).memory_report()
        assert large["phys_dist_entries"] > small["phys_dist_entries"]
        assert large["total_bytes_est"]   > small["total_bytes_est"]


# ─── Output / usability: export helpers ──────────────────────────────────────

import json
import csv
import io
import tempfile

try:
    from main import (
        save_results_csv, save_results_json, save_trace,
        _metrics_to_row, _CSV_FIELDS,
    )
    HAS_MAIN = True
except ImportError:
    HAS_MAIN = False

skip_no_main = pytest.mark.skipif(not HAS_MAIN, reason="main.py import failed")


def _dummy_row(circuit="test.qasm", router="dSABRE", strategy="A_random"):
    return {
        "circuit": circuit, "router": router, "strategy": strategy,
        "eprs": 5, "teles": 5, "burst_saves": 2, "catcomms": 0,
        "ls": 10, "cost": 55.0, "backup_activations": 0,
        "aborted": 0, "compile_time_ms": 12.5, "total_time_ms": 37.0,
    }


@skip_no_main
class TestExportHelpers:

    def _fresh_path(self, suffix=".csv"):
        """Return a path to a file that does not yet exist."""
        import tempfile
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        os.unlink(path)   # delete so save_results_csv sees a new file
        return path

    def test_save_results_csv_creates_file(self):
        path = self._fresh_path(".csv")
        try:
            save_results_csv([_dummy_row()], path)
            assert os.path.exists(path)
            with open(path) as fh:
                reader = list(csv.DictReader(fh))
            assert len(reader) == 1
            assert reader[0]["circuit"] == "test.qasm"
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_save_results_csv_appends(self):
        path = self._fresh_path(".csv")
        try:
            save_results_csv([_dummy_row()], path)
            save_results_csv([_dummy_row(circuit="b.qasm")], path)
            with open(path) as fh:
                reader = list(csv.DictReader(fh))
            assert len(reader) == 2
            assert reader[1]["circuit"] == "b.qasm"
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_save_results_csv_header_only_once(self):
        path = self._fresh_path(".csv")
        try:
            save_results_csv([_dummy_row()], path)
            save_results_csv([_dummy_row()], path)
            with open(path) as fh:
                lines = fh.readlines()
            header_lines = [l for l in lines if "circuit" in l]
            assert len(header_lines) == 1
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_save_results_csv_all_fields_present(self):
        path = self._fresh_path(".csv")
        try:
            save_results_csv([_dummy_row()], path)
            with open(path) as fh:
                header = fh.readline().strip().split(",")
            assert set(_CSV_FIELDS) == set(header)
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_save_results_json_creates_array(self):
        path = self._fresh_path(".json")
        try:
            save_results_json([_dummy_row()], path)
            with open(path) as fh:
                data = json.load(fh)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["router"] == "dSABRE"
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_save_results_json_merges_existing(self):
        path = self._fresh_path(".json")
        try:
            save_results_json([_dummy_row()], path)
            save_results_json([_dummy_row(circuit="b.qasm")], path)
            with open(path) as fh:
                data = json.load(fh)
            assert len(data) == 2
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_save_results_json_handles_corrupt_file(self):
        import tempfile as _tf
        fd, path = _tf.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            f.write("not valid json {{{")
        try:
            save_results_json([_dummy_row()], path)
            with open(path) as fh:
                data = json.load(fh)
            assert len(data) == 1
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_metrics_to_row_fields(self):
        m = {"eprs": 3, "teles": 3, "burst_saves": 1, "catcomms": 0,
             "ls": 7, "cost": 39.0, "backup_activations": 0,
             "aborted": False, "compile_time": 0.05, "trace": None,
             "failure_log": []}
        row = _metrics_to_row(m, "c.qasm", "BurstDSABRE", "B_sabre_locked", 0.1)
        assert row["eprs"] == 3
        assert row["compile_time_ms"] == pytest.approx(50.0, abs=0.1)
        assert row["total_time_ms"]   == pytest.approx(100.0, abs=0.1)
        assert row["aborted"] == 0

    def test_save_trace_none_is_noop(self):
        path = self._fresh_path(".json")   # already deleted — does not exist
        save_trace({"trace": None}, path)
        assert not os.path.exists(path)

    def test_save_trace_writes_events(self):
        path = self._fresh_path(".json")
        try:
            metrics = {"trace": [
                ("SWAP", 0, 1, 0),
                ("TELE", 0, 1, 10, 0, 1),
            ]}
            save_trace(metrics, path)
            with open(path) as fh:
                data = json.load(fh)
            assert len(data) == 2
            assert data[0][0] == "SWAP"
            assert data[1][0] == "TELE"
        finally:
            if os.path.exists(path): os.unlink(path)


# ─── Trace log integration ────────────────────────────────────────────────────

@skip_no_qiskit
class TestTraceLog:

    def _route_with_trace(self, router_cls, arch, qc, layout=None):
        dag = circuit_to_dag(qc)
        if layout is None:
            layout = _layout_sequential(dag, arch)
        config = HardwareConfig(trace_routing=True)
        if router_cls is BurstDSABRE:
            router = router_cls(arch, config)
        else:
            router = router_cls(arch, config)
        return router.route(dag, layout)

    def test_trace_is_none_by_default(self):
        arch = _simple_arch()
        qc = QuantumCircuit(2); qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        config = HardwareConfig(trace_routing=False)
        m, _ = General_dSABRE_Router(arch, config).route(
            dag, _layout_sequential(dag, arch)
        )
        assert m["trace"] is None

    def test_trace_enabled_is_list(self):
        arch = _simple_arch()
        qc = QuantumCircuit(2); qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        m, _ = General_dSABRE_Router(arch, HardwareConfig(trace_routing=True)).route(
            dag, _layout_sequential(dag, arch)
        )
        assert isinstance(m["trace"], list)

    def test_trace_swap_count_matches_ls(self):
        arch = _simple_arch()
        qc = QuantumCircuit(6)
        for i in range(5):
            qc.cx(i, i + 1)
        dag = circuit_to_dag(qc)
        config = HardwareConfig(trace_routing=True)
        m, _ = General_dSABRE_Router(arch, config).route(
            dag, _layout_sequential(dag, arch)
        )
        swap_events = [e for e in m["trace"] if e[0] == "SWAP"]
        assert len(swap_events) == m["ls"]

    def test_trace_tele_count_matches_teles(self):
        arch = _simple_arch()
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.cx(i, (i + 1) % 4)
        dag = circuit_to_dag(qc)
        config = HardwareConfig(trace_routing=True)
        m, _ = General_dSABRE_Router(arch, config).route(
            dag, _layout_sequential(dag, arch)
        )
        tele_events = [e for e in m["trace"] if e[0] == "TELE"]
        assert len(tele_events) == m["teles"]

    def test_trace_burst_swap_count_matches_ls(self):
        arch = _simple_arch()
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.cx(i, (i + 1) % 4)
        dag = circuit_to_dag(qc)
        config = HardwareConfig(trace_routing=True)
        m, _ = BurstDSABRE(arch, config).route(
            dag, _layout_sequential(dag, arch)
        )
        swap_events = [e for e in m["trace"] if e[0] == "SWAP"]
        assert len(swap_events) == m["ls"]

    def test_trace_burst_tele_count_matches_teles(self):
        arch = _simple_arch()
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.cx(i, (i + 1) % 4)
        dag = circuit_to_dag(qc)
        config = HardwareConfig(trace_routing=True)
        m, _ = BurstDSABRE(arch, config).route(
            dag, _layout_sequential(dag, arch)
        )
        tele_events = [e for e in m["trace"] if e[0] == "TELE"]
        assert len(tele_events) == m["teles"]

    def test_trace_swap_event_structure(self):
        """Each SWAP event must be (type, phys_a, phys_b, core_id)."""
        arch = _simple_arch()
        qc = QuantumCircuit(6)
        for i in range(5):
            qc.cx(i, i + 1)
        dag = circuit_to_dag(qc)
        m, _ = General_dSABRE_Router(arch, HardwareConfig(trace_routing=True)).route(
            dag, _layout_sequential(dag, arch)
        )
        for ev in (e for e in m["trace"] if e[0] == "SWAP"):
            assert len(ev) == 4
            _, pa, pb, core_id = ev
            assert arch.core_of(pa) == core_id
            assert arch.core_of(pb) == core_id

    def test_trace_tele_event_structure(self):
        """Each TELE event must be (type, virt, p_src, p_dst, src_core, next_core)."""
        arch = _simple_arch()
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.cx(i, (i + 1) % 4)
        dag = circuit_to_dag(qc)
        m, _ = General_dSABRE_Router(arch, HardwareConfig(trace_routing=True)).route(
            dag, _layout_sequential(dag, arch)
        )
        for ev in (e for e in m["trace"] if e[0] == "TELE"):
            assert len(ev) == 6
            _, virt, p_src, p_dst, src_core, next_core = ev
            assert arch.core_of(p_src) == src_core
            assert arch.core_of(p_dst) == next_core
