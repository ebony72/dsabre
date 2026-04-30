"""
Commutativity-aware hypergraph of a quantum circuit (Hong et al. DAC 2026).

Each qubit wire is partitioned into SuperNodes: maximal runs of gates sharing
identical eigen-directions on that wire (mutually commuting on that wire).
A SuperNode is ready iff its wire-predecessor is fully executed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from architecture import DistributedArchitecture


# Eigen-direction table: gate_type -> [(role, eigen_dirs), …] per qubit position.
# Directions encode commutativity: two gates on the same qubit commute iff their
# eigen-dirs agree on that wire.

EIGEN_DIRS: Dict[str, List[Tuple[str, FrozenSet[str]]]] = {
    # ── single-qubit ─────────────────────────────────────────────────────────
    "x":    [("target", frozenset({"x+", "x-"}))],
    "y":    [("target", frozenset({"y+", "y-"}))],
    "z":    [("target", frozenset({"0",  "1" }))],
    "h":    [("target", frozenset({"h+", "h-"}))],
    "s":    [("target", frozenset({"0",  "1" }))],
    "sdg":  [("target", frozenset({"0",  "1" }))],
    "t":    [("target", frozenset({"0",  "1" }))],
    "tdg":  [("target", frozenset({"0",  "1" }))],
    "sx":   [("target", frozenset({"x+", "x-"}))],
    "sxdg": [("target", frozenset({"x+", "x-"}))],
    "rx":   [("target", frozenset({"x+", "x-"}))],
    "ry":   [("target", frozenset({"y+", "y-"}))],
    "rz":   [("target", frozenset({"0",  "1" }))],
    "u1":   [("target", frozenset({"0",  "1" }))],
    "u2":   [("target", frozenset({"u2a","u2b"}))],
    "u3":   [("target", frozenset({"u3a","u3b"}))],
    "u":    [("target", frozenset({"u3a","u3b"}))],
    "p":    [("target", frozenset({"0",  "1" }))],
    "id":   [("target", frozenset({"*"       }))],
    "reset":[("target", frozenset({"R"       }))],
    # ── two-qubit ────────────────────────────────────────────────────────────
    "cx":   [("control", frozenset({"0","1"})),  ("target", frozenset({"x+","x-"}))],
    "cz":   [("control", frozenset({"0","1"})),  ("target", frozenset({"0", "1" }))],
    "cy":   [("control", frozenset({"0","1"})),  ("target", frozenset({"y+","y-"}))],
    "ch":   [("control", frozenset({"0","1"})),  ("target", frozenset({"h+","h-"}))],
    "cp":   [("control", frozenset({"0","1"})),  ("target", frozenset({"0", "1" }))],
    "crx":  [("control", frozenset({"0","1"})),  ("target", frozenset({"x+","x-"}))],
    "cry":  [("control", frozenset({"0","1"})),  ("target", frozenset({"y+","y-"}))],
    "crz":  [("control", frozenset({"0","1"})),  ("target", frozenset({"0", "1" }))],
    "swap": [("swap0",   frozenset({"sw"    })), ("swap1",  frozenset({"sw"     }))],
    "iswap":[("isw0",    frozenset({"isw"   })), ("isw1",   frozenset({"isw"    }))],
    "ecr":  [("ecr0",    frozenset({"ecr0"  })), ("ecr1",   frozenset({"ecr1"   }))],
    "rzz":  [("rzz0",    frozenset({"0","1" })), ("rzz1",   frozenset({"0","1"  }))],
    "rxx":  [("rxx0",    frozenset({"x+","x-"})),("rxx1",   frozenset({"x+","x-"}))],
    "ryy":  [("ryy0",    frozenset({"y+","y-"})),("ryy1",   frozenset({"y+","y-"}))],
    # ── three-qubit ──────────────────────────────────────────────────────────
    "ccx":  [("ctrl0",   frozenset({"0","1"})), ("ctrl1",  frozenset({"0","1" })),
             ("target",  frozenset({"x+","x-"}))],
    "ccz":  [("ctrl0",   frozenset({"0","1"})), ("ctrl1",  frozenset({"0","1" })),
             ("target",  frozenset({"0", "1" }))],
    "cswap":[("control", frozenset({"0","1"})), ("swap0",  frozenset({"sw"    })),
             ("swap1",   frozenset({"sw"    }))],
    "rccx": [("ctrl0",   frozenset({"0","1"})), ("ctrl1",  frozenset({"0","1" })),
             ("target",  frozenset({"x+","x-"}))],
}




@dataclass(eq=False)
class Gate:
    """
    A quantum gate at a fixed position in the original circuit sequence.

    Attributes
    ----------
    gate_id   : Unique integer, stable for the lifetime of the hypergraph.
    gate_type : Lowercase gate name, e.g. "cx", "ccx", "rx".
    qubits    : Logical qubit indices — controls listed first, then target.
    params    : Optional rotation angles / phase values.
    label     : Optional display-name override.
    """

    gate_id:   int
    gate_type: str
    qubits:    List[int]
    params:    List[float] = field(default_factory=list)
    label:     str         = field(default="")

    def _dirs(self) -> List[FrozenSet[str]]:
        spec = EIGEN_DIRS.get(self.gate_type.lower())
        if spec is None:
            # Unknown gate: give each qubit a unique fingerprint so it never
            # merges into another SuperNode.
            return [frozenset({f"unk_{self.gate_type}_{self.gate_id}_{i}"})
                    for i in range(len(self.qubits))]
        dirs = [d for _, d in spec]
        while len(dirs) < len(self.qubits):
            dirs.append(dirs[-1])
        return dirs[:len(self.qubits)]

    def eigen_dir_on(self, qubit: int) -> FrozenSet[str]:
        return self._dirs()[self.qubits.index(qubit)]

    def commutes_with(self, other: "Gate") -> bool:
        shared = set(self.qubits) & set(other.qubits)
        if not shared:
            return True
        return all(self.eigen_dir_on(q) == other.eigen_dir_on(q) for q in shared)

    @property
    def name(self) -> str:
        return self.label or self.gate_type.upper()

    def __repr__(self) -> str:
        p = f"({','.join(f'{x:.3g}' for x in self.params)})" if self.params else ""
        return f"G{self.gate_id}:{self.name}{p}{self.qubits}"

    def __hash__(self):   return id(self)
    def __eq__(self, o):  return self is o


@dataclass(eq=False)
class SuperNode:
    """
    A maximal run of gates sharing identical eigen-directions on a single qubit wire
    (Eq. 3, Hong et al. DAC 2026).  Gates within the same SuperNode commute on that
    wire; global commutativity is not guaranteed.

    A SuperNode is ready iff its wire-predecessor is fully executed (or absent).
    """

    node_id:    int
    qubit:      int
    eigen_dirs: FrozenSet[str]
    gates:      List[Gate]             = field(default_factory=list)
    _executed:  Set[int]               = field(default_factory=set, repr=False)
    pred:       Optional["SuperNode"]  = field(default=None,        repr=False)
    succ:       Optional["SuperNode"]  = field(default=None,        repr=False)

    def mark_executed(self, gate_id: int) -> None:
        self._executed.add(gate_id)

    @property
    def pending_gates(self) -> List[Gate]:
        return [g for g in self.gates if g.gate_id not in self._executed]

    @property
    def is_fully_executed(self) -> bool:
        return len(self._executed) >= len(self.gates)

    @property
    def is_ready(self) -> bool:
        return self.pred is None or self.pred.is_fully_executed

    def __repr__(self) -> str:
        pred_s = f"SN{self.pred.node_id}" if self.pred else "HEAD"
        succ_s = f"SN{self.succ.node_id}" if self.succ else "TAIL"
        pending = [g.gate_id for g in self.pending_gates]
        return (f"SN{self.node_id}(q{self.qubit}|{sorted(self.eigen_dirs)}|"
                f"{pred_s}→{succ_s}|pending={pending})")

    def __hash__(self):  return id(self)
    def __eq__(self, o): return self is o


class EigenClusterHypergraph:
    """Commutativity-aware hypergraph of a quantum circuit (Hong et al. DAC 2026)."""

    def __init__(self) -> None:
        self.num_qubits:  int                        = 0
        self.nodes:       List[SuperNode]            = []
        self._g2sn:       Dict[int, List[SuperNode]] = {}  # gate_id -> SNs (one per qubit)
        self._gates:      Dict[int, Gate]            = {}
        self.hyperedges:  Dict[int, List[SuperNode]] = {}  # gate_id -> SNs for 2q+ gates
        self._id_ctr = itertools.count()
        # Incrementally-maintained ready sets: O(1) front-layer queries.
        # At most one SN per qubit can be ready at a time (linear wire invariant).
        self._ready_snodes:      Set[SuperNode]                   = set()
        self._qubit_to_ready_sn: Dict[int, Optional[SuperNode]]   = {}

    def _new_node(self, qubit: int, eigen_dirs: FrozenSet[str],
                  pred: Optional[SuperNode]) -> SuperNode:
        sn = SuperNode(next(self._id_ctr), qubit, eigen_dirs, pred=pred)
        if pred is not None:
            pred.succ = sn
        self.nodes.append(sn)
        return sn

    def build(self, circuit: List[Gate], num_qubits: int) -> None:
        """
        Greedy SuperNode merging (Section 3.1, Hong et al. DAC 2026). O(k·n).
        For each gate, merge into the frontier SN if eigen-dirs match; else create
        a new SN with a wire edge from the current frontier. Initialises the
        incremental ready-set after all gates are processed.
        """
        self.num_qubits = num_qubits
        frontier: Dict[int, Optional[SuperNode]] = {q: None for q in range(num_qubits)}

        # Accept Qiskit Qubit objects or plain ints; map to stable integers.
        qubit_map: Dict[object, int] = {}
        next_idx = [0]

        def _to_int(q) -> int:
            if isinstance(q, int):
                return q
            if q not in qubit_map:
                qubit_map[q] = next_idx[0]
                next_idx[0] += 1
            return qubit_map[q]

        for gate in circuit:
            gate.qubits = [_to_int(q) for q in gate.qubits]
            self._gates[gate.gate_id] = gate
            node_for_qubit: Dict[int, SuperNode] = {}

            for q in gate.qubits:
                f_dirs = gate.eigen_dir_on(q)
                prev   = frontier[q]
                if prev is not None and prev.eigen_dirs == f_dirs:
                    prev.gates.append(gate)
                    node_for_qubit[q] = prev
                else:
                    sn = self._new_node(q, f_dirs, prev)
                    sn.gates.append(gate)
                    frontier[q] = sn
                    node_for_qubit[q] = sn

            self._g2sn[gate.gate_id] = [node_for_qubit[q] for q in gate.qubits]
            if len(gate.qubits) > 1:
                self.hyperedges[gate.gate_id] = [node_for_qubit[q] for q in gate.qubits]

        self._ready_snodes = set()
        self._qubit_to_ready_sn = {q: None for q in range(num_qubits)}
        for sn in self.nodes:
            if sn.pred is None and not sn.is_fully_executed:
                self._ready_snodes.add(sn)
                self._qubit_to_ready_sn[sn.qubit] = sn

    def _is_executed(self, gate_id: int) -> bool:
        return all(gate_id in sn._executed for sn in self._g2sn.get(gate_id, []))

    def _ready_nodes(self) -> List[SuperNode]:
        return list(self._ready_snodes)

    def _ready_sn_for(self, qubit: int) -> Optional[SuperNode]:
        return self._qubit_to_ready_sn.get(qubit)

    def get_front_layer(self) -> List[Gate]:
        """
        All pending gates whose every per-qubit SN is ready (ALL-SN rule).
        Two front-layer gates may share a qubit if they occupy the same SN
        (same eigen-dirs); process one at a time and re-query after each removal.
        Global commutativity between front-layer gates is NOT guaranteed.
        """
        seen:   Set[int]   = set()
        result: List[Gate] = []
        for sn in self._ready_snodes:
            for g in sn.pending_gates:
                if g.gate_id in seen:
                    continue
                if all(sn2.is_ready for sn2 in self._g2sn[g.gate_id]):
                    seen.add(g.gate_id)
                    result.append(g)
        return result

    def remove_gate(self, gate_id: int) -> None:
        """
        Mark gate as executed. O(k) where k = gate arity.
        When a SuperNode becomes fully executed, it is retired from the ready-set
        and its wire successor is promoted. Idempotent.
        """
        if gate_id not in self._gates:
            raise KeyError(f"gate_id={gate_id} not found in hypergraph")
        if self._is_executed(gate_id):
            return
        for sn in self._g2sn[gate_id]:
            was_done = sn.is_fully_executed
            sn.mark_executed(gate_id)
            if not was_done and sn.is_fully_executed:
                self._ready_snodes.discard(sn)
                if self._qubit_to_ready_sn.get(sn.qubit) is sn:
                    self._qubit_to_ready_sn[sn.qubit] = None
                succ = sn.succ
                if succ is not None and not succ.is_fully_executed:
                    self._ready_snodes.add(succ)
                    self._qubit_to_ready_sn[succ.qubit] = succ

    def remove_gates(self, gate_ids) -> None:
        for gid in gate_ids:
            self.remove_gate(gid)

    def execute_front_layer(self) -> List[Gate]:
        fl = self.get_front_layer()
        self.remove_gates(g.gate_id for g in fl)
        return fl

    def execute_local_front(self, layout: Dict[int, int]) -> List[Gate]:
        local = self.get_local_front_layer(layout)
        self.remove_gates(g.gate_id for g in local)
        return local

    def is_remote(self, gate: Gate, layout: Dict[int, int]) -> bool:
        if len(gate.qubits) < 2:
            return False
        cores = {layout[q] for q in gate.qubits if q in layout}
        return len(cores) > 1

    def get_remote_front_layer(self, layout: Dict[int, int]) -> List[Gate]:
        return [g for g in self.get_front_layer() if self.is_remote(g, layout)]

    def get_local_front_layer(self, layout: Dict[int, int]) -> List[Gate]:
        return [g for g in self.get_front_layer() if not self.is_remote(g, layout)]

    def num_pending(self) -> int:
        return sum(1 for gid in self._gates if not self._is_executed(gid))

    def is_done(self) -> bool:
        return self.num_pending() == 0

    def get_tele_gain_block(
        self,
        lq_idx:   int,
        new_phys: int,
        old_phys: int,
        idx2phys: Dict[int, int],
        arch:     "DistributedArchitecture",
        max_depth: int = 30,
    ) -> int:
        """
        Count pending 2q gates on lq_idx's wire whose distance δ = d(old,partner) -
        d(new,partner) > 0, used to score burst teleportation candidates.

        Walk SNs from the ready frontier. For each pending 2q gate:
          δ > 0 → count it and continue.
          δ = 0 → skip.
          δ < 0 → rescue pass: count front-ready gates in the same SN with δ > 0,
                  then stop (do not advance to the next SN).

        SN membership guarantees rescue gates commute with the harmful gate on
        lq_idx's wire; ALL-SN readiness ensures they are unblocked on all other wires.
        """
        cur: Optional[SuperNode] = self._qubit_to_ready_sn.get(lq_idx)
        if cur is None:
            return 0

        def gate_dist(phys_a: int, phys_b: int) -> Optional[int]:
            ca, cb = arch.core_of(phys_a), arch.core_of(phys_b)
            if ca == cb:
                return arch.intra_dist[ca][phys_a][phys_b]
            return arch.phys_dist.get(phys_a, {}).get(phys_b)

        def delta_for(g: Gate) -> Optional[int]:
            partneridx  = g.qubits[1] if g.qubits[0] == lq_idx else g.qubits[0]
            partnerphys = idx2phys.get(partneridx)
            if partnerphys is None:
                return None
            d_old = gate_dist(old_phys, partnerphys)
            d_new = gate_dist(new_phys, partnerphys)
            if d_old is None or d_new is None:
                return None
            return d_old - d_new

        def is_front(g: Gate) -> bool:
            return all(sn.is_ready for sn in self._g2sn[g.gate_id])

        count = 0
        depth = 0

        while cur is not None and depth < max_depth:
            found_bad = False
            depth += 1

            for g in cur.pending_gates:
                if len(g.qubits) < 2:
                    continue
                d = delta_for(g)
                if d is None:
                    continue
                if d > 0:
                    count += 1
                elif d < 0:
                    found_bad = True
                    already_counted: set = set()
                    for g2 in cur.pending_gates:
                        if g2 is g or len(g2.qubits) < 2 or g2.gate_id in already_counted:
                            continue
                        if not is_front(g2):
                            continue
                        d2 = delta_for(g2)
                        if d2 is not None and d2 > 0:
                            count += 1
                            already_counted.add(g2.gate_id)
                    break

            if found_bad:
                break

            cur = cur.succ

        return count