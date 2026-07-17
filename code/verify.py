"""
Structural / invariant-based equivalence checker for dSABRE output.

A dSABRE run produces a trace of SWAP and TELE actions (when
`HardwareConfig.trace_routing=True`).  Together with the input DAG and the
initial layout, this trace fully determines the output circuit on physical
qubits.  This module replays the trace against the DAG and verifies four
invariants whose conjunction implies that the output implements the same
unitary as the input, up to the final qubit permutation:

  (I1) Gate preservation: every 2q gate in the input is executed exactly
       once, in an order consistent with the DAG's partial order.  No gate
       is dropped or duplicated.  1q gates are also accounted for.
  (I2) Connectivity legality: each executed 2q gate acts on two physical
       qubits adjacent in the architecture graph (intra-core edge or
       inter-core EPR edge).  Every SWAP is along an intra-core edge of the
       declared core.  Every TELE endpoint pair is an inter-core link of
       the declared core pair.
  (I3) Permutation tracking: the logical-to-physical map v2p is updated
       atomically and consistently with each SWAP / TELE.  The final v2p
       matches the layout returned by the router.
  (I4) Pre-gate adjacency: a 2q gate becomes executable only when its two
       logical qubits map to adjacent physical qubits.
  (I5) Teleport port legality: at each TELE, the source-side comm port of
       the link must be free (or hold the moving qubit itself) and the
       moving qubit must sit at or adjacent to that port; the destination
       port must be free.  Both ports carry EPR halves during the protocol,
       so an occupied port makes the teleport physically unexecutable.

Usage
-----
    cfg = HardwareConfig(..., trace_routing=True)
    router = General_dSABRE_Router(arch, cfg)
    metrics, final_layout = router.route(dag, initial_layout)
    report = verify_routing(dag, arch, initial_layout, final_layout,
                            metrics["trace"])
    assert report.ok, report.failures
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class VerifyReport:
    ok: bool
    n_gates_1q: int = 0
    n_gates_2q: int = 0
    n_swaps: int = 0
    n_teles: int = 0
    failures: List[str] = field(default_factory=list)

    def __bool__(self):
        return self.ok


def verify_routing(dag, arch, initial_layout, final_layout, trace) -> VerifyReport:
    """Verify a routed dSABRE output against its input DAG.

    Parameters
    ----------
    dag : DAGCircuit
        The input DAG passed to `router.route`.
    arch : DistributedArchitecture
    initial_layout : dict {virtual_qubit -> physical_qubit}
        Same layout passed to `router.route`.
    final_layout : dict {virtual_qubit -> physical_qubit}
        Returned by `router.route`.
    trace : list of tuples
        ("SWAP", a, b, core_id) or
        ("TELE", virt, p_src, p_comm_dst, src_core, next_core)
    """
    fails: List[str] = []

    # ── State: virtual → physical (v2p) and inverse (p2v). ────────────────────
    v2p: Dict[Any, int] = dict(initial_layout)
    p2v: Dict[int, Any] = {p: None for p in arch.Gr.nodes}
    for v, p in v2p.items():
        if p2v.get(p) is not None:
            fails.append(f"initial_layout: physical {p} occupied by two logicals")
        p2v[p] = v

    # ── Index input DAG. ──────────────────────────────────────────────────────
    # We walk in topological order, tracking which gates have been executed.
    # A gate becomes executable when (a) all predecessors are executed and
    # (b) its physical qubits (under current v2p) are adjacent.
    op_nodes = list(dag.topological_op_nodes())
    pending = list(op_nodes)
    expected_1q = sum(1 for n in op_nodes if len(n.qargs) < 2)
    expected_2q = sum(1 for n in op_nodes if len(n.qargs) == 2)

    executed_ids = set()
    pred_remaining: Dict[int, int] = {}
    for n in op_nodes:
        # Count predecessors that are op nodes.  Qiskit returns fresh
        # DAGOpNode wrappers on each iteration, so identity is by `_node_id`
        # (the persistent rustworkx node id) — same convention as router.py.
        preds = [p for p in dag.predecessors(n) if hasattr(p, "qargs")]
        pred_remaining[n._node_id] = len(preds)

    def _try_drain():
        """Execute every front-layer gate that is now legally adjacent.

        Returns the number of 2q gates executed (1q always execute).
        """
        progress = True
        n2q_exec = 0
        while progress:
            progress = False
            for n in list(pending):
                if n._node_id in executed_ids:
                    continue
                if pred_remaining[n._node_id] > 0:
                    continue
                if len(n.qargs) < 2:
                    _commit(n)
                    progress = True
                    continue
                p1, p2 = v2p[n.qargs[0]], v2p[n.qargs[1]]
                if arch.Gr.has_edge(p1, p2):
                    _commit(n)
                    n2q_exec += 1
                    progress = True
        return n2q_exec

    def _commit(n):
        executed_ids.add(n._node_id)
        pending.remove(n)
        for succ in dag.successors(n):
            if hasattr(succ, "qargs") and succ._node_id in pred_remaining:
                pred_remaining[succ._node_id] -= 1

    # Drain anything already legal under the initial layout.
    _try_drain()

    n_swaps = 0
    n_teles = 0

    # ── Replay trace. ─────────────────────────────────────────────────────────
    for step_i, ev in enumerate(trace or []):
        kind = ev[0]
        if kind == "SWAP":
            _, a, b, core_id = ev
            # (I2) intra-core edge of the declared core.
            if a not in arch.intra[core_id] or b not in arch.intra[core_id]:
                fails.append(f"[{step_i}] SWAP {a}<->{b}: endpoints not in core {core_id}")
                continue
            if not arch.intra[core_id].has_edge(a, b):
                fails.append(f"[{step_i}] SWAP {a}<->{b}: not an intra-core edge in core {core_id}")
                continue
            # (I3) atomic permutation update.
            va, vb = p2v[a], p2v[b]
            p2v[a], p2v[b] = vb, va
            if va is not None:
                v2p[va] = b
            if vb is not None:
                v2p[vb] = a
            n_swaps += 1

        elif kind == "TELE":
            _, virt, p_src, p_comm_dst, src_core, next_core = ev
            # (I2) declared inter-core link must exist (in either direction).
            cu = arch.core_of(p_comm_dst)
            if cu != next_core:
                fails.append(
                    f"[{step_i}] TELE: p_comm_dst {p_comm_dst} not in next_core {next_core}"
                )
            # The teleport in router.py first moves `virt` to the staging slot
            # n_s adjacent to the src comm port, then writes it onto p_comm_dst
            # in the neighbour core; intermediate SWAPs were already emitted.
            # So at the moment of TELE, virt is somewhere in src_core and the
            # destination slot must be free.
            if v2p.get(virt) is None or arch.core_of(v2p[virt]) != src_core:
                fails.append(
                    f"[{step_i}] TELE: virt {virt} not in src_core {src_core} "
                    f"(currently at {v2p.get(virt)})"
                )
            if p2v[p_comm_dst] is not None:
                fails.append(
                    f"[{step_i}] TELE: destination {p_comm_dst} not free "
                    f"(holds {p2v[p_comm_dst]})"
                )
            # (I5) source-side port legality.  The trace does not record the
            # source port, but p_comm_dst identifies the inter-core link.
            p_now = v2p.get(virt)
            src_ports = [u for u, v in arch.inter_links_between(src_core, next_core)
                         if v == p_comm_dst]
            if not src_ports:
                fails.append(
                    f"[{step_i}] TELE: no {src_core}->{next_core} link "
                    f"ends at {p_comm_dst}"
                )
            elif not any(
                p_now == u
                or (p2v.get(u) is None
                    and p_now is not None
                    and arch.intra[src_core].has_edge(p_now, u))
                for u in src_ports
            ):
                fails.append(
                    f"[{step_i}] TELE: source port(s) {src_ports} of link to "
                    f"{p_comm_dst} occupied or not adjacent to virt {virt} "
                    f"at {p_now}"
                )
            # (I3) permutation update: virt moves to p_comm_dst, old slot frees.
            old = v2p[virt]
            p2v[old] = None
            v2p[virt] = p_comm_dst
            p2v[p_comm_dst] = virt
            n_teles += 1
        else:
            fails.append(f"[{step_i}] unknown trace event: {ev!r}")
            continue

        # After each physical action, drain any gates that became legal.
        _try_drain()

    # ── Final state checks. ───────────────────────────────────────────────────
    # (I1) every gate executed exactly once.
    remaining = [n for n in pending]
    if remaining:
        # Helpful diagnostic: show the first few unexecuted gates and why.
        diag = []
        for n in remaining[:5]:
            if len(n.qargs) < 2:
                diag.append(f"1q gate on {n.qargs}")
            else:
                p1 = v2p.get(n.qargs[0])
                p2 = v2p.get(n.qargs[1])
                adj = arch.Gr.has_edge(p1, p2) if p1 is not None and p2 is not None else False
                diag.append(
                    f"2q gate {n.qargs} -> ({p1},{p2}) adj={adj} "
                    f"preds_left={pred_remaining[n._node_id]}"
                )
        fails.append(
            f"{len(remaining)} input gate(s) never executed; first few: {diag}"
        )

    # (I3) final layout matches.
    for v, p in final_layout.items():
        if v2p.get(v) != p:
            fails.append(
                f"final_layout mismatch for virt {v}: replay={v2p.get(v)}, router={p}"
            )
            break

    return VerifyReport(
        ok=(not fails),
        n_gates_1q=expected_1q,
        n_gates_2q=expected_2q,
        n_swaps=n_swaps,
        n_teles=n_teles,
        failures=fails,
    )
