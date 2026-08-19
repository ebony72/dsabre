"""
telegate_router.py — dSABRE with a telegate (cat-comm) macro-action.

Why this exists
---------------
A reviewer asked why \\dSABRE{} uses teledata only when
\\TeleSABRE{} supports both primitives.  Section VI ("Telegate and
communication bursts") answers that scoring a *single* telegate is not the
hard part -- it "consumes no destination capacity and enters Eq. 1 with
c_cap = 0, keeping only the port staging d_prep its two link qubits need;
its progress term is the front-layer gate it retires outright rather than a
distance reduction" -- and that the hard part is the *grouped* telegate,
which needs commutation-derived bursts and an entanglement lifetime model.

This module implements exactly the easy half, verbatim from that paragraph,
so the claim can be measured rather than asserted.  Nothing here is
amortised: one telegate serves one gate and consumes one EPR pair, the same
unit `_apply_teleport` charges for one teledata hop.

The action
----------
For a front-layer gate g = (q_a @ core A, q_b @ core B) with A, B *adjacent*
in the core graph, and a link (p_src_port in A, p_dst_port in B):

  1. evict both link endpoints (they must hold the EPR halves),
  2. SWAP q_a to a neighbour of p_src_port inside A       (cat-entangler arm),
  3. SWAP q_b to a neighbour of p_dst_port inside B       (remote gate arm),
  4. consume one EPR pair: cat-entangle q_a onto p_dst_port,
  5. execute g locally in B against the cat qubit, then disentangle.

Neither operand changes core, and the cat qubit is measured out in the same
atomic action, so **core occupancy is unchanged** -- which is why the
capacity term drops out.  It also means a telegate can never violate safe
mode's free(c) >= 1 invariant, so the termination argument of Appendix C is
untouched by adding it.

Restriction to adjacent cores is deliberate (and is what the user asked to
test): a telegate across k hops needs k EPR pairs *held simultaneously*
along a chain of intermediate link qubits, which is the lifetime/buffer
assumption the paper's Section IV-B measures the price of.  A one-hop
telegate holds one pair for the duration of one gate.

Scoring
-------
    score_telegate = w_dst * d_dst + d_src + bias                 (cost)
                     - w_prog * d_phys(p_a, p_b)                  (progress)

against the unchanged teleport score

    score_teleport = d_prep + c_cap - dF - w_e * dE .

Both cost one EPR pair, so the EPR term cancels on both sides and the two
lists are directly comparable; the best-scoring action of either kind wins.
A telegate gets **no** lookahead credit (dE = 0) because it moves no qubit
between cores: that asymmetry is not an oversight, it is the amortisation
trade-off made explicit.  A teleport whose extended set is full of follow-on
gates in the destination core earns a large -w_e*dE and outranks the
telegate; a one-off remote gate does not, and the telegate wins.

Implementation note
-------------------
`router.py` is untouched.  Everything here is a subclass override of three
hooks that `route()` already calls through `self`:
`_generate_candidates` (append telegate candidates to the same sorted list),
`_apply_teleport` (dispatch on action type), and `_init_dag_state` /
`_rebuild_wdag` (capture the retiring-DAG proxy, since a telegate retires
its gate and `route()` keeps `wdag` in a local).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import networkx as nx

from config import HardwareConfig
from dsabre_ext import dSABRE_BFSExt
from router import _RetiringDAG


# ── The action ────────────────────────────────────────────────────────────────

@dataclass
class TelegateAction:
    """One cat-comm macro-action: execute `node` remotely, both operands stay.

    Field names deliberately mirror `TeleportAction` where the meaning is the
    same, so `route()`'s post-application bookkeeping (`applied.node`) works
    for either kind without a special case.

    node        : the front-layer gate this action RETIRES outright
    virt        : the operand that is cat-entangled into the far core
    partner     : the other operand, which the gate is executed against
    p_src       : `virt`'s physical slot at scoring time
    n_s         : source-side staging slot (neighbour of p_comm_src)
    p_comm_src  : source-side link endpoint (holds one EPR half)
    p_comm_dst  : destination-side link endpoint (holds the cat qubit)
    src_core    : core `virt` stays in
    next_core   : core `partner` stays in (adjacent to src_core)
    d_src/d_dst : the two halves of d_prep, kept for instrumentation
    score       : lower is better; competes directly with TeleportAction.score
    """
    __slots__ = ['node', 'virt', 'partner', 'p_src', 'n_s',
                 'p_comm_src', 'p_comm_dst', 'src_core', 'next_core',
                 'd_src', 'd_dst', 'gate_dist', 'score']
    node:       Any
    virt:       Any
    partner:    Any
    p_src:      int
    n_s:        int
    p_comm_src: int
    p_comm_dst: int
    src_core:   int
    next_core:  int
    d_src:      int
    d_dst:      int
    gate_dist:  int
    score:      float


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TelegateConfig(HardwareConfig):
    """`HardwareConfig` plus the telegate knobs.

    Defaults reproduce the published router exactly when `telegate=False`:
    no candidate of the new kind is ever generated, and every other code path
    is the inherited one.
    """
    # Master switch.  False -> pure teledata, i.e. the published \dSABRE{}.
    telegate: bool = False
    # Maximum core-graph distance a telegate may span.  1 is the primitive
    # this module implements (one EPR pair, one link, held for one gate);
    # larger values are NOT supported and raise, rather than silently
    # charging one pair for a k-hop cat state.
    telegate_max_core_dist: int = 1
    # Weight on the progress term: the gate's current d_phys, which the
    # telegate drives to "executed" in one action.  1.0 puts it in the same
    # units as a teleport's dF.
    telegate_progress: float = 1.0
    # Weight on the DESTINATION-side staging SWAPs (bringing the far operand
    # next to the link endpoint).  1.0 charges them in full; 0.0 is the
    # literal reading of Section VI ("only the port staging its two link
    # qubits need"), which ignores what the far operand must walk.
    telegate_dst_weight: float = 1.0
    # Flat additive offset on every telegate score.  > 0 discourages,
    # < 0 encourages.  Exists to measure how sharp the crossover is.
    telegate_bias: float = 0.0
    # Amortisation penalty.  A telegate leaves `virt` where it is, so every
    # LATER gate this qubit has with a partner in the far core still costs a
    # full EPR pair, whereas one teledata hop would have made them all local.
    # The teleport score already prices that as its -w_e*dE lookahead bonus,
    # but only at w_e = 0.25; this knob adds the symmetric penalty
    # + telegate_amort_weight * max(0, dE) to the telegate, where dE is the
    # SAME extended-set distance reduction that teleporting `virt` onto the
    # far endpoint would earn.  0.0 relies on the teleport's bonus alone.
    telegate_amort_weight: float = 0.0
    # Selective rule (as opposed to the discouraging ones above).  When True
    # a telegate is offered ONLY for gates whose cat-entangled operand would
    # gain nothing in the extended set by relocating (dE <= threshold), i.e.
    # exactly the gates a teledata hop cannot amortise.  This is the
    # structurally correct criterion rather than a tuned constant: it does
    # not make telegate cheaper, it removes it from the competition wherever
    # teledata has a future to amortise over.
    telegate_require_no_lookahead: bool = False
    telegate_lookahead_threshold: float = 0.0
    # When True only qargs[0] (the CX control) may be the cat-entangled
    # operand.  False also allows the target, which costs two local
    # Hadamards (CX = (I(x)H) CZ (I(x)H)) and no extra EPR pair.
    telegate_control_only: bool = False


# ── The router ────────────────────────────────────────────────────────────────

class dSABRE_Telegate(dSABRE_BFSExt):
    """dSE (BFS extended set) with the single-gate telegate macro-action."""

    # -- wdag capture -------------------------------------------------------
    # A telegate retires its gate, but `route()` holds the `_RetiringDAG`
    # proxy in a local.  Both hooks that create one are overridden to stash
    # it; the proxy is what keeps `_indeg` / `_remaining` / `_retired_log`
    # in step, so retiring through it (never through the bare DAG) is
    # required for checkpoint-rollback to stay correct.

    def _init_dag_state(self, real_dag):
        super()._init_dag_state(real_dag)
        self._wdag = _RetiringDAG(real_dag, self)

    def _rebuild_wdag(self, dag, mark):
        w = super()._rebuild_wdag(dag, mark)
        self._wdag = w
        return w

    # -- candidate generation ----------------------------------------------

    def _generate_candidates(self, front_inter, extended, l2p, p2l,
                             committed_nid=None):
        cands = super()._generate_candidates(front_inter, extended, l2p, p2l,
                                             committed_nid=committed_nid)
        if not getattr(self.config, "telegate", False):
            return cands
        tg = self._telegate_candidates(front_inter, extended, l2p, p2l)
        if not tg:
            return cands
        self._tg_offered += len(tg)
        cands = list(cands) + tg
        cands.sort(key=lambda a: a.score)
        return cands

    def _telegate_candidates(self, front_inter, extended, l2p, p2l):
        arch, cfg = self.arch, self.config
        if cfg.telegate_max_core_dist != 1:
            raise ValueError(
                "telegate_max_core_dist > 1 needs a multi-hop cat state "
                "(k EPR pairs held simultaneously); not implemented here.")
        out = []
        free = {}
        for node in front_inter:
            q1, q2 = node.qargs[0], node.qargs[1]
            p1, p2 = l2p[q1], l2p[q2]
            c1, c2 = arch.core_of(p1), arch.core_of(p2)
            self._tg_front_inter += 1
            if arch.core_dist[c1][c2] != 1:
                continue
            self._tg_front_adj += 1
            gate_dist = arch.phys_dist.get(p1, {}).get(p2, 999)
            # Either operand may carry the cat state.  Note the two sides
            # SCORE IDENTICALLY at telegate_dst_weight = 1: `d_src`'s staging
            # term is min over N(p_comm_src) of d(p_a, .), which is exactly
            # what the opposite assignment contributes as its `d_dst`, and the
            # two eviction costs are common to both.  So `telegate_control_only`
            # changes nothing unless dst_weight != 1 -- the choice of which
            # operand is shared is free in this cost model, which is also why
            # the CX = CZ-up-to-Hadamards rewrite never has to be invoked.
            sides = [(q1, p1, c1, q2, p2, c2)]
            if not cfg.telegate_control_only:
                sides.append((q2, p2, c2, q1, p1, c1))
            for (virt, p_src, src_c, partner, p_par, dst_c) in sides:
                for (p_comm_src, p_comm_dst) in arch.inter_links_between(src_c, dst_c):
                    # Capacity enters ONLY as "can the endpoint be vacated at
                    # all", never as cfg.cap_penalty: a telegate returns both
                    # cores to their entry occupancy.
                    if p2l[p_comm_src] is not None:
                        if src_c not in free:
                            free[src_c] = self._free_slots(src_c, arch, p2l)
                        if free[src_c] == 0:
                            continue
                    if p2l[p_comm_dst] is not None:
                        if dst_c not in free:
                            free[dst_c] = self._free_slots(dst_c, arch, p2l)
                        if free[dst_c] == 0:
                            continue
                    n_s = min(arch.intra[src_c].neighbors(p_comm_src),
                              key=lambda n: self._idist(src_c, p_src, n))
                    d_src = (self._idist(src_c, p_src, n_s)
                             + self._evict_cost(p_comm_src, arch, p2l)
                             + self._evict_cost(p_comm_dst, arch, p2l))
                    d_dst = min(self._idist(dst_c, p_par, nb)
                                for nb in arch.intra[dst_c].neighbors(p_comm_dst))
                    amort = 0.0
                    if cfg.telegate_amort_weight or cfg.telegate_require_no_lookahead:
                        # What a teledata hop of `virt` onto this endpoint
                        # would have bought the extended set -- the future
                        # this telegate declines to buy.
                        dE = self._delta_front(virt, p_comm_dst, extended, l2p,
                                               decay=cfg.lookahead_decay)
                        if (cfg.telegate_require_no_lookahead
                                and dE > cfg.telegate_lookahead_threshold):
                            continue
                        amort = cfg.telegate_amort_weight * max(0.0, dE)
                    score = (d_src + cfg.telegate_dst_weight * d_dst
                             - cfg.telegate_progress * gate_dist
                             + amort + cfg.telegate_bias)
                    out.append(TelegateAction(
                        node, virt, partner, p_src, n_s, p_comm_src, p_comm_dst,
                        src_c, dst_c, d_src, d_dst, gate_dist, score))
        return out

    # -- execution ----------------------------------------------------------

    def _apply_teleport(self, a, l2p, p2l, metrics, partner_phys=None):
        """Dispatch: `route()` calls this for whichever action won the sort."""
        if isinstance(a, TelegateAction):
            return self._apply_telegate(a, l2p, p2l, metrics,
                                        partner_phys=partner_phys)
        return super()._apply_teleport(a, l2p, p2l, metrics,
                                       partner_phys=partner_phys)

    def _walk_to_port_neighbour(self, virt, p_port, core, l2p, p2l, metrics):
        """SWAP `virt` to a neighbour of `p_port` without crossing `p_port`.

        `p_port` must already be free (it holds an EPR half for the duration
        of the action), so no SWAP may push a bystander back onto it.  The
        destination is whichever port neighbour is reachable most cheaply in
        `intra[core] - {p_port}`; on a topology where the port is a cut
        vertex the nearest neighbour in the unrestricted graph can sit in
        another component, which is why every neighbour is tried.

        Returns the final slot, or None if no legal walk exists.
        """
        arch, cfg = self.arch, self.config
        cur = l2p[virt]
        if cur == p_port:
            return None
        nbrs = list(arch.intra[core].neighbors(p_port))
        if cur in nbrs:
            return cur
        restricted = nx.restricted_view(arch.intra[core], [p_port], [])
        best = None
        for nb in nbrs:
            try:
                path = nx.shortest_path(restricted, cur, nb)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if best is None or len(path) < len(best):
                best = path
        if best is None:
            return None
        for i in range(len(best) - 1):
            u, v = best[i], best[i + 1]
            if u == p_port or v == p_port:
                return None
            qu, qv = p2l[u], p2l[v]
            if qu is not None: l2p[qu] = v
            if qv is not None: l2p[qv] = u
            p2l[u], p2l[v] = qv, qu
            metrics["ls"] += 1
            metrics["cost"] += cfg.cost_local_swap
            if cfg.trace_routing:
                metrics["trace"].append(("SWAP", u, v, core))
        return best[-1]

    def _apply_telegate(self, a: TelegateAction, l2p, p2l, metrics,
                        partner_phys=None) -> bool:
        """Atomically execute one telegate and retire its gate.

        Same transactional shape as `_apply_teleport`: every precondition is
        re-checked against the live state, every failure path is a complete
        no-op via `rollback()`, and the DAG retirement is the LAST step --
        `_snapshot` restores layout and counters but cannot un-retire a gate,
        so nothing that can fail may follow it.
        """
        arch, cfg = self.arch, self.config
        rollback = self._snapshot(l2p, p2l, metrics)

        # ── Action validation ─────────────────────────────────────────────
        if a.virt not in l2p or a.partner not in l2p:
            return rollback()
        if arch.core_of(l2p[a.virt]) != a.src_core:
            return rollback()
        if arch.core_of(l2p[a.partner]) != a.next_core:
            return rollback()
        if arch.core_of(a.p_comm_src) != a.src_core:
            return rollback()
        if arch.core_of(a.p_comm_dst) != a.next_core:
            return rollback()
        if (a.p_comm_src, a.p_comm_dst) not in set(
                arch.inter_links_between(a.src_core, a.next_core)):
            return rollback()

        # ── Vacate both link endpoints ────────────────────────────────────
        # Occupancy of each core is preserved: the evictee lands on a free
        # slot of the SAME core, and the cat qubit that transiently occupies
        # p_comm_dst is measured out before this action returns.  So unlike
        # a teleport, no capacity has to be secured in the far core.
        self._evict(a.p_comm_src, a.src_core, l2p, p2l, metrics,
                    partner_phys=partner_phys)
        if p2l[a.p_comm_src] is not None:
            return rollback()
        self._evict(a.p_comm_dst, a.next_core, l2p, p2l, metrics,
                    partner_phys=partner_phys)
        if p2l[a.p_comm_dst] is not None:
            return rollback()

        # ── Stage both operands next to their own endpoint ────────────────
        stage_src = self._walk_to_port_neighbour(a.virt, a.p_comm_src,
                                                 a.src_core, l2p, p2l, metrics)
        if stage_src is None:
            return rollback()
        stage_dst = self._walk_to_port_neighbour(a.partner, a.p_comm_dst,
                                                 a.next_core, l2p, p2l, metrics)
        if stage_dst is None:
            return rollback()

        # ── Pre-telegate validation ───────────────────────────────────────
        if (p2l[a.p_comm_src] is not None
                or p2l[a.p_comm_dst] is not None
                or p2l[stage_src] != a.virt
                or p2l[stage_dst] != a.partner
                or not arch.intra[a.src_core].has_edge(stage_src, a.p_comm_src)
                or not arch.intra[a.next_core].has_edge(stage_dst, a.p_comm_dst)):
            return rollback()

        # ── Commit ────────────────────────────────────────────────────────
        # One EPR pair, exactly as a teledata hop costs one, so the reported
        # EPR column stays a single comparable unit (this is also how
        # \TeleSABRE{}'s teledata + telegate total is formed).  `teles` is
        # deliberately NOT incremented: it counts relocations, and this
        # action relocates nothing.
        metrics["catcomms"] += 1
        metrics["eprs"]     += 1
        metrics["cost"]     += cfg.cost_teleport
        if cfg.trace_routing:
            metrics["trace"].append(
                ("TGATE", a.virt, a.partner, stage_src, a.p_comm_src,
                 a.p_comm_dst, stage_dst, a.src_core, a.next_core))
            metrics["trace"].append(
                ("GATE", a.node.qargs[0], a.node.qargs[1], a.next_core))

        # Layout maps stayed mutually consistent.  Only one direction is
        # needed here, unlike in `_apply_teleport`: every write this action
        # makes goes through `_evict` / `_walk_to_port_neighbour`, which move
        # qubits in pairs, so no slot can be orphaned.  Checked BEFORE the
        # retirement below, which `rollback()` cannot undo.
        for logical, physical in l2p.items():
            if p2l.get(physical) != logical:
                return rollback()

        self._wdag.remove_op_node(a.node)
        return True

    # -- instrumentation ----------------------------------------------------

    # Whole-instance totals.  `route()` is called once per pass by
    # `run_sabre_passes` / `run_protocol_ex`, and only the winning FORWARD
    # pass's metrics are reported -- but a telegate taken in a backward pass
    # still shapes the layout the next forward pass starts from, so the
    # protocol-level counts have to survive the per-pass reset.
    _tg_all_telegates = 0
    _tg_all_teledata = 0
    _tg_all_front_inter = 0
    _tg_all_front_adj = 0

    def route(self, dag, initial_layout):
        self._tg_offered = self._tg_front_inter = self._tg_front_adj = 0
        metrics, layout = super().route(dag, initial_layout)
        metrics["telegates"]      = metrics.get("catcomms", 0)
        metrics["teledata"]       = metrics.get("teles", 0)
        metrics["tg_offered"]     = self._tg_offered
        # How often the adjacency restriction bites: front-layer inter-core
        # gates seen while scoring, and the subset whose two cores are
        # neighbours (the only ones a one-hop telegate can serve).
        metrics["tg_front_inter"] = self._tg_front_inter
        metrics["tg_front_adj"]   = self._tg_front_adj
        self._tg_all_telegates   += metrics["telegates"]
        self._tg_all_teledata    += metrics["teledata"]
        self._tg_all_front_inter += self._tg_front_inter
        self._tg_all_front_adj   += self._tg_front_adj
        return metrics, layout
