"""
dSABRE — Distributed SABRE router for multi-core quantum processors.

The routing loop alternates between two modes:
  - Intra-core: when both qubits of a pending gate are in the same core,
    execute SWAP-based routing using the SABRE heuristic on local gates.
  - Inter-core: when qubits straddle cores, score teleportation candidates
    and execute the best-scoring move (fewest EPR pairs, best lookahead).

Deadlock recovery: if no progress is made for `deadlock_limit` consecutive
iterations, the router restores a checkpoint and executes a forced hop
(backup plan).  After `max_backup_attempts` failed recoveries the route is
marked aborted.

Extended sets.  Two are built, on different axes and with different costs:
`_top_ext` fills the inter-core set in topological order and `_bfs_ext`
(dsabre_ext.py) fills it by BFS layer -- `_inter_ext` is the hook route()
calls, so a subclass selects one by overriding that; `_get_local_extended`
builds the per-core intra sets.

Incremental bookkeeping (2026-08-07).  Costs that used to be Theta(N) on
*every* iteration are now O(1) amortised: remaining in-degrees are carried
across calls instead of rebuilt, the unrouted-gate count is maintained
rather than recounted, checkpoints record a mark in a retirement log
instead of deep-copying the DAG, `_best_intra_swap` scores only the gates a
candidate SWAP can move, and `_get_local_extended` stops once no core can
admit another gate.  None of this changes what the router decides: the
output is bit-identical to the pre-optimisation implementation, frozen as
`_baseline_router.py` and checked by `verify_router.py` across the
25/36/64/360-qubit suites.  The measured effect is 3.5-5.6x less compile
time; see the complexity appendix of the paper.

What safe mode makes dead code.  Under the five hypotheses route() checks
at entry -- r >= 2, connected core graph, connected core coupling graphs,
M >= r+3, and P - n >= r*K + 1 -- every core keeps at least one free slot
at every point of the run, and every scored teleport candidate therefore
executes exactly as scored.  The paper's Appendix C proves this; the
measurement is no rejection in 1.08M scored candidates over four
architectures.  Unreachable in consequence:

  _force_make_room            both call sites in `_apply_teleport` are
                              guarded by free(c) == 0
  `_apply_teleport`'s 18      every `return rollback()` path in it; the
  rollback paths              method itself is of course still required
  _fallback_local_swap        reached only when `_safe_progress` returns
                              False, which requires safe_mode off
  _backup_plan's greedy       the safe branch at its head returns first
  cross-core hops
  _route_gate_transaction     called only after that branch has failed,
                              and with it `_relay_room_to` and
                              `_find_nearest_slack_core`
  _safe_drain                 its two backup-exhaustion callers cannot
                              fire (safe mode raises `max_backups` to the
                              gate count); its ITERATION_LIMIT caller can,
                              but only when `max_iterations` is set
                              explicitly rather than derived from
                              `iterations_bound`
  every metrics["aborted"]    that is the termination-with-success theorem

None of them is replaced by an assertion, so that a violated hypothesis
degrades to the pre-safe-mode behaviour instead of aborting, and so
score-only mode (safe_mode=False), where all of them are live, still
works.  Still required in safe mode: `_snapshot`/`rollback`
(checkpoint-rollback is one of the hypotheses, and deadlock recovery does
fire), `_make_layout_safe`, `_relay_slack_to`, `_find_donor_core`,
`_plan_meeting_core`, `_safe_pick_gate` and `_safe_route_gate`.
"""

import time
from collections import deque
from copy import deepcopy

import networkx as nx

from config import HardwareConfig
from architecture import DistributedArchitecture
from actions import TeleportAction


class _RetiringDAG:
    """Thin proxy over a DAGCircuit that reports gate retirements to the router.

    Every attribute except `remove_op_node` forwards untouched, so helpers
    that only read the DAG (`_front_2q`, `_backup_plan`, `_get_local_extended`)
    operate on it exactly as on the real object.

    Interception is what keeps the maintained in-degrees, the unrouted-gate
    counter, and the retirement log in step with the DAG, wherever the
    retirement happens.
    """

    def __init__(self, dag, router):
        self._dag = dag
        self._router = router

    def __getattr__(self, name):
        return getattr(self._dag, name)

    def remove_op_node(self, node):
        # Successors must be read before the node is spliced out.
        self._router._on_retire(self._dag, node)
        self._dag.remove_op_node(node)


class General_dSABRE_Router:

    def __init__(self, arch: DistributedArchitecture, config: HardwareConfig):
        self.arch = arch
        self.config = config

    # ── Distance helpers ───────────────────────────────────────────────────────

    def _idist(self, core_id: int, u: int, v: int) -> int:
        return self.arch.intra_dist[core_id][u][v]

    def _ipath(self, core_id: int, u: int, v: int) -> list:
        return self.arch.intra_path[core_id][u][v]

    def _free_slots(self, core_id, arch, p2l) -> int:
        return sum(1 for p in arch.core_qubits(core_id) if p2l[p] is None)

    def _evict_cost(self, p_comm, arch, p2l) -> int:
        """SWAPs needed to free p_comm (0 if already empty)."""
        if p2l[p_comm] is None:
            return 0
        ci = arch.core_of(p_comm)
        return min(
            (self._idist(ci, p_comm, pq)
             for pq in arch.core_qubits(ci) if p2l[pq] is None),
            default=999,
        )

    def _gate_dist(self, node, l2p) -> int:
        p1, p2 = l2p[node.qargs[0]], l2p[node.qargs[1]]
        return self.arch.phys_dist.get(p1, {}).get(p2, 999)

    # ── Lookahead scoring ──────────────────────────────────────────────────────

    def _delta_front(self, virt, new_phys, gates, l2p, decay=1.0) -> float:
        """Sum of distance reductions for gates involving `virt` if it moves to new_phys.

        Each gate is discounted by decay^depth, where depth is its DAG distance
        from the front layer (stored as the second element of each (gate, depth)
        pair in `gates`).  decay=1.0 gives a flat sum.
        """
        arch = self.arch
        old_phys = l2p[virt]
        old_core = arch.core_of(old_phys)
        new_core = arch.core_of(new_phys)
        delta = 0.0
        for g, depth in gates:
            if len(g.qargs) < 2:
                continue
            q1, q2 = g.qargs[0], g.qargs[1]
            if virt not in (q1, q2):
                continue
            partner = l2p[q2 if virt == q1 else q1]
            if partner is None:
                continue
            p_core = arch.core_of(partner)
            if old_core == p_core:
                old_dist = self._idist(old_core, old_phys, partner)
            else:
                old_dist = arch.phys_dist.get(old_phys, {}).get(partner, 999)
                if old_dist == 999: continue
            if new_core == p_core:
                new_dist = self._idist(new_core, new_phys, partner)
            else:
                new_dist = arch.phys_dist.get(new_phys, {}).get(partner, 999)
                if new_dist == 999: continue
            delta += (decay ** depth) * (old_dist - new_dist)
        return delta

    # ── Candidate generation ───────────────────────────────────────────────────

    def _inter_ext(self, dag, front, size):
        """Inter-core extended set, memoised on the working DAG's generation.

        Subclasses select a construction by overriding `_build_inter_ext`.
        The default is the topological fill of `_top_ext`; `dSABRE_BFSExt`
        swaps in the BFS-layer fill of `_bfs_ext`.

        `E` depends on the DAG alone, never on the layout, so it survives
        every SWAP and every teleport and needs rebuilding only when a gate
        leaves the working DAG.  `_dag_gen` counts those events; keying on it
        rather than on the retirement count matters because a rollback can
        return the log to a length it already held with different contents.
        Since 2026-08-08 the intra-core scorer reads this set too
        (`_get_local_extended`), so both scorers share one construction and
        one memo.  Callers must treat the returned list as read-only.
        """
        key = (self._dag_gen, size)
        if key != self._ext_key:
            self._ext_val = self._build_inter_ext(dag, front, size)
            self._ext_key = key
        return self._ext_val

    def _build_inter_ext(self, dag, front, size):
        """The construction `_inter_ext` memoises. Override this, not it."""
        return self._top_ext(dag, front, size)

    def _top_ext(self, dag, front, size):
        """Topological fill ("top-ext"): up to `size` (gate, depth) pairs.

        depth is the gate's DAG distance from the front layer (front = depth 0;
        immediate successors = depth 1).  Depths come from a single topological
        pass tracking the longest predecessor chain.  Interleaves all wires
        uniformly, so a deep chain on one wire can crowd out a shallow gate on
        another -- which is what `_bfs_ext` exists to avoid.
        """
        front_nids = {n._node_id for n in front}
        # depth[nid] = DAG distance from the nearest front-layer ancestor
        node_depth: dict[int, int] = {n._node_id: 0 for n in front}
        ext = []
        for n in dag.topological_op_nodes():
            nid = n._node_id
            if nid not in node_depth:
                d = max(
                    (node_depth[p._node_id] for p in dag.predecessors(n)
                     if getattr(p, '_node_id', None) in node_depth),
                    default=None,
                )
                if d is None:
                    continue  # not reachable from front
                node_depth[nid] = d + 1
            if nid in front_nids:
                continue
            if len(n.qargs) == 2:
                ext.append((n, node_depth[nid]))
                if len(ext) >= size:
                    break
        return ext

    def _generate_candidates(self, front_inter, extended, l2p, p2l, committed_nid=None):
        """Score all feasible one-hop teleportation moves and return sorted list.

        Scoring formula for each candidate:
            score = d_prep + cap - dF - weight_extended * dE
                    - commit_bonus + priority_penalty

        where:
          d_prep    = intra-core SWAPs to reach the comm port + eviction cost
          cap       = penalty if destination core is nearly full
          dF        = distance reduction on front-layer gates (immediate reward)
          dE        = distance reduction on extended-layer gates (lookahead)
          commit_bonus = cfg.commit_bonus if this candidate continues the gate
                      (by DAG node id) that the PREVIOUS iteration teleported,
                      else 0 -- see the config field's docstring.
          priority_penalty = cfg.cheapest_first_weight * the gate's OWN current
                      qubit-pair phys_dist -- 0 when the weight is 0 (default);
                      otherwise gates already close to resolution outscore
                      farther ones regardless of commit state.

        Lower score → better move.  The top candidate is executed each iteration.
        """
        arch = self.arch
        cfg  = self.config
        # Capacity as a legality condition (safe mode, the default): a
        # teleport into `next_c` is legal only if it leaves the destination
        # at or above `tier1_floor - 1`.  Score-only mode
        # (`safe_mode=False`) keeps the pre-2026-08-09 ">= 1" rule.
        min_dst_free = self._tier1_floor() if cfg.safe_mode else 1

        def score_nodes(node_list, free_cache):
            out = []
            for node in node_list:
                q1, q2 = node.qargs[0], node.qargs[1]
                p1, p2 = l2p[q1], l2p[q2]
                c1, c2 = arch.core_of(p1), arch.core_of(p2)
                commit_bonus = (cfg.commit_bonus
                                if committed_nid is not None and node._node_id == committed_nid
                                else 0.0)
                priority_penalty = (cfg.cheapest_first_weight
                                    * arch.phys_dist.get(p1, {}).get(p2, 999))
                # Front-layer gates are qubit-disjoint (each wire contributes
                # at most one), so `virt` below can only ever match `node`'s
                # own qargs -- passing the rest of front_inter to _delta_front
                # for dF would always find nothing else there, just at
                # O(|front_inter|) cost instead of O(1).
                dF_gates = [(node, 0)]

                for (virt, p_src, src_c, tgt_c) in [(q1, p1, c1, c2), (q2, p2, c2, c1)]:
                    for next_c in arch.core_graph.neighbors(src_c):
                        if free_cache.get(next_c, 0) < min_dst_free:
                            continue
                        for (p_comm_src, p_comm_dst) in arch.inter_links_between(src_c, next_c):
                            n_s = min(
                                arch.intra[src_c].neighbors(p_comm_src),
                                key=lambda n: self._idist(src_c, p_src, n),
                            )
                            d_prep   = (self._idist(src_c, p_src, n_s)
                                        + self._evict_cost(p_comm_src, arch, p2l)
                                        + self._evict_cost(p_comm_dst, arch, p2l))
                            cap      = cfg.cap_penalty * max(0, cfg.capacity_threshold - free_cache[next_c])
                            dF = self._delta_front(virt, p_comm_dst, dF_gates, l2p)
                            dE = self._delta_front(virt, p_comm_dst, extended, l2p,
                                                   decay=cfg.lookahead_decay)
                            score = (d_prep + cap - dF - cfg.weight_extended * dE
                                    - commit_bonus + priority_penalty)
                            out.append(
                                TeleportAction(node, virt, p_src, n_s,
                                              p_comm_src, p_comm_dst, src_c, next_c, tgt_c, score)
                            )
            return out

        if committed_nid is not None and cfg.commit_hard_lock:
            locked = [n for n in front_inter if n._node_id == committed_nid]
            if locked:
                # Fast path: only the locked gate's own two qubits can act, so
                # only their neighbouring cores' free-slot counts are ever
                # looked up -- skip the full free_cache scan over every core
                # in the chip (matters on architectures with many cores).
                node = locked[0]
                c1 = arch.core_of(l2p[node.qargs[0]])
                c2 = arch.core_of(l2p[node.qargs[1]])
                relevant = (set(arch.core_graph.neighbors(c1))
                           | set(arch.core_graph.neighbors(c2)))
                local_free_cache = {c: self._free_slots(c, arch, p2l) for c in relevant}
                candidates = score_nodes(locked, local_free_cache)
                if candidates:
                    candidates.sort(key=lambda a: a.score)
                    return candidates
                # Locked gate has no legal move this iteration: release the
                # lock for this iteration only and fall through to full scoring.

        free_cache = {c: self._free_slots(c, arch, p2l) for c in range(arch.num_cores)}
        candidates = score_nodes(front_inter, free_cache)
        candidates.sort(key=lambda a: a.score)
        return candidates

    # ── Incremental DAG state ──────────────────────────────────────────────────
    #
    # `_dag_gen` counts working-DAG mutations; `_ext_key`/`_ext_val` are the
    # `_inter_ext` memo keyed on it.  Declared here so an instance is valid
    # before `route()` seeds them.
    _dag_gen = 0
    _ext_key = None
    _ext_val = None
    #
    # Three quantities that used to be recomputed from the whole DAG on every
    # iteration.  All are seeded in route() and maintained by `_on_retire`,
    # which `_RetiringDAG` calls on every removal.  Correctness rests on gates
    # leaving the DAG only from the front layer -- true of the drain loop,
    # `_backup_plan`, and `_route_gate_transaction` -- which makes the retired
    # set downward closed.  A surviving node therefore has no retired
    # successors, and `remove_op_node`'s splicing never creates an edge between
    # two survivors, so decrementing a retiring node's successors keeps
    # `_indeg` exactly equal to a from-scratch rebuild on the reduced DAG.

    def _init_dag_state(self, real_dag):
        """Seed maintained in-degrees, the unrouted-gate count, and the log."""
        self._bump_dag_gen()
        self._indeg = {
            n._node_id: sum(1 for p in real_dag.predecessors(n)
                            if getattr(p, 'qargs', None))
            for n in real_dag.topological_op_nodes()
        }
        self._remaining = len(real_dag.op_nodes())
        self._retired_log = []

    def _bump_dag_gen(self):
        """Invalidate the `_inter_ext` memo: the working DAG changed."""
        self._dag_gen = getattr(self, "_dag_gen", 0) + 1
        self._ext_key = None
        self._ext_val = None

    def _on_retire(self, real_dag, node):
        """Called by the proxy just before `node` leaves the DAG."""
        self._bump_dag_gen()
        self._remaining -= 1
        self._retired_log.append(node._node_id)
        for succ in real_dag.successors(node):
            sid = getattr(succ, '_node_id', None)
            if sid is not None and getattr(succ, 'qargs', None):
                self._indeg[sid] -= 1

    def _rebuild_wdag(self, dag, mark):
        """Reconstruct the working DAG as of retirement `mark`.

        Replaces `deepcopy(ckpt_wdag)`: replays the first `mark` retirements,
        in their original order, onto a fresh copy of the untouched input.
        Checkpointing is then O(1) and only a rollback pays O(N) -- the right
        trade, since checkpoints outnumber rollbacks by orders of magnitude.
        """
        self._n_wdag_rebuilds += 1
        self._bump_dag_gen()
        _t0 = time.perf_counter()
        fresh = deepcopy(dag)
        by_id = {n._node_id: n for n in fresh.op_nodes()}
        for nid in self._retired_log[:mark]:
            victim = by_id.get(nid)
            if victim is not None:
                fresh.remove_op_node(victim)
        del self._retired_log[mark:]
        self._indeg = {
            n._node_id: sum(1 for p in fresh.predecessors(n)
                            if getattr(p, 'qargs', None))
            for n in fresh.topological_op_nodes()
        }
        self._remaining = len(fresh.op_nodes())
        self._t_wdag_rebuild += time.perf_counter() - _t0
        return _RetiringDAG(fresh, self)

    # ── Transactions ───────────────────────────────────────────────────────────

    def _snapshot(self, l2p, p2l, metrics):
        """Capture the routing state; return a `rollback()` closure.

        `rollback()` restores both layout maps and every scalar counter, drops
        any trace entries appended since, and returns False -- so a failed
        precondition is just `return rollback()`.  The trace is truncated
        rather than copied, keeping a snapshot O(qubits) however long the trace
        has grown.  Snapshots nest: an inner rollback restores the inner entry
        state, leaving an enclosing one free to restore its own later.
        """
        # Transaction counters live on the router, not in `metrics`: a rollback
        # restores every scalar metric, so a counter kept there would undo its
        # own increment and always read 0.
        self._n_snapshots += 1
        _t0 = time.perf_counter() if self.config.profile_transactions else 0.0
        saved_l2p = l2p.copy()
        saved_p2l = p2l.copy()
        # Scalars only: "trace" (list) is handled by truncation below, and no
        # dict-valued metric is mutated inside a transaction.
        saved_counters  = {k: v for k, v in metrics.items()
                           if not isinstance(v, (list, dict))}
        saved_trace_len = (len(metrics["trace"])
                           if metrics.get("trace") is not None else 0)
        if self.config.profile_transactions:
            self._t_snapshot += time.perf_counter() - _t0

        def rollback() -> bool:
            self._n_rollbacks += 1
            _t1 = time.perf_counter() if self.config.profile_transactions else 0.0
            l2p.clear();  l2p.update(saved_l2p)
            p2l.clear();  p2l.update(saved_p2l)
            metrics.update(saved_counters)
            if metrics.get("trace") is not None:
                del metrics["trace"][saved_trace_len:]
            if self.config.profile_transactions:
                self._t_rollback += time.perf_counter() - _t1
            return False

        return rollback

    # ── Physical operations ────────────────────────────────────────────────────

    def _evict(self, p_comm, core_id, l2p, p2l, metrics, partner_phys=None):
        """Move the qubit occupying p_comm to a free slot (if any).

        Default target: the nearest free slot to p_comm (minimises the SWAP
        chain below). If `config.evict_distance_aware` and the evicted qubit
        has a pending front-layer gate (`partner_phys` maps it to that gate's
        partner's physical position), a free slot that would INCREASE the
        evicted qubit's own phys_dist to its partner is deprioritised first;
        nearest-to-p_comm still breaks ties among slots that don't worsen it.
        See `config.HardwareConfig.evict_distance_aware`'s docstring.
        """
        if p2l[p_comm] is None:
            return
        free_slots = [pq for pq in self.arch.core_qubits(core_id) if p2l[pq] is None]
        partner = (partner_phys.get(p2l[p_comm])
                  if self.config.evict_distance_aware and partner_phys else None)
        if partner is not None:
            old_dist = self.arch.phys_dist.get(p_comm, {}).get(partner, 999)
            free_dst = min(
                free_slots,
                key=lambda pq: (self.arch.phys_dist.get(pq, {}).get(partner, 999) > old_dist,
                                self._idist(core_id, p_comm, pq)),
                default=None,
            )
        else:
            free_dst = min(free_slots, key=lambda pq: self._idist(core_id, p_comm, pq),
                           default=None)
        if free_dst is not None:
            path = self._ipath(core_id, p_comm, free_dst)
            for i in reversed(range(len(path) - 1)):
                a, b = path[i], path[i + 1]
                qa, qb = p2l[a], p2l[b]
                if qa is not None: l2p[qa] = b
                if qb is not None: l2p[qb] = a
                p2l[a], p2l[b] = qb, qa
                metrics["ls"]   += 1
                metrics["cost"] += self.config.cost_local_swap
                if self.config.trace_routing:
                    metrics["trace"].append(("SWAP", a, b, core_id))

    def _apply_teleport(self, a: TeleportAction, l2p, p2l, metrics, partner_phys=None) -> bool:
        """Atomically execute one teleport macro-action.

        Steps: secure a free slot in both cores, evict both comm ports, SWAP
        `a.virt` to the staging slot without touching the source port, then
        teleport it onto the destination port.  Every step is checked; on any
        failure the layout, counters, and trace are restored to their state
        at entry and False is returned (the action is a complete no-op).

        In safe mode none of those failure paths can fire: under the entry
        hypotheses they are assertions, not branches the router takes (see
        the module docstring, and the proposition in the paper's Appendix
        C).  They divide into the six checks that restate how the candidate
        was built -- exact because a rejected attempt restores the state its
        whole list was scored in, so attempt k begins where attempt 1 did --
        the three capacity checks, excluded by every core keeping a free
        slot, the three staging checks, excluded by intra-core connectivity
        plus the port-neighbour retry below, the final port/staging
        validation, and the two map-consistency postconditions.  In
        score-only mode they are live and do the work they look like they do.

        `partner_phys`, if given, maps a logical qubit with a pending
        front-layer gate to that gate's partner's physical position -- passed
        through to `_evict`/`_force_make_room` for distance-aware eviction.
        """
        arch = self.arch
        cfg  = self.config
        rollback = self._snapshot(l2p, p2l, metrics)

        # ── Action validation ─────────────────────────────────────────────
        # `inter_links_between` returns [] for non-adjacent cores, so the
        # link-membership check also enforces core adjacency.
        if a.virt not in l2p:
            return rollback()
        if arch.core_of(l2p[a.virt]) != a.src_core:
            return rollback()
        if arch.core_of(a.p_comm_src) != a.src_core:
            return rollback()
        if arch.core_of(a.p_comm_dst) != a.next_core:
            return rollback()
        if (a.p_comm_src, a.p_comm_dst) not in set(
                arch.inter_links_between(a.src_core, a.next_core)):
            return rollback()
        if (a.n_s not in arch.intra[a.src_core]
                or not arch.intra[a.src_core].has_edge(a.n_s, a.p_comm_src)):
            return rollback()

        # ── Secure capacity ───────────────────────────────────────────────
        # A full core leaves _evict below nowhere to swap the port occupant
        # to; relieve it with one real (legal) teleport hop first.  `a.virt`
        # itself must stay put (exclude_virt), or the staging-path lookup
        # below would operate on a qubit that already left this core.
        if (self._free_slots(a.src_core, arch, p2l) == 0
                and not self._force_make_room(a.src_core, l2p, p2l, metrics,
                                              exclude_virt=a.virt,
                                              partner_phys=partner_phys)):
            return rollback()
        if (self._free_slots(a.next_core, arch, p2l) == 0
                and not self._force_make_room(a.next_core, l2p, p2l, metrics,
                                              exclude_virt=a.virt,
                                              partner_phys=partner_phys)):
            return rollback()
        # Relieving one core can consume the other's vacancy: the relief hop
        # lands its evacuee in any neighbour with a free slot, including the
        # other core of this very move.  Neither port may be treated as
        # reserved unless both cores hold a vacancy NOW.
        if (self._free_slots(a.src_core, arch, p2l) == 0
                or self._free_slots(a.next_core, arch, p2l) == 0):
            return rollback()
        # Room-making moves other qubits, but must not have moved this
        # move's own qubit out of the source core.
        if a.virt not in l2p or arch.core_of(l2p[a.virt]) != a.src_core:
            return rollback()

        # ── Evict and reserve both comm ports ─────────────────────────────
        # _evict has no return value; verify success from p2l directly.
        self._evict(a.p_comm_src, a.src_core,  l2p, p2l, metrics, partner_phys=partner_phys)
        if p2l[a.p_comm_src] is not None:
            return rollback()
        self._evict(a.p_comm_dst, a.next_core, l2p, p2l, metrics, partner_phys=partner_phys)
        if p2l[a.p_comm_dst] is not None:
            return rollback()

        # ── Staging path that cannot reoccupy the source port ─────────────
        # Both comm ports must be free at teleport time (EPR halves).  A path
        # through the source port would push a bystander qubit back onto the
        # just-evicted port on its final SWAP, so when the port is a cut
        # vertex and no avoiding path exists, the candidate is not executable
        # -- fail it rather than fall back to the port-crossing path.
        current_phys = l2p[a.virt]
        if current_phys == a.p_comm_src:
            return rollback()
        if current_phys == a.n_s:
            staging_path = [current_phys]
        else:
            restricted = nx.restricted_view(arch.intra[a.src_core],
                                            [a.p_comm_src], [])
            try:
                staging_path = nx.shortest_path(restricted, current_phys, a.n_s)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # `a.n_s` was picked by the caller as the port neighbour nearest
                # `virt` in the UNRESTRICTED core graph, so on a topology whose
                # port is a cut vertex it can sit in a different component of
                # `intra - {port}` than `virt` does.  Some port neighbour always
                # shares virt's component (every component of `intra - {port}`
                # touches the port, since `intra` is connected), so retry over
                # the reachable ones instead of failing the whole action.  Grid
                # cores have no articulation points, so this never fires there
                # -- hence the safe_mode gate, which keeps score-only output
                # bit-identical.
                staging_path = None
                if cfg.safe_mode:
                    best = None
                    for nb in arch.intra[a.src_core].neighbors(a.p_comm_src):
                        if nb == current_phys:
                            best = [current_phys]
                            break
                        try:
                            p = nx.shortest_path(restricted, current_phys, nb)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
                        if best is None or len(p) < len(best):
                            best = p
                    staging_path = best
                if staging_path is None:
                    return rollback()

        for i in range(len(staging_path) - 1):
            u, v = staging_path[i], staging_path[i + 1]
            if u == a.p_comm_src or v == a.p_comm_src:
                return rollback()
            qu, qv = p2l[u], p2l[v]
            if qu is not None: l2p[qu] = v
            if qv is not None: l2p[qv] = u
            p2l[u], p2l[v] = qv, qu
            metrics["ls"]   += 1
            metrics["cost"] += cfg.cost_local_swap
            if cfg.trace_routing:
                metrics["trace"].append(("SWAP", u, v, a.src_core))

        # ── Pre-teleport validation ───────────────────────────────────────
        # Both ports free and the staging slot holds exactly a.virt -- the
        # teleport must never move a bystander (or nothing) while recording
        # a.virt in the trace.  `staging_slot` is a.n_s except on the
        # cut-vertex retry above, which may land on a different port neighbour.
        staging_slot = staging_path[-1]
        if (p2l[a.p_comm_src] is not None
                or p2l[a.p_comm_dst] is not None
                or p2l[staging_slot] != a.virt
                or l2p.get(a.virt) != staging_slot
                or not arch.intra[a.src_core].has_edge(staging_slot,
                                                       a.p_comm_src)):
            return rollback()

        # ── Commit ────────────────────────────────────────────────────────
        p2l[staging_slot] = None
        p2l[a.p_comm_dst] = a.virt
        l2p[a.virt] = a.p_comm_dst
        hop_cost = (cfg.cost_teleport
                    + cfg.cost_teleport_per_hop
                    * arch.core_dist[a.src_core][a.next_core])
        metrics["teles"] += 1
        metrics["eprs"]  += 1
        metrics["cost"]  += hop_cost
        if cfg.trace_routing:
            # Record the slot virt actually teleports from -- not the stale
            # a.p_src captured at scoring time.
            metrics["trace"].append(
                ("TELE", a.virt, staging_slot, a.p_comm_dst,
                 a.src_core, a.next_core)
            )

        # ── Postcondition: both layout maps stayed mutually consistent ────
        for logical, physical in l2p.items():
            if p2l.get(physical) != logical:
                return rollback()
        for physical, logical in p2l.items():
            if logical is not None and l2p.get(logical) != physical:
                return rollback()
        return True

    # ── Intra-core helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _front_2q(dag):
        return [n for n in dag.front_layer() if len(n.qargs) == 2]

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        """Per-core intra-core lookahead set `E_c`.

        `E_c` is the inter-core extended set `E` restricted to core `c`: the
        gates of `E` with both operands resident in that core, carrying `E`'s
        own depths.  One construction therefore feeds both scorers, and
        `dep(g)` means the same thing in each.

        This replaced a separate taint-propagated sweep on 2026-08-08 (kept as
        `_taint_local_extended`, and still selectable with
        `HardwareConfig(local_ext_mode="taint")`).  Measured over the 25/36/64,
        heavy-hex ring/star and 100/200/360-qubit suites -- 38 circuits where
        both complete -- the shared set costs $4.8\\%$ fewer EPR pairs, runs
        $7.1\\times$ faster, and aborts less often than the sweep it replaces.

        Cost is `O(L)` given `E`, which `_inter_ext` memoises on the working
        DAG's generation, so a run of iterations that retires nothing builds
        it once.  The sweep it replaces was `Theta(N_r)` per call.
        """
        if self.config.local_ext_mode == "taint":
            return self._taint_local_extended(wdag, front, core_ids, l2p)

        arch = self.arch
        shared = self._inter_ext(wdag, front, self.config.lookahead_size)
        ext = {ci: [] for ci in core_ids}
        for g, dep in shared:
            c1 = arch.core_of(l2p[g.qargs[0]])
            if c1 in ext and c1 == arch.core_of(l2p[g.qargs[1]]):
                ext[c1].append((g, dep))
        return ext

    def _taint_local_extended(self, wdag, front, core_ids, l2p):
        """Single-pass intra-core lookahead for a collection of cores.

        The pre-2026-08-08 construction, superseded as the default by
        `_get_local_extended` above but reachable through
        `HardwareConfig(local_ext_mode="taint")`.  Every number published
        before that date was produced with it, and `verify_router.py` runs in
        this mode so its diff against `_baseline_router.py` stays meaningful.

        Returns dict mapping core_id -> list of (gate, depth) pairs
        (up to lookahead_size each).  depth is the gate's DAG distance from
        the front layer, tracked per qubit as the number of intra-core gates
        on that qubit's path since the front.  Tainted-qubit propagation is
        shared across all cores in one traversal.

        The scan stops once no core can admit another gate.  That exit is
        exact rather than a cutoff: `l2p` is fixed for the duration of the
        call, so each qubit's core is fixed and `tainted` only grows; a gate
        joins core c only with two untainted operands in c, so once every
        still-requested core holds fewer than two untainted qubits, no later
        node in the order can be admitted.  `live[c]` counts those qubits.
        Without it the scan runs to the end of the topological order on
        72-100% of calls, because taint propagation frequently makes a core's
        size-L quota unreachable.  Cores with no pending gate are still
        counted in `live`, which can only delay the exit, never advance it.
        """
        arch = self.arch
        tainted = set()
        front_ids = {id(n) for n in front}
        ext = {ci: [] for ci in core_ids}
        remaining = set(core_ids)
        # qubit_depth[q] = depth of the last gate seen on qubit q (0 for front)
        qubit_depth: dict = {q: 0 for n in front for q in n.qargs}

        live = {ci: 0 for ci in remaining}
        for q, p in l2p.items():
            c = arch.core_of(p)
            if c in live:
                live[c] += 1
        eligible = {c for c in remaining if live[c] >= 2}

        for n in wdag.topological_op_nodes():
            if not eligible:
                break
            if len(n.qargs) < 2:
                continue
            q1, q2 = n.qargs[0], n.qargs[1]
            c1, c2 = arch.core_of(l2p[q1]), arch.core_of(l2p[q2])
            # A cross-core gate and a gate on an already-tainted wire are
            # handled identically: taint both operands and move on.
            if c1 != c2 or q1 in tainted or q2 in tainted:
                for q in (q1, q2):
                    if q not in tainted:
                        tainted.add(q)
                        c = arch.core_of(l2p[q])
                        if c in live:
                            live[c] -= 1
                            if live[c] < 2:
                                eligible.discard(c)
                continue
            if id(n) in front_ids:
                continue
            if c1 in remaining:
                depth = max(qubit_depth.get(q1, 0), qubit_depth.get(q2, 0)) + 1
                ext[c1].append((n, depth))
                qubit_depth[q1] = qubit_depth[q2] = depth
                if len(ext[c1]) >= self.config.lookahead_size:
                    remaining.discard(c1)
                    eligible.discard(c1)
        return ext

    def _fallback_local_swap(self, front_inter, wdag, l2p, p2l, metrics):
        """SABRE-style SWAP when no teleportation candidates are available.

        Not required in safe mode: both call sites are
        `_safe_progress(...) or _fallback_local_swap(...)`, and with
        `front_inter` non-empty `_safe_progress` returns False only when
        safe mode is off (module docstring).
        """
        arch = self.arch
        involved = {l2p[n.qargs[0]] for n in front_inter} | {l2p[n.qargs[1]] for n in front_inter}

        core_ids = {arch.core_of(u) for u in involved}
        local_ext_cache = self._get_local_extended(wdag, front_inter, core_ids, l2p)

        best_swap, min_H = None, float("inf")
        for u in involved:
            ci = arch.core_of(u)
            local_ext = local_ext_cache[ci]
            comm_ports = arch._core_comm_ports[ci]
            if not comm_ports:
                continue
            for v in arch.intra[ci].neighbors(u):
                d_before = min(self._idist(ci, u, cp) for cp in comm_ports)
                d_after  = min(self._idist(ci, v, cp) for cp in comm_ports)
                delta_Hf = d_after - d_before
                delta_He = sum(
                    (self.config.lookahead_decay ** depth) * (
                        self._idist(ci,
                                    v if l2p[n.qargs[0]] == u else (u if l2p[n.qargs[0]] == v else l2p[n.qargs[0]]),
                                    v if l2p[n.qargs[1]] == u else (u if l2p[n.qargs[1]] == v else l2p[n.qargs[1]]))
                        - self._idist(ci, l2p[n.qargs[0]], l2p[n.qargs[1]])
                    )
                    for n, depth in local_ext
                )
                H = delta_Hf + self.config.weight_extended * (delta_He / max(len(local_ext), 1))
                if H < min_H:
                    min_H, best_swap = H, (u, v, ci)

        if not best_swap:
            return False
        u, v, ci = best_swap
        qu, qv = p2l[u], p2l[v]
        if qu is not None: l2p[qu] = v
        if qv is not None: l2p[qv] = u
        p2l[u], p2l[v] = qv, qu
        metrics["ls"]   += 1
        metrics["cost"] += self.config.cost_local_swap
        if self.config.trace_routing:
            metrics["trace"].append(("SWAP", u, v, ci))
        return True

    # ── Deadlock recovery ──────────────────────────────────────────────────────

    def _force_make_room(self, core_id, l2p, p2l, metrics, exclude_virt=None, partner_phys=None):
        """Teleport a qubit out of a full core to free one slot.

        Not required in safe mode.  Its two callers in `_apply_teleport` are
        guarded by `free(core) == 0`, and no core ever reaches zero once the
        legality floor is enforced; its third caller is in `_backup_plan`'s
        greedy branch, which safe mode does not reach either (module
        docstring).  It is the only step inside a macro-action that can
        spend an EPR pair besides the teleport itself, which is why safe
        mode's ordinary iterations cost exactly one.

        `exclude_virt`, when given, must not be the qubit relieved: used by
        `_apply_teleport` when it calls this mid-move, so the relief hop
        can't sweep away the very qubit it is already routing. `partner_phys`
        is forwarded to `_evict` for distance-aware eviction.
        """
        metrics["force_make_room"] += 1
        arch = self.arch
        if self._free_slots(core_id, arch, p2l) > 0:
            return True
        outgoing = (
            [(u, v) for u, v in arch.inter_core_links
             if arch.core_of(u) == core_id and self._free_slots(arch.core_of(v), arch, p2l) > 0
             and p2l[u] != exclude_virt]
            + [(v, u) for u, v in arch.inter_core_links
               if arch.core_of(v) == core_id and self._free_slots(arch.core_of(u), arch, p2l) > 0
               and p2l[v] != exclude_virt]
        )
        if not outgoing:
            return False
        cp_out, cp_in = outgoing[0]
        neighbor_core = arch.core_of(cp_in)
        self._evict(cp_in, neighbor_core, l2p, p2l, metrics, partner_phys=partner_phys)
        # The core is full (the free-slot early exit did not fire), so the
        # out-port occupant is itself the cheapest evacuee and no staging
        # slot can be cleared for it: teleport it straight off the port.
        # A qubit sitting on the source port is the one placement the
        # teleport protocol still allows when that port is not free.
        virt = p2l[cp_out]
        p2l[cp_out] = None
        if virt is not None: l2p[virt] = cp_in
        p2l[cp_in] = virt
        metrics["teles"] += 1
        metrics["eprs"]  += 1
        metrics["cost"]  += self.config.cost_teleport
        if self.config.trace_routing:
            metrics["trace"].append(
                ("TELE", virt, cp_out, cp_in, core_id, neighbor_core)
            )
        return True

    # ── Invariant-preserving relay (2026-08-03) ────────────────────────────────
    #
    # Precondition (*): every core has >=1 free physical qubit, and the total
    # free count across the chip is >= num_cores + 1.  By pigeonhole this
    # guarantees at least one core has >=2 free ("slack") at all times, since
    # the total free count is exactly conserved by every teleport and SWAP
    # (neither creates nor destroys a qubit or a slot) -- so if (*) held when
    # backup_plan was first invoked, it still holds now, regardless of what
    # ordinary routing or earlier backup_plan calls have done since.
    #
    # Given that, routing ANY remote gate to a shared core WITHOUT ever letting
    # a core's free count touch 0 reduces to: before teleporting a qubit INTO
    # core X, make sure X currently has >=2 free (so it still has >=1 after).
    # If X doesn't, relay the chip's slack to X one hop at a time along a
    # shortest core-graph path, by teleporting an arbitrary (non-gate) qubit
    # OUT of each core along that path and INTO its slack-holding predecessor.
    # Termination: the relay-search is a BFS over a finite connected graph and
    # always finds a slack core under (*); each payload hop strictly reduces
    # the gate's core-distance, so the whole procedure finishes in a bounded
    # number of teleports.  This is what `_force_make_room` approximates for a
    # single hop (and gives up beyond direct neighbours) and what the old
    # greedy `_backup_plan` cross-core loop did not attempt at all -- it only
    # ever checked "does next_core have >=1 free", which can still drop a core
    # to exactly 0.

    def _find_nearest_slack_core(self, start_core, p2l, min_free=2):
        """BFS outward from `start_core` over the core graph for the nearest
        core (possibly `start_core` itself) with >= `min_free` free physical
        qubits.  Returns None only if no such core exists anywhere on the
        chip (i.e. precondition (*) does not actually hold).

        Not required in safe mode: its only caller is `_relay_room_to`
        (module docstring).  The safe-mode transaction relays through
        `_relay_slack_to`/`_find_donor_core` instead, which apply the
        reserve-aware donor threshold this one does not."""
        arch = self.arch
        if self._free_slots(start_core, arch, p2l) >= min_free:
            return start_core
        visited = {start_core}
        queue = deque([start_core])
        while queue:
            c = queue.popleft()
            for nb in arch.core_graph.neighbors(c):
                if nb in visited:
                    continue
                visited.add(nb)
                if self._free_slots(nb, arch, p2l) >= min_free:
                    return nb
                queue.append(nb)
        return None

    def _relay_room_to(self, target_core, l2p, p2l, metrics, min_free=2, protect=()):
        """Ensure `target_core` has >= `min_free` free slots by relaying the
        chip's slack there one hop at a time, never dropping any core below
        1 free.

        Not required in safe mode: its callers are `_route_gate_transaction`
        and `_backup_plan`'s relay branch, neither of which safe mode reaches
        (module docstring).  `protect` qubits (the gate's own two qubits) are never moved
        as filler.

        The relay is a single transaction: a chain that gets partway and then
        stalls (no slack core, no eligible filler, or a hop its own
        preconditions reject) rolls the whole chain back and returns False,
        rather than leaving the chip rearranged by teleports that bought
        nothing.  A caller treats False as backup_plan failing this attempt.
        """
        arch = self.arch
        if self._free_slots(target_core, arch, p2l) >= min_free:
            return True
        rollback = self._snapshot(l2p, p2l, metrics)
        source = self._find_nearest_slack_core(target_core, p2l, min_free)
        if source is None:
            return rollback()
        path = arch.core_path[source][target_core]
        for i in range(len(path) - 1):
            cur, nxt = path[i], path[i + 1]
            # Slack currently sits at `cur`; pull an arbitrary non-protected
            # qubit out of `nxt` into `cur`, handing the slack to `nxt`.
            filler = next(
                (p2l[p] for p in arch.core_qubits(nxt)
                 if p2l[p] is not None and p2l[p] not in protect),
                None,
            )
            if filler is None:
                return rollback()
            links = arch.inter_links_between(nxt, cur)
            if not links:
                return rollback()
            p_comm_src, p_comm_dst = links[0]
            p_src = l2p[filler]
            n_s = min(arch.intra[nxt].neighbors(p_comm_src),
                      key=lambda n: self._idist(nxt, p_src, n))
            if not self._apply_teleport(
                    TeleportAction(None, filler, p_src, n_s, p_comm_src, p_comm_dst,
                                  nxt, cur, cur, score=0.0),
                    l2p, p2l, metrics,
            ):
                return rollback()
            metrics["relay_hops"] += 1
        # The relay is only worth its teleports if it actually delivered the
        # slack it was called for.
        if self._free_slots(target_core, arch, p2l) < min_free:
            return rollback()
        return True

    # ── Capacity-safe routing (2026-08-07, config.safe_mode) ───────────────────
    #
    # Invariant (†): free(c) >= core_reserve for every core, at every boundary
    # between top-level actions.  Feasibility (F†): F = P - n >= r*K + 1, so by
    # pigeonhole some core always holds >= r+1 free ("donor") -- a slot it can
    # give away without breaching (†) itself.  F is exactly conserved, so (F†)
    # is a property of the architecture and circuit, checked once in route().
    #
    # (†) is what makes the OPTIMAL relocation plan always legal.  Walking a
    # qubit along a shortest core path touches each intermediate core twice:
    # on arrival its free count drops to >= r-1 >= 1 (still enough to clear the
    # destination port), and departure restores it.  Only the endpoint keeps
    # the deficit, which is what `need` reserves for.  So a remote gate at core
    # distance d costs exactly d teleports -- the information-theoretic lower
    # bound -- whenever the meeting core already has the headroom, and d plus a
    # relay otherwise.  Under the weaker "free >= 1" precondition the same
    # procedure pays ~3d, because the slot a hop vacates always reappears one
    # core BEHIND the qubit while the next hop needs it one core AHEAD.
    #
    # See SAFE_DSABRE.md for the derivation, the termination-with-success
    # theorem, and the measured cost of the legality rule.

    def _tier1_floor(self) -> int:
        """Free slots a destination needs for an ordinary teleport to be legal."""
        cfg = self.config
        return (cfg.core_reserve + 1 if cfg.tier1_floor is None
                else cfg.tier1_floor)

    def _tier1_is_strict(self) -> bool:
        """True when Tier 1 alone maintains free >= core_reserve.

        When it does not, `_safe_route_gate` must restore the reserve itself
        before it can plan -- its `d`-optimal hop sequence assumes it.
        """
        return self._tier1_floor() >= self.config.core_reserve + 1

    @staticmethod
    def iterations_bound(remaining_2q: int, deadlock_limit: int) -> int:
        """Worst-case main-loop iterations still needed, in safe mode.

        Every iteration either retires at least one operation or increments the
        no-progress counter; after `deadlock_limit` of the latter the
        guaranteed transaction fires and retires a gate outright (§5 of
        SAFE_DSABRE.md).  Charging each gate its stall window, its own
        transaction and one progressing iteration -- the accounting the
        paper's proof sketch uses -- no gate absorbs more than
        `deadlock_limit + 2` iterations, and

            I_remaining <= remaining_2q * (deadlock_limit + 2)

        Only 2-qubit gates count: 1-qubit gates and already-adjacent
        intra-core 2q gates are drained inside the iteration that finds them,
        without consuming one of their own.

        This is a genuine bound, not an estimate -- it holds at every step, so
        `router.iterations_bound(router._remaining_2q(wdag), L)` is a live
        "iterations still required" figure.  It is loose by construction: it
        assumes every remaining gate stalls for the full deadlock window, which
        measurement puts 1-3 orders of magnitude above the actual count.

        `config.max_iterations = None` sets the budget to this bound at entry,
        so in safe mode the loop terminates by the theorem rather than by a
        budget it can exhaust.  Outside safe mode there is no bound --
        `max_iterations` is then a budget the route can hit and abort on,
        which is the difference safe mode removes.
        """
        return remaining_2q * (deadlock_limit + 2)

    @staticmethod
    def deadlock_limit_for(arch: DistributedArchitecture) -> int:
        """`deadlock_limit` derived from the architecture, not tuned per suite.

        This is what `config.deadlock_limit = None` selects, and what every
        benchmark driver in this repository passes.  It exists so the router
        has no per-suite parameter: the rule reads two diameters off the
        architecture and nothing off the circuit or the results.

        A gate about to resolve by ordinary means should not be interrupted,
        and resolving one takes at most `diam(core graph)` teleports, each
        preceded by up to `diam(core)` intra-core SWAP iterations to reach the
        staging slot, plus the teleport itself.  The factor 4 is the smallest
        integer covering the values that were hand-tuned before the rule
        existed (50 / 100 / 200 at 25q / 64q / 360q, against 56 / 84 / 252
        here).

        Measured on every suite in the paper -- 25q, 36q, 64q, 100q, 200q,
        360q and the two heavy-hex core graphs, per SabreLayout seed, safe
        mode -- this rule reproduces the hand-tuned runs' EPR counts
        **exactly**, best and median alike, at equal or fewer iterations
        (`probe_derived_deadlock.py --rule arch`).  So adopting it costs
        nothing and removes the last per-suite constant.

        A constant `L = 10` was measured too, on the argument that in safe
        mode the stall window is pure waste: recovery cannot fail, and the
        work done while stalling is discarded by the checkpoint restore, so
        EPR should not depend on how long the router thrashes first.  That
        holds up to 200 qubits -- identical EPR on 25q through 200q, at 20-35%
        fewer iterations -- and then breaks: on the 360-qubit QPEexact it
        loses the seed that produced the reported route, 1938 -> 2735 EPR
        (+41%), while the medians move only +2%.  Cutting the window short
        costs a best-of-three winner there, so the rule below is what ships.
        """
        diam_core = max(max(d.values()) for d in arch.core_dist.values())
        diam_intra = max(max(max(d.values()) for d in arch.intra_dist[c].values())
                         for c in range(arch.num_cores))
        return 4 * max(1, diam_core) * (diam_intra + 1)

    @staticmethod
    def iterations_estimate(iterations_so_far: int, retired_2q: int,
                            remaining_2q: int) -> float:
        """Live linear extrapolation of iterations still needed.

        `iterations_bound` is valid at every step but loose by 1-2 orders of
        magnitude (measured 18-200x on the 64q suite), because it assumes every
        remaining gate stalls for a full deadlock window.  Iteration count is
        driven by how many gates are *remote under the current layout*, which
        no function of (n, |G_2q|) alone predicts -- measured
        iterations/|G_2q| ranges 0.5-5.5 across the 64q suite.  So pair the
        bound with this: extrapolate from the rate observed so far, which is
        self-correcting as routing proceeds.  Returns inf before any gate has
        retired.
        """
        if retired_2q <= 0:
            return float("inf")
        return iterations_so_far * (remaining_2q / retired_2q)

    def _free_by_core(self, p2l) -> list:
        """Free-slot count per core, one pass over p2l."""
        arch = self.arch
        occ = [0] * arch.num_cores
        for p, lq in p2l.items():
            if lq is not None:
                occ[arch.core_of(p)] += 1
        return [len(arch.core_qubits(c)) - occ[c] for c in range(arch.num_cores)]

    def _find_donor_core(self, target_core, p2l, min_free):
        """Nearest core to `target_core`, EXCLUDING it, with >= min_free free.

        `_find_nearest_slack_core` can return `target_core` itself, which is
        useless when the point is to move a slot INTO it.
        """
        arch = self.arch
        visited = {target_core}
        queue = deque([target_core])
        while queue:
            c = queue.popleft()
            for nb in arch.core_graph.neighbors(c):
                if nb in visited:
                    continue
                visited.add(nb)
                if self._free_slots(nb, arch, p2l) >= min_free:
                    return nb
                queue.append(nb)
        return None

    def _relay_slack_to(self, target_core, l2p, p2l, metrics, need, protect=()):
        """Raise free(target_core) to >= `need` without breaching (†).

        Unlike `_relay_room_to`, the donor threshold and the target requirement
        are separate: a donor must hold >= core_reserve + 1 so that giving one
        slot away leaves it at the reserve, while `need` is whatever the caller
        must make room for.  Each chain hands the slack along one core at a
        time; every intermediate is transiently +1 and returns to its entry
        count, so no core ever drops below its entry value.

        One transaction: a chain that stalls rolls the whole thing back.
        """
        arch = self.arch
        donor_floor = self.config.core_reserve + 1
        if self._free_slots(target_core, arch, p2l) >= need:
            return True
        rollback = self._snapshot(l2p, p2l, metrics)
        for _ in range(need + arch.num_cores):
            if self._free_slots(target_core, arch, p2l) >= need:
                return True
            source = self._find_donor_core(target_core, p2l, donor_floor)
            if source is None:
                return rollback()
            path = arch.core_path[source][target_core]
            for i in range(len(path) - 1):
                cur, nxt = path[i], path[i + 1]
                # Slack sits at `cur`; pull a non-protected qubit out of `nxt`
                # into `cur`, handing the slack on to `nxt`.
                filler = next(
                    (p2l[p] for p in arch.core_qubits(nxt)
                     if p2l[p] is not None and p2l[p] not in protect),
                    None,
                )
                if filler is None:
                    return rollback()
                links = arch.inter_links_between(nxt, cur)
                if not links:
                    return rollback()
                p_comm_src, p_comm_dst = links[0]
                p_src = l2p[filler]
                n_s = min(arch.intra[nxt].neighbors(p_comm_src),
                          key=lambda n: self._idist(nxt, p_src, n))
                if not self._apply_teleport(
                        TeleportAction(None, filler, p_src, n_s,
                                       p_comm_src, p_comm_dst,
                                       nxt, cur, cur, score=0.0),
                        l2p, p2l, metrics,
                ):
                    return rollback()
                metrics["relay_hops"] += 1
        return rollback()

    def _plan_meeting_core(self, node, l2p, p2l):
        """Cheapest safe way to co-locate `node`'s two operands.

        Returns (meeting_core, movers, need) minimising

            d_C(a,m) + d_C(b,m) + relay(m)

        over three families, all of which keep (†):
          * m = a or m = b -- one operand moves, the meeting core must reach
            reserve+1 so it is still at the reserve once the arrival lands;
          * m = any core already holding reserve+2 -- both operands move and no
            relay is needed at all, which can beat the two-body options when a
            roomy core sits on a shortest path.
        relay(m) is the exact cost of the shortfall: the BFS distance to the
        nearest donor, which is optimal for the single slot m in {a,b} can
        need (a chain through a non-donor core costs at least as much, by the
        triangle inequality).
        """
        arch = self.arch
        r = self.config.core_reserve
        q1, q2 = node.qargs[0], node.qargs[1]
        a, b = arch.core_of(l2p[q1]), arch.core_of(l2p[q2])
        free = self._free_by_core(p2l)

        def relay_cost(m, need):
            if free[m] >= need:
                return 0
            donor = self._find_donor_core(m, p2l, r + 1)
            if donor is None:
                return None
            return arch.core_dist[donor][m] * (need - free[m])

        best = None
        for m, movers in ((a, (q2,)), (b, (q1,))):
            rc = relay_cost(m, r + 1)
            if rc is None:
                continue
            cost = arch.core_dist[a][m] + arch.core_dist[b][m] + rc
            if best is None or cost < best[0]:
                best = (cost, m, movers, r + 1)
        for m in range(arch.num_cores):
            if m in (a, b) or free[m] < r + 2:
                continue
            cost = arch.core_dist[a][m] + arch.core_dist[b][m]
            if best is not None and cost >= best[0]:
                continue
            best = (cost, m, (q1, q2), r + 2)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _safe_route_gate(self, node, wdag, l2p, p2l, metrics) -> bool:
        """Execute `node` under (†), atomically.

        Either the gate retires and (†) still holds, or nothing happened.
        Under (†)+(F†) on a connected core graph this cannot fail; the rollback
        paths are assertions, and `metrics["safe_route_failed"]` counts any
        that fire so a violated hypothesis is visible rather than silent.
        """
        arch = self.arch
        q1, q2 = node.qargs[0], node.qargs[1]
        rollback = self._snapshot(l2p, p2l, metrics)

        # With a relaxed `tier1_floor`, ordinary routing is allowed to spend the
        # reserve down (never to zero), so restore it before planning -- the
        # hop sequence below is only legal from a state satisfying (†).  A
        # donor always exists under (F†): if every core held <= core_reserve
        # the total would be <= core_reserve*K < F.  No-op when Tier 1 is
        # strict, which is why the fast path costs a single boolean.
        if not self._tier1_is_strict():
            if not self._make_layout_safe(l2p, p2l, metrics, protect=(q1, q2)):
                metrics["safe_route_failed"] += 1
                return rollback()

        if arch.core_of(l2p[q1]) != arch.core_of(l2p[q2]):
            plan = self._plan_meeting_core(node, l2p, p2l)
            if plan is None:
                metrics["safe_route_failed"] += 1
                return rollback()
            m, movers, need = plan
            if not self._relay_slack_to(m, l2p, p2l, metrics, need,
                                        protect=(q1, q2)):
                metrics["safe_route_failed"] += 1
                return rollback()
            for x in movers:
                # Bounded by the core-graph diameter; the +1 guard is defensive.
                for _ in range(arch.num_cores + 1):
                    cur = arch.core_of(l2p[x])
                    if cur == m:
                        break
                    nxt = arch.core_path[cur][m][1]
                    hopped = False
                    for p_comm_src, p_comm_dst in arch.inter_links_between(cur,
                                                                           nxt):
                        p_src = l2p[x]
                        n_s = min(arch.intra[cur].neighbors(p_comm_src),
                                  key=lambda n: self._idist(cur, p_src, n))
                        if self._apply_teleport(
                                TeleportAction(node, x, p_src, n_s,
                                               p_comm_src, p_comm_dst,
                                               cur, nxt, m, score=0.0),
                                l2p, p2l, metrics,
                        ):
                            hopped = True
                            break
                    if not hopped:
                        metrics["safe_route_failed"] += 1
                        return rollback()
                if arch.core_of(l2p[x]) != m:
                    metrics["safe_route_failed"] += 1
                    return rollback()

        ci = arch.core_of(l2p[q1])
        p1, p2 = l2p[q1], l2p[q2]
        try:
            path = self._ipath(ci, p1, p2)
        except Exception:
            metrics["safe_route_failed"] += 1
            return rollback()
        for i in range(len(path) - 2):
            u, v = path[i], path[i + 1]
            qu, qv = p2l[u], p2l[v]
            if qu is not None: l2p[qu] = v
            if qv is not None: l2p[qv] = u
            p2l[u], p2l[v] = qv, qu
            metrics["ls"]   += 1
            metrics["cost"] += self.config.cost_local_swap
            if self.config.trace_routing:
                metrics["trace"].append(("SWAP", u, v, ci))
        if not arch.Gr.has_edge(l2p[q1], l2p[q2]):
            metrics["safe_route_failed"] += 1
            return rollback()

        wdag.remove_op_node(node)
        metrics["safe_routes"] += 1
        if self.config.trace_routing:
            metrics["trace"].append(("GATE", q1, q2, ci))
        return True

    def _safe_pick_gate(self, front, l2p, p2l):
        """Front-layer gate with the cheapest safe plan (ties: most stuck)."""
        best, best_key = None, None
        for n in front:
            plan = self._plan_meeting_core(n, l2p, p2l)
            if plan is None:
                continue
            m, movers, _ = plan
            cost = (self.arch.core_dist[self.arch.core_of(l2p[n.qargs[0]])][m]
                    + self.arch.core_dist[self.arch.core_of(l2p[n.qargs[1]])][m])
            key = (cost, -self._gate_dist(n, l2p))
            if best_key is None or key < best_key:
                best, best_key = n, key
        return best if best is not None else (front[0] if front else None)

    def _safe_progress(self, front_inter, wdag, l2p, p2l, metrics) -> bool:
        """One guaranteed gate, if safe mode is on. False when it is off."""
        if not self.config.safe_mode or not front_inter:
            return False
        node = self._safe_pick_gate(front_inter, l2p, p2l)
        return (node is not None
                and self._safe_route_gate(node, wdag, l2p, p2l, metrics))

    def _safe_drain(self, wdag, l2p, p2l, metrics) -> bool:
        """Retire every remaining gate with `_safe_route_gate` alone.

        The escape hatch that replaces `ITERATION_LIMIT` as an abort: heuristic
        search is abandoned, and each remaining gate is executed by the
        guaranteed transaction.  Cost is bounded by 2*diam(core graph) EPRs per
        remote gate; termination is the strictly decreasing gate count.

        Mostly not required in safe mode.  The two backup-exhaustion callers
        cannot fire, since safe mode raises `max_backups` to the unrouted-gate
        count and every recovery retires one.  The `ITERATION_LIMIT` caller
        can, but only when a driver sets `max_iterations` to a finite value
        below the theorem's bound instead of leaving it None (module
        docstring).
        """
        arch = self.arch
        guard = self._remaining + 1
        while self._remaining > 0 and guard > 0:
            guard -= 1
            # Drain to a fixed point.  The loop must NOT stop the moment
            # `_front_2q` is empty: retiring the front's 1q gates can expose
            # more 1q gates, leaving the front temporarily free of 2q gates
            # while the DAG is far from empty.  Bailing out there made
            # `_safe_drain` report failure on a circuit it had merely not
            # finished draining -- the cause of `random_100`'s ITERATION_LIMIT
            # abort, with `safe_route_failed` still 0.
            progress = True
            while progress:
                progress = False
                for n in list(wdag.front_layer()):
                    if len(n.qargs) < 2:
                        metrics["1q_gates"] += 1
                        wdag.remove_op_node(n)
                        progress = True
                ready = [n for n in self._front_2q(wdag)
                         if arch.Gr.has_edge(l2p[n.qargs[0]], l2p[n.qargs[1]])]
                for n in ready:
                    wdag.remove_op_node(n)
                    if self.config.trace_routing:
                        metrics["trace"].append(
                            ("GATE", n.qargs[0], n.qargs[1],
                             arch.core_of(l2p[n.qargs[0]]))
                        )
                    progress = True
            if self._remaining == 0:
                break
            # After the fixed point the front holds no 1q gate, so a non-empty
            # DAG has a 2q gate in front and this cannot be empty.
            front = self._front_2q(wdag)
            if not front:
                return False
            node = self._safe_pick_gate(front, l2p, p2l)
            if node is None or not self._safe_route_gate(node, wdag, l2p, p2l,
                                                         metrics):
                return False
        return self._remaining == 0

    def _make_layout_safe(self, l2p, p2l, metrics, protect=()) -> bool:
        """Bring an entry layout up to (†) by relaying slack into short cores.

        Pass 1 of `run_sabre_passes` starts from `sabre_locked_boundary_layout`,
        which already reserves >= 2 slots per core on every published suite; in
        safe mode pass 1 also EXITS safe, so passes 2 and 3 inherit a safe state
        and this is a no-op.  It exists so an externally supplied (or archived
        pre-safe-mode) layout is repaired rather than rejected.
        """
        arch = self.arch
        r = self.config.core_reserve
        for _ in range(arch.num_cores + 1):
            short = [c for c in range(arch.num_cores)
                     if self._free_slots(c, arch, p2l) < r]
            if not short:
                return True
            fixed_any = False
            for c in short:
                if self._relay_slack_to(c, l2p, p2l, metrics, r,
                                        protect=protect):
                    fixed_any = True
            if not fixed_any:
                return False
        return not [c for c in range(arch.num_cores)
                    if self._free_slots(c, arch, p2l) < r]

    # ── Whole-gate transactional recovery (2026-08-05) ─────────────────────────
    #
    # _backup_plan's cross-core branches (both the greedy and the relay-mode
    # variants) move only q1, hop by hop, and simply `break` out of the loop
    # the moment a hop fails -- so a stall midway leaves q1 parked wherever it
    # got to, still not adjacent to q2, having spent whatever teleports it
    # already used. Two things follow: (1) the router never tries moving q2
    # instead, even when THAT direction is the one with room, and (2) a
    # failure is not atomic -- the partial move stays applied.
    #
    # _route_gate_transaction below fixes both. It wraps one full attempt
    # (relay room to every next core via the existing invariant-preserving
    # `_relay_room_to`, teleport the mover hop by hop, then intra-core SWAP it
    # onto the gate and execute) in a single snapshot: either the whole thing
    # lands and the gate executes, or every mutation is rolled back and the
    # state is bit-identical to entry. `_backup_plan` tries mover=q1 and, if
    # that fails, mover=q2 -- so a one-sided bottleneck (room exists heading
    # one way, not the other) no longer aborts recovery outright.
    #
    # Measured 2026-08-05 (see the capacity-safe fallback investigation):
    # bit-for-bit unchanged on circuits where backup never fires, and net
    # -8.2% total EPR on the 64q suite's 4 circuits where it does (ae -30%,
    # multiplier -14%, qft -8%, qaoa -5%, qpeexact +2% the one regression).
    # More importantly, on random_80 (10-core H-grid, 23,381 CX) -- the one
    # archived instance where the OLD backup_plan hits DEADLOCK_BACKUP_FAILED
    # outright (the greedy hop has no legal move, independent of iteration
    # budget) -- this transaction never fails to make progress on any of the
    # 3 SabreLayout candidates; given enough iterations (a separate, ordinary
    # budget knob already scaled per suite via HardwareConfig) all 3 complete.
    # The old branches are kept below as the fallback for the rare case
    # neither operand's transaction can be validated (e.g. the active core
    # graph is disconnected, or no port-avoiding staging path exists).

    def _route_gate_transaction(self, node, mover, anchor, wdag, l2p, p2l,
                                metrics) -> bool:
        """Atomically bring `mover` to `anchor`'s core and execute `node`.

        Either the gate executes (True) or the layout, counters, and trace
        are exactly as at entry (False). `wdag` is only mutated on success
        (the gate's removal is the transaction's last step), so a failed
        attempt never needs to touch it.

        Not required in safe mode: `_backup_plan` calls this only after
        `_safe_route_gate` has already failed, which under the entry
        hypotheses it cannot (module docstring).  It remains the best
        recovery available when one of them is violated, and the only one in
        score-only mode.
        """
        arch = self.arch
        rollback = self._snapshot(l2p, p2l, metrics)
        target_core = arch.core_of(l2p[anchor])

        # `anchor` never changes core here: relay hops protect it via
        # `protect=`. `mover` is protected from being CHOSEN as filler the
        # same way, but `_relay_room_to`'s hops run through `_apply_teleport`,
        # whose own internal `_force_make_room` (called if a relay hop's
        # source/destination core is itself full) has no knowledge of
        # `mover` -- only of the filler qubit that specific call is actively
        # relocating (`exclude_virt`). It can therefore incidentally displace
        # `mover` to a different core as a side effect. Recompute cur_core/
        # next_core from mover's actual position AFTER the relay rather than
        # trusting the pre-relay values (`KeyError` in `_idist` otherwise,
        # from treating a now-foreign physical qubit as being in `cur_core`)
        # -- bounded by the core-graph diameter (the `num_cores + 1` guard is
        # defensive, not load-bearing).
        for _ in range(arch.num_cores + 1):
            cur_core = arch.core_of(l2p[mover])
            if cur_core == target_core:
                break
            next_core = arch.core_path[cur_core][target_core][1]
            if not self._relay_room_to(next_core, l2p, p2l, metrics,
                                       min_free=2, protect=(mover, anchor)):
                return rollback()
            cur_core = arch.core_of(l2p[mover])
            if cur_core == target_core:
                break
            next_core = arch.core_path[cur_core][target_core][1]
            hopped = False
            for p_comm_src, p_comm_dst in arch.inter_links_between(cur_core,
                                                                   next_core):
                p_src = l2p[mover]
                n_s = min(arch.intra[cur_core].neighbors(p_comm_src),
                          key=lambda n: self._idist(cur_core, p_src, n))
                if self._apply_teleport(
                        TeleportAction(node, mover, p_src, n_s,
                                       p_comm_src, p_comm_dst,
                                       cur_core, next_core, target_core,
                                       score=0.0),
                        l2p, p2l, metrics,
                ):
                    hopped = True
                    break
            if not hopped:
                return rollback()
        if arch.core_of(l2p[mover]) != target_core:
            return rollback()

        p1, p2 = l2p[node.qargs[0]], l2p[node.qargs[1]]
        ci = arch.core_of(p1)
        try:
            path = self._ipath(ci, p1, p2)
        except Exception:
            return rollback()
        for i in range(len(path) - 2):
            a, b = path[i], path[i + 1]
            qa, qb = p2l[a], p2l[b]
            if qa is not None: l2p[qa] = b
            if qb is not None: l2p[qb] = a
            p2l[a], p2l[b] = qb, qa
            metrics["ls"]   += 1
            metrics["cost"] += self.config.cost_local_swap
            if self.config.trace_routing:
                metrics["trace"].append(("SWAP", a, b, ci))
        if not arch.Gr.has_edge(l2p[node.qargs[0]], l2p[node.qargs[1]]):
            return rollback()

        wdag.remove_op_node(node)
        if self.config.trace_routing:
            metrics["trace"].append(
                ("GATE", node.qargs[0], node.qargs[1], ci)
            )
        return True

    def _backup_plan(self, wdag, l2p, p2l, metrics):
        """Force the most-stuck gate closer to completion via greedy hops."""
        arch = self.arch
        front = self._front_2q(wdag)
        if not front:
            return False
        if self.config.safe_mode:
            # Guaranteed path first: under (†) this cannot fail, so everything
            # below -- the greedy cross-core hops, `_route_gate_transaction`,
            # and through it `_relay_room_to` -- is unreachable in safe mode
            # (module docstring; the paper's Appendix C proves the ordinary
            # scored path never needs them either).  They stay as a fallback
            # rather than an assertion so a violated hypothesis degrades to
            # today's behaviour instead of aborting the route.
            node = self._safe_pick_gate(front, l2p, p2l)
            if node is not None and self._safe_route_gate(node, wdag, l2p, p2l,
                                                          metrics):
                return True
        stuck = max(front, key=lambda n: self._gate_dist(n, l2p))
        q1, q2 = stuck.qargs[0], stuck.qargs[1]
        p1, p2 = l2p[q1], l2p[q2]
        c1, c2 = arch.core_of(p1), arch.core_of(p2)

        if c1 == c2:
            try:
                path = self._ipath(c1, p1, p2)
            except Exception:
                return False
            for i in range(len(path) - 2):
                a, b = path[i], path[i + 1]
                qa, qb = p2l[a], p2l[b]
                if qa is not None: l2p[qa] = b
                if qb is not None: l2p[qb] = a
                p2l[a], p2l[b] = qb, qa
                metrics["ls"]   += 1
                metrics["cost"] += self.config.cost_local_swap
                if self.config.trace_routing:
                    metrics["trace"].append(("SWAP", a, b, arch.core_of(a)))
            if arch.Gr.has_edge(l2p[q1], l2p[q2]):
                wdag.remove_op_node(stuck)
                if self.config.trace_routing:
                    metrics["trace"].append(
                        ("GATE", q1, q2, arch.core_of(l2p[q1]))
                    )
            return True

        # Whole-gate transaction first, either direction: succeeds whenever
        # EITHER operand can reach the other's core, atomically, and retires
        # the gate outright rather than merely hopping it partway.
        if self._route_gate_transaction(stuck, q1, q2, wdag, l2p, p2l, metrics):
            return True
        if self._route_gate_transaction(stuck, q2, q1, wdag, l2p, p2l, metrics):
            return True

        moved_any = False
        if self.config.backup_relay_mode:
            # Invariant-preserving cross-core hop: relay the chip's slack to
            # next_core (guaranteeing it has >=2 free, hence >=1 after q1
            # arrives) before every hop, instead of the >=1-free check below,
            # which can still leave a core at exactly 0 free.
            for hop_idx in range(len(arch.core_path[c1][c2]) - 1):
                cur_core  = arch.core_of(l2p[q1])
                next_core = arch.core_path[c1][c2][hop_idx + 1]
                if cur_core == c2:
                    break
                if not self._relay_room_to(next_core, l2p, p2l, metrics,
                                           min_free=2, protect=(q1, q2)):
                    break
                links = arch.inter_links_between(cur_core, next_core)
                if not links:
                    break
                p_comm_src, p_comm_dst = links[0]
                n_s = min(
                    arch.intra[cur_core].neighbors(p_comm_src),
                    key=lambda n: self._idist(cur_core, l2p[q1], n),
                )
                if not self._apply_teleport(
                        TeleportAction(stuck, q1, l2p[q1], n_s, p_comm_src, p_comm_dst,
                                       cur_core, next_core, c2, score=0.0),
                        l2p, p2l, metrics,
                ):
                    break
                moved_any = True
        else:
            for hop_idx in range(len(arch.core_path[c1][c2]) - 1):
                cur_core  = arch.core_of(l2p[q1])
                next_core = arch.core_path[c1][c2][hop_idx + 1]
                if cur_core == c2:
                    break
                hopped = False
                for p_comm_src, p_comm_dst in arch.inter_links_between(cur_core, next_core):
                    if (self._free_slots(next_core, arch, p2l) < 1
                            and not self._force_make_room(next_core, l2p, p2l, metrics)):
                        continue
                    n_s = min(
                        list(arch.intra[cur_core].neighbors(p_comm_src)),
                        key=lambda n: self._idist(cur_core, l2p[q1], n),
                    )
                    if not self._apply_teleport(
                            TeleportAction(stuck, q1, l2p[q1], n_s, p_comm_src, p_comm_dst,
                                           cur_core, next_core, c2, score=0.0),
                            l2p, p2l, metrics,
                    ):
                        continue  # rolled back cleanly: try the next link
                    moved_any = hopped = True
                    break
                if not hopped:
                    break

        np1, np2 = l2p[q1], l2p[q2]
        nc1, nc2 = arch.core_of(np1), arch.core_of(np2)
        if nc1 == nc2 and arch.Gr.has_edge(np1, np2):
            wdag.remove_op_node(stuck)
            if self.config.trace_routing:
                metrics["trace"].append(("GATE", q1, q2, nc1))
            moved_any = True
        elif nc1 == nc2:
            try:
                path = self._ipath(nc1, np1, np2)
                for i in range(len(path) - 2):
                    a, b = path[i], path[i + 1]
                    qa, qb = p2l[a], p2l[b]
                    if qa is not None: l2p[qa] = b
                    if qb is not None: l2p[qb] = a
                    p2l[a], p2l[b] = qb, qa
                    metrics["ls"]   += 1
                    metrics["cost"] += self.config.cost_local_swap
                    if self.config.trace_routing:
                        metrics["trace"].append(("SWAP", a, b, arch.core_of(a)))
                moved_any = True
                if arch.Gr.has_edge(l2p[q1], l2p[q2]):
                    wdag.remove_op_node(stuck)
                    if self.config.trace_routing:
                        metrics["trace"].append(
                            ("GATE", q1, q2, arch.core_of(l2p[q1]))
                        )
            except Exception:
                pass

        return moved_any

    # ── Intra-core SWAP selection ──────────────────────────────────────────────

    @staticmethod
    def _merge_ix(a, b):
        """Union of two ascending index lists, ascending, without duplicates."""
        if not b:
            return a
        if not a:
            return b
        out, i, j = [], 0, 0
        while i < len(a) and j < len(b):
            if a[i] < b[j]:
                out.append(a[i]); i += 1
            elif a[i] > b[j]:
                out.append(b[j]); j += 1
            else:
                out.append(a[i]); i += 1; j += 1
        out.extend(a[i:]); out.extend(b[j:])
        return out

    def _best_intra_swap(self, ci, local_front, local_ext, l2p, p2l):
        """Lowest-scoring intra-core SWAP in core `ci`, or None.

        Scores every candidate on Eq. H_intra, but only against the gates the
        candidate can actually move.  A SWAP exchanges the contents of two
        slots, so a gate's distance term changes only if one of its operands
        sits on u or v; every other term of both sums is exactly zero.

        Front-layer gates are qubit-disjoint (each wire contributes at most
        one), so at most *two* front gates are affected whatever |F_c| is --
        which is what turns the per-core cost from O(d|F_c|(|F_c|+L)) into
        O(d(|F_c|+L)), and removes a factor M from the chip-wide bound.
        Extended-set gates are not disjoint, so they are indexed by qubit.

        Identical to scoring the full sums, not merely equivalent: the omitted
        terms are integer zeros; the surviving terms are accumulated in their
        original list order, so float addition is never reordered; `involved`
        is built in the original order, so the `score < min_score` first-wins
        tie-break is unchanged; and the normalisers use the full lengths.
        """
        arch  = self.arch
        decay = self.config.lookahead_decay
        w_e   = self.config.weight_extended
        n_f   = max(len(local_front), 1)
        n_e   = max(len(local_ext), 1)

        # qubit -> index of its front gate (disjointness makes this a function)
        front_ix = {}
        for i, n in enumerate(local_front):
            front_ix[n.qargs[0]] = i
            front_ix[n.qargs[1]] = i
        # qubit -> ascending indices of the extended gates it appears in
        ext_ix: dict = {}
        for i, (n, _d) in enumerate(local_ext):
            ext_ix.setdefault(n.qargs[0], []).append(i)
            ext_ix.setdefault(n.qargs[1], []).append(i)

        involved = set()
        for n in local_front:
            involved.update([l2p[n.qargs[0]], l2p[n.qargs[1]]])

        best_swap, min_score = None, float("inf")
        for u in involved:
            for v in arch.Gr.neighbors(u):
                if arch.core_of(v) != ci:
                    continue
                qu, qv = p2l[u], p2l[v]

                def moved(q, _u=u, _v=v):
                    p = l2p[q]
                    return _v if p == _u else (_u if p == _v else p)

                f_ix = []
                if qu is not None and qu in front_ix:
                    f_ix.append(front_ix[qu])
                if qv is not None and qv in front_ix and front_ix[qv] not in f_ix:
                    f_ix.append(front_ix[qv])
                delta_Hf = 0
                for i in sorted(f_ix):
                    n = local_front[i]
                    q1, q2 = n.qargs[0], n.qargs[1]
                    delta_Hf += (self._idist(ci, moved(q1), moved(q2))
                                 - self._idist(ci, l2p[q1], l2p[q2]))

                e_ix = self._merge_ix(ext_ix.get(qu, []) if qu is not None else [],
                                      ext_ix.get(qv, []) if qv is not None else [])
                delta_He = 0
                for i in e_ix:
                    n, depth = local_ext[i]
                    q1, q2 = n.qargs[0], n.qargs[1]
                    delta_He += (decay ** depth) * (
                        self._idist(ci, moved(q1), moved(q2))
                        - self._idist(ci, l2p[q1], l2p[q2])
                    )

                score = (delta_Hf / n_f) + w_e * (delta_He / n_e)
                if score < min_score:
                    min_score, best_swap = score, (u, v)
        return best_swap

    # ── Main routing loop ──────────────────────────────────────────────────────

    def route(self, dag, initial_layout):
        """Route `dag` under `initial_layout`.

        Raises
        ------
        ValueError
            If `dag` holds an operation on more than two qubits, or if
            `initial_layout` is not an injective map into this architecture's
            physical qubits.  Both are caller errors that routing has no way
            to express as a result -- see the checks below.

        Returns
        -------
        (metrics, final_layout)
            metrics["eprs"]     : EPR pairs consumed
            metrics["ls"]       : local SWAPs applied
            metrics["aborted"]  : True if routing failed
            metrics["compile_time"] : wall-clock seconds
        """
        arch = self.arch

        # A >2q operation is neither drained as a 1q gate nor picked up by
        # _front_2q, so it blocks its wires forever and the run ends in
        # DEADLOCK_BACKUP_FAILED -- an input error disguised as a hard
        # circuit.  Barriers are the likely source: every benchmark driver
        # strips them, so this fires only when one forgets.
        for node in dag.op_nodes():
            if len(node.qargs) > 2:
                raise ValueError(
                    f"routing supports 1- and 2-qubit operations only; "
                    f"'{node.name}' acts on {len(node.qargs)} qubits. "
                    f"Decompose it (and strip barriers) before routing."
                )

        l2p  = initial_layout.copy()
        p2l  = {p: None for p in arch.Gr.nodes}
        # Validate as p2l is built.  The assignment below is last-write-wins:
        # a physical qubit used twice would silently drop one logical qubit,
        # and routing then reports a clean success -- aborted=False, every
        # gate retired -- for a start state where two qubits share one site.
        # An off-chip physical instead surfaces much later, as a bare KeyError
        # from inside a distance lookup.
        for lq, p in l2p.items():
            if p not in p2l:
                raise ValueError(
                    f"initial_layout puts logical qubit {lq} on physical "
                    f"qubit {p}, which is not in this architecture"
                )
            if p2l[p] is not None:
                raise ValueError(
                    f"initial_layout puts logical qubits {p2l[p]} and {lq} "
                    f"on the same physical qubit {p}"
                )
            p2l[p] = lq
        # Wrapped so every retirement updates the maintained in-degrees,
        # unrouted-gate counter, and retirement log.
        _real = deepcopy(dag)
        self._init_dag_state(_real)
        wdag = _RetiringDAG(_real, self)
        n_2q_at_entry = sum(1 for n in _real.op_nodes() if len(n.qargs) == 2)
        self._n_snapshots = self._n_rollbacks = self._n_wdag_rebuilds = 0
        self._t_snapshot = self._t_rollback = self._t_wdag_rebuild = 0.0

        metrics = {
            "ls": 0, "teles": 0, "catcomms": 0, "eprs": 0,
            "cost": 0, "1q_gates": 0, "aborted": False,
            "compile_time": 0.0, "backup_activations": 0,
            # ── Mechanism instrumentation ─────────────────────────────────────
            # force_make_room : calls to _force_make_room (deadlock recovery)
            "force_make_room": 0,
            # relay_hops : filler teleports issued by _relay_room_to /
            # _relay_slack_to
            "relay_hops": 0,
            # safe_routes / safe_route_failed : gates retired by the guaranteed
            # transaction, and attempts that hit one of its assertion paths --
            # the latter must stay 0 while (†) and (F†) hold (config.safe_mode)
            "safe_routes": 0,
            "safe_route_failed": 0,
            "trace": [] if self.config.trace_routing else None,
        }
        failure_log = []

        # ── Capacity-safe preconditions ───────────────────────────────────
        # Every hypothesis of the termination-with-success theorem
        # (SAFE_DSABRE.md §5) is checked here.  A guarantee whose hypotheses go
        # unchecked is not a guarantee: each of these, violated, turns
        # `_safe_route_gate` from "cannot fail" into "fails, or raises
        # KeyError from a distance lookup on an unreachable core".
        if self.config.safe_mode:
            r = self.config.core_reserve
            if r < 2:
                # `_safe_route_gate` relays only at the meeting core, which is
                # sound only from free >= 2: with a reserve of 1 an
                # intermediate core reaches 0 as the mover arrives and can
                # strand it there.  A reserve-1 guarantee needs a relay before
                # EVERY hop (SAFE_DSABRE.md §2.1) -- a different procedure, not
                # a parameter of this one.
                raise ValueError(
                    f"safe_mode requires core_reserve >= 2 (got {r}): the "
                    f"guaranteed transaction's hop sequence is only legal from "
                    f"a state with two free slots per core."
                )
            if not self.config.enable_deadlock_recovery:
                raise ValueError(
                    "safe_mode requires enable_deadlock_recovery=True: the "
                    "guaranteed transaction is reached through the deadlock "
                    "path, so disabling recovery removes the guarantee."
                )
            if not nx.is_connected(arch.core_graph):
                raise ValueError(
                    "safe_mode requires a connected core graph; this one has "
                    f"{nx.number_connected_components(arch.core_graph)} "
                    "components, so a gate straddling two of them can never "
                    "be routed."
                )
            bad_intra = [c for c in range(arch.num_cores)
                         if not nx.is_connected(arch.intra[c])]
            if bad_intra:
                raise ValueError(
                    f"safe_mode requires every core's coupling graph to be "
                    f"connected; cores {bad_intra} are not, so a qubit cannot "
                    f"always reach a comm port."
                )
            # A relay must always find a filler qubit that is not one of the
            # gate's own two operands.  On the relay path every core has
            # occupancy >= kappa - (need - 1) >= kappa - r - 1, and at most one
            # protected qubit sits there, so kappa >= r + 3 leaves one spare.
            small = [c for c in range(arch.num_cores)
                     if len(arch.core_qubits(c)) < r + 3]
            if small:
                raise ValueError(
                    f"safe_mode with core_reserve={r} requires every core to "
                    f"hold at least {r + 3} physical qubits (so a relay can "
                    f"always find a non-gate filler); cores {small} are "
                    f"smaller."
                )
            total_free = len(p2l) - len(l2p)
            if total_free < r * arch.num_cores + 1:
                raise ValueError(
                    f"safe_mode needs P - n >= core_reserve*K + 1 = "
                    f"{r * arch.num_cores + 1}, but this architecture leaves "
                    f"only {total_free} free slots for {len(l2p)} qubits on "
                    f"{arch.num_cores} cores. Lower core_reserve, use a larger "
                    f"architecture, or route with safe_mode=False."
                )
            if not self._make_layout_safe(l2p, p2l, metrics):
                raise ValueError(
                    "safe_mode could not bring the initial layout up to "
                    f"free >= {r} per core: "
                    f"{[self._free_slots(c, arch, p2l) for c in range(arch.num_cores)]}"
                )
        _t_start = time.perf_counter()

        iteration        = 0
        last_remaining   = self._remaining
        no_progress_iters = 0
        backup_attempts  = 0
        # `deadlock_limit = None` means "derive it from the architecture"
        # (`deadlock_limit_for`), which is what every benchmark driver here
        # passes so that no suite carries a hand-tuned value.  An explicit int
        # is honoured unchanged, so pre-rule configs reproduce bit-for-bit.
        deadlock_limit = (self.deadlock_limit_for(arch)
                          if self.config.deadlock_limit is None
                          else self.config.deadlock_limit)
        # `max_iterations = None` sets the budget to the worst case the
        # termination theorem allows, so it cannot bind: in safe mode the loop
        # then stops because every gate is routed, not because a budget ran
        # out.  In score-only mode there is no such theorem, and None simply
        # means "do not cap".
        max_iterations = (self.iterations_bound(n_2q_at_entry, deadlock_limit)
                          if self.config.max_iterations is None
                          else self.config.max_iterations)
        # Safe mode: every recovery retires a gate, so the useful bound on
        # activations is the gate count, not a hand-tuned constant -- capping
        # below it would abort a route the guarantee says must finish.  In
        # score-only mode there is no such theorem to derive a bound from, so
        # None means "do not cap", exactly as it does for `max_iterations`:
        # `max_backups` stays None, the test below is skipped, and the budget
        # that ends a stuck route is `max_iterations` alone.  That is not a
        # licence to run forever -- an activation costs `deadlock_limit`
        # no-progress iterations, so activations are bounded by
        # max_iterations / deadlock_limit whether or not this cap is set.
        max_backups = (max(self.config.max_backup_attempts or 0,
                           self._remaining + 1)
                       if self.config.safe_mode
                       else self.config.max_backup_attempts)   # None = uncapped
        committed_nid    = None  # DAG node id of the gate the last teleport advanced
        extended_cache   = None

        # Output-stream accounting must be checkpointed alongside the layout:
        # deadlock recovery discards the work done since the checkpoint, so
        # ops emitted on the abandoned branch are not part of the compiled
        # circuit and must leave the trace/counters too, or the emitted
        # stream stops being replayable against the restored state.
        ckpt_counter_keys = ("ls", "teles", "catcomms", "eprs", "cost", "1q_gates")

        ckpt_l2p   = l2p.copy()
        ckpt_p2l   = p2l.copy()
        ckpt_mark  = len(self._retired_log)        # O(1) checkpoint
        ckpt_counters  = {k: metrics[k] for k in ckpt_counter_keys}
        ckpt_trace_len = len(metrics["trace"]) if metrics["trace"] is not None else 0
        prev_remaining = last_remaining

        while self._remaining > 0:
            if iteration >= max_iterations:
                remaining = self._remaining
                elapsed   = time.perf_counter() - _t_start
                # Safe mode: the budget running out stops the heuristic search,
                # not the route.  Every remaining gate is retired by the
                # guaranteed transaction instead.
                if (self.config.safe_mode
                        and self._safe_drain(wdag, l2p, p2l, metrics)):
                    failure_log.append(
                        ("ITERATION_LIMIT_SAFE_DRAIN", iteration, remaining, elapsed))
                    break
                failure_log.append(("ITERATION_LIMIT", iteration, remaining, elapsed))
                metrics["aborted"] = True
                metrics["compile_time"] = elapsed
                break
            iteration += 1

            # Drain executable gates (1q and adjacent intra-core 2q).
            progress = True
            while progress:
                progress = False
                for node in list(wdag.front_layer()):
                    if len(node.qargs) < 2:
                        metrics["1q_gates"] += 1
                        wdag.remove_op_node(node)
                        progress = True
                front = self._front_2q(wdag)
                if not front:
                    break
                intra_exec = [
                    (n, l2p[n.qargs[0]], l2p[n.qargs[1]]) for n in front
                    if (arch.core_of(l2p[n.qargs[0]]) == arch.core_of(l2p[n.qargs[1]])
                        and arch.Gr.has_edge(l2p[n.qargs[0]], l2p[n.qargs[1]]))
                ]
                if intra_exec:
                    for n, p1, p2 in intra_exec:
                        wdag.remove_op_node(n)
                        if self.config.trace_routing:
                            metrics["trace"].append(
                                ("GATE", n.qargs[0], n.qargs[1], arch.core_of(p1))
                            )
                    progress = True

            if self._remaining == 0:
                break

            current_remaining = self._remaining
            if current_remaining < prev_remaining:
                extended_cache = None
            prev_remaining = current_remaining

            front       = self._front_2q(wdag)
            front_intra = [n for n in front
                           if arch.core_of(l2p[n.qargs[0]]) == arch.core_of(l2p[n.qargs[1]])]
            front_inter = [n for n in front if n not in front_intra]
            # {virt: phys of its front-layer gate partner} -- front-layer gates
            # are qubit-disjoint (each wire contributes at most one), so this is
            # unambiguous.  Used for distance-aware eviction (config.evict_distance_aware).
            partner_phys = {}
            for n in front:
                q1, q2 = n.qargs[0], n.qargs[1]
                partner_phys[q1] = l2p[q2]
                partner_phys[q2] = l2p[q1]

            if front_intra:
                # Intra-core SABRE: route each active core independently and in parallel.
                core_to_gates = {}
                for n in front_intra:
                    ci = arch.core_of(l2p[n.qargs[0]])
                    core_to_gates.setdefault(ci, []).append(n)

                local_ext_cache = self._get_local_extended(wdag, front, core_to_gates.keys(), l2p)

                for ci, local_front in core_to_gates.items():
                    local_ext = local_ext_cache[ci]
                    best_swap = self._best_intra_swap(ci, local_front, local_ext,
                                                      l2p, p2l)

                    if best_swap:
                        u, v = best_swap
                        qu, qv = p2l[u], p2l[v]
                        if qu is not None: l2p[qu] = v
                        if qv is not None: l2p[qv] = u
                        p2l[u], p2l[v] = qv, qu
                        metrics["ls"]   += 1
                        metrics["cost"] += self.config.cost_local_swap
                        if self.config.trace_routing:
                            metrics["trace"].append(("SWAP", u, v, ci))

            elif front_inter:
                # Inter-core teleportation: score candidates, execute best.
                if extended_cache is None:
                    extended_cache = self._inter_ext(wdag, front, self.config.lookahead_size)
                candidates = self._generate_candidates(front_inter, extended_cache, l2p, p2l,
                                                       committed_nid=committed_nid)
                if not candidates:
                    # In safe mode an empty list means no neighbouring core has
                    # headroom -- which a local SWAP cannot change, so go
                    # straight to the guaranteed transaction rather than
                    # burning iterations until deadlock_limit fires.
                    if not (self._safe_progress(front_inter, wdag, l2p, p2l,
                                                metrics)
                            or self._fallback_local_swap(front_inter, wdag,
                                                         l2p, p2l, metrics)):
                        remaining = self._remaining
                        elapsed   = time.perf_counter() - _t_start
                        failure_log.append(("NO_ACTIONS_NO_FALLBACK", iteration, remaining, elapsed))
                        metrics["aborted"] = True
                        break
                else:
                    # A candidate can be rejected atomically (rolled back) if
                    # its preconditions no longer hold at execution time; fall
                    # through the score-sorted list until one commits.
                    applied = None
                    for cand in candidates:
                        if self._apply_teleport(cand, l2p, p2l, metrics,
                                                partner_phys=partner_phys):
                            applied = cand
                            break
                    if applied is not None:
                        committed_nid  = applied.node._node_id
                        # The extended set is NOT invalidated here.  A teleport
                        # removes no gate, and _inter_ext reads only (dag,
                        # front, size), so a rebuild would return the same
                        # list -- measured, 4-55% of all rebuilds were this.
                        # Retirement is still caught by the current_remaining
                        # test above, and rollback by the reset below.
                    elif not (self._safe_progress(front_inter, wdag, l2p, p2l,
                                                  metrics)
                              or self._fallback_local_swap(front_inter, wdag,
                                                           l2p, p2l, metrics)):
                        remaining = self._remaining
                        elapsed   = time.perf_counter() - _t_start
                        failure_log.append(("ALL_CANDIDATES_REJECTED", iteration, remaining, elapsed))
                        metrics["aborted"] = True
                        break

            remaining = self._remaining
            if remaining < last_remaining:
                last_remaining    = remaining
                no_progress_iters = 0
                ckpt_l2p   = l2p.copy()
                ckpt_p2l   = p2l.copy()
                ckpt_mark  = len(self._retired_log)
                ckpt_counters  = {k: metrics[k] for k in ckpt_counter_keys}
                ckpt_trace_len = len(metrics["trace"]) if metrics["trace"] is not None else 0
            else:
                no_progress_iters += 1

            if no_progress_iters >= deadlock_limit:
                if not self.config.enable_deadlock_recovery:
                    elapsed = time.perf_counter() - _t_start
                    failure_log.append(("DEADLOCK_NO_RECOVERY", iteration, remaining, elapsed))
                    metrics["aborted"] = True
                    break
                backup_attempts += 1
                elapsed = time.perf_counter() - _t_start
                if max_backups is not None and backup_attempts > max_backups:
                    if (self.config.safe_mode
                            and self._safe_drain(wdag, l2p, p2l, metrics)):
                        failure_log.append(
                            ("BACKUP_EXHAUSTED_SAFE_DRAIN", iteration, remaining, elapsed))
                        break
                    failure_log.append(("DEADLOCK_BACKUP_EXHAUSTED", iteration, remaining, elapsed))
                    metrics["aborted"] = True
                    break
                l2p        = ckpt_l2p.copy()
                p2l        = ckpt_p2l.copy()
                wdag       = self._rebuild_wdag(dag, ckpt_mark)
                for k in ckpt_counter_keys:
                    metrics[k] = ckpt_counters[k]
                if metrics["trace"] is not None:
                    del metrics["trace"][ckpt_trace_len:]
                metrics["backup_activations"] += 1
                extended_cache = None
                committed_nid  = None  # l2p/wdag were rolled back to the checkpoint
                if not self._backup_plan(wdag, l2p, p2l, metrics):
                    if (self.config.safe_mode
                            and self._safe_drain(wdag, l2p, p2l, metrics)):
                        failure_log.append(
                            ("BACKUP_FAILED_SAFE_DRAIN", iteration, remaining, elapsed))
                        break
                    failure_log.append(("DEADLOCK_BACKUP_FAILED", iteration, remaining, elapsed))
                    metrics["aborted"] = True
                    break
                remaining = last_remaining = self._remaining
                no_progress_iters = 0
                ckpt_l2p   = l2p.copy()
                ckpt_p2l   = p2l.copy()
                ckpt_mark  = len(self._retired_log)
                ckpt_counters  = {k: metrics[k] for k in ckpt_counter_keys}
                ckpt_trace_len = len(metrics["trace"]) if metrics["trace"] is not None else 0

        metrics["compile_time"] = time.perf_counter() - _t_start
        metrics["failure_log"]  = failure_log
        # Iterations actually consumed, against the safe-mode worst case
        # computed from the gate count at entry (see `iterations_bound`).
        metrics["iterations"]       = iteration
        metrics["iterations_bound"] = self.iterations_bound(n_2q_at_entry,
                                                            deadlock_limit)
        metrics["deadlock_limit"]   = deadlock_limit
        # Transaction accounting.  `snapshots`/`rollbacks` cover every nested
        # transaction (`_apply_teleport`, `_relay_slack_to`, `_safe_route_gate`
        # ...); `wdag_rebuilds` are the O(N) whole-DAG replays a deadlock
        # checkpoint restore costs, which is the expensive kind.  The *_s
        # timings are only populated when config.profile_transactions is set,
        # so the hot path pays nothing by default (wdag_rebuild_s is always
        # timed -- it is rare and dominant when it happens).
        metrics["snapshots"]       = self._n_snapshots
        metrics["rollbacks"]       = self._n_rollbacks
        metrics["wdag_rebuilds"]   = self._n_wdag_rebuilds
        metrics["wdag_rebuild_s"]  = round(self._t_wdag_rebuild, 4)
        metrics["snapshot_s"]      = round(self._t_snapshot, 4)
        metrics["rollback_s"]      = round(self._t_rollback, 4)
        return metrics, l2p
