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
"""

import time
from collections import deque
from copy import deepcopy

import networkx as nx

from config import HardwareConfig
from architecture import DistributedArchitecture
from actions import TeleportAction


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

    def _extended_2q(self, dag, front, size):
        """Return up to `size` (gate, depth) pairs beyond the front layer.

        depth is the gate's DAG distance from the front layer (front = depth 0;
        immediate successors = depth 1).  Base implementation computes depths
        via a single topological pass tracking the longest predecessor chain.
        dSABRE_BurstExt overrides this with a BFS-layer expansion.
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
                        if free_cache.get(next_c, 0) < 1:
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

        `partner_phys`, if given, maps a logical qubit with a pending
        front-layer gate to that gate's partner's physical position -- passed
        through to `_evict`/`_force_make_room` for distance-aware eviction.
        """
        arch = self.arch
        cfg  = self.config

        # Transaction snapshot: layouts, scalar counters, trace length.  The
        # trace is truncated (not copied) on rollback, so the snapshot stays
        # O(qubits) no matter how long the routing trace has grown.
        saved_l2p = l2p.copy()
        saved_p2l = p2l.copy()
        saved_counters  = {k: v for k, v in metrics.items()
                           if not isinstance(v, (list, dict))}
        saved_trace_len = (len(metrics["trace"])
                           if metrics.get("trace") is not None else 0)

        def rollback() -> bool:
            l2p.clear();  l2p.update(saved_l2p)
            p2l.clear();  p2l.update(saved_p2l)
            metrics.update(saved_counters)
            if metrics.get("trace") is not None:
                del metrics["trace"][saved_trace_len:]
            return False

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
            try:
                staging_path = nx.shortest_path(
                    nx.restricted_view(arch.intra[a.src_core], [a.p_comm_src], []),
                    current_phys, a.n_s,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
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
        # a.virt in the trace.
        if (p2l[a.p_comm_src] is not None
                or p2l[a.p_comm_dst] is not None
                or p2l[a.n_s] != a.virt
                or l2p.get(a.virt) != a.n_s):
            return rollback()

        # ── Commit ────────────────────────────────────────────────────────
        p2l[a.n_s] = None
        p2l[a.p_comm_dst] = a.virt
        l2p[a.virt] = a.p_comm_dst
        hop_cost = (cfg.cost_teleport
                    + cfg.cost_teleport_per_hop
                    * arch.core_dist[a.src_core][a.next_core])
        metrics["teles"] += 1
        metrics["eprs"]  += 1
        metrics["cost"]  += hop_cost
        if cfg.trace_routing:
            # Record a.n_s, the slot virt actually teleports from -- not the
            # stale a.p_src captured at scoring time.
            metrics["trace"].append(
                ("TELE", a.virt, a.n_s, a.p_comm_dst, a.src_core, a.next_core)
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
        """Single-pass intra-core lookahead for a collection of cores.

        Returns dict mapping core_id -> list of (gate, depth) pairs
        (up to lookahead_size each).  depth is the gate's DAG distance from
        the front layer, tracked per qubit as the number of intra-core gates
        on that qubit's path since the front.  Tainted-qubit propagation is
        shared across all cores in one traversal.
        """
        arch = self.arch
        tainted = set()
        front_ids = {id(n) for n in front}
        ext = {ci: [] for ci in core_ids}
        remaining = set(core_ids)
        # qubit_depth[q] = depth of the last gate seen on qubit q (0 for front)
        qubit_depth: dict = {q: 0 for n in front for q in n.qargs}
        for n in wdag.topological_op_nodes():
            if not remaining:
                break
            if len(n.qargs) < 2:
                continue
            q1, q2 = n.qargs[0], n.qargs[1]
            c1, c2 = arch.core_of(l2p[q1]), arch.core_of(l2p[q2])
            if c1 != c2:
                tainted.update([q1, q2])
                continue
            if q1 in tainted or q2 in tainted:
                tainted.update([q1, q2])
                continue
            if id(n) in front_ids:
                continue
            if c1 in remaining:
                depth = max(qubit_depth.get(q1, 0), qubit_depth.get(q2, 0)) + 1
                ext[c1].append((n, depth))
                qubit_depth[q1] = qubit_depth[q2] = depth
                if len(ext[c1]) >= self.config.lookahead_size:
                    remaining.discard(c1)
        return ext

    def _fallback_local_swap(self, front_inter, wdag, l2p, p2l, metrics):
        """SABRE-style SWAP when no teleportation candidates are available."""
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
        chip (i.e. precondition (*) does not actually hold)."""
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
        1 free.  `protect` qubits (the gate's own two qubits) are never moved
        as filler.  Returns False only if no slack core exists at all (i.e.
        precondition (*) is violated) -- the caller should treat that as
        backup_plan failing this attempt, exactly as the old code did.
        """
        arch = self.arch
        if self._free_slots(target_core, arch, p2l) >= min_free:
            return True
        source = self._find_nearest_slack_core(target_core, p2l, min_free)
        if source is None:
            return False
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
                return False
            links = arch.inter_links_between(nxt, cur)
            if not links:
                return False
            p_comm_src, p_comm_dst = links[0]
            p_src = l2p[filler]
            n_s = min(arch.intra[nxt].neighbors(p_comm_src),
                      key=lambda n: self._idist(nxt, p_src, n))
            if not self._apply_teleport(
                    TeleportAction(None, filler, p_src, n_s, p_comm_src, p_comm_dst,
                                  nxt, cur, cur, score=0.0),
                    l2p, p2l, metrics,
            ):
                return False
            metrics["relay_hops"] += 1
        return True

    def _backup_plan(self, wdag, l2p, p2l, metrics):
        """Force the most-stuck gate closer to completion via greedy hops."""
        arch = self.arch
        front = self._front_2q(wdag)
        if not front:
            return False
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

    # ── Main routing loop ──────────────────────────────────────────────────────

    def route(self, dag, initial_layout):
        """Route `dag` under `initial_layout`.

        Returns
        -------
        (metrics, final_layout)
            metrics["eprs"]     : EPR pairs consumed
            metrics["ls"]       : local SWAPs applied
            metrics["aborted"]  : True if routing failed
            metrics["compile_time"] : wall-clock seconds
        """
        arch = self.arch
        l2p  = initial_layout.copy()
        p2l  = {p: None for p in arch.Gr.nodes}
        for lq, p in l2p.items():
            p2l[p] = lq
        wdag = deepcopy(dag)

        metrics = {
            "ls": 0, "teles": 0, "catcomms": 0, "eprs": 0,
            "cost": 0, "1q_gates": 0, "aborted": False,
            "compile_time": 0.0, "backup_activations": 0,
            # ── Mechanism instrumentation ─────────────────────────────────────
            # force_make_room : calls to _force_make_room (deadlock recovery)
            "force_make_room": 0,
            # relay_hops : filler teleports issued by _relay_room_to (only
            # nonzero when config.backup_relay_mode is True)
            "relay_hops": 0,
            "trace": [] if self.config.trace_routing else None,
        }
        failure_log = []
        _t_start = time.perf_counter()

        iteration        = 0
        last_remaining   = len(list(wdag.op_nodes()))
        no_progress_iters = 0
        backup_attempts  = 0
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
        ckpt_wdag  = deepcopy(wdag)
        ckpt_counters  = {k: metrics[k] for k in ckpt_counter_keys}
        ckpt_trace_len = len(metrics["trace"]) if metrics["trace"] is not None else 0
        prev_remaining = last_remaining

        while wdag.op_nodes():
            if iteration >= self.config.max_iterations:
                remaining = len(list(wdag.op_nodes()))
                elapsed   = time.perf_counter() - _t_start
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

            if not wdag.op_nodes():
                break

            current_remaining = len(list(wdag.op_nodes()))
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
                    involved = set()
                    for n in local_front:
                        involved.update([l2p[n.qargs[0]], l2p[n.qargs[1]]])
                    local_ext = local_ext_cache[ci]

                    best_swap, min_score = None, float("inf")
                    for u in involved:
                        for v in arch.Gr.neighbors(u):
                            if arch.core_of(v) != ci:
                                continue
                            delta_Hf = sum(
                                self._idist(ci,
                                            v if l2p[n.qargs[0]] == u else (u if l2p[n.qargs[0]] == v else l2p[n.qargs[0]]),
                                            v if l2p[n.qargs[1]] == u else (u if l2p[n.qargs[1]] == v else l2p[n.qargs[1]]))
                                - self._idist(ci, l2p[n.qargs[0]], l2p[n.qargs[1]])
                                for n in local_front
                            )
                            delta_He = sum(
                                (self.config.lookahead_decay ** depth) * (
                                    self._idist(ci,
                                                v if l2p[n.qargs[0]] == u else (u if l2p[n.qargs[0]] == v else l2p[n.qargs[0]]),
                                                v if l2p[n.qargs[1]] == u else (u if l2p[n.qargs[1]] == v else l2p[n.qargs[1]]))
                                    - self._idist(ci, l2p[n.qargs[0]], l2p[n.qargs[1]])
                                )
                                for n, depth in local_ext
                            )
                            score = ((delta_Hf / max(len(local_front), 1))
                                     + self.config.weight_extended * (delta_He / max(len(local_ext), 1)))
                            if score < min_score:
                                min_score, best_swap = score, (u, v)

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
                    extended_cache = self._extended_2q(wdag, front, self.config.lookahead_size)
                candidates = self._generate_candidates(front_inter, extended_cache, l2p, p2l,
                                                       committed_nid=committed_nid)
                if not candidates:
                    if not self._fallback_local_swap(front_inter, wdag, l2p, p2l, metrics):
                        remaining = len(list(wdag.op_nodes()))
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
                        extended_cache = None
                    elif not self._fallback_local_swap(front_inter, wdag, l2p, p2l, metrics):
                        remaining = len(list(wdag.op_nodes()))
                        elapsed   = time.perf_counter() - _t_start
                        failure_log.append(("ALL_CANDIDATES_REJECTED", iteration, remaining, elapsed))
                        metrics["aborted"] = True
                        break

            remaining = len(list(wdag.op_nodes()))
            if remaining < last_remaining:
                last_remaining    = remaining
                no_progress_iters = 0
                ckpt_l2p   = l2p.copy()
                ckpt_p2l   = p2l.copy()
                ckpt_wdag  = deepcopy(wdag)
                ckpt_counters  = {k: metrics[k] for k in ckpt_counter_keys}
                ckpt_trace_len = len(metrics["trace"]) if metrics["trace"] is not None else 0
            else:
                no_progress_iters += 1

            if no_progress_iters >= self.config.deadlock_limit:
                if not self.config.enable_deadlock_recovery:
                    elapsed = time.perf_counter() - _t_start
                    failure_log.append(("DEADLOCK_NO_RECOVERY", iteration, remaining, elapsed))
                    metrics["aborted"] = True
                    break
                backup_attempts += 1
                elapsed = time.perf_counter() - _t_start
                if backup_attempts > self.config.max_backup_attempts:
                    failure_log.append(("DEADLOCK_BACKUP_EXHAUSTED", iteration, remaining, elapsed))
                    metrics["aborted"] = True
                    break
                l2p        = ckpt_l2p.copy()
                p2l        = ckpt_p2l.copy()
                wdag       = deepcopy(ckpt_wdag)
                for k in ckpt_counter_keys:
                    metrics[k] = ckpt_counters[k]
                if metrics["trace"] is not None:
                    del metrics["trace"][ckpt_trace_len:]
                metrics["backup_activations"] += 1
                extended_cache = None
                committed_nid  = None  # l2p/wdag were rolled back to the checkpoint
                if not self._backup_plan(wdag, l2p, p2l, metrics):
                    failure_log.append(("DEADLOCK_BACKUP_FAILED", iteration, remaining, elapsed))
                    metrics["aborted"] = True
                    break
                remaining = last_remaining = len(list(wdag.op_nodes()))
                no_progress_iters = 0
                ckpt_l2p   = l2p.copy()
                ckpt_p2l   = p2l.copy()
                ckpt_wdag  = deepcopy(wdag)
                ckpt_counters  = {k: metrics[k] for k in ckpt_counter_keys}
                ckpt_trace_len = len(metrics["trace"]) if metrics["trace"] is not None else 0

        metrics["compile_time"] = time.perf_counter() - _t_start
        metrics["failure_log"]  = failure_log
        return metrics, l2p
