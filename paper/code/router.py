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
from copy import deepcopy

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

        Gates deeper in the list are discounted by decay^i (decay=1.0 → flat).
        """
        arch = self.arch
        old_phys = l2p[virt]
        old_core = arch.core_of(old_phys)
        new_core = arch.core_of(new_phys)
        delta = 0.0
        for i, g in enumerate(gates):
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
            delta += (decay ** i) * (old_dist - new_dist)
        return delta

    # ── Candidate generation ───────────────────────────────────────────────────

    def _extended_2q(self, dag, front, size):
        """Return up to `size` 2-qubit gates from the DAG beyond the front layer.

        Base implementation uses topological order.  dSABRE_BurstExt overrides
        this with a BFS-layer expansion that respects DAG dependency order and
        prioritises gates that share qubits with the front layer.
        """
        front_nids = {n._node_id for n in front}
        ext = []
        for n in dag.topological_op_nodes():
            if len(n.qargs) == 2 and n._node_id not in front_nids:
                ext.append(n)
                if len(ext) >= size:
                    break
        return ext

    def _generate_candidates(self, front_inter, extended, l2p, p2l):
        """Score all feasible one-hop teleportation moves and return sorted list.

        Scoring formula for each candidate:
            score = d_prep + cap - hop_gain - dF - weight_extended * dE

        where:
          d_prep    = intra-core SWAPs to reach the comm port + eviction cost
          cap       = penalty if destination core is nearly full
          hop_gain  = reward for moving closer to target core
          dF        = distance reduction on front-layer gates (immediate reward)
          dE        = distance reduction on extended-layer gates (lookahead)

        Lower score → better move.  The top candidate is executed each iteration.

        Also generates proactive congestion-relief moves: idle qubits in
        high-demand cores are offered as candidates to neighbouring cores with
        free capacity, preventing routing bottlenecks before they form.
        """
        arch = self.arch
        cfg  = self.config
        candidates = []
        free_cache = {c: self._free_slots(c, arch, p2l) for c in range(arch.num_cores)}

        for node in front_inter:
            q1, q2 = node.qargs[0], node.qargs[1]
            p1, p2 = l2p[q1], l2p[q2]
            c1, c2 = arch.core_of(p1), arch.core_of(p2)

            for (virt, p_src, src_c, tgt_c) in [(q1, p1, c1, c2), (q2, p2, c2, c1)]:
                for next_c in arch.core_graph.neighbors(src_c):
                    if free_cache[next_c] < 1:
                        continue
                    for (lq_src, lq_dst) in arch.inter_links_between(src_c, next_c):
                        n_s = min(
                            arch.intra[src_c].neighbors(lq_src),
                            key=lambda n: self._idist(src_c, p_src, n),
                        )
                        d_prep   = (self._idist(src_c, p_src, n_s)
                                    + self._evict_cost(lq_src, arch, p2l)
                                    + self._evict_cost(lq_dst, arch, p2l))
                        cap      = cfg.cap_penalty * max(0, cfg.capacity_threshold - free_cache[next_c])
                        hop_gain = cfg.hop_gain * (arch.core_dist[src_c][tgt_c]
                                                   - arch.core_dist[next_c][tgt_c])
                        dF = self._delta_front(virt, lq_dst, front_inter, l2p)
                        dE = self._delta_front(virt, lq_dst, extended, l2p,
                                               decay=cfg.lookahead_decay)
                        score = d_prep + cap - hop_gain - dF - cfg.weight_extended * dE
                        candidates.append(
                            TeleportAction(node, virt, p_src, n_s,
                                           lq_src, lq_dst, src_c, next_c, tgt_c, score)
                        )

        # ── Proactive congestion relief ────────────────────────────────────────
        if not cfg.enable_congestion_relief:
            candidates.sort(key=lambda a: a.score)
            return candidates
        demand = {c: 0 for c in range(arch.num_cores)}
        horizon = min(len(extended), cfg.demand_lookahead)
        for node in front_inter + extended[:horizon]:
            c1b = arch.core_of(l2p[node.qargs[0]])
            c2b = arch.core_of(l2p[node.qargs[1]])
            if c1b != c2b:
                demand[arch.core_path[c1b][c2b][1]] += 1
                demand[arch.core_path[c2b][c1b][1]] += 1

        next_use_depth = {}
        for depth, node in enumerate(front_inter + extended):
            for q in node.qargs:
                if q not in next_use_depth:
                    next_use_depth[q] = depth
        max_depth = len(front_inter) + len(extended) + 1

        core_busyness = {
            c: (len(arch.core_qubits(c)) - free_cache[c]) + demand[c]
            for c in range(arch.num_cores)
        }
        front_qubits = {l2p[n.qargs[0]] for n in front_inter} | {l2p[n.qargs[1]] for n in front_inter}

        for c_cong, d in demand.items():
            free = free_cache[c_cong]
            if d >= cfg.demand_threshold and free <= cfg.congestion_threshold:
                for c_relief in arch.core_graph.neighbors(c_cong):
                    if free_cache[c_relief] < cfg.relief_space_req:
                        continue
                    victims = sorted(
                        [p for p in arch.core_qubits(c_cong)
                         if p2l[p] is not None and p not in front_qubits],
                        key=lambda p: next_use_depth.get(p2l[p], max_depth),
                        reverse=True,
                    )[:2]
                    gradient = core_busyness[c_cong] - core_busyness[c_relief]
                    for v_phys in victims:
                        virt = p2l[v_phys]
                        depth_score = next_use_depth.get(virt, max_depth)
                        for (lq_src, lq_dst) in arch.inter_links_between(c_cong, c_relief):
                            n_s = min(
                                arch.intra[c_cong].neighbors(lq_src),
                                key=lambda n: self._idist(c_cong, v_phys, n),
                            )
                            d_prep = (self._idist(c_cong, v_phys, n_s)
                                      + self._evict_cost(lq_src, arch, p2l)
                                      + self._evict_cost(lq_dst, arch, p2l))
                            cap = cfg.cap_penalty * max(0, cfg.capacity_threshold - free_cache[c_relief])
                            dE  = self._delta_front(virt, lq_dst, extended, l2p,
                                                    decay=cfg.lookahead_decay)
                            score = (d_prep + cap - cfg.weight_extended * dE
                                     - cfg.relief_bonus * (d - free + 1)
                                     - cfg.relief_depth_weight * depth_score
                                     - cfg.relief_gradient_weight * gradient)
                            candidates.append(
                                TeleportAction(None, virt, v_phys, n_s,
                                               lq_src, lq_dst, c_cong, c_relief, c_relief, score)
                            )

        candidates.sort(key=lambda a: a.score)
        return candidates

    # ── Physical operations ────────────────────────────────────────────────────

    def _local_swap_path(self, p_src, p_dst, core_id, l2p, p2l, metrics):
        if p_src == p_dst:
            return
        path = self._ipath(core_id, p_src, p_dst)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            qa, qb = p2l[a], p2l[b]
            if qa is not None: l2p[qa] = b
            if qb is not None: l2p[qb] = a
            p2l[a], p2l[b] = qb, qa
            metrics["ls"]   += 1
            metrics["cost"] += self.config.cost_local_swap
            if self.config.trace_routing:
                metrics["trace"].append(("SWAP", a, b, core_id))

    def _evict(self, p_comm, core_id, l2p, p2l, metrics):
        """Move the qubit occupying p_comm to the nearest free slot (if any)."""
        if p2l[p_comm] is None:
            return
        free_dst = min(
            (pq for pq in self.arch.core_qubits(core_id) if p2l[pq] is None),
            key=lambda pq: self._idist(core_id, p_comm, pq),
            default=None,
        )
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

    def _apply_teleport(self, a: TeleportAction, l2p, p2l, metrics):
        """Execute teleportation: evict comm qubits, SWAP to staging slot, move."""
        self._evict(a.p_comm_src, a.src_core,  l2p, p2l, metrics)
        self._evict(a.p_comm_dst, a.next_core, l2p, p2l, metrics)
        self._local_swap_path(l2p[a.virt], a.n_s, a.src_core, l2p, p2l, metrics)
        virt = p2l[a.n_s]
        p2l[a.n_s] = None
        if virt is not None: l2p[virt] = a.p_comm_dst
        p2l[a.p_comm_dst] = virt
        hop_cost = (self.config.cost_teleport
                    + self.config.cost_teleport_per_hop
                    * self.arch.core_dist[a.src_core][a.next_core])
        metrics["teles"] += 1
        metrics["eprs"]  += 1
        metrics["cost"]  += hop_cost
        if self.config.trace_routing:
            metrics["trace"].append(
                ("TELE", a.virt, a.p_src, a.p_comm_dst, a.src_core, a.next_core)
            )

    # ── Intra-core helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _front_2q(dag):
        return [n for n in dag.front_layer() if len(n.qargs) == 2]

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        """Single-pass intra-core lookahead for a collection of cores.

        Returns dict mapping core_id -> list of lookahead gates (up to lookahead_size each).
        Tainted-qubit propagation is shared across all cores in one DAG traversal.
        """
        arch = self.arch
        tainted = set()
        front_ids = {id(n) for n in front}
        ext = {ci: [] for ci in core_ids}
        remaining = set(core_ids)
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
                ext[c1].append(n)
                if len(ext[c1]) >= self.config.lookahead_size:
                    remaining.discard(c1)
        return ext

    def _fallback_local_swap(self, front_inter, wdag, l2p, p2l, metrics, node_decay):
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
                    (self.config.lookahead_decay ** i) * (
                        self._idist(ci,
                                    v if l2p[n.qargs[0]] == u else (u if l2p[n.qargs[0]] == v else l2p[n.qargs[0]]),
                                    v if l2p[n.qargs[1]] == u else (u if l2p[n.qargs[1]] == v else l2p[n.qargs[1]]))
                        - self._idist(ci, l2p[n.qargs[0]], l2p[n.qargs[1]])
                    )
                    for i, n in enumerate(local_ext)
                )
                H = max(node_decay.get(u, 1.0), node_decay.get(v, 1.0)) * (
                    delta_Hf + self.config.weight_extended * (delta_He / max(len(local_ext), 1))
                )
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
        node_decay[u] = node_decay.get(u, 1.0) + 0.1
        node_decay[v] = node_decay.get(v, 1.0) + 0.1
        return True

    # ── Deadlock recovery ──────────────────────────────────────────────────────

    def _force_make_room(self, core_id, l2p, p2l, metrics):
        """Teleport any qubit out of a full core to free one slot."""
        metrics["force_make_room"] += 1
        arch = self.arch
        if self._free_slots(core_id, arch, p2l) > 0:
            return True
        outgoing = (
            [(u, v) for u, v in arch.inter_core_links
             if arch.core_of(u) == core_id and self._free_slots(arch.core_of(v), arch, p2l) > 0]
            + [(v, u) for u, v in arch.inter_core_links
               if arch.core_of(v) == core_id and self._free_slots(arch.core_of(u), arch, p2l) > 0]
        )
        if not outgoing:
            return False
        cp_out, cp_in = outgoing[0]
        neighbor_core = arch.core_of(cp_in)
        src = min(
            [p for p in arch.core_qubits(core_id) if p2l[p] is not None],
            key=lambda p: self._idist(core_id, p, cp_out),
        )
        n_s = min(
            list(arch.intra[core_id].neighbors(cp_out)),
            key=lambda n: self._idist(core_id, src, n),
        )
        self._evict(cp_out, core_id, l2p, p2l, metrics)
        self._evict(cp_in, neighbor_core, l2p, p2l, metrics)
        self._local_swap_path(src, n_s, core_id, l2p, p2l, metrics)
        virt = p2l[n_s]
        p2l[n_s] = None
        if virt is not None: l2p[virt] = cp_in
        p2l[cp_in] = virt
        metrics["teles"] += 1
        metrics["eprs"]  += 1
        metrics["cost"]  += self.config.cost_teleport
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
            return True

        moved_any = False
        for hop_idx in range(len(arch.core_path[c1][c2]) - 1):
            cur_core  = arch.core_of(l2p[q1])
            next_core = arch.core_path[c1][c2][hop_idx + 1]
            if cur_core == c2:
                break
            hopped = False
            for lq_src, lq_dst in arch.inter_links_between(cur_core, next_core):
                if (self._free_slots(next_core, arch, p2l) < 1
                        and not self._force_make_room(next_core, l2p, p2l, metrics)):
                    continue
                n_s = min(
                    list(arch.intra[cur_core].neighbors(lq_src)),
                    key=lambda n: self._idist(cur_core, l2p[q1], n),
                )
                self._apply_teleport(
                    TeleportAction(stuck, q1, l2p[q1], n_s, lq_src, lq_dst,
                                   cur_core, next_core, c2, score=0.0),
                    l2p, p2l, metrics,
                )
                moved_any = hopped = True
                break
            if not hopped:
                break

        np1, np2 = l2p[q1], l2p[q2]
        nc1, nc2 = arch.core_of(np1), arch.core_of(np2)
        if nc1 == nc2 and arch.Gr.has_edge(np1, np2):
            wdag.remove_op_node(stuck)
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
            # relief_candidates : total proactive-relief candidates generated
            # relief_picks      : iterations whose chosen teleport came from relief
            # force_make_room   : calls to _force_make_room (deadlock recovery)
            "relief_candidates": 0, "relief_picks": 0, "force_make_room": 0,
            "trace": [] if self.config.trace_routing else None,
        }
        failure_log = []
        _t_start = time.perf_counter()

        node_decay       = {p: 1.0 for p in arch.Gr.nodes}
        iteration        = 0
        last_remaining   = len(list(wdag.op_nodes()))
        no_progress_iters = 0
        backup_attempts  = 0
        extended_cache   = None

        ckpt_l2p   = l2p.copy()
        ckpt_p2l   = p2l.copy()
        ckpt_wdag  = deepcopy(wdag)
        ckpt_decay = node_decay.copy()
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
                        node_decay[p1] = node_decay[p2] = 1.0
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
                                (self.config.lookahead_decay ** i) * (
                                    self._idist(ci,
                                                v if l2p[n.qargs[0]] == u else (u if l2p[n.qargs[0]] == v else l2p[n.qargs[0]]),
                                                v if l2p[n.qargs[1]] == u else (u if l2p[n.qargs[1]] == v else l2p[n.qargs[1]]))
                                    - self._idist(ci, l2p[n.qargs[0]], l2p[n.qargs[1]])
                                )
                                for i, n in enumerate(local_ext)
                            )
                            score = max(node_decay[u], node_decay[v]) * (
                                (delta_Hf / max(len(local_front), 1))
                                + self.config.weight_extended * (delta_He / max(len(local_ext), 1))
                            )
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
                        node_decay[u] += 0.1
                        node_decay[v] += 0.1

            elif front_inter:
                # Inter-core teleportation: score candidates, execute best.
                if extended_cache is None:
                    extended_cache = self._extended_2q(wdag, front, self.config.lookahead_size)
                candidates = self._generate_candidates(front_inter, extended_cache, l2p, p2l)
                # Instrumentation: count proactive-relief candidates (those have node=None).
                metrics["relief_candidates"] += sum(1 for c in candidates if c.node is None)
                if not candidates:
                    if not self._fallback_local_swap(front_inter, wdag, l2p, p2l, metrics, node_decay):
                        remaining = len(list(wdag.op_nodes()))
                        elapsed   = time.perf_counter() - _t_start
                        failure_log.append(("NO_ACTIONS_NO_FALLBACK", iteration, remaining, elapsed))
                        metrics["aborted"] = True
                        break
                else:
                    best = candidates[0]
                    if best.node is None:
                        metrics["relief_picks"] += 1
                    self._apply_teleport(best, l2p, p2l, metrics)
                    node_decay[best.p_comm_src] = node_decay[best.p_comm_dst] = 1.0
                    extended_cache = None

            remaining = len(list(wdag.op_nodes()))
            if remaining < last_remaining:
                last_remaining    = remaining
                no_progress_iters = 0
                ckpt_l2p   = l2p.copy()
                ckpt_p2l   = p2l.copy()
                ckpt_wdag  = deepcopy(wdag)
                ckpt_decay = node_decay.copy()
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
                node_decay = ckpt_decay.copy()
                metrics["backup_activations"] += 1
                extended_cache = None
                if not self._backup_plan(wdag, l2p, p2l, metrics):
                    failure_log.append(("DEADLOCK_BACKUP_FAILED", iteration, remaining, elapsed))
                    metrics["aborted"] = True
                    break
                remaining = last_remaining = len(list(wdag.op_nodes()))
                no_progress_iters = 0
                ckpt_l2p   = l2p.copy()
                ckpt_p2l   = p2l.copy()
                ckpt_wdag  = deepcopy(wdag)
                ckpt_decay = node_decay.copy()

        metrics["compile_time"] = time.perf_counter() - _t_start
        metrics["failure_log"]  = failure_log
        return metrics, l2p
