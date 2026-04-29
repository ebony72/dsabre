import time
from copy import deepcopy
from config import HardwareConfig
from architecture import DistributedArchitecture
from actions import TeleportAction


class General_dSABRE_Router:
    def __init__(self, arch: DistributedArchitecture, config: HardwareConfig):
        self.arch = arch
        self.config = config

    def _idist(self, core_id: int, u: int, v: int) -> int:
        return self.arch.intra_dist[core_id][u][v]

    def _ipath(self, core_id: int, u: int, v: int) -> list:
        return self.arch.intra_path[core_id][u][v]

    def _free_slots(self, core_id, arch, p2l):
        return sum(1 for p in arch.core_qubits(core_id) if p2l[p] is None)

    def _evict_cost(self, p_comm, arch, p2l):
        if p2l[p_comm] is None:
            return 0
        ci = arch.core_of(p_comm)
        best = 999
        for pq in arch.core_qubits(ci):
            if p2l[pq] is None:
                d = self._idist(ci, p_comm, pq)
                if d < best:
                    best = d
        return best

    def _gate_dist(self, node, l2p):
        p1, p2 = l2p[node.qargs[0]], l2p[node.qargs[1]]
        return self.arch.phys_dist.get(p1, {}).get(p2, 999)

    def _delta_front(self, virt, new_phys, gates, l2p, decay=1.0):
        """
        Sum of (old_dist - new_dist) over gates involving virt.
        Gates at list position i are weighted by decay^i so that deeper lookahead
        gates have less influence on the score (decay=1.0 gives flat weighting).
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
                if old_dist == 999:
                    continue
            if new_core == p_core:
                new_dist = self._idist(new_core, new_phys, partner)
            else:
                new_dist = arch.phys_dist.get(new_phys, {}).get(partner, 999)
                if new_dist == 999:
                    continue
            delta += (decay ** i) * (old_dist - new_dist)
        return delta

    def _generate_candidates(self, front_inter, extended, l2p, p2l):
        arch = self.arch
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
                        neighbors_s = list(arch.intra[src_c].neighbors(lq_src))
                        n_s = min(neighbors_s, key=lambda n: self._idist(src_c, p_src, n))
                        d_prep = (self._idist(src_c, p_src, n_s)
                                  + self._evict_cost(lq_src, arch, p2l)
                                  + self._evict_cost(lq_dst, arch, p2l))
                        cap = self.config.cap_penalty * max(0, self.config.capacity_threshold - free_cache[next_c])
                        hop_gain = self.config.hop_gain * (arch.core_dist[src_c][tgt_c] - arch.core_dist[next_c][tgt_c])
                        dF = self._delta_front(virt, lq_dst, front_inter, l2p)
                        dE = self._delta_front(virt, lq_dst, extended, l2p,
                                               decay=self.config.lookahead_decay)
                        score = d_prep + cap - hop_gain - dF - self.config.weight_extended * dE
                        candidates.append(TeleportAction(node, virt, p_src, n_s, lq_src, lq_dst, src_c, next_c, tgt_c, score))

        # Proactive congestion relief: preemptively move idle qubits out of
        # high-demand cores before they become bottlenecks.
        demand = {c: 0 for c in range(arch.num_cores)}
        horizon = min(len(extended), self.config.demand_lookahead)
        lookahead_window = front_inter + extended[:horizon]

        next_use_depth = {}
        for depth, node in enumerate(front_inter + extended):
            for q in node.qargs:
                if q not in next_use_depth:
                    next_use_depth[q] = depth
        max_depth = len(front_inter) + len(extended) + 1

        for node in lookahead_window:
            c1, c2 = arch.core_of(l2p[node.qargs[0]]), arch.core_of(l2p[node.qargs[1]])
            if c1 != c2:
                demand[arch.core_path[c1][c2][1]] += 1
                demand[arch.core_path[c2][c1][1]] += 1

        core_busyness = {}
        for c in range(arch.num_cores):
            occupied = len(arch.core_qubits(c)) - free_cache[c]
            core_busyness[c] = occupied + demand[c]

        front_qubits = {l2p[n.qargs[0]] for n in front_inter} | {l2p[n.qargs[1]] for n in front_inter}

        for c_cong, d in demand.items():
            free = free_cache[c_cong]
            if d >= self.config.demand_threshold and free <= self.config.congestion_threshold:
                for c_relief in arch.core_graph.neighbors(c_cong):
                    if free_cache[c_relief] >= self.config.relief_space_req:
                        victims = [p for p in arch.core_qubits(c_cong)
                                   if p2l[p] is not None and p not in front_qubits]
                        victims.sort(key=lambda p: next_use_depth.get(p2l[p], max_depth), reverse=True)
                        victims = victims[:2]
                        gradient = core_busyness[c_cong] - core_busyness[c_relief]
                        sink_bonus = 1.0 * gradient
                        for v_phys in victims:
                            virt = p2l[v_phys]
                            depth_score = next_use_depth.get(virt, max_depth)
                            inactivity_bonus = 0.5 * depth_score
                            for (lq_src, lq_dst) in arch.inter_links_between(c_cong, c_relief):
                                neighbors_s = list(arch.intra[c_cong].neighbors(lq_src))
                                n_s = min(neighbors_s, key=lambda n: self._idist(c_cong, v_phys, n))
                                d_prep = (self._idist(c_cong, v_phys, n_s)
                                          + self._evict_cost(lq_src, arch, p2l)
                                          + self._evict_cost(lq_dst, arch, p2l))
                                cap = self.config.cap_penalty * max(0, self.config.capacity_threshold - free_cache[c_relief])
                                relief_bonus = self.config.relief_bonus * (d - free + 1)
                                dE = self._delta_front(virt, lq_dst, extended, l2p,
                                                       decay=self.config.lookahead_decay)
                                score = (d_prep + cap - self.config.weight_extended * dE
                                         - relief_bonus - inactivity_bonus - sink_bonus)
                                candidates.append(TeleportAction(None, virt, v_phys, n_s, lq_src, lq_dst,
                                                                  c_cong, c_relief, c_relief, score))

        candidates.sort(key=lambda a: a.score)
        return candidates

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
            metrics["ls"] += 1
            metrics["cost"] += self.config.cost_local_swap
            if self.config.trace_routing:
                metrics["trace"].append(("SWAP", a, b, core_id))

    def _evict(self, p_comm, core_id, l2p, p2l, metrics):
        if p2l[p_comm] is None:
            return
        free_dst = min(
            (pq for pq in self.arch.core_qubits(core_id) if p2l[pq] is None),
            key=lambda pq: self._idist(core_id, p_comm, pq),
            default=None
        )
        if free_dst is not None:
            path = self._ipath(core_id, p_comm, free_dst)
            for i in reversed(range(len(path) - 1)):
                a, b = path[i], path[i + 1]
                qa, qb = p2l[a], p2l[b]
                if qa is not None: l2p[qa] = b
                if qb is not None: l2p[qb] = a
                p2l[a], p2l[b] = qb, qa
                metrics["ls"] += 1
                metrics["cost"] += self.config.cost_local_swap
                if self.config.trace_routing:
                    metrics["trace"].append(("SWAP", a, b, core_id))

    def _apply_teleport(self, a: TeleportAction, l2p, p2l, metrics):
        self._evict(a.p_comm_src, a.src_core, l2p, p2l, metrics)
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

    @staticmethod
    def _front_2q(dag):
        return [n for n in dag.front_layer() if len(n.qargs) == 2]

    def _extended_2q(self, dag, front, size):
        front_set = set(id(n) for n in front)
        ext = []
        for n in dag.topological_op_nodes():
            if len(n.qargs) == 2 and id(n) not in front_set:
                ext.append(n)
                if len(ext) >= size:
                    break
        return ext

    def _get_local_extended(self, wdag, front, core_id, l2p):
        """Intra-core lookahead gates not blocked by any upstream inter-core gate."""
        arch = self.arch
        tainted = set()
        ext = []
        front_ids = {id(n) for n in front}

        for n in wdag.topological_op_nodes():
            if len(n.qargs) < 2:
                continue
            q1, q2 = n.qargs[0], n.qargs[1]
            c1, c2 = arch.core_of(l2p[q1]), arch.core_of(l2p[q2])
            if c1 != c2:
                tainted.add(q1)
                tainted.add(q2)
                continue
            if q1 in tainted or q2 in tainted:
                tainted.add(q1)
                tainted.add(q2)
                continue
            if id(n) in front_ids:
                continue
            if c1 == core_id:
                ext.append(n)
                if len(ext) >= self.config.lookahead_size:
                    break

        return ext

    def _fallback_local_swap(self, front_inter, wdag, l2p, p2l, metrics, node_decay):
        arch, best_swap, min_H = self.arch, None, float("inf")
        involved = set()
        for n in front_inter:
            involved.update([l2p[n.qargs[0]], l2p[n.qargs[1]]])

        local_ext_cache = {}
        for u in involved:
            ci = arch.core_of(u)
            if ci not in local_ext_cache:
                local_ext_cache[ci] = self._get_local_extended(wdag, front_inter, ci, l2p)
            local_ext = local_ext_cache[ci]

            comm_ports = arch._core_comm_ports[ci]
            if not comm_ports:
                continue

            for v in arch.intra[ci].neighbors(u):
                d_before = min(self._idist(ci, u, cp) for cp in comm_ports)
                d_after  = min(self._idist(ci, v, cp) for cp in comm_ports)
                delta_Hf = d_after - d_before

                delta_He = 0
                for _ei, n in enumerate(local_ext):
                    p0, p1 = l2p[n.qargs[0]], l2p[n.qargs[1]]
                    dist_before = self._idist(ci, p0, p1)
                    np0 = v if p0 == u else (u if p0 == v else p0)
                    np1 = v if p1 == u else (u if p1 == v else p1)
                    dist_after = self._idist(ci, np0, np1)
                    delta_He += (self.config.lookahead_decay ** _ei) * (dist_after - dist_before)

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
        metrics["ls"] += 1
        metrics["cost"] += self.config.cost_local_swap
        if self.config.trace_routing:
            metrics["trace"].append(("SWAP", u, v, ci))
        node_decay[u] = node_decay.get(u, 1.0) + 0.1
        node_decay[v] = node_decay.get(v, 1.0) + 0.1
        return True

    def _force_make_room(self, core_id, l2p, p2l, metrics):
        arch = self.arch
        if self._free_slots(core_id, arch, p2l) > 0:
            return True
        outgoing = (
            [(u, v) for u, v in arch.inter_core_links
             if arch.core_of(u) == core_id and self._free_slots(arch.core_of(v), arch, p2l) > 0] +
            [(v, u) for u, v in arch.inter_core_links
             if arch.core_of(v) == core_id and self._free_slots(arch.core_of(u), arch, p2l) > 0]
        )
        if not outgoing:
            return False
        cp_out, cp_in = outgoing[0]
        neighbor_core = arch.core_of(cp_in)
        src = min(
            [p for p in arch.core_qubits(core_id) if p2l[p] is not None],
            key=lambda p: self._idist(core_id, p, cp_out)
        )
        n_s = min(
            list(arch.intra[core_id].neighbors(cp_out)),
            key=lambda n: self._idist(core_id, src, n)
        )
        self._evict(cp_out, core_id, l2p, p2l, metrics)
        self._evict(cp_in, neighbor_core, l2p, p2l, metrics)
        self._local_swap_path(src, n_s, core_id, l2p, p2l, metrics)
        virt = p2l[n_s]
        p2l[n_s] = None
        if virt is not None: l2p[virt] = cp_in
        p2l[cp_in] = virt
        metrics["teles"] += 1
        metrics["eprs"] += 1
        metrics["cost"] += self.config.cost_teleport
        return True

    def _backup_plan(self, wdag, l2p, p2l, metrics):
        arch, front = self.arch, self._front_2q(wdag)
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
                metrics["ls"] += 1
                metrics["cost"] += self.config.cost_local_swap
                if self.config.trace_routing:
                    metrics["trace"].append(("SWAP", a, b, arch.core_of(a)))
            np1, np2 = l2p[q1], l2p[q2]
            if arch.Gr.has_edge(np1, np2):
                wdag.remove_op_node(stuck)
            return True

        moved_any = False
        for hop_idx in range(len(arch.core_path[c1][c2]) - 1):
            cur_core = arch.core_of(l2p[q1])
            next_core = arch.core_path[c1][c2][hop_idx + 1]
            if cur_core == c2:
                break
            hopped = False
            for lq_src, lq_dst in arch.inter_links_between(cur_core, next_core):
                if self._free_slots(next_core, arch, p2l) < 1 and not self._force_make_room(next_core, l2p, p2l, metrics):
                    continue
                n_s = min(
                    list(arch.intra[cur_core].neighbors(lq_src)),
                    key=lambda n: self._idist(cur_core, l2p[q1], n)
                )
                self._apply_teleport(
                    TeleportAction(stuck, q1, l2p[q1], n_s, lq_src, lq_dst, cur_core, next_core, c2, score=0.0),
                    l2p, p2l, metrics
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
        elif nc1 == nc2 and not arch.Gr.has_edge(np1, np2):
            try:
                path = self._ipath(nc1, np1, np2)
                for i in range(len(path) - 2):
                    a, b = path[i], path[i + 1]
                    qa, qb = p2l[a], p2l[b]
                    if qa is not None: l2p[qa] = b
                    if qb is not None: l2p[qb] = a
                    p2l[a], p2l[b] = qb, qa
                    metrics["ls"] += 1
                    metrics["cost"] += self.config.cost_local_swap
                    if self.config.trace_routing:
                        metrics["trace"].append(("SWAP", a, b, arch.core_of(a)))
                moved_any = True
                if arch.Gr.has_edge(l2p[q1], l2p[q2]):
                    wdag.remove_op_node(stuck)
            except Exception:
                pass

        return moved_any

    def route(self, dag, initial_layout):
        arch = self.arch
        l2p = initial_layout.copy()
        p2l = {p: None for p in arch.Gr.nodes}
        for lq, p in l2p.items():
            p2l[p] = lq
        wdag = deepcopy(dag)

        metrics = {
            "ls": 0, "teles": 0, "catcomms": 0, "eprs": 0,
            "cost": 0, "1q_gates": 0, "aborted": False,
            "compile_time": 0.0, "backup_activations": 0,
            "trace": [] if self.config.trace_routing else None,
        }
        failure_log = []
        _t_start = time.perf_counter()
        node_decay = {p: 1.0 for p in arch.Gr.nodes}
        iteration = 0
        last_remaining = len(list(wdag.op_nodes()))
        no_progress_iters = 0
        backup_attempts = 0

        ckpt_l2p   = l2p.copy()
        ckpt_p2l   = p2l.copy()
        ckpt_wdag  = deepcopy(wdag)
        ckpt_decay = node_decay.copy()

        extended_cache = None
        prev_remaining = last_remaining

        while wdag.op_nodes():
            if iteration >= self.config.max_iterations:
                remaining = len(list(wdag.op_nodes()))
                elapsed = time.perf_counter() - _t_start
                failure_log.append(("ITERATION_LIMIT", iteration, remaining, elapsed))
                metrics["aborted"] = True
                metrics["compile_time"] = elapsed
                break
            iteration += 1

            progress = True
            while progress:
                progress = False

                front_all = list(wdag.front_layer())
                for node in front_all:
                    if len(node.qargs) < 2:
                        metrics["1q_gates"] += 1
                        wdag.remove_op_node(node)
                        progress = True

                front = self._front_2q(wdag)
                if not front:
                    break

                intra_exec = []
                for node in front:
                    p1, p2 = l2p[node.qargs[0]], l2p[node.qargs[1]]
                    if arch.core_of(p1) == arch.core_of(p2) and arch.Gr.has_edge(p1, p2):
                        intra_exec.append((node, p1, p2))

                if intra_exec:
                    for node, p1, p2 in intra_exec:
                        wdag.remove_op_node(node)
                        node_decay[p1] = node_decay[p2] = 1.0
                    progress = True

            if not wdag.op_nodes():
                break

            current_remaining = len(list(wdag.op_nodes()))
            if current_remaining < prev_remaining:
                extended_cache = None
            prev_remaining = current_remaining

            front = self._front_2q(wdag)
            front_intra = [n for n in front
                           if arch.core_of(l2p[n.qargs[0]]) == arch.core_of(l2p[n.qargs[1]])]
            front_inter = [n for n in front if n not in front_intra]

            if front_intra:
                involved = set()
                for n in front_intra:
                    involved.update([l2p[n.qargs[0]], l2p[n.qargs[1]]])
                local_ext_cache = {
                    core_id: self._get_local_extended(wdag, front, core_id, l2p)
                    for core_id in range(arch.num_cores)
                }

                best_swap, min_score = None, float("inf")
                for u in involved:
                    ci = arch.core_of(u)
                    local_front = [n for n in front_intra if arch.core_of(l2p[n.qargs[0]]) == ci]
                    local_ext = local_ext_cache[ci]

                    for v in arch.Gr.neighbors(u):
                        if arch.core_of(v) != ci:
                            continue

                        delta_Hf = 0
                        for n in local_front:
                            p0, p1 = l2p[n.qargs[0]], l2p[n.qargs[1]]
                            dist_before = self._idist(ci, p0, p1)
                            np0 = v if p0 == u else (u if p0 == v else p0)
                            np1 = v if p1 == u else (u if p1 == v else p1)
                            dist_after = self._idist(ci, np0, np1)
                            delta_Hf += (dist_after - dist_before)

                        delta_He = 0
                        for _ei, n in enumerate(local_ext):
                            p0, p1 = l2p[n.qargs[0]], l2p[n.qargs[1]]
                            dist_before = self._idist(ci, p0, p1)
                            np0 = v if p0 == u else (u if p0 == v else p0)
                            np1 = v if p1 == u else (u if p1 == v else p1)
                            dist_after = self._idist(ci, np0, np1)
                            delta_He += (self.config.lookahead_decay ** _ei) * (dist_after - dist_before)

                        score = max(node_decay[u], node_decay[v]) * (
                            (delta_Hf / max(len(local_front), 1)) +
                            self.config.weight_extended * (delta_He / max(len(local_ext), 1))
                        )

                        if score < min_score:
                            min_score, best_swap = score, (u, v)

                if best_swap:
                    u, v = best_swap
                    qu, qv = p2l[u], p2l[v]
                    if qu is not None: l2p[qu] = v
                    if qv is not None: l2p[qv] = u
                    p2l[u], p2l[v] = qv, qu
                    metrics["ls"] += 1
                    metrics["cost"] += self.config.cost_local_swap
                    if self.config.trace_routing:
                        metrics["trace"].append(("SWAP", u, v, arch.core_of(u)))
                    node_decay[u] += 0.1
                    node_decay[v] += 0.1

            elif front_inter:
                if extended_cache is None:
                    extended_cache = self._extended_2q(wdag, front, self.config.lookahead_size)

                candidates = self._generate_candidates(front_inter, extended_cache, l2p, p2l)
                if not candidates:
                    if not self._fallback_local_swap(front_inter, wdag, l2p, p2l, metrics, node_decay):
                        remaining = len(list(wdag.op_nodes()))
                        elapsed = time.perf_counter() - _t_start
                        failure_log.append(("NO_ACTIONS_NO_FALLBACK", iteration, remaining, elapsed))
                        metrics["aborted"] = True
                        break
                else:
                    best = candidates[0]
                    self._apply_teleport(best, l2p, p2l, metrics)
                    node_decay[best.p_comm_src] = node_decay[best.p_comm_dst] = 1.0
                    extended_cache = None

            remaining = len(list(wdag.op_nodes()))
            if remaining < last_remaining:
                last_remaining = remaining
                no_progress_iters = 0
                ckpt_l2p   = l2p.copy()
                ckpt_p2l   = p2l.copy()
                ckpt_wdag  = deepcopy(wdag)
                ckpt_decay = node_decay.copy()
            else:
                no_progress_iters += 1

            if no_progress_iters >= self.config.deadlock_limit:
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
        metrics["failure_log"] = failure_log
        return metrics, l2p
