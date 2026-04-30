from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from config import HardwareConfig
from architecture import DistributedArchitecture
from eigen_cluster_hypergraph import Gate, EigenClusterHypergraph


@dataclass
class BurstCandidate:
    trigger_gate: object
    virt_idx:     int
    psrc:         int
    ns:           int
    pcomm_src:    int
    pcomm_dst:    int
    src_core:     int
    next_core:    int
    tgt_core:     int
    score:        float
    burst_value:  int = 0


class BurstDSABRE:

    def __init__(
        self,
        arch: DistributedArchitecture,
        config: HardwareConfig,
        weight_burst: float = 4.0,
        max_burst_normaliser: int = 6,
    ):
        self.arch = arch
        self.config = config
        self.weight_burst = weight_burst
        self.max_burst_norm = max(max_burst_normaliser, 1)

    def _idist(self, core_id: int, u: int, v: int) -> int:
        return self.arch.intra_dist[core_id][u][v]

    def _ipath(self, core_id: int, u: int, v: int) -> list:
        return self.arch.intra_path[core_id][u][v]

    def _free_slots(self, core_id: int, p2l: dict) -> int:
        return sum(1 for p in self.arch.core_qubits(core_id) if p2l[p] is None)

    def _evict_cost(self, p_comm: int, p2l: dict) -> int:
        if p2l[p_comm] is None:
            return 0
        ci = self.arch.core_of(p_comm)
        best = 999
        for pq in self.arch.core_qubits(ci):
            if p2l[pq] is None:
                d = self._idist(ci, p_comm, pq)
                if d < best:
                    best = d
        return best

    def _build_hg(self, wdag) -> Tuple[
        EigenClusterHypergraph,
        Dict[int, object],
        List,
    ]:
        """
        Build the HG from wdag (the working copy, not the original dag).
        gid2node must reference wdag node objects for remove_op_node to work.
        Single-qubit gates must be included: they define SuperNode boundaries.
        """
        hg = EigenClusterHypergraph()
        circuit: List[Gate] = []
        gid2node: Dict[int, object] = {}
        idx2lq = list(wdag.qubits)
        qubit_to_idx: Dict[object, int] = {q: i for i, q in enumerate(idx2lq)}

        for gid, node in enumerate(wdag.topological_op_nodes()):
            qubits = [qubit_to_idx[q] for q in node.qargs]
            params = [float(p) for p in (node.op.params or [])]
            circuit.append(Gate(gid, node.op.name.lower(), qubits, params))
            gid2node[gid] = node

        hg.build(circuit, num_qubits=wdag.num_qubits())
        return hg, gid2node, idx2lq

    def _make_hg_layout(self, idx2lq: List, l2p: dict) -> Dict[int, int]:
        layout = {}
        for idx, lq in enumerate(idx2lq):
            phys = l2p.get(lq)
            if phys is not None:
                layout[idx] = self.arch.core_of(phys)
        return layout

    def _local_swap_path(
        self,
        start_p: int,
        target_p: int,
        core_id: int,
        l2p: dict,
        p2l: dict,
        metrics: dict,
        forbidden: set | None = None,
    ):
        if start_p == target_p:
            return

        effective_forbidden: set = (
            (set(forbidden) - {start_p, target_p}) if forbidden else set()
        )

        # Use the pre-cached path when it doesn't pass through any forbidden node.
        cached = self.arch.intra_path[core_id].get(start_p, {}).get(target_p)
        if cached is not None and not any(n in effective_forbidden for n in cached):
            path = cached
        elif effective_forbidden:
            core_nodes = set(self.arch.core_qubits(core_id))
            G = self.arch.Gr.subgraph(core_nodes).copy()
            G.remove_nodes_from(effective_forbidden)
            try:
                path = nx.shortest_path(G, source=start_p, target=target_p)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Forbidden nodes disconnect the graph; fall back to unconstrained
                # cached path so the staging swap can still proceed.
                path = (cached if cached is not None
                        else self.arch.intra_path[core_id][start_p][target_p])
        else:
            raise RuntimeError(
                f"No valid path from {start_p} to {target_p} in core {core_id} "
                f"(architecture intra_path incomplete — this is a construction bug)"
            )

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            qa, qb = p2l[a], p2l[b]
            if qa is not None:
                l2p[qa] = b
            if qb is not None:
                l2p[qb] = a
            p2l[a], p2l[b] = qb, qa
            metrics["ls"] += 1
            metrics["cost"] += self.config.cost_local_swap
            if self.config.trace_routing:
                metrics["trace"].append(("SWAP", a, b, core_id))

    def _evict(self, p_comm: int, core_id: int,
               l2p: dict, p2l: dict, metrics: dict) -> None:
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
                a, b   = path[i], path[i + 1]
                qa, qb = p2l[a], p2l[b]
                if qa is not None: l2p[qa] = b
                if qb is not None: l2p[qb] = a
                p2l[a], p2l[b] = qb, qa
                metrics["ls"]   += 1
                metrics["cost"] += self.config.cost_local_swap
                if self.config.trace_routing:
                    metrics["trace"].append(("SWAP", a, b, core_id))

    def _apply_teleport(self, cand: BurstCandidate, l2p, p2l, metrics, idx2lq):
        self._evict(cand.pcomm_src, cand.src_core, l2p, p2l, metrics)
        if p2l[cand.pcomm_src] is not None:
            return   # eviction failed (core full) — skip this teleport
        self._evict(cand.pcomm_dst, cand.next_core, l2p, p2l, metrics)
        if p2l[cand.pcomm_dst] is not None:
            return   # eviction failed — skip
        virt = idx2lq[cand.virt_idx]
        self._local_swap_path(l2p[virt], cand.ns, cand.src_core, l2p, p2l, metrics,
                              forbidden={cand.pcomm_src})
        assert p2l[cand.ns] == virt, "Teleport staging failed"
        assert p2l[cand.pcomm_src] is None
        assert p2l[cand.pcomm_dst] is None
        virt_q = p2l[cand.ns]
        p2l[cand.ns] = None
        if virt_q is not None:
            l2p[virt_q] = cand.pcomm_dst
        p2l[cand.pcomm_dst] = virt_q
        hop_cost = (self.config.cost_teleport
                    + self.config.cost_teleport_per_hop
                    * self.arch.core_dist[cand.src_core][cand.next_core])
        metrics['teles'] += 1
        metrics['eprs']  += 1
        metrics['cost']  += hop_cost
        if self.config.trace_routing:
            metrics["trace"].append(
                ("TELE", cand.virt_idx, cand.psrc, cand.pcomm_dst,
                 cand.src_core, cand.next_core)
            )

    def _generate_burst_candidates(
        self,
        hg:          EigenClusterHypergraph,
        front_inter: List[Gate],
        idx2lq:      List,
        l2p:         dict,
        p2l:         dict,
    ) -> List[BurstCandidate]:
        """
        For each inter-core front-layer gate, score teleporting either endpoint
        one hop along the core graph.
        score = d_prep + cap - hop_gain - weight_burst * burst_count  (lower = better)
        """
        arch = self.arch
        config = self.config
        idx2phys: Dict[int, int] = {idx: l2p[lq] for idx, lq in enumerate(idx2lq)}
        free_cache: Dict[int, int] = {c: self._free_slots(c, p2l) for c in range(arch.num_cores)}

        # Deduplicate: the same (v_idx, srcc, nextc, lqsrc, lqdst) can arise from
        # multiple front-inter gates; evaluate each combination only once.
        considered: set = set()
        candidates: List[BurstCandidate] = []

        for g in front_inter:
            qidxs = g.qubits[:2]
            plist = [l2p[idx2lq[qi]] for qi in qidxs]
            clist = [arch.core_of(p) for p in plist]

            for v_idx, psrc, srcc, tgtc in [
                (qidxs[0], plist[0], clist[0], clist[1]),
                (qidxs[1], plist[1], clist[1], clist[0]),
            ]:
                for nextc in arch.core_graph.neighbors(srcc):
                    for lqsrc, lqdst in arch.inter_links_between(srcc, nextc):
                        # Need 2 free slots if lqdst is occupied (eviction required).
                        required = 1 if p2l[lqdst] is None else 2
                        if free_cache[nextc] < required:
                            continue

                        key = (v_idx, srcc, nextc, lqsrc, lqdst)
                        if key in considered:
                            continue
                        considered.add(key)

                        neighbors_s = list(arch.intra[srcc].neighbors(lqsrc))
                        if not neighbors_s:
                            continue
                        # Capture psrc by value to avoid lambda closure over loop variable.
                        ns = min(neighbors_s,
                                 key=lambda n, _p=psrc: self._idist(srcc, _p, n))

                        raw_count = hg.get_tele_gain_block(
                            v_idx, lqdst, psrc, idx2phys, arch,
                            max_depth=config.max_burst_walk_depth,
                        )
                        burst_count = min(raw_count, self.max_burst_norm)

                        d_prep = (self._idist(srcc, psrc, ns)
                                  + self._evict_cost(lqsrc, p2l)
                                  + self._evict_cost(lqdst, p2l))
                        cap = config.cap_penalty * max(0, config.capacity_threshold - free_cache[nextc])
                        hop_gain = config.hop_gain * (arch.core_dist[srcc][tgtc] - arch.core_dist[nextc][tgtc])
                        score = d_prep + cap - hop_gain - self.weight_burst * burst_count

                        candidates.append(BurstCandidate(
                            trigger_gate=g,
                            virt_idx=v_idx,
                            psrc=psrc,
                            ns=ns,
                            pcomm_src=lqsrc,
                            pcomm_dst=lqdst,
                            src_core=srcc,
                            next_core=nextc,
                            tgt_core=tgtc,
                            score=score,
                            burst_value=raw_count,
                        ))

        candidates.sort(key=lambda c: c.score)
        return candidates

    def _fallback_local_swap(
        self,
        front_inter: List[Gate],
        idx2lq:      List,
        l2p:         dict,
        p2l:         dict,
        metrics:     dict,
        node_decay:  dict,
    ) -> bool:
        arch = self.arch
        min_H = float("inf")
        best_swap = None

        for g in front_inter:
            for qi in g.qubits[:2]:
                phys = l2p[idx2lq[qi]]
                ci = arch.core_of(phys)
                comm_ports = arch._core_comm_ports[ci]
                if not comm_ports:
                    continue
                for v in arch.intra[ci].neighbors(phys):
                    d_before = min(self._idist(ci, phys, cp) for cp in comm_ports)
                    d_after  = min(self._idist(ci, v, cp) for cp in comm_ports)
                    H = max(node_decay.get(phys, 1.0),
                            node_decay.get(v, 1.0)) * (d_after - d_before)
                    if H < min_H:
                        min_H, best_swap = H, (phys, v, ci)

        if best_swap is None:
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
        node_decay[u]   = node_decay.get(u, 1.0) + 0.1
        node_decay[v]   = node_decay.get(v, 1.0) + 0.1
        return True

    def route(self, dag, initial_layout: dict):
        arch   = self.arch
        config = self.config

        # Build HG from wdag so gid2node maps to wdag node objects.
        wdag = deepcopy(dag)
        hg, gid2node, idx2lq = self._build_hg(wdag)

        # Remap initial_layout from original dag qubits to wdag qubits.
        # zip-pairing is safe: both lists are in the same positional order.
        dag_to_wdag = dict(zip(dag.qubits, wdag.qubits))
        l2p = {dag_to_wdag[lq]: p for lq, p in initial_layout.items()}
        p2l = {p: None for p in arch.Gr.nodes}
        for lq, p in l2p.items():
            p2l[p] = lq

        metrics = {
            "ls": 0, "teles": 0, "eprs": 0, "burst_saves": 0,
            "catcomms": 0, "cost": 0, "1q_gates": 0, "aborted": False,
            "compile_time": 0.0, "backup_activations": 0,
            "trace": [] if self.config.trace_routing else None,
        }
        failure_log = []
        node_decay  = {p: 1.0 for p in arch.Gr.nodes}
        _t_start    = time.perf_counter()

        ckpt_hg       = deepcopy(hg)
        ckpt_wdag     = deepcopy(wdag)
        ckpt_l2p      = l2p.copy()
        ckpt_p2l      = p2l.copy()
        ckpt_decay    = node_decay.copy()
        ckpt_gid2node = gid2node.copy()

        iteration      = 0
        last_remaining = len(list(wdag.op_nodes()))
        no_progress    = 0
        backup_attempts = 0

        while wdag.op_nodes():

            if iteration >= config.max_iterations:
                failure_log.append(("ITERATION_LIMIT", iteration,
                                    len(list(wdag.op_nodes())),
                                    time.perf_counter() - _t_start))
                metrics["aborted"] = True
                break
            iteration += 1

            # Phase 1: drain all immediately executable gates.
            # hg_layout is stable throughout this phase (no core crossings here).
            hg_layout = self._make_hg_layout(idx2lq, l2p)
            progress = True
            while progress:
                progress = False
                for hg_gate in hg.get_local_front_layer(hg_layout):
                    dag_node = gid2node.get(hg_gate.gate_id)
                    if dag_node is None:
                        continue
                    if len(hg_gate.qubits) < 2:
                        try:
                            hg.remove_gate(hg_gate.gate_id)
                            wdag.remove_op_node(dag_node)
                            metrics["1q_gates"] += 1
                            progress = True
                        except Exception:
                            pass
                        break
                    p0 = l2p[idx2lq[hg_gate.qubits[0]]]
                    p1 = l2p[idx2lq[hg_gate.qubits[1]]]
                    if arch.Gr.has_edge(p0, p1):
                        try:
                            hg.remove_gate(hg_gate.gate_id)
                            wdag.remove_op_node(dag_node)
                            node_decay[p0] = node_decay[p1] = 1.0
                            progress = True
                        except Exception:
                            pass
                        break

            if not wdag.op_nodes():
                break

            # Phase 2: classify front layer into intra-core (non-adjacent) and inter-core.
            hg_layout = self._make_hg_layout(idx2lq, l2p)

            front_intra = []
            for hg_gate in hg.get_local_front_layer(hg_layout):
                if len(hg_gate.qubits) < 2:
                    continue
                p0 = l2p[idx2lq[hg_gate.qubits[0]]]
                p1 = l2p[idx2lq[hg_gate.qubits[1]]]
                if not arch.Gr.has_edge(p0, p1):
                    front_intra.append(hg_gate)

            front_inter = [
                g for g in hg.get_remote_front_layer(hg_layout)
                if len(g.qubits) >= 2
            ]

            # Phase 3a: intra-core SWAP for non-adjacent same-core gates.
            if front_intra:
                best_swap, min_score = None, float("inf")
                involved = set()
                for g in front_intra:
                    involved.add(l2p[idx2lq[g.qubits[0]]])
                    involved.add(l2p[idx2lq[g.qubits[1]]])

                for u in involved:
                    ci = arch.core_of(u)
                    local_ci = [g for g in front_intra
                                if arch.core_of(l2p[idx2lq[g.qubits[0]]]) == ci]
                    for v in arch.Gr.neighbors(u):
                        if arch.core_of(v) != ci:
                            continue
                        dHf = sum(
                            self._idist(
                                ci,
                                v if l2p[idx2lq[g.qubits[0]]] == u else
                                (u if l2p[idx2lq[g.qubits[0]]] == v
                                 else l2p[idx2lq[g.qubits[0]]]),
                                v if l2p[idx2lq[g.qubits[1]]] == u else
                                (u if l2p[idx2lq[g.qubits[1]]] == v
                                 else l2p[idx2lq[g.qubits[1]]]),
                            ) - self._idist(
                                ci,
                                l2p[idx2lq[g.qubits[0]]],
                                l2p[idx2lq[g.qubits[1]]],
                            )
                            for g in local_ci
                        )
                        score = (
                            max(node_decay.get(u, 1.0), node_decay.get(v, 1.0))
                            * (dHf / max(len(local_ci), 1))
                        )
                        if score < min_score:
                            min_score, best_swap = score, (u, v)

                if best_swap:
                    u, v   = best_swap
                    qu, qv = p2l[u], p2l[v]
                    if qu is not None: l2p[qu] = v
                    if qv is not None: l2p[qv] = u
                    p2l[u], p2l[v] = qv, qu
                    metrics["ls"]   += 1
                    metrics["cost"] += config.cost_local_swap
                    if config.trace_routing:
                        metrics["trace"].append(("SWAP", u, v, arch.core_of(u)))
                    node_decay[u]   = node_decay.get(u, 1.0) + 0.1
                    node_decay[v]   = node_decay.get(v, 1.0) + 0.1

            # Phase 3b: burst teleportation for inter-core gates.
            elif front_inter:
                candidates = self._generate_burst_candidates(
                    hg, front_inter, idx2lq, l2p, p2l,
                )
                if not candidates:
                    if not self._fallback_local_swap(
                        front_inter, idx2lq, l2p, p2l, metrics, node_decay,
                    ):
                        failure_log.append((
                            "NO_CANDIDATES_NO_FALLBACK", iteration,
                            len(list(wdag.op_nodes())),
                            time.perf_counter() - _t_start,
                        ))
                        metrics["aborted"] = True
                        break
                else:
                    best = candidates[0]
                    self._apply_teleport(best, l2p, p2l, metrics, idx2lq)
                    metrics["burst_saves"] += best.burst_value
                    node_decay[best.pcomm_src] = 1.0
                    node_decay[best.pcomm_dst] = 1.0
                    hg_layout = self._make_hg_layout(idx2lq, l2p)

            # Phase 4: progress tracking and deadlock recovery.
            remaining = len(list(wdag.op_nodes()))
            if remaining < last_remaining:
                last_remaining = remaining
                no_progress    = 0
                ckpt_hg       = deepcopy(hg)
                ckpt_wdag     = deepcopy(wdag)
                ckpt_l2p      = l2p.copy()
                ckpt_p2l      = p2l.copy()
                ckpt_decay    = node_decay.copy()
                ckpt_gid2node = gid2node.copy()
            else:
                no_progress += 1

            if no_progress >= config.deadlock_limit:
                backup_attempts += 1
                elapsed = time.perf_counter() - _t_start
                if backup_attempts > config.max_backup_attempts:
                    failure_log.append(("DEADLOCK_EXHAUSTED", iteration, remaining, elapsed))
                    metrics["aborted"] = True
                    break

                hg         = deepcopy(ckpt_hg)
                wdag       = deepcopy(ckpt_wdag)
                l2p        = ckpt_l2p.copy()
                p2l        = ckpt_p2l.copy()
                node_decay = ckpt_decay.copy()
                gid2node   = ckpt_gid2node.copy()
                metrics["backup_activations"] += 1
                no_progress = 0

                hg_layout   = self._make_hg_layout(idx2lq, l2p)
                stuck_inter = [g for g in hg.get_remote_front_layer(hg_layout)
                               if len(g.qubits) >= 2]

                if stuck_inter:
                    stuck = max(
                        stuck_inter,
                        key=lambda g: arch.phys_dist.get(
                            l2p[idx2lq[g.qubits[0]]], {}
                        ).get(l2p[idx2lq[g.qubits[1]]], 0),
                    )
                    qi0, qi1 = stuck.qubits[0], stuck.qubits[1]
                    p1, p2   = l2p[idx2lq[qi0]], l2p[idx2lq[qi1]]
                    c1, c2   = arch.core_of(p1), arch.core_of(p2)
                    if c1 != c2:
                        hop_path = arch.core_path[c1][c2]
                        next_c   = hop_path[1] if len(hop_path) > 1 else c2
                        links    = arch.inter_links_between(c1, next_c)
                        if links:
                            lq_src, lq_dst = links[0]
                            nbrs = list(arch.intra[c1].neighbors(lq_src))
                            if nbrs:
                                n_s = min(nbrs, key=lambda n: self._idist(c1, p1, n))
                                forced = BurstCandidate(
                                    trigger_gate=stuck,
                                    virt_idx=qi0,
                                    psrc=p1,
                                    ns=n_s,
                                    pcomm_src=lq_src,
                                    pcomm_dst=lq_dst,
                                    src_core=c1,
                                    next_core=next_c,
                                    tgt_core=c2,
                                    score=0.0,
                                )
                                self._apply_teleport(forced, l2p, p2l, metrics, idx2lq)
                last_remaining = len(list(wdag.op_nodes()))

        metrics["compile_time"] = time.perf_counter() - _t_start
        metrics["failure_log"]  = failure_log
        return metrics, l2p
