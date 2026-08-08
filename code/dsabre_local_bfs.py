"""dSABRE_LocalBFS — intra-core extended set built by a seeded BFS closure.

Motivation
----------
`General_dSABRE_Router._get_local_extended` builds the per-core intra-core
lookahead set `E_c` by sweeping the *global* topological order of the working
DAG and propagating taint: a qubit is tainted by the first cross-core gate on
its wire, and taint spreads along wires forever.  A core's size-`L` quota is
therefore frequently unreachable, and the sweep runs to the end of the order on
72-100% of calls -- the `Theta(N_r)` term that makes compile time quadratic in
gate count (Appendix "Computational Complexity").

The term is an artefact of the *traversal*, not of the set.  Unfolding the taint
recursion, the admitted set is

    G = { g : g and every 2-qubit ancestor of g are intra-core },

the maximal ancestor-closed set of core-local gates, of which `E_c` takes the
first `L` elements lying in core `c`.  Two consequences of the layout being
frozen for the duration of the call:

  * `G` does not cross cores.  A directed path leaves a wire only at a gate,
    and every gate of `G` has both wires in one core, so `G = |_|_c G_c`, each
    `G_c` confined to core `c`'s wires.  Taint never has to be shared.
  * `G_c` is seeded by `F_c`.  Walking back from any `g` in `G_c` along a wire
    stays inside `G_c` and ends at that wire's earliest unexecuted gate, which
    is in the front layer by definition.  So `G_c` is the forward closure of
    `F_c` through core-local gates.

A Kahn expansion seeded at the front layer, expanding along core-local wires and
*pruning* at the first gate that is cross-core, already tainted, or in a core
that has filled its quota, therefore emits exactly `G_c` -- in `O(L + M)` per
core rather than `Theta(N_r)`: at most `L` gates emitted, at most one
terminating gate per wire (`|F_c| <= M/2` wires), and `O(1)` blocked successors
per emitted node.  An empty frontier is an exact exhaustion certificate, which
is what the default router's live-qubit early exit approximates.

Pruning is sound because taint is downstream-monotone: every gate below a dead
point on a wire has a tainted operand in the sweep, and is unreachable through
the live closure here, so both constructions reject it.

Difference from the default router
----------------------------------
The *set* and the per-gate depths agree whenever `|G_c| <= L` (the case that
makes the sweep expensive).  `depth` is `max(d_q1, d_q2) + 1`, longest-path
-from-front, which is invariant across topological orders of `G_c`.  The two
can differ only in emission *order*, and that is observable only when
`|G_c| > L` truncates -- and, marginally, through the order in which
`_best_intra_swap` accumulates the float `Delta_E` sum.

Replicated quirk
----------------
`_get_local_extended`'s `if id(n) in front_ids: continue` guard never fires:
Qiskit's Rust-backed DAGCircuit returns a fresh Python wrapper on every
accessor call, so `id()` taken from `front_layer()` never matches a node from
`topological_op_nodes()`.  Front-layer gates are therefore admitted into `E_c`
with depth 1 in the shipped router, and every published number was produced
that way.  `front_skip=False` (the default here) reproduces that behaviour;
`front_skip=True` implements the guard as written.
"""

from dsabre_ext import dSABRE_BFSExt


class dSABRE_LocalBFS(dSABRE_BFSExt):
    """dSABRE whose intra-core extended set is a seeded BFS closure."""

    #: reproduce the shipped router's dead `id(n) in front_ids` guard
    front_skip = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.le_calls = 0     # calls to _get_local_extended
        self.le_visits = 0    # DAG nodes examined across those calls

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        """Forward closure of the front layer through core-local gates.

        Returns the same `{core_id: [(gate, depth), ...]}` shape as the
        default router's topological sweep.
        """
        arch  = self.arch
        L     = self.config.lookahead_size
        indeg = self._indeg

        ext       = {ci: [] for ci in core_ids}
        remaining = set(core_ids)
        front_nids = {n._node_id for n in front} if self.front_skip else ()
        qubit_depth = {q: 0 for n in front for q in n.qargs}

        dec   = {}
        queue = list(front)
        visits = 0

        def commit(node):
            """Advance both wires past `node`; enqueue whatever became ready."""
            for succ in wdag.successors(node):
                sid = getattr(succ, '_node_id', None)
                if sid is None or not getattr(succ, 'qargs', None):
                    continue
                d = dec.get(sid, 0) + 1
                dec[sid] = d
                if indeg[sid] - d == 0:
                    queue.append(succ)

        head = 0
        while head < len(queue) and remaining:
            n = queue[head]
            head += 1
            visits += 1

            if len(n.qargs) < 2:
                # 1q gates are transparent: they neither taint nor admit.
                commit(n)
                continue

            q1, q2 = n.qargs[0], n.qargs[1]
            c1 = arch.core_of(l2p[q1])
            if c1 != arch.core_of(l2p[q2]):
                # Cross-core: both wires die here.  Everything below is dead
                # too, so the walk simply stops -- no taint set needed.
                continue
            if c1 not in remaining:
                # Core already at quota, or never requested.  Every gate below
                # this one lies in the same core, so none can be admitted.
                continue
            if self.front_skip and n._node_id in front_nids:
                commit(n)
                continue

            depth = max(qubit_depth.get(q1, 0), qubit_depth.get(q2, 0)) + 1
            ext[c1].append((n, depth))
            qubit_depth[q1] = qubit_depth[q2] = depth
            if len(ext[c1]) >= L:
                remaining.discard(c1)
            commit(n)

        self.le_calls  += 1
        self.le_visits += visits
        return ext


class dSABRE_LocalKahnLex(dSABRE_LocalBFS):
    """The same closure, emitted in the global topological order.

    `dSABRE_LocalBFS` walks the closure FIFO, which is a valid topological
    order but not Qiskit's.  The two therefore return the same *set* but can
    order it differently, which is observable once the size-`L` quota
    truncates.

    Qiskit's `topological_op_nodes` is a lexicographic topological sort keyed
    on the *highest* qubit index a node touches -- measured, and
    insertion-order independent: given the independent gates `cx(9,2)`,
    `cx(8,3)`, `cx(7,4)` it emits `(7,4), (8,3), (9,2)` whichever order they
    are written in, which rules out first-operand, last-operand and
    sorted-tuple keys.  The key is total on any ready set, since two
    *independent* nodes cannot share their highest qubit.

    Replaying that key over the closure alone reproduces the global order
    restricted to it.  A non-closure node can never release a closure node --
    the closure is ancestor-closed -- so emissions outside it neither reorder
    nor unblock anything inside it, and each closure choice therefore sees
    the same ready set and the same minimum.  The cost is
    `O((L+M) log(L+M))` instead of `O(L+M)`.
    """

    def _qkey(self, wdag, node):
        idx = self._qidx_cache
        best = -1
        for q in node.qargs:
            i = idx.get(q)
            if i is None:
                i = idx[q] = wdag.find_bit(q).index
            if i > best:
                best = i
        return best

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        import heapq

        if not hasattr(self, "_qidx_cache"):
            self._qidx_cache = {}
        arch  = self.arch
        L     = self.config.lookahead_size
        indeg = self._indeg

        ext       = {ci: [] for ci in core_ids}
        remaining = set(core_ids)
        qubit_depth = {q: 0 for n in front for q in n.qargs}

        dec  = {}
        heap = [(self._qkey(wdag, n), n._node_id, n) for n in front]
        heapq.heapify(heap)
        visits = 0

        def commit(node):
            for succ in wdag.successors(node):
                sid = getattr(succ, '_node_id', None)
                if sid is None or not getattr(succ, 'qargs', None):
                    continue
                d = dec.get(sid, 0) + 1
                dec[sid] = d
                if indeg[sid] - d == 0:
                    heapq.heappush(heap, (self._qkey(wdag, succ), sid, succ))

        while heap and remaining:
            _, _, n = heapq.heappop(heap)
            visits += 1

            if len(n.qargs) < 2:
                commit(n)
                continue
            q1, q2 = n.qargs[0], n.qargs[1]
            c1 = arch.core_of(l2p[q1])
            if c1 != arch.core_of(l2p[q2]) or c1 not in remaining:
                continue
            if self.front_skip and n._node_id in {m._node_id for m in front}:
                commit(n)
                continue

            depth = max(qubit_depth.get(q1, 0), qubit_depth.get(q2, 0)) + 1
            ext[c1].append((n, depth))
            qubit_depth[q1] = qubit_depth[q2] = depth
            if len(ext[c1]) >= L:
                remaining.discard(c1)
            commit(n)

        self.le_calls  += 1
        self.le_visits += visits
        return ext


class dSABRE_LocalScanCounted(dSABRE_BFSExt):
    """The shipped construction, instrumented with the same visit counter.

    Exists so the two traversals can be compared on work done, not only on
    wall time (which on this machine swings +/-30% run to run).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.le_calls = 0
        self.le_visits = 0

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        arch = self.arch
        tainted = set()
        front_ids = {id(n) for n in front}
        ext = {ci: [] for ci in core_ids}
        remaining = set(core_ids)
        qubit_depth = {q: 0 for n in front for q in n.qargs}

        live = {ci: 0 for ci in remaining}
        for q, p in l2p.items():
            c = arch.core_of(p)
            if c in live:
                live[c] += 1
        eligible = {c for c in remaining if live[c] >= 2}

        visits = 0
        for n in wdag.topological_op_nodes():
            if not eligible:
                break
            visits += 1
            if len(n.qargs) < 2:
                continue
            q1, q2 = n.qargs[0], n.qargs[1]
            c1, c2 = arch.core_of(l2p[q1]), arch.core_of(l2p[q2])
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

        self.le_calls  += 1
        self.le_visits += visits
        return ext


class dSABRE_SharedExt(dSABRE_BFSExt):
    """`E_c` is the restriction of the *inter-core* extended set `E` to core c.

    This is what Table~I of the paper already says `E_c` is -- "their
    restrictions to core c" -- and what the shipped router does not do: it
    builds a second, differently-constructed set (taint-propagated, per-core
    quota L, depth = longest path inside the local closure) and calls it
    `E_c`.

    Here both scorers read one set, built once by `_bfs_ext`: BFS layers over
    the remaining DAG, burst-priority within a layer, `dep(g)` = BFS layer
    index.  Every property Section III-E claims for the inter-core lookahead
    then holds for the intra-core one too -- in particular the size-L
    truncation cuts at a layer boundary, so the surviving gates are the
    shallow ones.

    Cost: `O(n+L)` for the one construction (`O(L)` if `E` is already cached,
    since it depends on the DAG alone) plus `O(L)` to partition it by core.
    No taint bookkeeping and no walk over the remaining DAG at all.

    `size_factor` scales how big the shared set is.  At 1 the two scorers read
    literally the same `L` gates, but a core sees only the ones with both
    operands resident in it, which on a K-core chip is a small fraction; at
    `K` each core gets a lookahead comparable in size to the shipped router's
    per-core quota.
    """

    size_factor = 1

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.le_calls = 0
        self.le_visits = 0
        self._dag_gen = 0        # bumped on every working-DAG mutation
        self._ext_key = None
        self._ext_val = None
        self.ext_builds = 0      # calls that had to run _bfs_ext
        self.ext_reuses = 0      # calls served from the memo

    # -- DAG generation counter -------------------------------------------
    # `E` depends on the DAG alone, never on the layout, so it needs
    # rebuilding only when the working DAG changes.  The two mutation points
    # are a retirement and a rollback; both bump the counter, which makes the
    # memo key exact.  (A retirement count alone is not: a rollback can bring
    # the log back to a length it already had with different contents.)

    def _init_dag_state(self, real_dag):
        self._dag_gen += 1
        self._ext_key = None
        return super()._init_dag_state(real_dag)

    def _on_retire(self, real_dag, node):
        self._dag_gen += 1
        return super()._on_retire(real_dag, node)

    def _rebuild_wdag(self, dag, mark):
        self._dag_gen += 1
        return super()._rebuild_wdag(dag, mark)

    def _inter_ext(self, dag, front, size):
        """`_bfs_ext`, memoised on the working DAG's generation.

        Both scorers read this one set: the teleport scorer directly, and the
        intra-core scorer through `_get_local_extended` below.  Within a run
        of iterations that retire nothing -- the common case, since a SWAP or
        a teleport removes no gate -- it is built once and partitioned `O(L)`
        per iteration.  Callers only read the returned list; it is shared, not
        copied.
        """
        key = (self._dag_gen, size)
        if key != self._ext_key:
            self._ext_val = self._bfs_ext(dag, front, size)
            self._ext_key = key
            self.ext_builds += 1
        else:
            self.ext_reuses += 1
        return self._ext_val

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        arch = self.arch
        size = self.config.lookahead_size * self.size_factor
        shared = self._inter_ext(wdag, front, size)

        ext = {ci: [] for ci in core_ids}
        for g, dep in shared:
            c1 = arch.core_of(l2p[g.qargs[0]])
            if c1 in ext and c1 == arch.core_of(l2p[g.qargs[1]]):
                ext[c1].append((g, dep))

        self.le_calls += 1
        self.le_visits += len(shared)
        return ext


class dSABRE_SharedExtK(dSABRE_SharedExt):
    """`dSABRE_SharedExt` with the shared set scaled by the core count, so a
    core's slice of it is comparable in size to the shipped per-core quota."""

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        self.size_factor = self.arch.num_cores
        return dSABRE_SharedExt._get_local_extended(
            self, wdag, front, core_ids, l2p)


class dSABRE_SharedExt2Q(dSABRE_SharedExt):
    """`dSABRE_SharedExt` with `dep(g)` counting *two-qubit* layers only.

    `_bfs_ext` increments `layer_depth` on every peeled layer, including ones
    made entirely of 1-qubit gates.  On opt3 native-gate circuits those are
    common -- 10.2% of the layers peeled on `qft_64`, which pushes the deepest
    `dep` in a size-20 set out to 30 -- so a gate one two-qubit gate beyond the
    front, but sitting behind a run of rz/sx, is discounted `gamma^5` rather
    than `gamma^1`.  At `gamma=0.9` that is a fivefold difference in weight for
    a gate that is, in routing terms, next.

    This variant advances the depth only on layers that contributed a 2Q gate,
    so `dep` is distance in two-qubit gates.  It changes the teleport scorer as
    well as the intra-core one, since both read this set.
    """

    def _bfs_ext(self, dag, front, size):
        front_nids = {n._node_id for n in front}
        front_qubits = {q for n in front for q in n.qargs}

        indeg = self._indeg
        dec = {}
        done = set()

        def commit(node):
            done.add(node._node_id)
            for succ in dag.successors(node):
                sid = getattr(succ, '_node_id', None)
                if sid is not None and getattr(succ, 'qargs', None) and sid not in done:
                    dec[sid] = dec.get(sid, 0) + 1

        for fn in front:
            commit(fn)

        ext = []
        current = list(front)
        layer_depth = 1

        while len(ext) < size:
            nxt_map = {}
            for node in current:
                for succ in dag.successors(node):
                    sid = getattr(succ, '_node_id', None)
                    if (sid is not None
                            and getattr(succ, 'qargs', None)
                            and sid not in done
                            and sid not in nxt_map
                            and indeg[sid] - dec.get(sid, 0) == 0):
                        nxt_map[sid] = succ

            if not nxt_map:
                break

            next_layer = list(nxt_map.values())
            for n in next_layer:
                commit(n)

            layer_2q = [n for n in next_layer if len(n.qargs) == 2]
            priority = [n for n in layer_2q if front_qubits & set(n.qargs)]
            rest = [n for n in layer_2q if n not in priority]

            for n in priority + rest:
                if len(ext) >= size:
                    break
                ext.append((n, layer_depth))

            current = next_layer
            # the one change: a layer of 1q gates alone is not a step in depth
            if layer_2q:
                layer_depth += 1

        return ext
