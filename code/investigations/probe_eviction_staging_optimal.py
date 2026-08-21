"""probe_eviction_staging_optimal.py -- how much does the greedy staging/eviction
plan (TODO.md item 1) cost against the true optimum?

Setting (see investigations/WRITEUP_eviction_recovery_2026-08.md for the full
derivation): inside one core's coupling graph G, given a fixed port t and the
gate operand q currently at s, the router picks n_s = argmin_{n in N(t)} of the
UNRESTRICTED distance from s to n, then executes evict(t) followed by a
staging walk of q to n_s in G - t.  That is one feasible plan; it is not
obviously the cheapest, and the eviction walk can shift q as a side effect if
s lies on the path from t to the nearest hole.

This script runs the *shipped, unmodified* production router (dSABRE_BFSExt,
safe_mode=True, exactly benchmark.py's 64q configuration) over real 64q-suite
circuits, and at every APPLIED teleport hooks `_apply_teleport` right before
it mutates state to:

  1. read off the shipped plan's actual cost (real local-SWAP count spent on
     the source-core eviction + staging, isolated from the destination-core
     eviction, which is a separate single-target shortest-path problem and
     already optimal on its own -- see the writeup);
  2. compute the true optimum by exact BFS over the *reduced* state (which
     vertices are free, and where q sits) -- this abstraction is valid
     because only q's identity is ever distinguished; every other qubit is
     interchangeable for the purpose of "is t empty and is q at some n_s in
     N(t)", so the state space is (free-set, q-position) rather than a full
     permutation, and is small enough (core graphs here have 16 sites) for
     BFS to be exact and fast;
  3. cross-check that BFS optimum against the closed-form decision rule from
     the writeup (try each candidate hole f, use a t-f geodesic through s if
     one exists and helps, otherwise the plain d(t,f)+d(s,n_s) baseline).

The router's own trajectory is completely unaffected -- this class only reads
state before calling the real `_apply_teleport`, never alters it.

Usage:  python3 probe_eviction_staging_optimal.py [--circuits a,b,c] [--max-samples N]
"""
import argparse
import glob
import os
import sys
import warnings
from collections import deque

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BFSExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from circuit_paths import circuits_path

_HW = HardwareConfig(deadlock_limit=None, max_backup_attempts=None, max_iterations=None)
SUFFIX = "_nativegates_ibm_qiskit_opt3_64.qasm"
CANON64 = ["ae", "ghz", "graphstate", "qft", "qnn", "random",
           "qpeexact", "qaoa", "multiplier"]


# ── Exact optimum: BFS over the (free-set, q-position) reduced state ───────────

def bfs_optimal_cost(adj, t, s, free0):
    """Min swaps so that t is free and q sits at some neighbour of t.

    `adj`: {vertex: [neighbours]}. `free0`: frozenset of free vertices at
    entry (must not contain s). Two concrete occupancy states are equivalent
    for this question iff they agree on (which vertices are free, where q
    is) -- every other qubit is interchangeable -- so that pair is the whole
    state; BFS over it is exact, not a heuristic.
    """
    targets_n = adj[t]  # any neighbour of t is an acceptable final q-position

    def is_goal(free, qpos):
        return t in free and qpos in targets_n

    start = (free0, s)
    if is_goal(*start):
        return 0, start
    visited = {start}
    frontier = [start]
    dist = 0
    while frontier:
        dist += 1
        nxt = []
        for free, qpos in frontier:
            for u in ([qpos] + list(free)):
                for v in adj[u]:
                    if v == u:
                        continue
                    # SWAP(u, v); only transitions that can change (free, qpos)
                    if u == qpos:
                        if v in free:
                            new_free = (free - {v}) | {u}
                            new_q = v
                        else:
                            new_free, new_q = free, v
                    else:  # u in free, v != qpos and v not in free (else no-op)
                        if v == qpos:
                            new_free = (free - {u}) | {v}
                            new_q = u
                        elif v in free:
                            continue  # hole-hole swap, no-op
                        else:
                            new_free = (free - {u}) | {v}
                            new_q = qpos
                    state = (new_free, new_q)
                    if state in visited:
                        continue
                    if is_goal(*state):
                        return dist, state
                    visited.add(state)
                    nxt.append(state)
        frontier = nxt
    return None, None  # unreachable (shouldn't happen on a connected core)


# ── Closed-form decision rule (the writeup's Section on the shift lemma) ──────

def closed_form_cost(dist_mat, adj, t, s, free0):
    """min over n_s in N(t), f in free0 of two options:

      A. evict along a t-f path that AVOIDS s entirely (q never moves), then
         stage q from s to n_s in G-t:
            d_{G-s}(t, f) + d_{G-t}(s, n_s)
         `d_{G-s}(t,f)` -- G with *s* deleted -- not the plain d(t,f):
         removing s can force a longer detour, and skipping that check was
         an actual bug caught by the BFS cross-check below (t on a grid
         boundary, s one of only two routes to the nearest hole).

      B. evict along a *shortest* t-f path that happens to pass through s,
         which shifts q one step toward f for free -- worth using only when
         it costs nothing extra (d(t,s)+d(s,f) == d(t,f)); a forced detour
         to manufacture this is never worth it; see the writeup's parity
         argument (grids are bipartite, so a detour costs >=2 while the
         shift saves at most 1).
    """
    dist_full = dist_mat["full"]
    dist_minus_t = dist_mat["minus_t"]
    dist_minus_s = dist_mat["minus_s"]
    dts = dist_full[t].get(s)
    best = None
    for f in free0:
        dtf = dist_full[t].get(f)
        if dtf is None:
            continue
        dsf = dist_full[s].get(f)
        on_path = dts is not None and dsf is not None and dts + dsf == dtf

        d_avoid = dist_minus_s.get(t, {}).get(f)
        if d_avoid is not None:
            for n_s in adj[t]:
                stage = dist_minus_t.get(s, {}).get(n_s)
                if stage is not None:
                    cand = d_avoid + stage
                    if best is None or cand < best:
                        best = cand

        if on_path:
            for sp in adj[s]:
                if (dist_full[t].get(sp) == dts + 1
                        and dist_full[sp].get(f) == dsf - 1):
                    for n_s in adj[t]:
                        stage = dist_minus_t.get(sp, {}).get(n_s)
                        if stage is not None:
                            cand = dtf + stage
                            if best is None or cand < best:
                                best = cand
    return best


def all_pairs_bfs(adj, exclude=None):
    """All-pairs shortest paths on `adj`, optionally with one vertex deleted."""
    verts = [v for v in adj if v != exclude]
    out = {}
    for src in verts:
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == exclude or v in d:
                    continue
                d[v] = d[u] + 1
                q.append(v)
        out[src] = d
    return out


# ── Hook into the shipped router ────────────────────────────────────────────

class AuditRouter(dSABRE_BFSExt):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.samples = []
        self.max_samples = None  # set by the caller after construction

    def _apply_teleport(self, action, l2p, p2l, metrics, partner_phys=None):
        arch = self.arch
        core = action.src_core
        t = action.p_comm_src
        s = l2p.get(action.virt)
        if (self.max_samples is None or len(self.samples) < self.max_samples) and s is not None:
            adj = {v: list(arch.intra[core].neighbors(v)) for v in arch.core_qubits(core)}
            free0 = frozenset(v for v in adj if p2l.get(v) is None)
            if s not in free0 and t in adj:
                dist_full = all_pairs_bfs(adj)
                dist_minus_t = all_pairs_bfs(adj, exclude=t)
                dist_minus_s = all_pairs_bfs(adj, exclude=s)
                shipped_ns = action.n_s
                d_evict_dst = self._evict_cost(action.p_comm_dst, arch, p2l)
                ls_before = metrics["ls"]

                def run_real():
                    return super(AuditRouter, self)._apply_teleport(
                        action, l2p, p2l, metrics, partner_phys=partner_phys)

                ok = run_real()
                if ok:
                    ls_after = metrics["ls"]
                    shipped_src_cost = (ls_after - ls_before) - d_evict_dst
                    opt_cost, _ = bfs_optimal_cost(adj, t, s, free0)
                    cf_cost = closed_form_cost(
                        {"full": dist_full, "minus_t": dist_minus_t,
                         "minus_s": dist_minus_s}, adj, t, s, free0)
                    self.samples.append(dict(
                        core_size=len(adj), n_free=len(free0),
                        shipped_ns=shipped_ns,
                        shipped_cost=shipped_src_cost,
                        optimal_cost=opt_cost,
                        closed_form_cost=cf_cost,
                    ))
                return ok
        return super()._apply_teleport(action, l2p, p2l, metrics, partner_phys=partner_phys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default=",".join(CANON64))
    ap.add_argument("--max-samples", type=int, default=2000)
    args = ap.parse_args()

    arch = build_h_grid_architecture(r=2, s=3, m=4)
    circuit_dir = circuits_path("qasm_64")
    names = args.circuits.split(",")

    router = AuditRouter(arch, _HW)
    router.max_samples = args.max_samples

    for cname in names:
        if router.max_samples is not None and len(router.samples) >= router.max_samples:
            break
        pattern = os.path.join(circuit_dir, f"{cname}{SUFFIX}")
        matches = glob.glob(pattern)
        if not matches:
            print(f"[{cname}: not found at {pattern}]", flush=True)
            continue
        qasm_path = matches[0]
        qc = QuantumCircuit.from_qasm_file(qasm_path)
        qc = qc.remove_final_measurements(inplace=False)
        qc = PassManager([RemoveBarriers()]).run(qc)
        dag = circuit_to_dag(qc)
        rev_dag = circuit_to_dag(qc.reverse_ops())

        n_before = len(router.samples)
        sl_layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        layout = sl_layouts[0]
        m = run_sabre_passes(router, dag, rev_dag, layout)
        n_after = len(router.samples)
        status = "aborted" if (m is None or m.get("aborted")) else f"eprs={m['eprs']}"
        print(f"{cname}: {status}, +{n_after - n_before} samples "
              f"(total {n_after})", flush=True)

    samples = router.samples
    print(f"\n{len(samples)} teleport actions sampled.\n")
    if not samples:
        return

    n_bfs_missing = sum(1 for s in samples if s["optimal_cost"] is None)
    n_cf_mismatch = sum(1 for s in samples
                        if s["optimal_cost"] is not None and s["closed_form_cost"] is not None
                        and s["optimal_cost"] != s["closed_form_cost"])
    n_cf_missing = sum(1 for s in samples if s["closed_form_cost"] is None)
    print(f"BFS optimum unreachable (should be 0): {n_bfs_missing}")
    print(f"Closed-form vs BFS mismatches (should be 0): {n_cf_mismatch}")
    print(f"Closed-form returned None (should be 0): {n_cf_missing}")

    valid = [s for s in samples if s["optimal_cost"] is not None]
    n_tight = sum(1 for s in valid if s["shipped_cost"] == s["optimal_cost"])
    total_shipped = sum(s["shipped_cost"] for s in valid)
    total_optimal = sum(s["optimal_cost"] for s in valid)
    gap = [s["shipped_cost"] - s["optimal_cost"] for s in valid]
    print(f"\nGreedy already optimal:  {n_tight}/{len(valid)} "
          f"({100*n_tight/len(valid):.1f}%)")
    print(f"Total shipped src-core cost: {total_shipped}")
    print(f"Total optimal  src-core cost: {total_optimal}")
    if total_optimal > 0:
        print(f"Excess: {total_shipped - total_optimal} "
              f"({100*(total_shipped-total_optimal)/total_optimal:.1f}% over optimal)")
    print(f"Max single-action gap: {max(gap)}")
    from collections import Counter
    print(f"Gap histogram: {dict(sorted(Counter(gap).items()))}")


if __name__ == "__main__":
    main()
