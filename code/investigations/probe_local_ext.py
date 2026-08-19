"""probe_local_ext.py — audit the intra-core extended set, scan vs BFS closure.

Runs the *shipped* router (so the routing trajectory is unchanged) and, at every
`_get_local_extended` call, also builds the seeded-BFS closure of
`dsabre_local_bfs.dSABRE_LocalBFS` on the same DAG, layout and core set.  For
each call it records

  * whether the two agree as ordered lists (gate id + depth),
  * whether they agree as sets (order-insensitive),
  * DAG nodes examined by each traversal,
  * whether the scan ran to the end of the topological order,
  * whether the shipped `id(n) in front_ids` guard fired even once.

Usage:  python3 probe_local_ext.py [--suite 25|64] [--circuits a,b,c]
"""
import argparse
import glob
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BFSExt
from dsabre_local_bfs import dSABRE_LocalBFS, dSABRE_LocalKahnLex
from layout import sabre_locked_boundary_layout, run_sabre_passes
from circuit_paths import circuits_path

SUITES = {
    "25": dict(d=circuits_path("qasm_25"),
               suffix="_nativegates_ibm_qiskit_opt3_25.qasm",
               arch=lambda: build_b_grid_architecture(r=2, s=2, m=4),
               hw=lambda: HardwareConfig()),
    "64": dict(d=circuits_path("qasm_64"),
               suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
               arch=lambda: build_h_grid_architecture(r=2, s=3, m=4),
               hw=lambda: HardwareConfig(deadlock_limit=100,
                                         max_backup_attempts=100,
                                         max_iterations=20000)),
}
CANON64 = {"ae", "ghz", "graphstate", "qft", "qnn", "random",
           "qpeexact", "qaoa", "multiplier"}


class AuditRouter(dSABRE_BFSExt):
    """Shipped router; every E_c call is mirrored against the BFS closure."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reset_audit()

    def reset_audit(self):
        self.n_calls = 0
        self.n_list_eq = 0        # identical as ordered lists
        self.n_set_eq = 0         # identical as sets, order aside
        self.n_truncated = 0      # some core hit the size-L quota
        self.scan_visits = 0
        self.bfs_visits = 0
        self.n_scan_to_end = 0    # scan reached the end of the order
        self.n_front_guard = 0    # `id(n) in front_ids` actually fired
        self.n_front_in_ext = 0   # a front gate landed in E_c
        self.n_lex_eq = 0         # lexicographic closure == scan, as a list
        self.n_untruncated_setdiff = 0

    # -- shipped construction, instrumented ---------------------------------
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
        ran_to_end = True
        for n in wdag.topological_op_nodes():
            if not eligible:
                ran_to_end = False
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
                self.n_front_guard += 1
                continue
            if c1 in remaining:
                depth = max(qubit_depth.get(q1, 0), qubit_depth.get(q2, 0)) + 1
                ext[c1].append((n, depth))
                qubit_depth[q1] = qubit_depth[q2] = depth
                if len(ext[c1]) >= self.config.lookahead_size:
                    remaining.discard(c1)
                    eligible.discard(c1)

        # -- mirror: seeded closures, FIFO and lexicographic ------------------
        before = self.le_visits
        bfs = dSABRE_LocalBFS._get_local_extended(self, wdag, front, core_ids, l2p)
        bfs_visits = self.le_visits - before
        lex = dSABRE_LocalKahnLex._get_local_extended(self, wdag, front, core_ids, l2p)

        def key(lst):
            return [(g._node_id, d) for g, d in lst]

        list_eq = all(key(ext[c]) == key(bfs[c]) for c in core_ids)
        set_eq = all(sorted(key(ext[c])) == sorted(key(bfs[c])) for c in core_ids)
        lex_eq = all(key(ext[c]) == key(lex[c]) for c in core_ids)
        truncated = any(len(ext[c]) >= self.config.lookahead_size for c in core_ids)
        self.n_lex_eq += lex_eq
        # appendix claim: the constructions can differ only under truncation
        if not set_eq and not truncated:
            self.n_untruncated_setdiff += 1

        front_nids = {n._node_id for n in front}
        in_ext = any(g._node_id in front_nids for c in core_ids for g, _ in ext[c])

        self.n_calls += 1
        self.n_list_eq += list_eq
        self.n_set_eq += set_eq
        self.n_truncated += truncated
        self.scan_visits += visits
        self.bfs_visits += bfs_visits
        self.n_scan_to_end += ran_to_end
        self.n_front_in_ext += in_ext
        return ext

    # `dSABRE_LocalBFS._get_local_extended` writes these onto self
    le_calls = 0
    le_visits = 0
    front_skip = dSABRE_LocalBFS.front_skip
    _qkey = dSABRE_LocalKahnLex._qkey


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="25", choices=list(SUITES))
    ap.add_argument("--circuits", default="")
    args = ap.parse_args()

    s = SUITES[args.suite]
    arch = s["arch"]()
    hw = s["hw"]()
    files = sorted(glob.glob(os.path.join(s["d"], "*.qasm")))
    if args.suite == "64":
        files = [f for f in files
                 if os.path.basename(f).replace(s["suffix"], "") in CANON64]
    if args.circuits:
        want = set(args.circuits.split(","))
        files = [f for f in files
                 if os.path.basename(f).replace(s["suffix"], "") in want]

    hdr = (f"{'circuit':<12} {'calls':>7} {'list=':>7} {'set=':>7} {'trunc':>6} "
           f"{'lex=':>7} {'scan/call':>10} {'bfs/call':>9} {'ratio':>7} "
           f"{'toEnd':>6} {'guard':>6} {'F in Ec':>8} {'!trunc\u0394':>8}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    tot = dict(calls=0, list_eq=0, set_eq=0, scan=0, bfs=0)
    for f in files:
        cname = os.path.basename(f).replace(s["suffix"], "")
        qc, dag = load_qasm(f)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        r = AuditRouter(arch, hw)
        r.reset_audit()
        t0 = time.perf_counter()
        run_sabre_passes(r, dag, rev_dag, layouts[0])
        el = time.perf_counter() - t0
        c = max(r.n_calls, 1)
        print(f"{cname:<12} {r.n_calls:>7} "
              f"{100*r.n_list_eq/c:>6.1f}% {100*r.n_set_eq/c:>6.1f}% "
              f"{100*r.n_truncated/c:>5.1f}% "
              f"{100*r.n_lex_eq/c:>6.1f}% "
              f"{r.scan_visits/c:>10.1f} {r.bfs_visits/c:>9.1f} "
              f"{r.scan_visits/max(r.bfs_visits,1):>6.1f}x "
              f"{100*r.n_scan_to_end/c:>5.1f}% "
              f"{r.n_front_guard:>6} {100*r.n_front_in_ext/c:>7.1f}% "
              f"{r.n_untruncated_setdiff:>8}"
              f"   ({el:.1f}s)", flush=True)
        tot["calls"] += r.n_calls
        tot["lex_eq"] = tot.get("lex_eq", 0) + r.n_lex_eq
        tot["setdiff"] = tot.get("setdiff", 0) + r.n_untruncated_setdiff
        tot["list_eq"] += r.n_list_eq
        tot["set_eq"] += r.n_set_eq
        tot["scan"] += r.scan_visits
        tot["bfs"] += r.bfs_visits

    c = max(tot["calls"], 1)
    print("-" * len(hdr), flush=True)
    print(f"{'TOTAL':<12} {tot['calls']:>7} "
          f"{100*tot['list_eq']/c:>6.1f}% {100*tot['set_eq']/c:>6.1f}% "
          f"{'':>6} {100*tot.get('lex_eq',0)/c:>6.1f}% "
          f"{tot['scan']/c:>10.1f} {tot['bfs']/c:>9.1f} "
          f"{tot['scan']/max(tot['bfs'],1):>6.1f}x"
          f"  |  set-difference without truncation: {tot.get('setdiff',0)}",
          flush=True)


if __name__ == "__main__":
    main()
