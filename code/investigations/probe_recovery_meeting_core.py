"""probe_recovery_meeting_core.py -- how much of TODO.md item 2 (the optimal
recovery transaction) is still open, measured on the shipped router?

Recovery (`_safe_route_gate`) is two phases:

  Phase 1 (`_make_layout_safe`): bring every core up to `free >= core_reserve`
  by relaying vacancies in from donors, one core at a time, each via its own
  independent nearest-donor search.  This is NOT a rare edge case: the
  shipped default `tier1_floor=2` is deliberately looser than
  `core_reserve+1` (SAFE_DSABRE.md Sec 10.4-10.5 -- the strict floor costs
  +20.8% EPR gmean against +4.2%), so ordinary routing only guarantees
  `free >= 1` per core, and Phase 1 has real repair work to do on *every*
  `_safe_route_gate` call, not just when an externally-supplied layout is
  unsafe. Earlier in this same investigation it was claimed Phase 1 is
  provably inert under the shipped config -- that was wrong, and this script
  exists partly to correct it with a measurement.

  Phase 2 (`_plan_meeting_core`): pick a meeting core `m`, top it up if
  needed, walk the non-resident operand(s) there, execute the gate.  The
  shipped implementation already searches three families -- m=a, m=b, and
  any third core that ALREADY holds enough free slots with no relay needed
  -- which is more than the paper's Algorithm 2 admits ("for simplicity, the
  meeting core is selected from the two operand cores"). What it does not
  search is a third core that would need relay too.

This script runs the shipped, unmodified production router (dSABRE_BFSExt,
safe_mode=True, exactly bench_large.py's scalability-suite configuration)
over real qft/qpeexact circuits at 100q/200q/360q -- the paper's own choice
of "where recovery shows" -- and at every Phase 1 / Phase 2 invocation
compares the real cost paid against two provably-optimal oracles computed
from the SAME state:

  Phase 1: an exact min-cost transportation solve (donors' surplus over the
  reserve as supply, deficient cores' shortfall as demand, core-graph
  distance as cost) -- see `phase1_optimal_cost`.

  Phase 2: exhaustive search over every third core `m`, with relay costed by
  the same transportation model specialised to a single sink (optimal for
  one sink by an exchange argument: take the k cheapest available donor
  units) -- see `extended_meeting_core_search`.

Both oracles use the identical "supply_c = max(0, free_c - core_reserve)"
model that the real `_relay_slack_to`/donor-floor=`core_reserve+1` mechanism
actually realises (any core starting above the reserve can give away exactly
its surplus, in any order, since the donor-floor check only ever requires
`>= core_reserve+1` *before* each individual hop). Neither oracle changes
what the router does -- both are read-only side computations hung off
`_make_layout_safe` / `_plan_meeting_core`.

Usage:  python3 probe_recovery_meeting_core.py [--suite 100|200|360|all] [--circuits qft,qpeexact]
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

import networkx as nx
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

SUITES = {
    "100q": dict(circuit_dir=circuits_path("qasm_100"),
                suffix="_nativegates_ibm_qiskit_opt3_100.qasm",
                arch=lambda: build_h_grid_architecture(r=2, s=3, m=5)),
    "200q": dict(circuit_dir=circuits_path("qasm_200"),
                suffix="_nativegates_ibm_qiskit_opt3_200.qasm",
                arch=lambda: build_h_grid_architecture(r=3, s=4, m=5)),
    "360q": dict(circuit_dir=circuits_path("qasm_360"),
                suffix="_nativegates_ibm_qiskit_opt3_360.qasm",
                arch=lambda: build_h_grid_architecture(r=4, s=5, m=5)),
}
DEFAULT_CIRCUITS = ["qft", "qpeexact"]


# ── Optimal oracles, on core-level (free_by_core) state only ───────────────────

def min_relay_cost_to_target(core_dist, free_by_core, r, target, k):
    """Exact min hop-cost to source k units into `target`.

    Optimal by an exchange argument: every unit is interchangeable and
    lands on the same sink, so the cheapest way to get k of them is to take
    the k cheapest available (donor, one-unit) prices, where a core with
    `free_c` above the reserve r can supply `free_c - r` units total (each
    single extraction only needs the donor at >= r+1 *at the time*, which
    holds for every extraction up to exactly that many).
    """
    if k <= 0:
        return 0
    units = []
    for c, f in enumerate(free_by_core):
        if c == target:
            continue
        supply = max(0, f - r)
        if supply > 0:
            units.extend([core_dist[c][target]] * supply)
    units.sort()
    if len(units) < k:
        return None
    return sum(units[:k])


def phase1_optimal_cost(core_dist, free_by_core, r, num_cores):
    """Exact min total hop-cost to bring every core to >= r free slots.

    Min-cost transportation: donors' surplus over r is supply, deficient
    cores' shortfall is demand, core-graph distance is cost. Solved as a
    max-flow-min-cost on a super-source/super-sink graph.
    """
    short = [c for c in range(num_cores) if free_by_core[c] < r]
    if not short:
        return 0
    donors = [c for c in range(num_cores) if free_by_core[c] > r]
    total_supply = sum(free_by_core[c] - r for c in donors)
    total_demand = sum(r - free_by_core[c] for c in short)
    if total_supply < total_demand:
        return None  # would violate Eq. (12)/feasibility -- shouldn't happen
    G = nx.DiGraph()
    for c in donors:
        G.add_edge("SRC", ("D", c), capacity=free_by_core[c] - r, weight=0)
    for c in short:
        G.add_edge(("T", c), "SNK", capacity=r - free_by_core[c], weight=0)
    for c in donors:
        for c2 in short:
            if c2 == c:
                continue
            G.add_edge(("D", c), ("T", c2),
                      capacity=min(free_by_core[c] - r, r - free_by_core[c2]),
                      weight=core_dist[c][c2])
    flow_dict = nx.max_flow_min_cost(G, "SRC", "SNK")
    return nx.cost_of_flow(G, flow_dict)


def extended_meeting_core_search(core_dist, free_by_core, r, a, b, num_cores):
    """min over ALL m (not just {a,b} or free third cores) of
    d_C(a,m) + d_C(b,m) + relay(m, needed(m) - free(m)).
    Returns {m: cost} for every feasible m.
    """
    costs = {}
    for m in range(num_cores):
        needed = (r + 1) if m in (a, b) else (r + 2)
        k = max(0, needed - free_by_core[m])
        relay = min_relay_cost_to_target(core_dist, free_by_core, r, m, k) if k > 0 else 0
        if relay is None:
            continue
        costs[m] = core_dist[a][m] + core_dist[b][m] + relay
    return costs


# ── Hook into the shipped router ────────────────────────────────────────────

class AuditRouter(dSABRE_BFSExt):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.phase1_samples = []
        self.phase2_samples = []

    def _make_layout_safe(self, l2p, p2l, metrics, protect=()):
        arch = self.arch
        r = self.config.core_reserve
        free_before = self._free_by_core(p2l)
        short_before = [c for c in range(arch.num_cores) if free_before[c] < r]
        relay_before = metrics.get("relay_hops", 0)

        ok = super()._make_layout_safe(l2p, p2l, metrics, protect=protect)

        if short_before:
            actual = metrics.get("relay_hops", 0) - relay_before
            optimal = phase1_optimal_cost(arch.core_dist, free_before, r, arch.num_cores)
            self.phase1_samples.append(dict(
                n_short=len(short_before), actual_hops=actual, optimal_hops=optimal,
            ))
        return ok

    def _plan_meeting_core(self, node, l2p, p2l):
        arch = self.arch
        r = self.config.core_reserve
        q1, q2 = node.qargs[0], node.qargs[1]
        a = arch.core_of(l2p[q1])
        b = arch.core_of(l2p[q2])

        plan = super()._plan_meeting_core(node, l2p, p2l)  # read-only: no state mutated

        if plan is not None and a != b:
            free = self._free_by_core(p2l)
            costs = extended_meeting_core_search(arch.core_dist, free, r, a, b, arch.num_cores)
            shipped_m = plan[0]
            shipped_cost = costs.get(shipped_m)
            best_cost = min(costs.values()) if costs else None
            self.phase2_samples.append(dict(
                shipped_m=shipped_m, shipped_in_ab=shipped_m in (a, b),
                shipped_cost=shipped_cost, best_cost=best_cost,
                n_candidates=len(costs),
            ))
        return plan


def run_suite(suite_name, s, circuits, router):
    for cname in circuits:
        pattern = os.path.join(s["circuit_dir"], f"{cname}{s['suffix']}")
        matches = glob.glob(pattern)
        if not matches:
            print(f"  [{suite_name}/{cname}: not found at {pattern}]", flush=True)
            continue
        qasm_path = matches[0]
        qc = QuantumCircuit.from_qasm_file(qasm_path)
        qc = qc.remove_final_measurements(inplace=False)
        qc = PassManager([RemoveBarriers()]).run(qc)
        dag = circuit_to_dag(qc)
        rev_dag = circuit_to_dag(qc.reverse_ops())

        n1, n2 = len(router.phase1_samples), len(router.phase2_samples)
        sl_layouts = sabre_locked_boundary_layout(qc, dag, router.arch, seed=0)
        m = run_sabre_passes(router, dag, rev_dag, sl_layouts[0])
        status = "aborted" if (m is None or m.get("aborted")) else f"eprs={m['eprs']}"
        print(f"  {suite_name}/{cname}: {status}, safe_routes={m.get('safe_routes', 0) if m else '?'}, "
              f"+{len(router.phase1_samples)-n1} phase1, +{len(router.phase2_samples)-n2} phase2",
              flush=True)


def summarize(name, samples, actual_key, optimal_key):
    valid = [s for s in samples if s[optimal_key] is not None]
    print(f"\n{name}: {len(samples)} invocations, {len(valid)} with a comparable optimum")
    if not valid:
        return
    n_tight = sum(1 for s in valid if s[actual_key] == s[optimal_key])
    total_actual = sum(s[actual_key] for s in valid)
    total_optimal = sum(s[optimal_key] for s in valid)
    print(f"  already optimal: {n_tight}/{len(valid)} ({100*n_tight/len(valid):.1f}%)")
    print(f"  total actual={total_actual}, total optimal={total_optimal}", end="")
    if total_optimal > 0:
        print(f"  (excess {total_actual-total_optimal}, "
              f"{100*(total_actual-total_optimal)/total_optimal:.1f}% over optimal)")
    else:
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["100", "200", "360", "all"], default="all")
    ap.add_argument("--circuits", default=",".join(DEFAULT_CIRCUITS))
    args = ap.parse_args()
    circuits = args.circuits.split(",")
    suite_names = (["100q", "200q", "360q"] if args.suite == "all" else [f"{args.suite}q"])

    all_phase1, all_phase2 = [], []
    for sname in suite_names:
        s = dict(SUITES[sname])
        s["arch"] = s["arch"]()
        router = AuditRouter(s["arch"], _HW)
        print(f"\n{'='*60}\n  {sname}\n{'='*60}", flush=True)
        run_suite(sname, s, circuits, router)
        all_phase1 += router.phase1_samples
        all_phase2 += router.phase2_samples

    summarize("Phase 1 (restore-invariant relay)", all_phase1, "actual_hops", "optimal_hops")

    print()
    valid2 = [s for s in all_phase2 if s["best_cost"] is not None and s["shipped_cost"] is not None]
    print(f"Phase 2 (meeting-core choice): {len(all_phase2)} invocations, "
          f"{len(valid2)} with a comparable optimum")
    if valid2:
        n_tight = sum(1 for s in valid2 if s["shipped_cost"] == s["best_cost"])
        n_third_better = sum(1 for s in valid2
                             if s["best_cost"] < s["shipped_cost"] and not s["shipped_in_ab"])
        n_third_wins = sum(1 for s in valid2 if s["best_cost"] < s["shipped_cost"])
        total_shipped = sum(s["shipped_cost"] for s in valid2)
        total_best = sum(s["best_cost"] for s in valid2)
        print(f"  shipped already optimal: {n_tight}/{len(valid2)} ({100*n_tight/len(valid2):.1f}%)")
        print(f"  a strictly cheaper m exists: {n_third_wins}/{len(valid2)}")
        print(f"  total shipped={total_shipped}, total best={total_best}", end="")
        if total_best > 0:
            print(f"  (excess {total_shipped-total_best}, "
                  f"{100*(total_shipped-total_best)/total_best:.1f}% over optimal)")
        else:
            print()


if __name__ == "__main__":
    main()
