# TODO

## After the revision is submitted — two optimality gaps the paper leaves open (2026-08-18)

Both are places where the shipped plan is provably correct but deliberately
not cost-minimal, and the paper says so in as many words. Neither is a bug;
each is a measurable question with a concrete baseline to beat.

### 1. Optimal eviction and staging plan

`d_prep = d_intra(p1, n_s) + d_evict(pi_s) + d_evict(pi_d)` (§III-D) is built
from three choices made independently and greedily: the staging neighbour
`n_s` (a neighbour of `pi_s` minimising `d_intra(p1, n_s)`), and the free slot
walked to each port to clear it. They interact — the SWAPs that clear `pi_s`
move `q1` too, sometimes towards `n_s` and sometimes away; one vacancy can
occasionally serve both evictions; and the eviction order matters.

- **Question.** What is the minimum-SWAP joint plan for
  {evict `pi_s`, evict `pi_d`, stage `q1` at `n_s`}, and how much of the
  local-SWAP count does it recover against the greedy plan?
- **Catch.** `d_prep` is not only the execution cost, it is the price the
  ranking uses, so a better plan reorders candidates as well as making the
  chosen one cheaper. Expect EPR counts to move, not just SWAP counts.
- **Code.** `code/router.py`: `_evict_cost` (the scored estimate),
  `_evict` (the applied plan), and the candidate loop in the teleport scorer.

### 2. Optimal recovery plan (Algorithm 2)

§III-F states the transaction's choices — nearest donor, meeting core
restricted to the two operand cores, shortest-path vacancy walks — are
"fixed to make completion provable" and explicitly "does not claim they are
cheap".

- **Question.** What is the cheapest transaction that retires a blocked
  inter-core gate and returns with Eq. (13) restored? A third meeting core
  can beat both operand cores, and the vacancy walks look like a min-cost
  flow over free slots rather than a set of independent shortest paths.
- **Constraint.** Any alternative must keep what the proof uses: every
  teleport it emits still obeying Eq. (14), the reserve restored on exit, and
  strict progress (one gate retired per transaction). A cheaper plan that
  weakens any of those costs the theorem, not just EPR pairs.
- **Where it would show.** Recovery fires only at scale — 6, 10 and 66 gates
  over the reported 100-, 200- and 360-qubit routes (Table V) — so the
  benchmark for this is the scalability series, not the 64-qubit suite.
- **Code.** `code/router.py`: `_safe_route_gate`, `_plan_meeting_core`,
  `_relay_slack_to`, `_relay_room_to`.
