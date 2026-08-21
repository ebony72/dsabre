# Eviction/staging and recovery: formalization + measurement (2026-08-21)

Investigation of the two post-submission optimality gaps in [`TODO.md`](../../TODO.md):
"Optimal eviction and staging plan" and "Optimal recovery plan (Algorithm 2)".
Discussion-only — the shipped router, architecture and paper are all
untouched. Everything here lives in two read-only probe scripts that
subclass the production router and observe it; neither script mutates
`router.py`, `dsabre_ext.py`, `architecture.py`, or any file `verify_router.py`
checks.

- [`probe_eviction_staging_optimal.py`](probe_eviction_staging_optimal.py) — item 1
- [`probe_recovery_meeting_core.py`](probe_recovery_meeting_core.py) — item 2

## 1. Eviction and staging (`router.py`: `_evict_cost`, `_evict`, the candidate loop)

### Setup

The interaction TODO.md describes is confined to one core. `d_prep` is a sum
of three pieces — `d_intra(p1, n_s)`, `evict_cost(pi_s)`, `evict_cost(pi_d)` —
and the third lives in a different core with its own free-slot pool, so it's
separable and already optimal on its own (a single-target shortest-path
walk). The real joint problem is inside `src_core`'s coupling graph `G`:

- `t` — the port (`p_comm_src`), fixed.
- `s` — the gate operand `q`'s current position, `s ≠ t`.
- `F` — the core's free (hole) vertices, `F ≠ ∅`.
- Choose `n_s ∈ N(t)` and a sequence of edge-SWAPs so that the final state has
  `occ(t) = ⊥` and `occ(n_s) = q`, minimizing the SWAP count.

### The shift lemma

`_evict` walks the path from `t` to the chosen free slot in **reverse**
order (nearest-`f` end first, `(t,·)` last) — that's what makes it correct:
it's shifting the *hole* from `f` to `t`, not walking `t`'s occupant
forward. Direct consequence: **every vertex on the path shifts by exactly
one step, away from `t`, toward `f`.** If `s` lies on the chosen `t→f` path,
eviction moves `q` from `s` to the next vertex on that path — one step
closer to `f` — for free, as a side effect. Off the path, `q` doesn't move.
This is the literal mechanism behind TODO.md's "the SWAPs that clear `pi_s`
move `q1` too."

Because the shift is always exactly one edge regardless of the rest of the
path's length, deliberately routing eviction through `s` can save at most 1
swap on staging — and forcing a detour to pick up `s` costs at least +2 (a
grid is bipartite, so `d(t,s)+d(s,f) - d(t,f)` is always even; ≥2 whenever
it's nonzero). So a detour is never worth it: the only exploitable case is
when `s` already sits on *some* existing `t→f` geodesic.

### Closed-form decision rule

For each candidate hole `f` and port-neighbour `n_s`, take the better of:

- **A. evict avoiding `s` entirely**, `q` stays put:
  `d_{G-s}(t, f) + d_{G-t}(s, n_s)`.
  `d_{G-s}(t,f)` — not the plain `d(t,f)` — because deleting `s` can force a
  longer route; assuming it never does was a real bug caught by the BFS
  cross-check below (`t` on a grid boundary, `s` one of only two routes to
  the nearest hole; the naive version returned a cost that wasn't
  achievable).
- **B. evict via a shortest `t→f` path through `s`**, using the shift:
  `d(t,f) + d_{G-t}(s', n_s)`, `s'` = the neighbour of `s` one step closer
  to `f` — only when `d(t,s)+d(s,f) = d(t,f)` (costs nothing extra; per the
  parity argument, a forced detour never pays for itself).

Minimize over `f ∈ F`, `n_s ∈ N(t)`. This closed form was checked against an
exact BFS over the reduced state `(free-set, q-position)` — valid because
every qubit other than `q` is interchangeable for this question, so that
pair *is* the whole state, small enough (16-site cores) for exhaustive BFS
— on every sample below: **0 mismatches out of 13,898.**

### Measurement — real 64q suite, all 9 canonical circuits

Hooked `_apply_teleport` to read the pre-action state, let the real code
execute unchanged, then isolate the source-core portion of the actual cost
(total local-SWAP delta minus the separately-optimal destination-port
eviction) and compare against the BFS optimum:

| | value |
|---|---|
| teleport actions sampled | 13,898 |
| BFS optimum unreachable | 0 |
| closed-form vs. BFS mismatches | 0 |
| greedy already optimal | 12,933 / 13,898 (93.1%) |
| total shipped src-core cost | 31,153 |
| total optimal src-core cost | 29,587 |
| excess | 1,566 (**5.3% over optimal**) |
| max single-action gap | 4 |
| gap histogram | {0: 12933, 1: 560, 2: 307, 4: 98} |

The greedy plan is already exactly optimal in 93% of cases. The other 7%
carry the full gap: a fixed, fast (`O(\lvert N(t)\rvert \cdot \lvert F\rvert)`,
no search) closed-form replacement for the current `n_s`-then-evict
heuristic would recover essentially all of the 5.3%, since it matched the
true optimum on every one of the 13,898 samples.

## 2. Recovery transaction (`router.py`: `_safe_route_gate`, `_plan_meeting_core`, `_relay_slack_to`, `_make_layout_safe`)

### Correction to the earlier discussion

Mid-discussion this session, before measuring anything, it was claimed that
Phase 1 (`_make_layout_safe`, restoring Eq. 13 everywhere) is "provably
inert under every config the paper ships." That's wrong. `config.py`'s
shipped default is `tier1_floor=2`, deliberately looser than the strict
`core_reserve+1=3` (SAFE_DSABRE.md §10.4–10.5: the strict floor costs +20.8%
EPR gmean against +4.2%, 124 guaranteed transactions against 9) — so
ordinary Tier-1 routing only guarantees `free ≥ 1` per core, not the
`≥ core_reserve` reserve, and `_safe_route_gate` always calls
`_make_layout_safe` first. Phase 1 runs on essentially every recovery
invocation, not a rare externally-supplied-layout edge case.

### Setup

Recovery is two phases, matching Algorithm 2's structure:

- **Phase 1** (lines 3–5): for every core below the reserve `r`, relay a
  vacancy in from its nearest donor (`free ≥ r+1`).
- **Phase 2** (lines 6–9): pick a meeting core `m`, top it up, walk the
  other operand there, execute.

`_plan_meeting_core` already searches three families — `m=a`, `m=b`, and
any third core that *already* holds `≥ r+2` free (no relay) — which is more
than the paper's Algorithm 2 admits ("for simplicity, the meeting core is
selected from the two operand cores"). The one family it doesn't search:
**a third core that would need relay too.**

Both remaining questions reduce to the same primitive: donors have supply
`max(0, free_c - r)` (any core above the reserve can give away exactly its
surplus, in any order — the donor-floor check only ever needs `≥ r+1`
*before* each individual extraction), and cost is core-graph distance.

- Phase 1's "restore everywhere" is a min-cost **transportation** problem
  (multiple donors, multiple deficits) — `phase1_optimal_cost`, solved as
  max-flow-min-cost on a super-source/super-sink graph.
- Phase 2's "extend to `m ∉ {a,b}`" only ever needs a **single-sink**
  version — `extended_meeting_core_search` — optimal by an exchange
  argument (take the `k` cheapest available donor units for that one `m`).

Extending the theorem to `m ∉ {a,b}` looks like a straightforward extension
of the existing proof (both operands walk to `m` instead of one; the same
per-hop reserve bookkeeping applies), not a new technique — this was not
re-derived in full here, just checked for plausibility.

### Measurement — real 100q/200q/360q scalability suites, qft + qpeexact

Hooked `_make_layout_safe` (Phase 1) and `_plan_meeting_core` (Phase 2),
both read-only: call the real method unchanged, capture the pre-state, and
score the real outcome against the oracle above on the identical state. 134
gates retired via `_safe_route_gate` across the six runs.

**Phase 1** (375 invocations with a real deficit to fix):

| | value |
|---|---|
| already optimal | 372 / 375 (99.2%) |
| total actual hops | 1,359 |
| total optimal hops | 1,355 |
| excess | 4 (**0.3% over optimal**) |

**Phase 2** (1,106 invocations — includes `_safe_pick_gate`'s scoring calls,
not just executed transactions):

| | value |
|---|---|
| shipped already optimal | 1,082 / 1,106 (97.8%) |
| a strictly cheaper `m` exists | 24 / 1,106 |
| total shipped cost | 3,646 |
| total best cost | 3,616 |
| excess | 30 (**0.8% over optimal**) |

### Reading

Both gaps are real but small — under 1% each, against 5.3% for item 1. The
nearest-donor-per-deficit greedy in Phase 1 is close to a full transportation
solve almost all the time in practice (multiple *simultaneous* deficits
sharing a donor pool, where a greedy assignment can lose to a joint one, are
apparently uncommon at these scales); implementing a min-cost-flow solve
inside the router's hot path would buy ~0.3% for real complexity. The
third-meeting-core-with-relay extension to Phase 2 is a comparably modest
win (0.8%) but is at least a clean, bounded, well-specified addition should
it ever be worth shipping — unlike item 1's closed-form rule, it would also
need the proof extension sketched above written out properly before it
could ship into a result-producing path.
