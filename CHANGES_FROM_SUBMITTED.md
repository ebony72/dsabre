# Changes from the Submitted Manuscript

Router- and layout-level algorithmic changes since the first submission
(commit `e1ba02b`, 2026-05-21), in the order that matters for reading them:
the layout fix first, because the relief investigation only makes sense
against it; node_decay's removal and its belated consequence second, because
it turned out to be the load-bearing fact behind congestion relief's
apparent value; relief's removal third, because it is downstream of both;
then, chronologically, the transactional recovery fix (§4), the `E_c`
performance/correctness fix (§5), and the capacity conditions that
supersede both and become the default router (§6) — read those three in
order, since each is a precondition for reading the next one's "before"
column correctly. §7 is a same-day, unrelated bookkeeping fix (an
experiment-setup mismatch, not an algorithm change) found while regenerating
data for §6.

This file is the answer to "what changed, and why" for anyone diffing the
submitted code against the current router. Paper-text changes are tracked by
the `\rev{}` blue-text convention in `paper/dsabre.tex` itself, so this file
covers the code — with one exception: §6.4 and §6.5 record corrections to
claims the *manuscript* made about the code, since those are the ones a
reviewer is most likely to test and the response letter has to own them.

---

## 1. Per-core corner reservation (`layout.py`, commit `6b1fc31`, 2026-07-17)

**Submitted version:** `sabre_locked_boundary_layout` removed the 4
globally-lowest-degree nodes of the *whole* architecture graph before running
SabreLayout. Under core-major qubit numbering those 4 nodes are always the 4
corners of core 0, so cores 1..K-1 received **zero** reserved escape slots
and regularly came out fully packed (verified this session: on the 64q
H-grid, GHZ and Random both land with 4 of 6 cores at 100% occupancy under
this rule).

**Current version:** `_per_core_reserved_corner_nodes` + `adaptive_corner_count`
reserve the k most-remote corner qubits of *every* core, k chosen per suite by
a fill-adaptive rule (largest k in 0..4 keeping usable-slot fill <= 80%).

**Effect measured at the time:** EPR ratio new/old layout (dSE, base router,
best-of-3 seeds, fwd-bwd-fwd): 25q 1.000, 36q 0.968, 64q 0.862 — an
improvement at every scale, driven by no longer starving cores 1..K-1 of
escape capacity.

## 1a. k-floor correction (`layout.py::adaptive_corner_count`, 2026-08-02)

`adaptive_corner_count` could still return k=0 in two cases: when k=0
happened to satisfy the 80%-fill target on its own (e.g. 64q_c33, fill
exactly 79.0% at k=0), or when *no* k in 0..4 satisfied the target and the
function fell back to k=0 regardless (e.g. 64q_b243, fill 88.9% even at
k=0 — a silent "give up," not a considered answer). Both leave a core able
to reach 100% occupancy in the initial layout.

**Fix:** k is now floored at 1 whenever reserving one slot per core is
physically feasible (`P - n >= K`, i.e. usable capacity at k=1 still covers
every logical qubit) — a hard structural guarantee, not a fill-optimisation
choice, since the reserved slots are removed from SabreLayout's coupling map
entirely and cannot be filled regardless of what the fill target says.

**Verified against every architecture in the paper** (`check_full_core_layouts.py`):
25q/36q/64q H-grid/100q/200q/360q/heavy-hex ring+star all already had k>=2
under the old rule — **k is unchanged for all of them**, floor never binds.
Only 64q_c33 (k: 0->1) and the uncited 64q_b243 (k: 0->1) are affected.
25q_c133 correctly stays at k=0: even k=1 is infeasible there (P-n=2 < K=3).

---

## 2. `enable_node_decay` removed (`router.py`, commit `e1d951e`, ~2026-07)

**Submitted version:** `HardwareConfig.enable_node_decay: bool = True`. The
router maintained a per-physical-node decay counter, incremented by 0.1 on
every SWAP touching that node and reset to 1.0 when a gate executed there or
a teleport departed from it; the intra-core SWAP score was multiplied by
`max(node_decay[u], node_decay[v])` for the candidate edge — a SABRE-inherited
anti-oscillation penalty discouraging the router from repeatedly SWAPping
through the same physical qubit.

**Current version:** removed entirely — no `node_decay` dict, no multiplier,
no config field. The removal predates this session; the reason it matters
*now* is a consequence discovered this session, not the original removal
rationale (see `TODO.md`/git history for that).

### The consequence discovered this session

The submitted manuscript's Table VI claimed disabling congestion relief cost
**+23.4%** gmean EPR at 64q (Full=134.8, No relief=166.4), with AE and QFT
"more than doubling." This was re-investigated because no non-failing
architecture available now — the current per-core layout, 9 occupancy-skew
profiles up to 100% single-core packing, and an independent degree-swept
synthetic family — showed relief helping at all; it costs -6.7% to -9.3% gmean
everywhere it was tested short of 64q_c33, an architecture where a separate
circuit (Multiplier) fails unconditionally regardless of relief.

Four controlled reproductions (`probe_relief_monolithic_layout.py`,
`probe_relief_exact_reproduction.py`, `probe_relief_isolate_node_decay.py`),
all on the real 96-physical 64q H-grid, all 6/6 circuits completing with no
aborts:

| layout | router | node_decay | budgets | relief's effect |
|---|---|---|---|---|
| current (per-core) | current | removed | 20000/100/100 | -6.7% |
| submitted (monolithic) | current | removed | 20000/100/100 | -2.8% |
| submitted (monolithic) | **legacy** | **off** | 10000/50/50 | **-9.3%** |
| submitted (monolithic) | **legacy** | **on** | 10000/50/50 | **+23.4%** (exact reproduction) |

The last two rows hold the router codebase, layout, and safety-limit budgets
identical and toggle only `enable_node_decay`. That is the entire cause of
the sign flip. The submitted +23.4% was real and is exactly reproducible, but
it was never evidence that congestion relief matters on a working
architecture — it was evidence that node_decay, once active, made relief
matter, for a reason that is plausible but not proven here: node_decay
discourages the cheap local remedy (repeated intra-core SWAPping), which
would otherwise handle the same congestion relief was invented to address.
Once node_decay was removed (for unrelated reasons, well before this
session), relief's apparent benefit should have been re-examined and was not.

**Historical reproduction, frozen and not to be modified:**
`code/_legacy_router_submitted.py`, `code/_legacy_dsabre_ext_submitted.py`,
`code/_legacy_config_submitted.py` are `git show e1ba02b:code/{router,
dsabre_ext,config}.py` verbatim, with only their cross-imports repointed at
each other instead of the current modules. They exist solely so the
`probe_relief_*.py` scripts above remain re-runnable as a permanent check
against this finding.

---

## 3. Congestion relief removed entirely (`router.py`, `config.py`, 2026-08-03)

**Decision:** remove the proactive congestion-relief mechanism from the
router and from the paper, following from §2 above plus the exhaustive
non-failing-architecture search that preceded it.

**What relief was:** each iteration, a demand vector over cores counted
front/lookahead inter-core gates whose shortest core-graph path crossed each
core; a core with demand >= `demand_threshold` and free slots <=
`congestion_threshold` triggered relief candidates that evicted its
most-idle qubits (by next-use depth) into a neighbour with slack
(>= `relief_space_req` free), scored by the same teleport objective with
`Delta_F=0`, hop-gain omitted, and three relief-specific bonus/weight terms.

**What was removed:**
- `router.py::_generate_candidates` — the entire "Proactive congestion
  relief" block (demand-vector computation, victim selection, relief
  candidate scoring); candidates are now only ever gate-driven.
- `router.py::route` — the `relief_candidates`/`relief_picks` instrumentation
  counters and their increments (per-iteration counting and the
  `best.node is None` relief-pick check).
- `config.py::HardwareConfig` — `enable_congestion_relief`, `relief_bonus`,
  `demand_lookahead`, `demand_threshold`, `congestion_threshold`,
  `relief_space_req`, `relief_depth_weight`, `relief_gradient_weight`.
- `ablate_common.py::summarise` — the `relief_candidates`/`relief_picks`
  fields on every result row.
- `actions.py::TeleportAction` — the docstring's "(None for proactive
  moves)" on `node`; every candidate now always has a real triggering gate.

**What was NOT touched:** checkpoint-rollback (`enable_deadlock_recovery`,
`_backup_plan`, `_force_make_room`) and hop-gain (`enable_hop_gain`,
`hop_gain`) are separate mechanisms with their own, independently-supported
evidence (hop-gain: +18.6% Multiplier, +9.4% QNN, +6.4% Random, +28.3% 100q
QFT, all on non-failing architectures — see the mechanism-necessity
investigation earlier this session). Neither is affected by this change.

**Downstream scripts fixed to not crash** on the removed `HardwareConfig`
fields: `ablate_occupancy.py` (relief configs dropped from `mech_configs`,
now `full`/`no_hop_gain` only), `ablate_occupancy_complete64.py`,
`ablate_occupancy_64q_c33_k1.py` (both marked historical/frozen — their
purpose was relief ablation on 64q_c33, now moot), `ablate_regular_cz.py`,
`regen_ablation_64q_full9.py`, `regen_ablation_64q_rest8.py`,
`regen_ablation_corners.py` (all had a `no_relief` config row constructed via
`dataclasses.replace(..., enable_congestion_relief=False)`, which raises
`TypeError` against the current dataclass — removed).

**Not yet done — every benchmark table needs regenerating:** `full` is now
what used to be a *different* run (relief was active in every published
number). The EPR-count preflight constants in `regen_ablation_64q_full9.py`
and `regen_ablation_64q_rest8.py` (`EXPECTED_FULL`) are stale and will reject
a re-run until updated. This affects, at minimum: Table III (main results,
all suites), Table IV (mechanism ablation — the "No congestion relief" row
disappears, the "Full" baseline changes), Table VI (heavy-hex ring/star),
the 100/200/360q scalability rows, `tab:regcz` (degree-swept synthetic —
since **removed from the paper**, 2026-08-10, see §6.6),
`tab:sensitivity` (cost-ratio sweep), the compile-time table, and both the
pytket-dqc and DMapS comparisons (which cite dSABRE's own EPR counts as the
comparison baseline). Given relief measured net-negative to EPR on every
non-c33 architecture, expect most numbers to move by a few percent in
dSABRE's favour, not against it — but this needs verifying per suite, not
assumed.

**`tab:regime` and Section III-F's "Where the congestion mechanisms are
load-bearing" paragraph are obsolete** and need removing or rewriting around
checkpoint-rollback alone: that section's entire argument was congestion
relief's completion value on 64q_c33, which no longer exists as a mechanism
to argue for.

---

## 4. Checkpoint–rollback made transactional (`router.py`, commit `2026-08-05`)

**Submitted version:** `_backup_plan`'s cross-core branches (greedy and
`backup_relay_mode` alike) moved only the gate's first operand, hop by hop,
and `break`d out of the loop on the first failed hop — leaving whatever
partial move it had made *applied, not rolled back*, and never trying the
other operand even when that direction had room.

**Current version:** `_route_gate_transaction` wraps one full attempt —
relay room to each next core (`_relay_room_to`), teleport the operand hop by
hop, SWAP it onto the gate, execute — in a single snapshot/rollback. Either
the whole thing lands and the gate executes, or every mutation is undone and
the state is bit-identical to entry. `_backup_plan` now tries `mover=q1`
then, if that fails, `mover=q2`, before falling through to the old
greedy/relay code as a last resort.

**Measured:** 64q suite, before vs after, identical protocol: −8.2% total
EPR on the 4 circuits where backup fires at all (`ae` −30%, `multiplier`
−14%, `qft` −8%, `qaoa` −5%, `qpeexact` +2% the one regression). On
`random_80` (10-core H-grid) — the one archived instance where the *old*
`_backup_plan` hit `DEADLOCK_BACKUP_FAILED` outright on a structural dead
end, not a budget issue — the new transaction never fails to make progress
on any of 3 layouts. Full derivation: `CAPACITY_SAFE_RECOVERY.md`.

This is the mechanism §6 below replaces with a version that additionally
*guarantees* success, not merely retries more carefully.

---

## 5. `E_c` redefined as the inter-core set restricted to the core (`router.py`, commit `89c088a`, 2026-08-08)

**Submitted version (and every number published through 2026-08-07):** the
per-core intra lookahead `E_c` was built by a *separate* sweep of the whole
topological order with taint propagation and a per-core quota of `L` —
`Theta(N_r)` per call, called once per iteration over `Theta(N)` iterations,
which is what made compile time quadratic in gate count (`N^{2.1}` measured,
against the linear bound Eq.~\ref{eq:complexity} derives).

**Current version:** Table I of the paper already *defines* `E_c` as the
restriction of the inter-core set `E` to core `c`; the code now literally
computes it that way. `_get_local_extended` partitions the already-built BFS
set `E` by core in one pass, so a single construction feeds both scorers and
`dep(g)` means the same depth in each, instead of two independent notions of
depth that happened to agree in expectation.

**This is an output-changing fix, not a refactor** — unlike the 2026-08-07
incremental-bookkeeping merge (checkpointing, maintained in-degrees, etc.),
which is bit-identical to what it replaced and needed no re-evaluation.
Evaluated against the old sweep over 8 suites (25/36/64q, heavy-hex ring and
star, 100/200/360q), 38 circuits where every arm completes: **−4.8% EPR,
7.1× faster**, and one seed-abort against the sweep's two. Gain grows with
circuit size (36× faster at 100q QFT, 54× at 200q, 43× at 360q) because
that is exactly where the removed `Theta(N_r)` term cost the most.

**The old construction is preserved**, reachable as
`HardwareConfig(local_ext_mode="taint")` — every number published before
2026-08-08 was produced with it, and `verify_router.py` still runs in that
mode so its diff against the frozen `_baseline_*.py` snapshots continues to
prove what it exists to prove, rather than reporting a deliberate change as
a regression.

**Effect on other published claims, already corrected in the current draft:**
`tab:main`, `tab:mech`, and `tab:fair` were regenerated for this
(dSABRE columns only — TeleSABRE reproduced every count on a full rerun,
and pytket-dqc's result files carry no dSABRE column to begin with, so
neither needed touching). Two consequences worth flagging explicitly for the
response letter: the "topological, not BFS" ablation margin at 64q grows
from +15.4% to +18.7%, and **the capacity-penalty ablation's abort count
drops from 3 circuits (submitted) to 2 (QNN, Multiplier)** — the first step
of a trend §6 continues.

---

## 6. Capacity promoted from a scoring penalty to a legality condition; the guaranteed router becomes the default (`router.py`, `config.py`, 2026-08-07 → 2026-08-09)

This is the largest behavioural change since submission, and the one this
section exists to justify for the response letter.

### 6.1 What motivated it

The complexity-analysis session (2026-08-07) asked whether the router's
termination argument extends to *success*. It does not: `route()` provably
terminates (three bounded counters force the loop to stop) but nothing
argues it terminates with the circuit fully routed. The concrete instance:
on a 360-qubit QFT<sup>†</sup>, the backward pass of the fwd→bwd→fwd protocol
aborted with `DEADLOCK_BACKUP_FAILED` at iteration 20,124, 4,535 operations
still unrouted, with more than half the iteration budget unused. Free slots
per core had drifted from an even `[24,16,19,25,15,27]` to `[72,0,0,51,0,3]`
— three of six cores fully saturated — because capacity was a *soft* score
term (`cap_penalty`) while a teleport's only legality condition was
`free(destination) >= 1`. A legal move could empty a core outright, and
nothing pushed back.

<sup>†</sup> **Architecture caveat, found while acting on item 1 of your last
message (§7 below):** this instance was measured on the 6-core, 81-qubit-per-core
architecture (`H_grid_2_3_9_9`, `build_h_grid_architecture(r=2,s=3,m=9)`) that
the project notes document as the standing 360q device, **not** the
20-core, 25-qubit-per-core architecture (`H_grid_4_5_5_5`, diameter 7) that
`sec:largecircuits` actually specifies for the QFT-scalability row. The two
had diverged in `bench_large.py`'s suite table; §7 fixes the wiring. The
motivating abort is real and independently reproducible at the commit that
found it, but it describes a related, wider-core stress configuration, not
literally the architecture behind the published scalability point — the
result of re-running *that* row under the invariant is in §7's own numbers,
not asserted here.

### 6.2 What changed

- **`config.core_reserve = 2`**: the guaranteed transaction's own
  requirement — the state it plans from. It is a *precondition* of that
  transaction, re-established on entry by `_make_layout_safe`, **not** an
  invariant held between actions: see §6.4, which corrects an earlier
  reading of this bullet that made it into the paper.
- **`config.tier1_floor = 2`** (default): an *ordinary* teleport is legal
  only if it leaves the destination `>= 1` free — the minimal change that
  stops a core ever reaching zero, which is what actually breaks routing (a
  qubit in a full core generally cannot be evicted, since clearing the
  outgoing comm port needs a free slot to displace its occupant into). A
  stricter alternative, `tier1_floor=None` (Tier 1 maintains the full
  reserve unaided), was measured and rejected: 5–80× the EPR cost of the
  shipped setting for no additional guarantee.
- **`_safe_route_gate`**: given the invariant, any remote gate can be
  executed by a transaction that provably cannot fail — pick a meeting core
  minimising hop count plus relay cost, relay slack to it if short, walk the
  operand(s) there, execute. Cost is exactly `d` teleports (the
  information-theoretic lower bound) when the meeting core already has
  headroom, `<= 2d` otherwise.
- **`_safe_drain`**: replaces every prior abort path
  (`ITERATION_LIMIT`, `DEADLOCK_BACKUP_EXHAUSTED`, `DEADLOCK_BACKUP_FAILED`,
  `NO_ACTIONS_NO_FALLBACK`) with draining the remainder via the guaranteed
  transaction, bounding worst-case cost at `2*diam(core graph)` EPR per
  remaining remote gate — the first non-vacuous EPR upper bound this router
  has had.
- **Five checked preconditions** (core-graph connectivity, per-core
  connectivity with `>= core_reserve+3` qubits, `P >= n + core_reserve*K+1`,
  recovery enabled, `core_reserve >= 2`) verified at `route()` entry, so an
  infeasible architecture raises immediately rather than discovering failure
  mid-route.
- **`config.safe_mode` default flipped `False -> True`, 2026-08-09** (this
  session): the above stops being opt-in and becomes what every table in the
  paper is produced with.

Full derivation, the termination-with-success theorem, and every
intermediate measurement (including two retracted and one confirmed
re-verification of concrete abort instances against later code) are in
`SAFE_DSABRE.md`.

### 6.3 Measured effect — EPR-neutral, and the ablation ranking changes

**Aggregate EPR**, all six published suites, matched against the *current*
(post-§5) non-safe baseline: **+0.2% geometric mean over 41 circuits**, every
suite within +/-2.6%, compile time neutral-to-better (fewer main-loop
iterations than the non-safe router on the suites checked; `_force_make_room`
and `backup_activations` are exactly zero on every circuit of the 25q/36q/64q
suites under the corrected setup — the guaranteed path is pure insurance at
this scale, never actually exercised).

**What is not neutral, and needs a response-letter paragraph of its own:**
the mechanism ablation's capacity-penalty row. This is the paper's own
"empirical core of the design claim" (Conclusion) and the top-ranked
mechanism in `tab:mech` and the 3rd contributions bullet. Three points now
exist for it:

| | submitted | current (post-§5, pre-safe-mode) | **post-safe-mode** |
|---|---|---|---|
| 64q, circuits aborting without `cap_penalty` | 3 of 9 | 2 of 9 | **0 of 9** |
| 64q gmEPR delta | (not separately reported) | +85.1%<sup>†</sup> (7 survivors) | **+9.4%** (all 9) |
| 25q gmEPR delta | +69.6% | +69.6% (unaffected by §5) | **−2.5%** |

<sup>†</sup> on the 7 circuits that complete without the penalty; not
comparable to the 9-circuit baseline.

This is not noise and not a bug — it is the mechanism working as designed.
The submitted router had exactly one thing standing between it and
infeasibility: the soft `cap_penalty` score term. Under the invariant, a
*hard* floor enforces feasibility unconditionally, independent of any score
term, so removing the soft preference no longer removes the only thing
keeping cores from emptying out — it only changes which *legal* move looks
best. **The capacity term's job has split into two mechanisms with very
different weight**: a small amount of legality (now load-bearing, previously
absent) and a larger amount of preference-among-legal-moves (what the
ablation now actually measures, and what shrank). The paper's current
framing — "the one term that encodes the architecture's finiteness... is the
one whose absence the router survives only by paying double, and on its
densest instances does not survive at all" — described the pre-invariant
router accurately and no longer describes the current one. It needs
rewriting, not just a table update.

**Resolved 2026-08-10 by ablating the two mechanisms independently**
(`code/ablate_capacity.py`, `results/results_ablate_capacity.json`,
`sec:capablation` / `tab:capablation` in the paper). The 2x2 is legality floor
{on, off} x `cap_penalty` {on, off}, plus the strict floor as a fifth arm.
25q is all 6 circuits; 64q is the 7 every arm completes, since `neither`
aborts two:

| arm | legality | `cap_penalty` | 25q gmEPR | Δ | 64q gmEPR | Δ | aborts |
|---|---|---|---|---|---|---|---|
| full | on | on | 15.3 | — | 110.6 | — | 0 |
| no_soft | on | off | 14.9 | −2.5% | 125.7 | +13.6% | 0 |
| no_legality | off | on | 15.1 | −1.1% | 112.3 | +1.5% | 0 |
| **neither** | off | off | **25.7** | **+67.7%** | **207.8** | **+87.8%** | **2** |
| strict | on (φ₁=r+1) | on | 15.4 | +0.4% | 134.5 | +21.5% | 0 |

Whole-suite 64q gmeans for the arms completing all 9: full 173.3
(= `tab:main`), no_soft 189.5 (+9.4%, the `tab:mech` row), no_legality 173.6
(+0.2%), strict 201.5 (+16.3% — corroborates the appendix's 16.0%).

Three things this settles:

1. **The two mechanisms are substitutes.** Either alone is nearly free;
   removing both costs +67.7% / +87.8% and aborts Multiplier and QNN. No
   single-mechanism ablation predicts the pair, which is why the one-row
   version of this table was uninformative either way.
2. **The submitted +69.6% is reproduced, not retracted.** That router had no
   legality condition, so its "no capacity penalty" row *was* the `neither`
   cell — measured here at +67.7%. The current −2.5% is a different quantity
   (what the score term is worth once a floor already holds feasibility).
   Both numbers are correct about their own router, and the response letter
   should say exactly that rather than presenting the change as a correction.
   The two circuits `neither` loses at 64q are also the same two as the
   post-§5 count in the table above.
3. **The legality floor's value is not in the EPR column.** The soft term
   alone avoids every abort on both suites — which is why the submitted
   design completed them — but carries no guarantee that it will. The floor's
   contribution is the abort column and §6.4's theorem, and the strict-floor
   row prices the alternative that would have made the reserve continuous:
   +21.5% at 64q for no additional guarantee.

The extended-set lookahead remains the largest *single scoring term*
(+41.4% at 64q, was +31.5%) and is unaffected by any of this, but it is no
longer the largest ablation effect in the paper — `neither` is. Claims of the
form "the largest single ablation effect" were corrected accordingly in the
abstract-adjacent contributions bullet and the Conclusion.

---

### 6.4 The invariant is (F†), not (†) — a correction to the theorem as first written (2026-08-10)

The two conditions above were conflated in the first write-up, and the
conflation propagated into the manuscript. Recording it because it is a
correctness point in a theorem statement, which is the first thing a reviewer
will check.

**What was claimed.** `dsabre.tex` Section III and `appendices.tex`
Appendix D both stated that `free(c) >= r` holds "at every boundary between
actions", and the theorem's proof inducted on it: *"preserved by every
ordinary action and restored by every transaction, hence holds at every
boundary."*

**Why it is false for what ships.** That describes `tier1_floor=None`, the
strict variant measured and rejected in §6.2. Under the shipped
`tier1_floor=2` an ordinary teleport needs two free slots at the destination
and leaves one, so it may take a core below the reserve. Measured over every
teleport of the 64q suite, some core sits below `r = 2` at **24.6 %** of
action boundaries (11 177 of 45 527), reaching 46.4 % on `ae`. No core ever
reaches 0, which is the floor Tier 1 actually enforces.

**What carries the proof instead.** Feasibility `F = P - n >= rK + 1`. `F` is
exactly conserved — a SWAP permutes occupancy inside a core, a teleport moves
one unit between two — so it holds at every boundary unconditionally, and by
pigeonhole guarantees a donor core with a slot to spare. That is what lets
`_make_layout_safe` restore the reserve on demand whenever a transaction
starts. The code always had this right (`router.py`'s comment at the
capacity-safe section head, and `config.py`'s `tier1_floor` docstring); only
the write-ups did not.

The theorem is unaffected in substance — `(F†)` is the stronger footing, being
conserved rather than merely re-established — but two things change in its
statement:

- the initial layout drops out of the hypotheses. It is *repaired* to the
  reserve at entry, which `(F†)` always permits, not assumed to satisfy it;
- the hypothesis list was wrong in the paper besides. It named connectivity,
  core size, feasibility and the initial layout, and omitted `core_reserve >= 2`
  and `enable_deadlock_recovery` — both of which `route()` does check, and both
  of which the guarantee genuinely needs.

**Fixed 2026-08-10** in `dsabre.tex` (Section III now states feasibility as
Eq. 8 and the reserve as Eq. 9, with the theorem inducting on the former),
`appendices.tex` (Appendix D §1 and the theorem), and `SAFE_DSABRE.md`
(§2, §3, §5 corrected inline; §10.6–10.7 add the current measurements, replacing the stale six-suite table that was deleted).
Equations 4–7 — the scoring rule itself — are untouched by any of this: safe
mode changes the *domain* of the arg min by one comparison per candidate and
leaves the objective alone.

### 6.5 Other paper-side corrections landed at the same time

Found while checking the above against the code, all fixed 2026-08-10:

- **The workflow described the wrong trigger order.** Section III's P4 and
  Figure 2 said the guaranteed transaction is a fallback reached "after
  `N_backup^max` failed recoveries". `_backup_plan` delegates to it
  *first*, on the first deadlock trigger; the greedy and
  `_route_gate_transaction` branches below it are unreachable in safe mode.
  Two further entry points went unmentioned entirely (`_safe_progress`, on an
  empty or fully-rejected candidate list). Measurement note: all 1 074
  transactions across the published suites arrive through `_backup_plan`, so
  the unmentioned paths are real but never taken on these circuits.
- **"Checkpoint–rollback fires only twice across every table here"** — stated
  in Section III and again in the Conclusion — is wrong and its parenthetical
  unsupported (the two nonzero 64q counts belong to `dS`, not the default
  `dSE`). Replaced with `tab:tier2`, a measured per-suite count.
- **Table I overstated its own generality.** Its caption claimed defaults are
  used unchanged in every experiment; `deadlock_limit` is 100 from the 36q
  suite up, and `max_backup_attempts` is raised to the unrouted-gate count in
  safe mode, since a fixed cap would abort a route the theorem says must
  finish. Both are now footnoted, and the Tier-1 destination floor is a listed
  parameter (`phi_1`) rather than a bare code identifier in the appendix prose.
- **`appendices.tex`'s `tab:safemode` mechanism counters were on an
  unlabelled convention** (one layout, not the whole search) and its
  guaranteed-transaction count did not reproduce. Re-measured whole-search and
  the convention stated in the caption.

### 6.6 Presentation changes to the ablation study (2026-08-10)

Not corrections — editorial decisions taken while landing §6.3's ablation,
recorded because they change what a reviewer comparing against the submitted
PDF will find:

- **Tables VI and VII merged.** The mechanism ablation and the new capacity
  2x2 shared a baseline, a protocol, and a row (`tab:mech`'s "No capacity
  penalty" *is* the 2x2's "no score term"), and reported that shared row on
  two different bases — whole-suite against matched-survivors — which is worse
  than either alone. Now one table, rows grouped into *scoring terms* and
  *capacity as a mechanism*, on the whole nine-circuit basis so every
  published number is unchanged. Only `Neither` lacks a whole-suite mean (it
  aborts two circuits); it carries a dagger to the matched figure.
- **The main-text ablation is 64q only.** The 25q suite runs far below the
  fill at which capacity binds and its column was mostly zeros. Moved to
  `app:mech25` rather than deleted, because the 25q `Neither` row (+67.7%) is
  what reconciles with the submitted +69.6% — the reviewers' own number is a
  25-qubit figure, so a like-for-like comparison has to exist somewhere.
- **The degree-swept synthetic suite is removed** (`app:regcz`, `tab:regcz`,
  and the roadmap entry). It was judged not to earn its space. The generator
  and driver survive in `code/gen_regular_cz.py`, `code/gen_regcz_scaling.py`,
  `code/ablate_regular_cz.py` and `code/circuits_regcz*/`, so the experiment is
  reproducible if it is ever wanted back. Note in passing that it had
  independently found §6.3's substitution effect — "the invariant has absorbed
  the term's entire job at high degree" — so that corroboration is no longer
  in the paper.

## 7. QFT-scalability architecture corrected to match `sec:largecircuits` (`bench_large.py`, 2026-08-09)

Found while regenerating data for §6: `bench_large.py`'s `SUITES["360q"]`
pointed at a 6-core, 81-qubit-per-core architecture
(`build_h_grid_architecture(r=2,s=3,m=9)`, diameter 3), not the 20-core,
25-qubit-per-core one (`r=4,s=5,m=5`, diameter 7, 500 physical qubits) that
`sec:largecircuits` specifies ("core size held at 5x5 throughout... 4x5
(20,500)... diameter 3->5->7"). 100q and 200q already matched (diameters 3
and 5, confirmed by direct computation). The correct device file,
`H_grid_4_5_5_5.json`, already existed in `~/Documents/telesabre/devices/`
and needed no generation — only the suite table's wiring was wrong.

The 2x3-of-9x9 architecture this used to point to is not fabricated or
wrong in general — it is the project's long-standing, separately-documented
360q device (`H_grid_2_3_9_9.json`, "preferred"), used throughout
`SAFE_DSABRE.md`'s safe-mode investigation including §6.1's motivating
abort. It simply is not the architecture the current paper draft's
scalability-series text describes, and the two had drifted apart without
either the code or the project notes flagging the mismatch.

**Corrected 360q numbers (dSE, this router):**

| circuit | CX | dSE EPR | dSE SWAP | Tier-2 activations |
|---|---|---|---|---|
| qft | 13,300 | **1890** | 15,606 | 35 |
| qpeexact | 13,979 | **1938** | 16,090 | 31 |

`force_make_room` is 0 on both — the invariant holds throughout, same as
every other suite. `dS` (topological extended set) gives 2326 / 2394
respectively, for whatever future ablation needs the comparison.

## 7a. Resolved: `bench_large.py`'s 200q architecture was transposed — TeleSABRE converges fine (found and fixed 2026-08-09)

**Corrected.** An earlier version of this section claimed TeleSABRE does not
converge at 200q. That was wrong, and the reason is now understood
precisely: `bench_large.py`'s `SUITES["200q"]` used
`build_h_grid_architecture(r=4, s=3, m=5)`, and the *actual* driver behind
`tab:main`'s published scalability row — `bench_scaling.py --design b`,
found by tracing the string `1232` through eight-plus past sessions back to
2026-07-29, where it has been byte-identical ever since (dSABRE's own number
moved release to release; TeleSABRE's never did — "ts unchanged" is even
written into `results_scaling_b.json`'s own metadata) — uses
`(r=3, s=4, m=5)` instead. **These are not the same graph.**
`build_h_grid_architecture`'s inter-core link placement is not symmetric
under a row/column swap: verified computationally, the two orderings' 17
inter-core links do not coincide as sets, despite identical core count,
core size, and core-graph diameter. `H_grid_3_4_5_5.json` — a separate file
sitting right next to the one `bench_large.py` pointed at — matches the
`(3,4,5)` construction exactly. `H_grid_4_3_5_5.json` matches `(4,3,5)`.
Both are legitimate device files; `bench_large.py` was simply pointed at
the wrong one for this specific row.

This is the same *class* of bug as §7 (a second candidate architecture
existing and the wrong one wired in), manifesting more subtly — not a
different `m`, a transposed `(r,s)` that happens to preserve every
aggregate property (core count, size, diameter) that a first check would
compare, which is exactly why it survived the diameter-based verification
that caught §7's 360q problem. `bench_large.py` fixed to
`build_h_grid_architecture(r=3, s=4, m=5)` /
`H_grid_3_4_5_5.json`, mirroring the file that has been correct all along.

**TeleSABRE, unchanged, reproduces 1232 EPR on the corrected architecture** —
see the confirmation run below. dSABRE's own numbers were re-derived on the
same corrected graph rather than assumed unchanged, since a
differently-oriented (if similarly-shaped) graph is not guaranteed to route
identically even for an architecture-agnostic router.

```
[200q, qft, corrected (r=3,s=4,m=5) architecture — 3 seeds each, best-of]
  ts:  eprs=1232  ls=7089   (seed 0)                    <- matches tab:main exactly
  dS:  eprs=745   ls=7004   (sl_seed0, 19.6s)
  dSE: eprs=766   ls=6482   (sl_seed2,  4.2s)            <- vs tab:main's published 825/7166
```

`dSE` moves from the published 825/7166 to 766/6482 (-7.2% EPR, -9.5% SWAP) —
consistent with every other suite in this document: TeleSABRE is unchanged
because it was not rerun, dSABRE improves because §4-§6's fixes postdate the
submission. This is the same direction and rough magnitude as the 25/36/64q
deltas elsewhere in this file, not an anomaly specific to 200q.
`qpeexact_200` was also computed (dS=1002/ls=9145, dSE=896/ls=6834) while
verifying the architecture fix, but is not part of any published table —
`tab:main`'s scalability row is QFT-only at 100/200/360q (confirmed by
exhaustive grep of both `dsabre.tex` and `appendices.tex`, see the
circuit-manifest work below) — so it is recorded here for completeness and
not carried further.

**360q's non-convergence is real and independently confirmed, not an
artefact of this bug.** `bench_scaling.py`'s own `(r,s,m)=(4,5,5)` for 360q
already matches what §7's fix uses — no transposition issue there — and,
decisively, `results_scaling_b.json` **itself** carries `"ts": null` for
360q and has since the file was first committed (`d81702c`, 2026-08-06):
the authoritative source already agrees with the direct re-test in this
section below. This is exactly what `tab:main`'s own `$^\ast$` marker on the
360q row already says. Nothing to fix here; the all-10-attempts trace below
is kept as corroborating detail, not as a finding that changes anything
published.

```
[360q, qft, seed 0 — all 10 of TeleSABRE's internal attempts, none succeeding]
attempt 1:  reached maximum iterations (7580)   Safety Valve activated 1
attempt 2:  reached maximum iterations (3489)   Safety Valve activated 1
attempt 3:  reached maximum iterations (7666)   Safety Valve activated 11
attempt 4:  reached maximum iterations (6745)   Safety Valve activated 5
attempt 5:  reached maximum iterations (7830)   Safety Valve activated 9
attempt 6:  reached maximum iterations (10961)  Safety Valve activated 27
attempt 7:  reached maximum iterations (8640)   Safety Valve activated 15
attempt 8:  reached maximum iterations (11894)  Safety Valve activated 25
attempt 9:  reached maximum iterations (9117)   Safety Valve activated 15
attempt 10: reached maximum iterations (2963)   Safety Valve activated 1
No successful runs :(
```

Matches what the paper's own related-work section names as TeleSABRE's
limitation: *"a safety valve that drops most of the lookahead once routing
stalls."* `max_iterations` is configured at 200,000; what actually
terminates each attempt (a few thousand iterations in, always under
"Safety Valve ON" with the remaining-gate count frozen across several
consecutive iterations first) is some other internal TS criterion —
`max_solving_deadlock_iterations: 1000` is the likely candidate, not traced
further into TS's own source since it does not change anything published.

**Going forward, `bench_scaling.py --design b` — not `bench_large.py` — is
the script to use for this row.** It is proven correct (it produced every
number currently in `tab:main`'s scalability section), and it differs from
`bench_large.py` in more than the one bug found here: it runs TeleSABRE
through `bench_large.py`'s own config template rather than `benchmark.py`'s
(the two spell the Hungarian-layout parameter differently — one is silently
ignored by TS, worth 410 vs 312 EPR on a 64q circuit per that script's own
comment), uses a 900 s TS timeout rather than 600 s, and gives the dSABRE
router `max_iterations=200,000` rather than `bench_large.py`'s
module-level `50,000`. None of these were shown to matter for the specific
numbers in this section, but they are open discrepancies between the two
scripts worth closing rather than carrying forward.

**Not investigated further**: retuning TS's own hyperparameters to find a
configuration that reconverges was judged out of scope — it risks producing
*a* successful number that is not verifiably *the* number that produced
1232, and the project's own convention (§5, §6) is not to regenerate other
compilers' data without a specific reason to. If an archived config or
command for the original 200q/360q TS runs exists, that would resolve this
directly; absent that, `tab:main`'s 200q TS figure and 360q's convergence
claim need re-deriving from whatever did produce them, or dropping with the
same `$^\ast$` non-convergence marker 360q's QFT point already carries in
the submitted-era table.

---

## 8. The three recovery budgets are derived, not hand-tuned (`router.py`, `config.py`, drivers, 2026-08-11)

**Submitted version, and the state up to today:** `deadlock_limit`,
`max_backup_attempts` and `max_iterations` were set per suite in each
driver — 50/50/10 000 at 25q, 100/100/20 000 at 36q, 64q and the heavy-hex
suites, 200/200/50 000 at 100–360q. Table I's footnote conceded
`L_deadlock` as "the only per-suite parameter in the paper".

**Current version:** all three accept `None`, meaning *derive me*, and every
driver here passes `None`:

| budget | derived as | values |
|---|---|---|
| `deadlock_limit` | `General_dSABRE_Router.deadlock_limit_for(arch)` = `4·diam(core graph)·(diam(core)+1)` | 56 B-grid, 84 @64q, 104 heavy-hex, 108/180/252 @100/200/360q |
| `max_backup_attempts` | unrouted-gate count (safe mode already did this at `route()` entry) | per circuit |
| `max_iterations` | `iterations_bound` = `|G_2q|·(L+1)`, the worst case the termination theorem allows | per circuit; cannot bind |

The rationale for `deadlock_limit` is the one that was already in
`deadlock_limit_for`'s docstring: a gate about to resolve unaided needs at
most `diam(core graph)` teleports, each preceded by up to `diam(core)`
intra-core SWAP iterations to reach a staging slot. The factor 4 is the
smallest integer covering the values that had been tuned by hand.

**Measured effect: none.** `probe_derived_deadlock.py --rule arch` reruns
every suite — 25q, 36q, 64q, 100q, 200q, 360q, heavy-hex ring and star, dSE,
three SabreLayout seeds × fwd→bwd→fwd — and reproduces the published EPR
counts **exactly**, best and per-seed median alike, at equal or fewer
iterations. Nothing in `tab:main` or any other table moves.

**A constant `L = 10` was measured and rejected.** The argument for it is
sound as far as it goes: in safe mode the stall window is pure waste, since
recovery cannot fail and the checkpoint restore discards whatever was done
while stalling, so EPR should not depend on how long the router thrashes
first. It holds through 200 qubits — identical EPR on every suite up to
there, at 20–35 % fewer iterations — and then breaks on the 360-qubit
QPEexact, where cutting the window short loses the seed that produced the
reported route: **1938 → 2735 EPR (+41 %)**, while the three seeds' median
moves only +2 %. So the loss is a best-of-three effect rather than a
systematic one, but the reported number is the best of three. This is a
correction to the sweep recorded in `deadlock_limit_for`'s docstring, which
had only 64q evidence behind "EPR is flat at L=10".

**Paper effect:** Table I's `L_deadlock` row now reads `4D_K(D_M+1)` and
`N_backup^max` reads `|W|`; the "only per-suite parameter" footnote is gone,
replaced by the derivation and the per-architecture values. §IV-A's
"defaults are used unchanged on every suite" now holds literally.

---

## Quick reference: what still differs from the submitted router

| mechanism | submitted | current | why |
|---|---|---|---|
| node_decay | on | **removed** | see §2; retired independently, cause of the relief mix-up |
| corner reservation | monolithic (whole-chip) | **per-core, fill-adaptive** | §1; fixed 2026-07-17 |
| k=0 fallback | (n/a, layout was monolithic) | **floored at k>=1 when feasible** | §1a; fixed 2026-08-02 |
| congestion relief | on | **removed** | §3; no non-failing evidence found, apparent benefit traced to node_decay |
| checkpoint-rollback | greedy, first-hop-`break`, no rollback | **transactional**, both operands tried, snapshot/rollback | §4; fixed 2026-08-05 |
| `E_c` (intra-core lookahead) | separate `Theta(N_r)`-per-call sweep | **restriction of `E` to core `c`** (as Table I already defined it), `O(1)` amortised | §5; commit `89c088a`, 2026-08-08. `local_ext_mode="taint"` reproduces the old behaviour |
| capacity | **soft score penalty only**, `free(dst)>=1` legal | **hard legality floor** (`tier1_floor=2`, so no core reaches 0) + a guaranteed-completion transaction, which re-establishes the `core_reserve=2` state on entry rather than the router holding it continuously | §6, §6.4; `safe_mode` default flipped 2026-08-09 |
| termination | guaranteed (budget-bounded); **success not guaranteed** | **success guaranteed**, given 5 checked preconditions, by induction on the conserved `F >= rK+1` | §6, §6.4; the paper's first non-vacuous completion guarantee |
| 360q scalability architecture | (n/a — added post-submission) | corrected `r=2,s=3,m=9` (diam 3) -> **`r=4,s=5,m=5`** (diam 7), matching `sec:largecircuits` | §7; fixed 2026-08-09 |
| TeleSABRE @ 200q/360q | reportedly converges (1232 EPR @ 200q) | **confirmed non-converging**, `"No successful runs"`, all 10 internal attempts, both circuits checked | §7a; needs the original config or a table-note fix, not a code fix |
| hop-gain | on | unchanged | independently evidenced, not implicated |
| max_iterations | 10000, hand-set per suite | **derived**: `iterations_bound`, so it cannot bind in safe mode | §8; 2026-08-11 |
| deadlock_limit | 50, hand-set per suite | **derived**: `4·diam(core graph)·(diam(core)+1)` | §8; reproduces every published number exactly |
| max_backup_attempts | 50, hand-set per suite | **derived**: the unrouted-gate count | §8 |
