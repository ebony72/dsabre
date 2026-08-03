# Teleport commitment / interleaving investigation (2026-08-03→04)

Does dSE (`dSABRE_BurstExt`) carry a remote gate to completion once it starts
teleporting it, or does it interleave with (and sometimes get set back by)
other pending gates? Four mitigation mechanisms were prototyped and ablated;
none is adopted as a default (all config toggles below default to their old
value — every existing published number is untouched). This file is the
record for anyone picking this back up.

All numbers are on the full published 64q suite (`ae, ghz, graphstate, qft,
qnn, random, qpeexact, qaoa, multiplier`), production layout search + fwd→bwd→fwd
protocol, unless noted otherwise. Diagnostic traces used a smaller 9/10-circuit
set (same suite minus `qnn`, since it's slow to trace) plus `wstate`/`vqe_su2`.

---

## 1. Diagnosis: the router has no memory of which gate it was mid-move on

Instrumented (via monkey-patching the real `_generate_candidates`/`_apply_teleport`,
not a reimplementation) to log, at every teleport-relevant iteration, every
pending inter-core front-layer gate's `arch.phys_dist` and which gate's
candidate actually won.

**Per-gate view** (of 1594 gates across 9 circuits that needed ≥2 teleport-relevant
iterations to clear):
- 90.8% had ≥1 *other* gate's teleport served while pending.
- 35.6% saw their own qubit-pair distance increase (backslide) at least once
  during their pending span.
- Of those backsliding gates, 98.6% had *at least one* backslide instance
  attributable to a different gate's action.

**Per-event view** (finer-grained: 1395 individual backslide *events*, not
gates — this is the number to use, not the per-gate 98.6% above, which only
asks "did at least one of this gate's several backslides come from someone
else" and over-counts relative to the true event-level split):
- **52.3%** of individual backslide events are caused by a different gate's
  action. Mean magnitude 1.12 (histogram: 651/730 exactly delta=1, 77 delta=2)
  — matches a same-core `_evict` shuffle (weight-1 intra-core edges), not a
  cross-core displacement. `_force_make_room` fired during only 1 of 730 such
  events.
- **47.7%** are caused by the gate's *own* chosen teleport backfiring. Mean
  magnitude **9.97**, with 649/665 events at *exactly* delta=10 — one
  full inter-core-link weight in `phys_dist`. Not explained by
  `_force_make_room` (0.8%) or the partner qubit also moving (0.6%). Traced
  one instance directly (`ae`, node 1523, iter 42): the winning candidate
  (score 31.3) moved the qubit core-graph *farther* from its partner, but
  every other candidate for that gate scored ~1000 (hit `_evict_cost`'s
  "no free slot anywhere in that core" sentinel of 999) — a genuine
  congestion effect, not a scorer bug: every option was bad, the chosen one
  was merely least-bad. Recurs for the same node across many consecutive
  iterations on `ae`, consistent with it being one of the gates that
  eventually trips a deadlock recovery.

Conclusion: the router re-scores every competing inter-core gate from
scratch every iteration, with no state tracking which one it was advancing.
Roughly half of backsliding is a small side effect of servicing someone
else; the other half is the scorer being *forced* into a locally bad move by
local congestion, which no "remember the gate" mechanism below addresses.

---

## 2. Mechanisms prototyped

All four are `HardwareConfig` fields in `code/config.py`, all default to
off/0.0 (verified byte-identical routing decisions vs. before these changes
existed, on `ae`/`graphstate`/`qft` at minimum).

### 2a. `commit_bonus: float = 0.0`
Score discount for a candidate continuing the gate the *previous* iteration's
teleport advanced (soft — can still lose to a better-scoring competitor).
Ablation (`ablate_commitment.py --bonuses 0,2,8`): gmean EPR **−4.3%** at
bonus=8, 4 circuits win / 4 lose / 1 tie — but the net "win" is almost
entirely `multiplier`'s outlier (−34.9%); **excluding it, gmean is +0.4%** (a
regression). `graphstate` +12.5%, `qft` +2.8%, `random` +4.6%.

### 2b. `commit_hard_lock: bool = False`
Hard variant: while the committed gate still has a legal candidate,
competing gates' candidates are never even generated (falls back to normal
scoring for that one iteration if the locked gate has no legal move). Same
`ablate_commitment.py --bonuses hard`: gmean EPR **−5.0%**, same 4/4/1 split
but more extreme both ways (`qft` flips to −6.9%, `qpeexact` worsens to
+16.1%, `multiplier` improves to −41.6%). Verified via a v2 diagnostic trace:
97.3% of "commit-active" iterations are honored (competitor never wins);
the ~2.7% overrides are almost entirely `_backup_plan` (deadlock recovery)
deliberately bypassing the lock, not the scoring mechanism failing. Wall-clock
time tracks the EPR direction, not uniformly faster (`ae` +18% wall time
despite better EPR, because the OTHER 2 of 3 SabreLayout seeds hit more
backup activations even though the winning seed improved).

A fast-path optimization for the locked branch (restrict `free_cache` to the
locked gate's neighboring cores; fix the `dF` term to O(1) via front-layer
qubit-disjointness — the second fix is universally valid, applied to both
branches) was verified behavior-identical but gave **no measurable wall-clock
speedup** (0.95–1.00× on a 10-core/80q architecture) — the actual expensive
part (skip scoring competitors) was already present from the first
implementation; the real bottleneck is elsewhere (checkpoint `deepcopy`,
extended-set BFS construction), untouched by this change.

### 2c. `cheapest_first_weight: float = 0.0` + `evict_distance_aware: bool = False`
Target the two root causes directly instead of re-litigating whose teleport
wins: (i) score penalty proportional to a gate's own current `phys_dist`, so
nearly-finished gates cut in line ahead of expensive ones; (ii) when
`_evict`/`_force_make_room` choose a free slot for a bystander qubit, prefer
one that doesn't worsen that bystander's own pending gate.
`ablate_priority_eviction.py` swept weight alone (0.2, 0.3) and combined with
eviction-awareness. **Best result of the whole investigation**:
`cheapest_first_weight=0.3` + `evict_distance_aware=True` → gmean EPR
**−4.2%**, SWAP −0.8%, **5 win / 2 lose / 2 tie**. `graphstate` ties baseline
exactly (cancels the priority term's standalone regression there), `qft`
becomes the single biggest winner (−16.5%, vs. being a loser under both
commit mechanisms), and the worst-case regression is much milder
(`qpeexact` +5.7% vs. hard-lock's +16.1%). Weight alone (no eviction
awareness) is worse-balanced (4/4/0-ish, `qpeexact` +6.2%).

### 2d. `backup_relay_mode: bool = False`
See §3 — replaces `_backup_plan`'s cross-core hop mechanism.

**Summary table** (gmean EPR / SWAP vs. baseline, full 9-circuit suite):

| mechanism | EPR | SWAP | win/lose/tie |
|---|---|---|---|
| `commit_bonus=8` | −4.3% | +1.0% | 4/4/1 (driven by 1 outlier) |
| `commit_hard_lock` | −5.0% | +0.1% | 4/4/1 (more extreme both ways) |
| `cheapest_first_weight=0.3` + `evict_distance_aware` | **−4.2%** | **−0.8%** | **5/2/2** |
| `backup_relay_mode` | −3.6% | −0.3% | 6/2/1 |

None recommended as a new default without further work — see §5.

---

## 3. Checkpoint-rollback: invariant-preserving gate routing

**Theory.** Precondition (\*): every core ≥1 free qubit, total free ≥ K+1
(K = core count). By pigeonhole this guarantees ≥1 core with ≥2 free
("slack") at all times — total free is exactly conserved by every teleport
and SWAP, so if (\*) held when a routing procedure starts, it still holds
throughout provided every teleport-in destination is checked to have ≥2 free
first (leaves ≥1 after). Built `_relay_room_to`/`_find_nearest_slack_core`
(BFS over the connected core graph for the nearest slack core, relayed in
one hop at a time by teleporting an arbitrary non-gate qubit) generalizing
`_force_make_room`, which only looks at *direct* neighbors and silently gives
up beyond that. Proved termination (finite BFS, strictly decreasing
core-distance per payload hop) and success (never violates (\*) given the
precondition).

**Optimality.** The unconstrained lower bound is $d = \mathrm{dist}_{\text{core}}(c(q),c(q'))$
teleports. This is **not always achievable** under (\*): every payload
teleport mechanically deposits the freed slot exactly *one hop behind* the
qubit's new position (never ahead — a qubit's departure frees its old core,
never the destination), so advancing the same qubit again always needs a
detour of ≥2 hops to get slack ahead of it again. Worst case (verified via
exact BFS over the (pos_q, pos_qprime, pos_slack) state space, `verify_optimal_epr.py`
in scratchpad, not checked into the repo) is close to $3d$, not $d$.

A proposed closed form $D = a+b+\min(a,b)$ (where $a,b$ = distances from
each gate qubit to the slack core, meeting there) was tested against the
same exact BFS and **only matches the trivial $a=b=1$ case** — true optimal
is roughly 3× higher once either distance reaches 2 (e.g. $a=b=2$: formula
says 6, exact optimum is 10; $a=b=3$: formula says 9, exact optimum is 17).
This remains **unresolved** — no revised construction was supplied before
the session closed. A follow-up correction (teleporting into $c'$ needs
$c'$ to have ≥2 free; a core hosting both gate qubits *without* an
intermediate relay needs ≥3 free beforehand) was confirmed correct, but is
already an automatic emergent property of the exact BFS model — not a new
constraint requiring a code or solver change.

**Implementation.** `backup_relay_mode=True` swaps `_backup_plan`'s
direct-neighbor-only cross-core loop for the BFS relay. Verified: default
mode is byte-identical to before this code existed. The new code's *own*
hops never create a fresh zero-free-core violation (checked on `ae`:
`violations_inside_backup_plan=0`). **Important limitation, found
empirically**: precondition (\*) essentially **never actually holds** when
`_backup_plan` fires on this suite — normal routing (via the *unmodified*
`_force_make_room` path elsewhere in `_apply_teleport`) already lets cores
hit 0 free before deadlock recovery ever runs (`precondition_held=0` /
`precondition_violated>0` on every circuit that exercises backup_plan: `ae`,
`qft`, `qpeexact`, `multiplier`). When the relay's own filler-teleport's
*source* core happens to already be at 0 free (inherited brokenness, not
caused by the relay), `_apply_teleport` silently falls back internally to
the old unguarded `_force_make_room` — this is where `qft`
(`violations_inside_backup_plan=2`) and `qpeexact` (`=1`) pick up genuine
new violations, and not coincidentally where the ablation shows `qpeexact`'s
one regression (+16.1% EPR). So the formal guarantee is correct but rarely
load-bearing in practice; the empirical win comes from BFS search being a
strictly more thorough heuristic than direct-neighbor search, not from the
invariant proof holding end-to-end.

Ablation (`ablate_backup_relay.py`): gmean EPR −3.6%, SWAP −0.3%, 6 win / 2
lose / 1 tie. `ae` −30.2%, `qft` −7.8% (both: fewer/cheaper backup
activations cascading into a much better downstream layout via the
fwd→bwd→fwd protocol — see the multi-pass note below). `qpeexact` +16.1% as
explained above.

**Multi-pass subtlety worth remembering**: a circuit's *reported*
`backup_activations` can be 0 in both conditions while EPR still differs
substantially, because `route_layout_set` only reports the single winning
pass's stats out of 9 total `route()` calls (3 SabreLayout seeds ×
fwd→bwd→fwd). The divergence can happen in an *upstream* pass (e.g. seed 0's
`pass1`) whose own backup activations reshape the layout that later passes
inherit, without that upstream pass itself being the one reported.

---

## 4. Config toggle reference (`code/config.py`)

| field | default | effect when enabled |
|---|---|---|
| `commit_bonus: float` | `0.0` | soft score discount for continuing the last-teleported gate |
| `commit_hard_lock: bool` | `False` | hard version — competitors' candidates not even generated |
| `cheapest_first_weight: float` | `0.0` | score penalty ∝ a gate's own current distance (favor near-finished gates) |
| `evict_distance_aware: bool` | `False` | eviction prefers a slot that doesn't worsen the evictee's own pending gate |
| `backup_relay_mode: bool` | `False` | BFS-relay replacement for `_backup_plan`'s cross-core hop |

## 5. Scripts

- `code/ablate_commitment.py` — sweeps `commit_bonus` values and `hard` lock; `--bonuses 0,2,8,hard`
- `code/ablate_priority_eviction.py` — sweeps `cheapest_first_weight` × `evict_distance_aware`
- `code/ablate_backup_relay.py` — greedy vs. relay `_backup_plan`
- `code/probe_hop_distance.py` — updated for the new `_generate_candidates(committed_nid=...)` signature; check this file first if adding more `_generate_candidates`/`_apply_teleport`/`_force_make_room` overrides elsewhere, since their signatures gained keyword args this session (`committed_nid`, `partner_phys`)

Diagnostic-only scripts (not checked in, lived in a scratch directory,
reproduce from this file's description if needed): the per-gate/per-event
teleport-commitment tracer, the hard-lock speedup A/B benchmark, the
relay-invariant verifier, and the exact BFS optimal-EPR solver.

## 6. If picking this back up

1. `cheapest_first_weight` + `evict_distance_aware` is the best-balanced
   candidate found — worth trying combined with `backup_relay_mode` (not
   yet tested together) and worth a run on the 25q/36q suites before any
   adoption decision, since everything above is 64q-only.
2. The self-caused-backslide congestion pattern (§1, the 47.7%/delta≈10
   half) isn't addressed by any of the four mechanisms — all of them
   arbitrate *between* gates, none reduces the chance that a gate's *own*
   best-available move is still a net regression under local congestion.
3. The optimal-EPR bound for the checkpoint-rollback procedure is still
   open — resolve by getting a concrete step sequence for the disputed
   formula and checking it against the exact BFS solver, or by conjecturing
   and verifying a corrected closed form.
