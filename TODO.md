# TODO

## Safe mode (2026-08-07→09) — shipped, not the default, one open re-verification

`config.safe_mode` (default `False`) landed on `tcad-revision`
(`c25c987`…`abd9bfd`): a capacity invariant (`free(c) >= core_reserve`,
`core_reserve=2`) promoted from a soft scoring penalty to a legality
condition, plus a guaranteed gate-execution transaction and a
termination-with-success theorem. Full derivation, measurements, and the
"should this be the default" analysis are in `SAFE_DSABRE.md`; the paper
side is `paper/appendices.tex`'s `app:safemode` (six subsections, including
`app:safemode_theory` on what's provably different from the default router,
independent of any benchmark).

**Where it stands, 2026-08-09:**
- Measured EPR-neutral: **+0.2% geometric mean over 41 circuits**, all six
  suites (25q/36q/64q/100q/200q/360q), each within ±2.6%. Re-measured once
  already against an unrelated concurrent change (`89c088a`, which redefined
  `E_c`) — the conclusion survived and tightened.
- `tier1_floor=2` (the "split" setting — Tier 1 only prevents a core hitting
  zero; the transaction re-establishes the fuller reserve itself when it
  runs) is now the shipped default. The alternative (`tier1_floor=None`,
  strict) costs 5-80x more for no additional guarantee — never use it.
- `safe_route_failed` is 0 in every run ever made, including a deliberately
  broken entry layout and heavy-hex architectures whose comm ports are
  articulation points (grid cores have none).
- One real bug found and fixed: `_safe_drain` gave up while the front layer
  was transiently all-1q, aborting `random_100` under the (now non-default)
  strict floor. Fixed in `dce7bb2`; `code/test_safe_drain.py` regression-tests
  it directly (fails on the parent commit).

**Re-verified 2026-08-09, mixed result — not just measurement, the abort
*claims themselves* needed re-checking.** Two of the three historical abort
anecdotes retracted, one confirmed:

| instance | historical claim | current code (2026-08-09) |
|---|---|---|
| `qft_360` pass 2, seed 0 | `DEADLOCK_BACKUP_FAILED` @ iter 20124 | **retracted** — completes, no abort |
| `qft_200`, 1 of 3 layouts | aborted | **retracted** — all 3 layouts complete |
| `random_200`, 3 of 3 layouts | aborted, no result | **confirmed** — still aborts on all 3, same failure mode (pass 1 hits the 50000-iteration budget), just ~100x faster to reach it, presumably from `89c088a`'s own Θ(N) DAG-bookkeeping fix rather than anything about safe mode |

`89c088a` (an unrelated, concurrent change to `E_c`'s construction) altered
pass 1's routing trajectory enough that `qft`-family circuits stopped landing
in the states that used to trigger a `DEADLOCK_BACKUP_FAILED`. That is a real
lesson, not just a footnote: **the flagship motivating example for this whole
mode stopped reproducing under a change that had nothing to do with it.**
`random_200` survived the same check, so "the default router cannot route
this circuit, safe mode can" is a confirmed, current fact for at least one
instance — but it needed re-checking, not assuming, and the theoretical
guarantee (`SAFE_DSABRE.md` §15) is deliberately built to not depend on
either outcome. See `SAFE_DSABRE.md` §10.1, §13.2, §13.4 for the full
derivation.

**Not yet done, in priority order:**
1. Decide whether the concrete-abort evidence should be supplemented with a
   synthetic worst-case instance built to violate the *default* router's
   free-slot invariant on purpose, rather than relying solely on an
   incidentally discovered one (`random_200`) that could in principle stop
   reproducing under some future unrelated change, the same way two of its
   siblings just did.
2. `qnn_200` (79,798 CX) — the one circuit in any suite still unmeasured
   under safe mode.
3. Re-sweep `deadlock_limit` under the `tier1_floor=2` split default — the
   "L=10 is free" measurement was taken under the strict floor, where the
   guaranteed transaction fired far more often.
4. A fallback policy for architectures below the feasibility line
   (`P < n + core_reserve*K + 1`): currently `route()` raises; a production
   default would need to degrade to the default router with a warning
   instead.
5. `code/layout.py`'s `adaptive_corner_count(reserve=)` parameter is
   implemented but **not committed** — that file carries ~240 lines of
   unrelated uncommitted work and committing it would sweep that in. Safe
   mode doesn't depend on it (`_make_layout_safe` repairs an unsafe entry
   layout at `route()` time regardless), so this is cleanup, not a blocker.
6. Full regeneration of every published table under `safe_mode=True`, if and
   when the default switches — not before, and not mid-revision (see
   `SAFE_DSABRE.md` §14 for the reasoning).

**Recommendation (`SAFE_DSABRE.md` §14): not the default during this
revision.** Coverage and EPR-neutrality are no longer objections; the
remaining ones are the page-budget cost of regenerating every table
mid-revision, thin test coverage on the escape-hatch code paths (of which the
`_safe_drain` bug is a concrete instance), and no fallback for infeasible
architectures. Revisit post-acceptance.

## Router review 2026-08-06 — 2 of 7 suggested fixes applied, 5 open

An external review of `code/router.py` (`_apply_teleport` focus) proposed seven
"mandatory" fixes plus an `l2p`/`p2l` bijection invariant test. Each was checked
against the code and, where reproducible, reproduced. Two were silent-wrong-answer
bugs and are **applied**; the rest are open, ordered below by whether they can move
a published number.

Baseline for any re-check: instrumenting every mutating router method with the
proposed bijection assertion and running the full 25q (6 circuits) and 64q (11
circuits) suites — dSE, layout candidate 0, single forward pass — gives **zero
violations** over ~5,000 teleports and ~35,000 SWAPs. The maps are not currently
being corrupted; the assertion is a tripwire for future edits, not a live bug.

### Applied 2026-08-06 (25q and 64q verified bit-identical)

- **`initial_layout` validation** (`route()`). `p2l` was built last-write-wins, so
  a repeated physical qubit silently dropped a logical one and routing returned
  `aborted=False` with every gate retired, for a start state where two qubits share
  one site. Now raises `ValueError`; an off-chip physical raises there too, instead
  of a bare `KeyError` from a distance lookup much later.
- **>2-qubit operation rejection** (`route()`). A 3-qarg node is neither drained as
  a 1q gate nor seen by `_front_2q`, so it blocked its wires and the run ended in
  `DEADLOCK_BACKUP_FAILED` — an input error read as a hard circuit. Now raises
  `ValueError`. Preflight over all 76 circuits in the 9 circuit dirs found zero >2q
  ops, so nothing published is affected; the likely future trigger is an unstripped
  `barrier`.

### Open — safe, no number moves

- **Teleport cost accounting is inconsistent.** `_apply_teleport` (router.py:412)
  charges `cost_teleport + cost_teleport_per_hop * core_dist`; `_force_make_room`
  (router.py:562) charges the base only. Same single hop measured at 23.0 vs 10.0
  with `cost_teleport_per_hop=4.0`. Harmless today — that field is 0.0 in
  `config.py` and no driver overrides it — so fix it before someone sets it nonzero.
- **Contiguous core-ID assumption.** `architecture.py:74` pre-seeds
  `_inter_links_between` over `range(num_cores)`, and `router.py:216` builds
  `free_cache` the same way. Non-contiguous IDs (e.g. `{0, 5}`) raise
  `KeyError: (0, 5)` at *construction*, so the router line is unreachable-bad and
  all three builders are contiguous anyway. Two-line hygiene fix
  (`for c in arch.intra`); lowest priority, since it fails loudly.

### Open — real, but they change results; gate behind a flag and regenerate

- **`_force_make_room` can evict the gate's own anchor.** It takes `exclude_virt`
  (one qubit), not a `protect` set, and `_backup_plan` (router.py:863) calls it on
  `next_core` with no exclusion at all — and on the last hop `next_core == c2`, the
  anchor's core. Reproduced: with the anchor on core 1's outgoing port,
  `_force_make_room(1)` teleports it to core 3, so the mover is then driven to a
  core the anchor has already left. Fix is a `protect` set replacing `exclude_virt`.
- **router.py:712's comment is false, from the same gap.** It claims "`anchor` never
  changes core here: relay hops protect it via `protect=`" — but the comment
  directly below explains why that reasoning fails for `mover` (`_force_make_room`
  inside `_apply_teleport` knows only the filler), and it fails identically for
  `anchor`. `target_core` is computed once at router.py:710 and never rechecked;
  what saves it is incidental — `_ipath` raises `KeyError` across cores and the
  `except Exception` at router.py:761 rolls the transaction back, discarding a whole
  relay chain. Fix the comment even if the code is left alone.
- **Caller-side EPR leak around `_force_make_room`.** The function is already atomic
  (its only `False` return, router.py:546, precedes every state mutation), so "make
  it transactional" is a no-op. The leak is at router.py:862: it spends a real EPR
  relocating a qubit, and if the `_apply_teleport` two lines later fails, that rolls
  back only its own changes — the room-making teleport stays applied and stays
  counted. Small blast radius: at 64q only ae/qft/qnn/qpeexact reach the backup path
  at all (2–3 activations each), and none of the largest circuits do.

### Rejected — not during the TCAD revision

- **Resetting `backup_attempts` after gate progress.** It is a deliberate global
  budget, not an oversight: `config.py:15` documents scaling it per suite and the
  drivers do (100 at 64q, 200 for pytket-large). Resetting on progress makes the
  abort criterion strictly weaker, so aborts flip to completions. `code/results/`
  holds 162 `aborted: true` entries, concentrated in exactly the ablation suites
  (freeslots, occupancy, sabrelayout, regcz) where the abort *is* the finding —
  changing this re-opens those tables and invites the reading that the success
  criterion was tuned. Termination does not need it; `max_iterations` backstops.

### The proposed invariant test

Worth adding, but not where the review aimed it. `_apply_teleport` already runs
exactly those two assertions post-commit (router.py:426-431), and the third
(`len(set(l2p.values())) == len(l2p)`) is redundant — if two logicals share a
physical, `p2l` matches at most one, so assertion 1 already fires. The *unguarded*
mutations are the intra-core SWAP block (router.py:1104), `_fallback_local_swap`
(router.py:513) and both `_backup_plan` SWAP chains (router.py:796, router.py:889).
Put it there behind a debug flag. Two notes: the existing check costs O(n) but
`_snapshot` already copies both maps, so it is not worth removing for speed; and
because it runs post-commit and rolls back to entry state, it cannot distinguish
"I broke it" from "it arrived broken" — an entry assertion under the same flag
would.

## Teleport-commitment mechanisms (2026-08-03→04, not adopted — see TELEPORT_COMMITMENT.md)

Diagnosed that dSE re-scores every pending inter-core gate from scratch each
iteration with no memory of which one it was mid-move on (90.8% of gates
needing >=2 teleport hops get interleaved with another gate's teleport;
~half of all backslide events are the gate's own scored-best move backfiring
under local congestion, not a side effect of someone else). Four mitigations
prototyped and ablated on the full 64q suite, all as `HardwareConfig` fields
defaulting off (no published number affected):

| mechanism | gmean EPR | win/lose/tie |
|---|---|---|
| `commit_bonus=8` (soft, favor last-moved gate) | -4.3% | 4/4/1, driven by one outlier (`multiplier`) |
| `commit_hard_lock` (hard version of same idea) | -5.0% | 4/4/1, more extreme both ways |
| `cheapest_first_weight=0.3` + `evict_distance_aware` | **-4.2%** | **5/2/2**, best-balanced |
| `backup_relay_mode` (BFS-relay `_backup_plan`, see below) | -3.6% | 6/2/1 |

**None adopted** — comparable gmean to what's already in the paper's
mechanism ablation, and every one regresses at least 2 of 9 circuits.
`cheapest_first_weight` + `evict_distance_aware` is the best-balanced if
revisited. Full diagnosis, per-mechanism detail, the invariant-preserving
checkpoint-rollback procedure (`backup_relay_mode`) and its correctness
caveat (its precondition never actually holds on this suite — normal
routing already lets cores hit 0 free before deadlock recovery fires, via
the untouched `_force_make_room` path), and an unresolved optimal-EPR bound
question are all in `TELEPORT_COMMITMENT.md`.

## Corrected 2026-07-28: TeleSABRE does converge on heavy-hex-ring QNN

`results_heavyhex.json` (run 2026-07-26) recorded `ts: null` for qnn, and the
paper claimed TeleSABRE "fails to converge on QNN, which it completes on the
H-grid". **That is wrong.** Re-running the identical architecture (verified
bit-identical: same `inter_core_links`, same `Gr`) reproduces every dSABRE
number exactly and gets TeleSABRE **260 EPR / 8135 SWAP**, deterministically,
on all three seeds in ~11 s each. The 2026-07-26 null was an artefact, most
likely the 300 s per-seed timeout firing under load.

Consequences, all applied: ring gmean is over six circuits, not five, and the
margin is **-43.0%** (was -48.2% on the five-circuit basis); the abstract and
contributions no longer cite heavy-hex QNN as a TeleSABRE non-convergence.
Old file kept as `results_heavyhex_2026-07-26_TSqnn-nonconvergent.json.bak`.

The other non-convergence claims were re-probed and **hold**: 64q H-grid
Random and QAOA, and heavy-hex-star AE, QFT and Random, all return
`Success: false` on all three seeds well inside the timeout. Before citing any
future `ts: null` as a router failure, re-probe it directly — a timeout and a
genuine failure are indistinguishable in the results JSON.

## Open after the 2026-07-17 benchmark regeneration

- ~~360q large-circuit row not regenerated.~~ **DONE 2026-07-18**: `tab:large`
  360q regenerated on the per-core layout (node_decay off), replay-verified —
  dSABRE 626/13433 EPR/SWAP (was 579/27489; SWAP -51%), TS -43.6%, pytket
  e-bits 195 (PartitioningHeterogeneous, Steiner-detached exceeded the 25-min
  cap). `$^{\S}$` deferred footnote removed. `results_360q.json` synced.
- **Steiner-detached does not scale.** `pytket-dqc`'s CoverEmbeddingSteinerDetached
  exceeded a 25-min budget on 64q qnn, 200q, 360q and is embedding-incompatible on
  random (arbitrary-phase); those cells use PartitioningHeterogeneous (footnoted). If
  a stronger scalable pytket number is wanted, give Steiner-detached a larger budget.
- **Congestion-relief contribution is now marginal.** Post-layout-fix ablation shows
  removing relief *lowers* 64q gmean EPR by 8.7% (was +23.4% in the old paper). Prose
  now frames relief as per-instance (64q Random) rather than a gmean driver. Worth a
  deeper look at whether relief earns its place, or should be retuned/removed.

## Layout-policy exploration beyond adaptive corners (2026-07-17, negative result)

Three capacity-envelope alternatives to the adopted per-core adaptive corner rule
were benchmarked on 25/36/64q (dS + dSE, identical best-of-3 SabreLayout ->
fwd/bwd/fwd protocol, all rows replay-verified):

1. **spread** — even per-core budget ceil(n/(K*0.8)); at 36/64q its budgets equal
   adaptive's, so those suites isolate reservation SHAPE (most-remote blob vs the
   4 spread-out corners).
2. **pack** — fewest connected cores C = ceil(n/0.8m), unused cores fully
   reserved, smallest budget on the most CENTRAL used core.
3. **lal** — `locality_aware_layout` (community partition), 3 seeds.

Result (gmean EPR vs adaptive; worst circuit):

| | spread | pack | lal |
|---|---|---|---|
| 25q dSE / dS | 1.36 / 1.36 (ghz 3x) | 1.36 / 1.32 (ghz 3x) | 1.63 / 1.87 (ghz 9x) |
| 36q dSE / dS | 0.98 / 0.98 | 0.93 / 0.94 (wstate 1.5x) | 2.20 / 2.14 |
| 64q dSE / dS | 1.04 / 0.96 | 1.01 / 0.96 (tails 1.2-1.3x) | 1.53 / 1.42 |

**None adopted.** Takeaways: (a) reservation shape is noise at equal budgets —
the fill head-room (budget) is what matters, and the corner set is as good as
any; (b) packing fewer cores helps mid-fill gmean (36q 0.93) and chain circuits
(ghz 64q: 6 vs 8 EPR) but blows up small sparse circuits at low fill (ghz 25q
3x) and carries 1.2-1.5x tails — unsafe as a default; (c) the interaction-graph
community layout is dominated everywhere. If revisited: a per-circuit policy
chooser (e.g. pack only when the interaction graph is chain-like) is the open
direction, not a new global rule.

## ~~Fix the corner-removal initial layout~~ (found 2026-07-15, RESOLVED 2026-07-17)

Fixed by porting CPHM's validated `sabre_adaptive_corner_layout` (Phase 13e,
`../cphm/code/layout_corners.py`) into `layout.py::sabre_locked_boundary_layout`
(same name/signature, callers unchanged): per-core fill-adaptive reservation of the
k most-remote corners of EVERY core, k = largest in 0..4 keeping usable-slot fill
<= 0.80 (k=4 at 25/36/100/200/360q, k=2 at 64q). Port verified AST-identical to the
CPHM implementation and producing identical corner sets on all six suite
architectures. Before/after on the frozen dqcbench circuit copies, identical
protocol and current router code (gmean EPR new/old): 25q 1.016 dS / 1.000 dSE;
36q 0.882 / 0.968; 64q 0.835 / 0.862 — matching CPHM's measurements (1.009 /
0.952 / 0.805 on its base router). All rows replay-verified.

**Headline tables in `code/results/` and the paper still carry old-layout
numbers**; regenerating them (and re-running 100q/200q/360q, where CPHM measured
0.736 / 0.546 ratios) remains open — see the original analysis below.

### Original analysis (kept for the re-run decision)

`layout.py::sabre_locked_boundary_layout` implements a **monolithic-chip** rule — "remove
the four corner nodes" via `sorted(min-degree nodes)[:4]`. On a multi-core architecture
with core-major qubit numbering, the four lowest-id minimum-degree nodes are always the
four corners of **core 0** (verified: nodes 0/3/12/15 on both the B-grid 2x2 m=4 and the
H-grid 2x3 m=4), so cores 1..N-1 get **zero** reserved communication/escape slots and
regularly come out fully packed.

The fix is per-core, fill-adaptive reservation: reserve the k most-remote corners of
EVERY core, with k chosen by head-room (largest k such that usable-slot fill
`nq / (ncores * (m^2 - k))` stays below ~0.80). Measured on the dSABRE base router
(CPHM off, identical best-of-3 SabreLayout seeds -> fwd/bwd/fwd protocol), EPR ratio vs
the current stock layout:

| reserved/core | 64q (6 cores, m=4) | 200q (12 cores, m=5) |
|---|---|---|
| 2 (SFC's `sabre_two_corner_layout`) | 0.805 (worst 0.997) | 0.673 (worst 0.965) |
| 3 | 0.717 (worst 1.069) | — |
| 4 (the stock rule properly generalized per core) | 0.767 (worst 1.005) | **0.546 (worst 0.911** — better on every circuit) |

The effect **grows with core count** (more cores with zero reserved slots under the stock
rule) and is fill-limited at m=4 (at 64q, 4/core pushes usable-slot fill to 89% and the
dense circuits ae/qft regress past 2/core; at 200q's m=5 cores, 4/core keeps fill at 79%
and strictly dominates). Neutral at low fill (25q 1.004), slightly noisy-harmful at 36q
(1.083, small-EPR circuits).

Sources (in `../cphm/`): `PROGRESS.md` Phase 13c/13d; `code/layout_corners.py`
(2-per-core port from SFC), `code/bench_corner_count.py` (dose-response),
`code/results/replace_layout_{25,36,64}q.json`, `scale_layout_{100,200}q.json`,
`corner_count_{64,200}q_pc*.json`. Original diagnosis: `SFC/compare_routers.py`
("dSE own-protocol initial mapping" comment block).

Implication: every dSABRE headline table was produced under the core-0-only layout; a
per-core reservation rescales the baseline by -20% (64q) to -45% (200q) EPR before any
router change. Adopting it means re-running the benchmark tables.

## ~~Teleport destination-port-occupied bug~~ (found 2026-07-17, RESOLVED 2026-07-17)

Root cause turned out to be neither a port bug nor teleport-specific: `route()`'s
deadlock recovery restored `l2p`/`p2l`/`wdag`/`node_decay` from the checkpoint but did
NOT roll back `metrics["trace"]` or the op counters (`ls`/`teles`/`eprs`/`cost`/
`1q_gates`). Ops emitted during the abandoned no-progress iterations (up to
`deadlock_limit` per episode) stayed in the trace and counters, so (a) the emitted op
stream was non-replayable past the first restore — the "destination not free" TELE
failures were replay-state divergence, not real port violations — and (b) reported
EPR/SWAP were inflated by the discarded search work (wstate_36 on B-grid m=4:
614 → 14 EPR; qaoa_36: 479 → 179, exactly one 300-iteration episode of oscillating
teleports that net to identity and therefore even replayed "ok" while inflating counts).

Fixed by checkpointing/restoring the trace length and counters in lockstep with the
layout (see commit "Roll back trace and op counters on deadlock recovery"). Routing
decisions are unchanged. **All 120 committed benchmark result rows have
`backup_activations == 0`, so no published table was affected** — the inflation only
manifests in deadlock-prone (very tight capacity) scenarios.
