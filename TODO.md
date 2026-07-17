# TODO

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
