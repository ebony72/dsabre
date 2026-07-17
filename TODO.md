# TODO

## Fix the corner-removal initial layout (found 2026-07-15, CPHM Phase 13)

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

## Teleport destination-port-occupied bug (found 2026-07-17, stress test)

Separate from the source-port bug fixed the same day in `code/router.py`
(`_apply_teleport` staging-path + full-source-core eviction, see the commit "Fix
teleport-legality bugs: staging path and full-source-core eviction"): `verify.py`'s I5
check can also fail on the **destination** side — `TELE: destination {p} not free
(holds {virt})` — meaning `_apply_teleport`'s `_evict(a.p_comm_dst, a.next_core, ...)`
sometimes doesn't actually clear the destination port before the write.

**Confirmed pre-existing**: reproduces identically on the unmodified pre-fix router
(commit `779c8c3`), so it predates and is independent of both fixes applied 2026-07-17.
Not caught by dqcbench's M1 pilot (qasm_25 / architecture A, 2x2 cores of 3x3) — only
surfaced under a deliberately tight stress scenario.

**Repro**: `build_b_grid_architecture(r=2, s=2, m=4)` (64 phys, 4 cores of 16),
`HardwareConfig(deadlock_limit=300, max_backup_attempts=300, max_iterations=20000,
trace_routing=True)`, `sabre_locked_boundary_layout(seed=0)`, a single
`router.route(dag, layout)` pass (both `General_dSABRE_Router` and
`dSABRE_BurstExt`) on `wstate_nativegates_ibm_qiskit_opt3_36.qasm` from dqcbench's
`circuits/qasm_36/`. Fails at trace index ~620 with `_free_slots(core)==0` /
`force_make_room` count >= 1 at the point of failure in every observed instance —
suggests the same "core is completely full" family as the source-side bug, but via a
different call path not yet isolated (candidate `next_core` free-slot check in
`_generate_candidates` runs before selection; something between there and the actual
`_evict(p_comm_dst, ...)` — possibly `_backup_plan`/`_force_make_room` activity in a
prior iteration, or a same-iteration interaction not yet identified — leaves the port
occupied by the time `_apply_teleport` writes to it). Root cause not yet diagnosed.

Plausibly related to the corner-removal layout bug above: if per-core reservation
were fixed, cores would fill up less often, which may make this destination-side
race harder to trigger — but the underlying legality bug would still need its own
fix independent of that.
