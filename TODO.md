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
