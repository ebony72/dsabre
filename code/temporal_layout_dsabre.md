# Temporal locality-greedy initial layout for dSABRE

Standalone drop-in initial-layout heuristic for dSABRE, ported from PORTER
(SFC). **Not wired into `benchmark.py`** — it is a self-contained module
plus this note. Core function has **no SFC dependency**; only the optional
`__main__` self-check imports SFC to cross-validate.

- Module: `temporal_layout_dsabre.py`
- Public API: `temporal_locality_layout(dag, arch, min_free_per_core=2, gamma=0.99) -> {Qubit: phys_int}`
- Same return shape as `layout.locality_aware_layout` / `sabre_locked_boundary_layout`

---

## What it does

A single deterministic, seed-free pass (~1 ms), three steps:

1. **Temporal weighting.** Each 2-qubit gate contributes `gamma**layer`
   (γ=0.99) to its pair's weight, where `layer` is the gate's ASAP level in
   the 2q-only DAG. Early interactions weigh more; repeated interactions on
   a pair accumulate.
2. **Core assignment.** Qubits are visited in first-2q-gate-appearance
   order. Each is placed in the core holding its heaviest already-placed
   neighbour if that core still has room above `min_free_per_core`, else in
   the core with the most remaining reserved capacity. Every core keeps
   `min_free_per_core` free slots by construction (dSABRE's teleport
   precondition holds from iteration 0 — no `repair_full_cores` needed).
3. **Physical placement.** Within each core, logicals sorted by descending
   total weight take physical slots sorted by ascending intra-core
   centrality (sum of local distances) — most-connected qubit → most
   central slot.

## How it differs from dSABRE's stock layout

| | dSABRE `sabre_locked_boundary_layout` | this module |
|---|---|---|
| Engine | Qiskit SabreLayout (stochastic, routing-based) | deterministic greedy, seed-free |
| Inter-core awareness | none — inter-core links are ordinary graph edges | explicit core-assignment step |
| Temporal structure | implicit via routing simulation | explicit γ-discount on gate ASAP layer |
| Headroom | k specific corner *positions* removed per core | `min_free_per_core` free *count* per core |
| Cost | ~30–100 ms + 3× full routing (best-of-3 seeds) | ~1 ms, no routing search |

The stock layout minimizes 2-qubit **swap distance** on the flat chip graph
and is blind to core boundaries, so it will place a tightly-coupled pair
across a costly inter-core link. This heuristic optimizes co-location by
core directly.

## Measured impact (dSE = `dSABRE_BurstExt`, EPR geometric mean)

Router held fixed; only the initial layout varies. `own` = dSABRE's own
best-of-3 SabreLayout; `PORTER f=2` = one shot of this heuristic; `union` =
best of both candidate sets.

| Suite | dSABRE own (×3) | this, f=2 (1 shot) | best-of-union |
|---|---|---|---|
| 25q | 49.1 | 45.8 | **44.4** (−10%) |
| 64q | 127.2 | 120.1 | **109.9** (−14%) |
| 80q | 126.5 | 144.7 ⚠️ | **118.5** (−6%) |

Per-circuit highlights (64q, EPR): ae 264→202, qft 259→211, ghz 9→6.

**Root cause** — inter-core cut (CX pairs crossing cores), the quantity
that drives EPR count:

| Circuit | dSABRE SabreLayout candidates | this, f=2 |
|---|---|---|
| ae_64 | 1334 / 1366 / 1496 | **1178** |
| qft_64 | 1439 / 1481 / 1498 | **1170** |

## Recommendation: add, do not replace

- **The union always wins** — every suite improves when both candidate sets
  are pooled and the router picks the best.
- **The single shot is not universally better** — 80q qnn regresses badly
  (1509 vs 842), and 25q ae/qft lose. A deterministic layout with no
  routing feedback can mispredict on some dense circuits; SabreLayout's
  stochastic best-of-3 catches those cases.
- So: append this as a **4th candidate** to the existing best-of portfolio
  (it's ~free), rather than swapping SabreLayout out.
- **Sweep `min_free_per_core` ∈ {2, 3}** when adding it: f=3 won ae_64
  (202 vs 219) and ghz_64 (6 vs 8), trading one extra free slot per core for
  relocation headroom. (f=4 was generally worse in this run.)

## Integration sketch (when/if you decide to wire it in)

In the dSE best-of loop (e.g. `benchmark.py` around the
`sabre_locked_boundary_layout(...)` call), extend the candidate list:

```python
from temporal_layout_dsabre import temporal_locality_layout
cands = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
for mf in (2, 3):
    try:
        cands.append(temporal_locality_layout(dag, arch, min_free_per_core=mf))
    except ValueError:
        pass  # circuit too dense to reserve mf free slots per core
# route every candidate, keep best EPR (existing logic)
```

## Verification

`python3 temporal_layout_dsabre.py` cross-checks the native port against
PORTER's own `compile_porter.build_initial_layout` — **exact match on all
18 benchmark circuits** (25q/64q/80q × ae/ghz/graphstate/qft/qnn/random).
Requires SFC on the path for the self-check only.

## Provenance / caveats

- Experiment harness: `porter_layout_into_dse.py` (feeds this layout into
  dSE across all three suites); summary: `summarize.py`.
- 80q `random` aborts under dSE for **both** layouts — a genuine dSE abort
  (matches `SFC/paper/compare_results.json`), not a layout effect.
- Ported from SFC `compile_porter.build_initial_layout` @ 2026-07-12
  version (label `porter.temporal_locality`).
