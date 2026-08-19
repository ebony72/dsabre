# dSABRE — Paper Directory

**IMPORTANT: Do not modify any files in this folder unless explicitly instructed by the user.**

This directory is the primary working directory for the dSABRE project.
All paths below are relative to `paper/` unless noted.

See also: `../CLAUDE.md` for the full project reference.

---

## Revision Workflow

**Do not work on the response letter before the revision itself is finished.**

Land the manuscript changes the review report asks for first — `dsabre.tex`,
`appendices.tex`, the generated tables, and the rebuilt PDF — and only then
write the point-by-point response to the reviewers. A response drafted ahead
of the revision cites text the paper does not yet contain, and has to be
rewritten once the edits actually land.

### Page budget: the main paper must stay at 12 pages

The submitted PDF is 18 pages. As of 2026-08-18 `dsabre.tex` builds to 12, and
**12 is now the target, not a floor to spend down** — an edit that pushes the
build to 13 has to be paid for by a cut of comparable length. Verify with
`pdfinfo dsabre.pdf | grep Pages` after any prose edit; the working tree's
`dsabre.pdf` is often a stale rebuild, so compile before trusting it.

Everything cut from `dsabre.tex` to reach the
limit goes into `appendices.tex` (the online appendices) — **nothing is
deleted outright**. Move the text, keep the numbers, and leave a one-sentence
pointer in the main paper (`\rev{... reported in the online appendices}`) so a
reviewer can still find the evidence they asked for. Cutting a result the
review report requested and not relocating it reads as a non-response.

When moving a block: cut prose + table together, renumber nothing by hand
(labels move with the block), and check that `\ref{}`s from the main text to
the moved label still resolve — `appendices.tex` is compiled both standalone
(`appendices_standalone.tex`) and, for cross-references, against `dsabre.aux`.

### Keep the abstract concise and short

Target **~200 words**. The abstract carries the motivation, the one-sentence
core idea, and the headline result against `\TeleSABRE{}` — nothing else.

It is not a summary of the contributions list. Baseline-by-baseline results
(`pytket-dqc`), secondary architectures (the heavy-hex ring and star), and
mechanism-by-mechanism accounting belong in the introduction's contribution
bullets and in Section IV, not here. Reviewer 3 asked for an abstract that
conveys the core idea; length works against that, so when a revision adds a
claim to the abstract, cut another one out rather than appending.

---

## File Structure

| File | Description |
|------|-------------|
| `router.py` | Self-contained `General_dSABRE_Router` — imports from {config,architecture,actions}.py |
| `dsabre_ext.py` | `dSABRE_BFSExt` — BFS extended-set variant (called "dSE" in benchmarks, "\dSABRE{}" in paper) |
| `_baseline_*.py` | Frozen pre-optimisation router/ext/architecture — reference for the verifiers, do not edit |
| `verify_router.py` | Diffs the router, pinned to score-only mode, against the baseline over all four suites |
| `verify_architecture.py` | Diffs the composed `phys_dist` against a dense all-pairs table |
| `config.py` | `HardwareConfig` — FEWER params than main repo (no `max_burst_walk_depth`) |
| `architecture.py` | `build_h_grid_architecture`, `DistributedArchitecture` |
| `actions.py` | `TeleportAction` and related primitives |
| `layout.py` | `locality_aware_layout(dag, arch, rng)`, `run_passes(router, dag, layout, n)` |
| `benchmark.py` | Full benchmark suite runner (B-grid 25q/36q, H-grid 64q) |
| `gen_tables.py` | Emits LaTeX table rows from results JSONs in `results/` |
| `dsabre.tex` | Main paper LaTeX source |
| `dsabre.bib` | Bibliography |
| `results/` | Benchmark results JSONs — named `results_{suite}.json` |

---

## Naming Conventions (Code vs Paper)

| Code name | Paper name | Description |
|-----------|-----------|-------------|
| `dS` / `General_dSABRE_Router` | `\dSABRE{}`-Topo | Topological-order extended set (`_top_ext`) |
| `dSE` / `dSABRE_BFSExt` | `\dSABRE{}` | BFS-layer extended set (`_bfs_ext`), production router |
| `TS` / `TeleSABRE` | `\TeleSABRE{}` | Reference C++ router |

**Extended-set method names (renamed 2026-08-07).** `_inter_ext` is the hook
`route()` calls; a subclass picks a construction by overriding it.
`_top_ext` (router.py) fills the inter-core set in topological order,
`_bfs_ext` (dsabre_ext.py) by BFS layer. `_get_local_extended` builds the
per-core *intra* sets and is a different axis — don't confuse it with the
two above. The old names were `_extended_2q` (both) and `dSABRE_BurstExt`;
`dSABRE_BurstExt` survives as an alias, since ~30 driver scripts import it.

**`dSE` is the default dSABRE router.** Every headline number, every
architecture comparison and every scalability row is `dSE`. Run `dS` *only*
where the two extended-set constructions are being compared — the
"Topological extended set (not BFS)" row of the mechanism ablation
(Table VI, row 3), which is computed from `benchmark.py`'s `dS` column over
the six-circuit ablation subset. A benchmark driver that emits a `dS` column
everywhere is doing double the work for one table cell; new drivers should
default to `dSE` alone.

---

## Key Invariants & Gotchas

### The default router is the optimised one (2026-08-07)

`router.py`, `dsabre_ext.py` and `architecture.py` carry the incremental
implementation: maintained in-degrees and gate counter, log-based
checkpoints, incremental `Δ_F`/`Δ_E` in `_best_intra_swap`, an exact early
exit in `_get_local_extended`, and a composed (not materialised) `phys_dist`.
Output is **bit-identical** to the pre-optimisation code — verified over 207
routing passes across the 25/36/64/360-qubit suites — at 3.2–5.6× less
compile time. ("Bit-identical" is under the mode the baseline implements,
score-only (`safe_mode=False`) — see the next bullet.)

- The pre-optimisation code is frozen as `_baseline_router.py`,
  `_baseline_dsabre_ext.py`, `_baseline_architecture.py`. **Do not edit them**;
  they exist so `verify_router.py` / `verify_architecture.py` can diff against
  the implementation that produced every published number. They predate
  `safe_mode` entirely — frozen 2026-08-07, before safe mode existed at all —
  and contain zero occurrences of it; they only ever ran what is now called
  score-only mode.
- **Re-run both verifiers after any router or architecture change — but know
  what a pass actually proves.** `verify_router.py` pins both routers to
  `local_ext_mode="taint", safe_mode=False`, the mode the frozen baseline
  implements, regardless of `HardwareConfig`'s current defaults. A `PASS`
  proves the **incremental rewrite** (maintained in-degrees, log-based
  checkpoints, composed `phys_dist`, etc.) is output-identical to the
  baseline in that mode — it does **not** exercise `safe_mode=True` (the
  shipped default since 2026-08-09) or the shared extended set (the default
  since 2026-08-08); both are deliberate behavioural changes from the
  baseline and each has its own check instead: `verify_shared_ext.py` for the
  shared set, `test_safe_drain.py` + `SAFE_DSABRE.md` §10 for safe mode. The
  verifier compares metrics, final layout, failure log and (on 25q) the full
  SWAP / teleport trace.
- **Do not read a bare `python3 verify_router.py` run as a verdict on the
  router as shipped.** It was silently comparing the wrong things between
  2026-08-09 (safe mode became the default) and the fix that pinned
  `safe_mode=False` explicitly — every safe-mode-only mechanism read as a
  547-diff `FAILED` on a clean tree (giveaway: `relay_hops: base=0 fast=1`).
  If a router change should be checked against the safe-mode default itself,
  that requires the safe-mode-specific coverage above, not this script.
- The router now carries per-route state, so one instance is safe to reuse
  sequentially (as `run_sabre_passes` does) but **not** across threads.
- Incremental in-degrees are correct only because gates leave the DAG solely
  from the front layer. Retiring a non-front node would desync them silently.
- `phys_dist` composition assumes unit-weight intra-core edges; a non-unit
  architecture falls back to the dense table automatically.

### Router API
- `router.py`'s `route()` takes a **DAGCircuit**, NOT a QuantumCircuit
  - Convert: `from qiskit.converters import circuit_to_dag; dag = circuit_to_dag(qc)`
- `locality_aware_layout(dag, arch, rng=...)` — `rng` must be a `random.Random` instance, NOT an int
  - Correct: `locality_aware_layout(dag, arch, rng=random.Random(seed))`
  - Wrong: `locality_aware_layout(dag, arch, rng=seed)` → AttributeError on `.shuffle`
- `run_passes(router, dag, layout, layout_passes)` is in `layout.py`, NOT router.py
- `p2v_to_layout(p2v, dag)` is defined inline in `benchmark.py`, NOT in layout.py

### HardwareConfig
- This directory's `config.py` does NOT have `max_burst_walk_depth` (main repo does)
- When benchmarking here, omit `max_burst_walk_depth` from HardwareConfig calls

### Import Paths
- Scripts here should use: `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`
- If also importing from the main repo root, insert `paper/` path LAST (higher priority)

### gen_tables.py
- Must use `_HERE = os.path.dirname(os.path.abspath(__file__))` for paths
- Hardcoded absolute paths will read from the wrong location

### Background Benchmark Scripts
- Always use `flush=True` on ALL print statements
- Save to JSON early and often; don't rely on stdout alone for results

### DO NOT regenerate benchmark circuits with the installed MQT Bench

`~/Documents/telesabre/circuits/` is **shared** by dsabre, cphm, SFC, dqcbench
and dqc-fusion. The paper's circuits are **MQT Bench v1.1.0** downloads (2024).
The installed generator is **v2.2.2**, and for the same benchmark name it emits
a *structurally different* circuit. Regenerating in place silently changes
published numbers for every project at once.

**`qnn_64` must have 8126 CX gates** (depth 650). It was broken between
2026-07-09 and 2026-07-27 and has since been **restored**:

| version | qnn @ 64 qubits |
|---|---|
| v1.1.0 (paper, Table VI) | **8126 CX**, depth 650 |
| v2.2.2 (installed) | **63 CX** — a linear chain, `reps=1` |

The difference is semantic, not cosmetic. v1.1.0 builds `qnn` from a
**ZZFeatureMap**, whose data encoding entangles *every* qubit pair; v2.2.2 uses
a **ZFeatureMap**, a product state, leaving only the ansatz's linear CX chain.
The v1.1.0 construction is

    ZZFeatureMap(n, reps=2) o RealAmplitudes(n, reps=1)

transpiled to `{rz,sx,x,cx}` at opt3 — verified to reproduce 1223 CX at n=25
and 8126 at n=64 exactly.

On 2026-07-09 a `cphm` session judged the deep circuit "a stale legacy file
structurally inconsistent with qnn_100/qnn_200" and overwrote it with the
v2.2.2 output, labelling the shallow one "CORRECTED" (`cphm/code/fix_qnn_64.py`).
It was not corrected — 8126 CX is the count in the submitted manuscript.
An SFC session flagged the regression on 2026-07-10. **Restored 2026-07-27**
from the `.bak`, preserving the original v1.1.0 header; the shallow file is kept
as `qnn_SHALLOW_v2.2.2_...qasm.bak` in case another project depends on it.

**Provenance check** — every legitimate suite circuit begins with

```
// Benchmark was created by MQT Bench on 2024-03-..
// MQT Bench version: 1.1.0
```

A suite `.qasm` with no header, or a different version, has been regenerated
and cannot be compared against published numbers. The current `qnn_64` has no
header and retains `creg meas[64]`, which the suite convention strips.

The whole `qnn` family is now the deep ZZ form and consistent across sizes —
1223 CX at 25q, 8126 at 64q, 19,898 at 100q, 79,798 at 200q, each entangling
every qubit pair. `qnn_100`/`qnn_200` were switched from the shallow v2.2.2
form on 2026-07-27 (`code/gen_deep_qnn.py`); their previous contents are kept
as `qnn_SHALLOW_v2.2.2_..._{100,200}.qasm.bak`. These are quadratic in n and
dwarf every other circuit in the suites (multiplier_64 is 13,040 CX), so budget
routing time accordingly and expect pytket-dqc's embedding distributors to fall
back on them.

**If you need a circuit that does not exist yet**, generate it under a *new*
name (as `code/gen_new_64q_circuits.py` and `code/gen_deep_qnn.py` do) and
never overwrite an existing one. Benchmark drivers should preflight CX counts
against the published table and abort on mismatch — see `EXPECTED_CX` in
`code/bench_heavyhex.py` and `code/bench_pytket_fair.py`, which read the
`.bak` explicitly rather than trusting the directory copy.

---

## TeleSABRE Setup

**Binary**: `~/Documents/telesabre/TeleSABRE`
**CLI format**: `<binary> <config.json> <device.json> <circuit.qasm>`
  - NOT: `--circuit`, `--device`, `--seed` flags

**Device files**: `~/Documents/telesabre/devices/`
  - `B_grid_2_2_4_4.json` — for 25q/36q suites
  - `H_grid_2_3_4_4.json` — for 64q suite
  - `H_grid_2_3_5_5.json` — for 100q circuits (150 physical)
  - `H_grid_4_3_5_5.json` — for 200q circuits (300 physical)
  - `H_grid_2_3_8_8.json` — for 360q circuits (384 physical)
  - `H_grid_2_3_9_9.json` — for 360q circuits (486 physical, preferred)

**Circuit files**: `~/Documents/telesabre/circuits/qasm_{n}/`

**Parsing TeleSABRE output** — extract ALL three metrics:
```python
td = tg = ls_ts = 0; ok = False
for line in proc.stdout.splitlines():
    l = line.strip()
    if   "Teledata:" in l: td    = int(l.split(":")[1])
    elif "Telegate:" in l: tg    = int(l.split(":")[1])
    elif "Swaps:"    in l: ls_ts = int(l.split(":")[1])   # ← NEVER omit this
    elif "Success: true" in l: ok = True
# Also read from JSON report for accuracy:
# rep["iterations"][-1]["swap_count"]
```
**CRITICAL**: `ls=0` for TeleSABRE is a parsing bug. TeleSABRE uses thousands of local SWAPs.
Always parse the "Swaps:" line. The EPR count = teledata + telegate.

**Config template** (use `save_report: True` to get initial layout p2v):
```python
{"config": {
    "name": "...", "seed": seed,
    "energy_type": "extended-set",
    "usage_penalties_reset_interval": 5,
    "optimize_initial": True, "initial_layout_type": "hungarian",
    "teleport_bonus": 100, "telegate_bonus": 100, "safety_valve_iters": 100,
    "extended_set_size": 20, "extended_set_factor": 0.05,
    "inter_core_edge_weight": 2, "full_core_penalty": 10,
    "max_solving_deadlock_iterations": 1000,
    "gate_usage_penalty": 0.0, "swap_usage_penalty": 0.002,
    "teledata_usage_penaly": 0.005, "telegate_usage_penalty": 0.005,
    "init_layout_hun_min_free_gate": 5, "init_layout_hun_min_free_qubit": 4,
    "enable_passing_core_emptying_teleport_possibility": False,
    "max_iterations": 200000,
    "save_report": True, "report_filename": rpt_path,
    "required_successes": 1, "max_attempts": 10,
}}
```

---

## Common Commands

```bash
# Run benchmark suite (from paper/)
python3 benchmark.py --suite 64q

# Regenerate LaTeX tables from results JSONs
python3 gen_tables.py

# Compile paper
pdflatex dsabre.tex && bibtex dsabre && pdflatex dsabre.tex && pdflatex dsabre.tex
```

---

## Paper Sync Rules

`router.py` here must stay in sync with `../router.py` (main repo).
Changes that MUST be kept in sync:
1. `_get_local_extended` — multi-core single-pass lookahead
2. `_fallback_local_swap` — single call with all core_ids
3. `front_intra` block — parallel per-core SWAP selection

---

## Compact Instructions

When compacting this session, ALWAYS preserve:
1. The "Key Invariants & Gotchas" section verbatim
2. The "TeleSABRE Setup" section (especially the parsing template and CRITICAL note)
3. The "Paper Sync Rules" section
4. The "Naming Conventions" table
5. Current task status from PROGRESS.md (re-read it after compaction)
