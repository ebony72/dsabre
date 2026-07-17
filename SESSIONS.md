# Session log — dSABRE

Running digest of Claude Code sessions in this project: **topic / achievement / open**,
oldest first. dSABRE is a SABRE-style router for multi-core **distributed** quantum
computers; this repo is the paper + code. **Project status: complete and frozen** (the
folder is marked "do not change unless informed").

> Summaries are derived from each session's first user message + final assistant message.
> Maintained by the weekly `session-digest` routine (see end). To refresh: ask Claude to
> "update SESSIONS.md".

_Last updated: 2026-06-02._

---

## 2026-04-28→05-02 — codebase review & continuation
- **Review dsabre codebase / continue work** — onboarding passes over the dSABRE code; mostly context-recovery (one ended at usage limit). Established where the router and paper live.

## 2026-05-03→05-05 — burst + commutativity, DQC survey
- **Plan gate-commutativity + burst communication in the router (05-03).** Diagnosed that `burst_ext_router.py` only re-bins the extended set (no EPR sharing / no commutativity), and `burst_router.py` uses ECH only as a teleport-scoring bonus. Produced a plan to genuinely exploit both. _Open then:_ ECH-for-DQC attempts keep underperforming.
- **Verify new `router_test.py` (05-03→04).** Blocker found: **ae +30.8% regression** before `router.py` can be replaced. _Open:_ is it the `_extended_2q` round-robin or `_get_local_extended` BFS optimism?
- **Test `_extended_2q` options B & C (05-05).** Burst-commit (post-hoc harvest) and Boolean burst tiebreaker variants evaluated; PDF rebuilt with updated 64q random row (dSABRE: 781 EPR / 4023 SWAP).
- **DQC compilation survey (05-05, ×2).** Plan + per-paper summary of distributed-QC compilation works (telegate/telestate, mapping/routing/scheduling, partition vs heuristic, architecture, open-source?). Two runs blocked on the 1M-context usage gate.

## 2026-05-06→05-10 — parallel intra-core routing, agent housekeeping
- **Optimize router parallelization for `front_intra` cores (05-07→09).** Route local gates per-core in parallel when several cores have intra 2q gates in the front. **Wrote the project `CLAUDE.md`** capturing the hard-won gotchas (`burst_router` `g.qubits[i]` vs `router` `n.qargs[i]`, `locality_aware_layout(rng=Random(seed))`, sys.path priority, TeleSABRE `ls=0` parsing bug).
- **Claude housekeeping (05-06, 05-09, 05-09→10).** Explained `~/.claude/projects/` layout & worktrees; created root + `paper/`-rooted `CLAUDE.md`; trimmed a dSABRE-Topo table column.

## 2026-05-10→05-13 — paper revision round 1, baselines
- **Extended-set variants / Algorithm 2 (05-10→11).** Decided **not** to expand `Q_F` per node; removed `dSEX`, kept `dSE` (fixed-Q_F BFS) as Alg. 2.
- **Compare with pytket-dqc (05-11).** Added Appendix B "Large-Circuit Scalability": 100q (−24.9% EPR vs TeleSABRE), 200q (TS times out, dSABRE completes), 360q (+26.1%, QPEexact regression).
- **dSABRE vs SABRE framing (05-11).** Rewrote SABRE subsection to close with bidirectional sweep; dSABRE section opens by contrasting departures.
- **Ablation / parameter sensitivity (05-11).** `cost_teleport` etc. sweeps; rewrote layout-passes mechanism (cold-start vs steady-state L1≈L2) and initial-mapping validity prose.
- **Reorg + equivalence checking (05-11, 05-12→13).** How to check output≡input for a circuit with teleportations; pushed a 48-file reorg (sources → `paper/code/` then `code/`).
- **Worktree vs direct-edit Q&A (05-11).** Clarified: avoid worktree selection when changes should land directly in `paper/`; worktrees are for branch-isolated work that needs a merge/PR.

## 2026-05-12→05-15 — consistency & polish
- **Verify 100q+ experiments use SL+sabre layout & consistent locality-aware partition (05-12→13).** Re-ran where needed; updated 25/36/64q tables (e.g. 64q gmean(6) −11.5%→−15.7%).
- **pytket-dqc deep dive (05-13).** Documented KaHyPar connectivity-metric partitioning + post-partition bundle extraction, vs dSABRE's bundles.
- **Alg. 1 "remove 1Q gates" → "execute" (05-12→13); Sec III/IV teleportation-vs-layout contradiction (05-13→14); abstract too long (05-14); remove node-decay discussion + trim Sec III-D/IV-C (05-15).** A sequence of correctness/clarity edits; abstract & conclusion tightened; overfull boxes fixed; stable at 14–15 pages.

## 2026-05-17→05-19 — related work (DMapS, TeleSABRE, FiDLS), link density
- **DMapS comparison (05-17→18).** Added related-work paragraph contrasting DMapS-R (swap-only cross-core primitive) with dSABRE; citation resolves.
- **Coworker vs Code for manuscript polish (05-18).** Guidance: Code when revision is coupled to benchmarks/tables; chat for pure prose.
- **TeleSABRE limitations (05-18).** Promoted congestion-control + rollback to a first-class contribution; folded BFS-extended into the heuristic contribution.
- **Fix dSABRE description: intra vs inter handled separately (05-18→19).** Corrected the "single loop" wording — dSABRE evaluates teleports only when `F_intra` is empty. Regenerated shared-KaHyPar tables (gmean −48/−17/−35%).
- **FiDLS review (05-15→19); enhanced B-grid with +4 quantum links (05-13→19).** Benchmarked the three compilers on the denser architecture; appendices split into `appendices.tex`.
- **Revise manuscript from docx suggestions (05-19).** Applied reviewer-style revision suggestions from a shared `.docx`; pushed `95e9a06`; Drive zips refreshed.

## 2026-05-19→05-21 — verification, figures, submission prep
- **Systematic experiment re-verification (05-19→20).** Re-ran all suites into `code/results_verify/`; updated only out-of-date appendix numbers, paper untouched.
- **Abstract/intro figure audit (05-19→20).** Fixed the "+81% if capacity removed" line and a "widens vs shifts" contradiction; dSABRE wins under both reporting choices.
- **Running-example ΔF + staging-cost double-count (05-20→21).** Checked code for `d_intra(p1,n_s)` overlap with `d(p1,p2)`; built the **arXiv submission bundle** (`arxiv_dsabre.tar.gz`, fields pre-filled).
- **Figure/table consistency double-check (05-20→21).** Found a striking `hop_gain` ablation result: **OFF beats ON by −76% EPR on QPEexact** with huge seed variance.

## 2026-05-21→05-22 — wrap-up & best practices
- **Best practices for finished sessions (05-21).** Wrote the **global `~/.claude/CLAUDE.md`** (durable patterns), removed all 11 worktrees, retired a stale memory.
- **Collect all table data → Excel/decks (05-20→21); arXiv/Drive zips (05-18→19).** Survey decks generated from `PHASE3_SYNTHESIS.md`.
- **Freeze the folder (05-22).** Added the "do not change unless informed" restriction to `CLAUDE.md`. Project handed off; SN-SABRE forks from here.

---

## Closing state
- **Paper:** complete, arXiv bundle built; headline ≈ −41% / −44% EPR vs TeleSABRE at 25/64q, 200q robustness advantage, 360q QPEexact regression acknowledged.
- **Unresolved when frozen:** the `router.py` replacement was still blocked by the ae regression; the `hop_gain` ablation showed layout-sensitivity worth more study — both carried forward conceptually into SN-SABRE rather than resolved here.

---

## Maintenance
Regenerated by the **`session-digest`** routine: `python3 ~/.claude/tools/session_digest_extract.py`
produces `/tmp/session_digest.json`; the routine folds new/changed sessions for this project
into the log. This project is frozen, so new sessions are unlikely. To trigger manually, ask
Claude to "update SESSIONS.md".
