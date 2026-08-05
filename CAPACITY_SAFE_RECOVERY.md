# Whole-gate transactional deadlock recovery (2026-08-05) — adopted

An external document, "the capacity-safe fallback design and code.md" (kept
alongside this file), proposed a capacity-invariant rewrite of the router:
enforce `free(core) >= 2` for every active core at every iteration boundary,
classify cores as active/restricted/dormant, and replace `_backup_plan` with
a "guaranteed" move-execute-return transaction. This file records what was
verified, what was rejected, and what was actually integrated into
`code/router.py`.

## Verdict on the source document

Partially rational, useful in one specific place. The core idea — recovery
should be an atomic, all-or-nothing transaction, not reactive room-making —
is sound and is what got adopted. The rest re-proposes machinery the router
already had (an atomic checked `_apply_teleport` already existed; see
`_apply_teleport.md`) or is actively worse:

- **The global `f(c) >= 2` boundary invariant was not adopted.** It would
  shrink the ordinary-routing action space that produces every published
  number (rejecting a legal `f=2` teleport unless a compound transaction is
  pre-validated) and adds an `nx.has_path` check per remote gate per
  iteration — expensive on `multiplier` (13,040 CX, tens of thousands of
  iterations). The free-slots ablation (`ablate_freeslots_*.json`, prior
  session) already found the real abort frontier is "~1 free/core + bulk
  slack ≈ 80% fill", with *central* cores wanting more slack than edge
  cores — a uniform per-core reserve is the wrong shape for that.
- **"Remove the 0.80 fill threshold" conflates two unrelated knobs.** That
  0.80 (`layout.py:adaptive_corner_count`) tunes the SabreLayout corner
  reservation count `k`, calibrated per suite by a dose-response study
  (k=3–4 often beats k=2). It is not the router's capacity mechanism.
- **Move-execute-*return* plans cost 2·d_C(c1,c2) teleports** to restore
  occupancy nothing needs once room is re-established live before each hop
  — the doc's own §4.3 flags this but the reference implementation's
  `_execute_teleport_transaction` hardcodes `forward_count = len(actions)//2`,
  so the "move and remain" optimisation it describes is dead code as given.
- The document's `_relay_room_to`-equivalent needs its own precondition
  (2 free per active core); the router's actual `_relay_room_to`
  (2026-08-03) already gets a stronger guarantee — *some* core has ≥2 free
  — from the weaker, cheaper invariant "total free ≥ num_cores + 1", which
  every teleport and SWAP conserves. No rewrite was needed there.

## What was extracted and integrated

`_backup_plan`'s cross-core branches (both the greedy and `backup_relay_mode`
variants) moved only the first operand, hop by hop, and `break`d out of the
loop on the first failed hop — leaving whatever partial move it had made
applied (not rolled back), and never trying the other operand even when that
direction had room.

`General_dSABRE_Router._route_gate_transaction` (new, `code/router.py`)
fixes both: one full attempt — relay room to each next core
(`_relay_room_to`), teleport the operand hop by hop, SWAP it onto the gate,
execute — wrapped in a single `_snapshot`/rollback. `_backup_plan` now tries
`mover=q1` then, if that fails, `mover=q2`, before falling through to the
existing greedy/relay code as a last resort. Ordinary routing (candidate
scoring, `_apply_teleport`) is untouched.

## Benchmark evidence

**64q suite**, dSE, fwd→bwd→fwd, best-of-3-SabreLayout (identical protocol,
before vs after): total EPR **4148 → 3808 (−8.2%)**. 8/9 circuits tied or
improved; qpeexact regressed 200→204 (+2%, the one loss). No circuit that
previously succeeded regressed into an abort.

| circuit | before | after |
|---|---|---|
| ghz, graphstate, random | unchanged | unchanged |
| qft | 208 | 192 |
| qpeexact | 200 | 204 |
| ae | 202 | 141 |
| qaoa | 542 | 517 |
| qnn | 501 | 497 |
| multiplier | 1761 | 1523 |

**random_80** (10-core H-grid, 23,381 CX) — the one archived instance
(`results_80q_dsabre.json`, 2026-07-28) where the *old* `_backup_plan`
aborted outright via `DEADLOCK_BACKUP_FAILED` on one of its three
SabreLayout candidates (the greedy hop had no legal move — not a budget
issue, a structural dead end). With the transactional recovery, none of the
3 layouts ever hits that failure mode; at the suite's standard
`max_iterations=20000` all 3 still run out of iteration budget but make
20–48% more progress than before in the same budget, and at
`max_iterations=80000` (an ordinary, already-precedented per-suite scaling —
see `HW_LARGE` in `benchmark.py`) all 3 complete cleanly with an empty
failure log.

## Not adopted

- Global capacity boundary invariant / active-restricted-dormant
  classification / per-iteration `nx.has_path` validation.
- Replacing the 0.80 layout fill threshold.
- Move-execute-return transactions (only move-execute-remain, i.e. the
  existing per-hop `_apply_teleport`, is used — no return leg).
- A separate `guaranteed_capacity_mode` config flag: the transaction is
  tried unconditionally, first, inside the existing `_backup_plan`, rather
  than gated behind a new toggle — there was no evidence a toggle was
  needed (no regression found on any suite tested).

## Bug found and fixed during full-suite regeneration (2026-08-05)

Regenerating the 200q suite (2x wider architecture than the 64q suite the
integration was originally validated on) crashed outright:
`KeyError` in `_idist` from inside `_route_gate_transaction`, on `qft`'s
`dS` router, pass 2 (bwd, reversed DAG).

Root cause: `_route_gate_transaction`'s loop computes `cur_core`/`next_core`
once per hop, then calls `_relay_room_to` to secure room at `next_core`
before teleporting `mover` there. `_relay_room_to`'s own hops are executed
via `_apply_teleport`, whose internal `_force_make_room` (invoked if a relay
hop's own source/destination core is itself full) only protects the filler
qubit *that specific call* is relocating (`exclude_virt`) — it has no idea
`mover` needs protecting too. So a relay hop can incidentally displace
`mover` to a different core as a side effect, and the code went on to use
the *pre-relay* `cur_core`/`next_core` regardless — treating a now-foreign
physical qubit as a member of a core it had just left, hence the `KeyError`
rather than a graceful failure.

This is the exact class of problem `apply_teleport_conflicts.docx` (§9,
"candidate data become stale during room-making") flagged for `_apply_teleport`
itself, showing up again one layer up, in code that didn't exist when that
doc was written. Fix: re-derive `cur_core`/`next_core` from `mover`'s actual
position immediately after `_relay_room_to` returns, instead of trusting the
pre-relay values. Worst case on a mismatch is now a clean transactional
rollback (falls through to the existing greedy `_backup_plan` fallback), not
a crash. 25q/36q/64q/360q/100q had already completed without ever triggering
this path (no behavior change there — the fix is a no-op when `mover`'s core
doesn't move mid-relay); 200q was re-run after the fix.

## Stale doc reference

`code/CLAUDE.md`'s "Paper Sync Rules" says `router.py` here must stay in
sync with `../router.py` (main repo). No such file exists relative to this
repo (`dsabre/router.py` and `dsabre/paper/router.py` are both absent); the
only sibling project with a same-named `router.py` (`../cphm/code/router.py`)
is a separate, unrelated experiment (Core-Path Health Monitor) layered on
top of a shared ancestor, not "the" main repo for this project. This change
was not propagated there — flagging here rather than guessing at an
out-of-scope repo.
