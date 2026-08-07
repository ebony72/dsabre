# Safe-dSABRE — a router that always terminates *with success* (2026-08-07)

**Implemented and tested** — see §10 for the result. `config.safe_mode`
(default `False`) gates every behavioural change; default-mode output is
unchanged and checked by `verify_router.py`.

**Headline:** `qft_360` pass 2 — the one archived instance of a genuine,
non-budget abort — now routes. The guaranteed transaction has **never failed**
in any run measured, and `_force_make_room` — the reactive room-making patch
the invariant is meant to make unnecessary — fires **0 times** in safe mode
against 21–222 in default mode.

Cost depends on one setting. With `tier1_floor` at its strict default the 64q
suite pays **+20.8 % EPR gmean**; splitting the Tier-1 floor from the Tier-2
reserve (`tier1_floor=2`, §10.4) brings that to **+4.2 %** — with *fewer*
iterations than the default router and 9 guaranteed transactions instead of
124. Use the split.

Follows on from the complexity-analysis session (commit `4ea5d6f`), whose
closing diagnosis was:

> Termination is guaranteed, but only by budget exhaustion. **Success is not
> guaranteed.** `qft_360` pass 2 aborts with `DEADLOCK_BACKUP_FAILED` at
> iteration 20 124 with 4 535 operations unrouted — not budget exhaustion, the
> cap was 50 000. Free slots per core had drifted from `[24,16,19,25,15,27]` to
> `[72,0,0,51,0,3]`, and the stuck gate sat between two of the zeros. The
> precondition every recovery mechanism assumes was violated at **all 62**
> backup activations.

The mechanism that allows the drift is that capacity is a *soft penalty*
(`cap_penalty × max(0, capacity_threshold − free)`) while candidate *legality*
requires only `free(dst) ≥ 1` ([router.py:227](code/router.py:227)). A legal
teleport may therefore take a core to zero, and nothing pushes back.

Safe-dSABRE promotes capacity from a penalty to a **legality condition**, and
adds a fallback that is *provably* able to execute any remote gate. Together
these give a termination-with-success theorem (§5), not just a stopping
argument.

---

## 0. Terminology

### The routing protocol (unchanged by this work)

**Pass 1 / 2 / 3** — `run_sabre_passes` ([layout.py:566](code/layout.py:566))
routes three times: forward, then the *reversed* circuit, then forward again.

| | routes | starting layout | its EPR count is |
|---|---|---|---|
| pass 1 (fwd) | the circuit | the initial layout | **reported** |
| pass 2 (bwd) | the circuit *reversed* | pass 1's final layout | **discarded** |
| pass 3 (fwd) | the circuit | pass 2's final layout | **reported** |

Pass 2 exists only to produce a better *layout* — this is SABRE's refinement
trick: routing the circuit backwards leaves qubits arranged well for routing it
forwards again. The result is the better of passes 1 and 3 by EPR. Two
consequences matter here: **each pass starts from the previous pass's routed
output**, not from the layout pass, and **an abort in pass 2 is invisible** in
the reported number as long as pass 1 succeeded.

**Seeds / best-of-3** — `sabre_locked_boundary_layout` returns three candidate
initial layouts (SabreLayout seeds 0, 1, 2). The whole 3-pass protocol runs on
each and the lowest EPR wins. So a seed whose pass 2 aborts can still be the
one that produces the reported number (`qft_360` seed 0 does exactly this).

### The two tiers (introduced by this work)

**Tier 1** — ordinary dSABRE. Each iteration scores every legal one-hop
teleport for the front layer and executes the best-scoring one
(`_generate_candidates` → `_apply_teleport`). Fast, heuristic, no guarantee: it
can stall, and before this work it could stall permanently.

**Tier 2** — the *guaranteed gate transaction*, `_safe_route_gate` (§4). Picks
a meeting core, relays slack to it if needed, walks one operand there hop by
hop, SWAPs the pair adjacent, executes the gate. Under (†) it cannot fail. It
is invoked when Tier 1 has no legal move, when Tier 1 has stalled for
`deadlock_limit` iterations, and — as `_safe_drain` — in place of every abort.

**`tier1_floor` vs `core_reserve`** — two different jobs (§10.4).
`core_reserve` (= 2) is what Tier 2 needs at its entry for its `d`-optimal plan
to be legal. `tier1_floor` is how many free slots a destination needs for an
*ordinary* teleport. Setting `tier1_floor = core_reserve + 1` makes Tier 1
maintain the reserve itself; setting it to 2 lets Tier 1 spend the reserve down
(never to zero) and has Tier 2 restore it on entry.

### Capacity vocabulary

| term | meaning |
|---|---|
| **free slot** | a physical qubit in a core holding no logical qubit |
| **reserve** | `core_reserve` free slots every core should hold at an action boundary — invariant (†) |
| **donor** | a core holding `≥ reserve + 1`, so it can give one slot away without breaching (†) |
| **slack / relay** | moving one free slot from a donor to a target core, by teleporting a filler qubit the other way, one core-graph hop at a time |
| **meeting core** | the core where a remote gate's two operands are brought together |
| **filler** | an arbitrary non-gate qubit a relay moves to shift a free slot |

### Transactions

Two distinct undo mechanisms, easy to confuse:

| | **rollback** | **checkpoint restore** |
|---|---|---|
| taken by | `_snapshot`, on entry to every transaction | the main loop, after every gate retirement |
| restores | layout maps, scalar counters, trace tail | the same, **plus the whole DAG** |
| costs | O(qubits) — two dict copies | O(N) — `_rebuild_wdag` deep-copies the input DAG and replays retirements |
| triggered by | any failed precondition inside `_apply_teleport`, `_relay_slack_to`, `_safe_route_gate` | `deadlock_limit` iterations with no progress ("backup activation") |
| counted as | `metrics["rollbacks"]` / `["snapshots"]` | `metrics["backup_activations"]` / `["wdag_rebuilds"]` |

Transactions nest: `_safe_route_gate` snapshots, then each `_relay_slack_to`
snapshots, then each `_apply_teleport` snapshots. An inner rollback leaves the
outer one free to restore its own entry state later. §12 has the measured
counts and costs.

## 1. Notation

| symbol | meaning |
|---|---|
| `G_C = (V,E)`, `K = \|V\|` | core graph, assumed connected |
| `d_C(·,·)`, `diam` | core-graph distance, diameter |
| `κ_c` | physical qubits in core `c`; `P = Σ κ_c` |
| `n` | logical qubits; `F = P − n` total free slots |
| `free(c)` | free physical qubits in core `c` |
| `c(q)` | core currently holding logical qubit `q` |

Two facts used throughout:

- **F is exactly conserved.** An intra-core SWAP moves no qubit between cores;
  a teleport moves exactly one, so `free(src) += 1`, `free(dst) -= 1`. Nothing
  in the router creates or destroys a qubit or a slot.
- **A teleport into `c` is mechanically executable iff `free(c) ≥ 1`** — that
  is enough to evict the destination comm port so the arriving qubit can land
  on it. (Plus a staging-path condition in the *source* core; see §6.)

---

## 2. The invariant

> **(†) Safe state.** `free(c) ≥ 2` for every core `c`.
>
> **(F†) Feasibility.** `F ≥ 2K + 1`, equivalently `P ≥ n + 2K + 1`.

(†) is required to hold at every **boundary between top-level actions** — after
the initial layout, and after each iteration of the main loop. Inside an atomic
transaction a core may dip to `free = 1`; it may never reach 0.

Why 2 rather than 1, and why the extra `+1` in (F†):

- **`free(c) ≥ 2` ⇒ every hop of a shortest-path relocation is legal.** This is
  the whole payoff, and §4 makes it precise: moving a qubit from `a` to `b`
  along a shortest core path touches each intermediate core exactly twice
  (arrive, leave). At arrival, `free` drops to ≥ 1 — still executable; on
  departure it is restored. Only the endpoint keeps the deficit. With a floor
  of 1 instead of 2, an arrival can drop a core to 0, and a qubit parked in a
  0-free core cannot generally leave it — which is exactly the `[72,0,0,51,0,3]`
  failure.
- **`F ≥ 2K + 1` ⇒ a *donor* core always exists.** If every core had
  `free ≤ 2` then `F ≤ 2K`. So under (†)+(F†) some core has `free ≥ 3`, i.e.
  a unit of slack that can be relayed away without breaking (†) at its source.
  This is the pigeonhole that replaces the old `total ≥ K + 1` argument at
  [router.py:704](code/router.py:704), one notch stronger.

Both are checkable up front: (F†) from the architecture and circuit, (†) from
the initial layout. §7 shows every published suite satisfies them with margin.

### 2.1 The reserve as a knob

Everything below is stated for a general reserve `r` (`config.core_reserve`),
with `r = 2` the proposal:

| | invariant | teleport legal iff | fallback cost | needs |
|---|---|---|---|---|
| `r = 0` | none | `free(dst) ≥ 1` | *may fail* | — (today) |
| `r = 1` | `free ≥ 1` | `free(dst) ≥ 2` | `≈ 3d` | `F ≥ K+1` |
| **`r = 2`** | **`free ≥ 2`** | **`free(dst) ≥ 3`** | **`d` … `2d`** | **`F ≥ 2K+1`** |
| `r ≥ 3` | `free ≥ r` | `free(dst) ≥ r+1` | `d` | `F ≥ rK+1` |

`r = 1` is "Fix A" from the complexity session and is enough for a *success*
guarantee, but its fallback pays the `≈3d` relay overhead derived in §4.4.
`r = 2` is the smallest reserve that makes the *optimal* `d`-teleport plan
always legal. `r ≥ 3` buys nothing further for the guarantee and costs usable
capacity, so it is only worth trying as a congestion-quality knob.

Note the shift in what "≥2 free" means relative to the 2026-08-03 discussion:
there the rule *"teleport into `c'` only if `free(c') ≥ 2`"* was the legality
condition under an invariant of `free ≥ 1`. Here `free ≥ 2` is the *invariant*,
so the corresponding legality condition moves up to `free(c') ≥ 3` — that is
what "maintained before and after each action" forces, since one teleport
consumes exactly one slot at the destination.

---

## 3. Tier 1 — ordinary routing, with capacity as a legality condition

One change to candidate generation. In `_generate_candidates`
([router.py:227](code/router.py:227)):

```python
if free_cache.get(next_c, 0) < 1:      # current
if free_cache.get(next_c, 0) < 3:      # Safe-dSABRE
```

A teleport into `c'` is legal iff `free(c') ≥ 3`, so the post-state is `≥ 2`
and **(†) is preserved by every ordinary action**:

- teleport: `dst` goes `≥3 → ≥2` ✓, `src` gains one ✓;
- intra-core SWAP and `_fallback_local_swap`: occupancy unchanged ✓.

Three things follow immediately.

**The soft penalty and the hard floor agree.** `capacity_threshold` is already
3 and `cap_penalty` 15.0 — the heuristic already charges 15 for landing on a
core with `free = 2`. The hard rule promotes the knob's own threshold to a
legality condition; it forbids what the score was already paying to avoid.
That is why the measured cost is small (§8).

**`_force_make_room` becomes dead code.** It fires only when
`free(src) == 0` or `free(dst) == 0` ([router.py:441](code/router.py:441)).
Under (†) both are `≥ 2` at Tier-1 execution time, and `≥ 1` inside every
transaction of §4. The one mechanism in the router with no invariant guarantee
— the reactive patch that `apply_teleport_conflicts.docx` §9 and the 2026-08-05
`KeyError` were both about — provably never runs. Convert it to an assertion.

**Starvation is not an abort.** When no candidate passes the floor, the router
does not fall back and does not abort: it calls Tier 2, which cannot fail.

---

## 4. Tier 2 — the safe gate transaction

This is the "execute the target remote gate in an optimal way" procedure, made
precise under (†).

### 4.1 Lower bound

Each teleport advances one qubit by one core hop. Co-locating `q` and `q'` at
core distance `d = d_C(a,b)` requires their core displacements to sum to at
least `d`, so **`d` teleports is a hard lower bound**, independent of capacity.

### 4.2 The plan

```
safe_route_gate(g = (q, q')):
    a, b = c(q), c(q')
    m    = argmin over cores of total(m)                  # §4.4, O(K)
    need = 2 + (number of operands that will arrive at m)  # 3 if one, 4 if two
    relay_slack_to(m, need, protect={q, q'})               # no-op if free(m) >= need
    for x in {q, q'} with c(x) != m:
        for (u, v) in consecutive pairs of core_path(c(x), m):
            teleport x from u to v                         # legal: free(v) >= 2
    swap q and q' adjacent inside m; execute g
```

```
relay_slack_to(target, need, protect):
    while free(target) < need:
        s = nearest core to target with free >= 3          # BFS on G_C
        for (u, v) in consecutive pairs of core_path(s, target):
            r = any qubit in v not in protect
            teleport r from v into u                       # slack moves u -> v
```

### 4.3 Correctness

**Every teleport is legal.** In `relay_slack_to`, a filler moves from `v` into
`u`, and `u` holds the slack at that moment (`free(u) ≥ 3` at the head of the
chain, `≥ 3` at each subsequent link because it has just received the slack).
In the main loop, `x` moves into `v` with `free(v) ≥ 2` by (†) — none of the
path cores has been touched yet. So no destination is ever below 1 free, and
`_force_make_room` never fires.

**(†) is restored on exit.**
- `relay_slack_to`: the donor `s` ends at `free ≥ 2`; every intermediate is
  transiently `+1` and returns to its entry value; `target` ends at `+1`.
- Relocation of `x` from `a` to `m`: `a` ends `+1`; each intermediate arrives at
  `≥1` and is restored to its entry value the moment `x` leaves; `m` ends at
  `free(m) − arrivals ≥ 2`, which is what `need` was chosen for.

**A donor always exists.** Under (†)+(F†), `Σ_c (free(c) − 2) ≥ 1`, so some core
has `free ≥ 3`. For `need = 4` (a third-core rendezvous, two arrivals) the
second unit needs a donor *other than* `target`, which requires `F ≥ 2K + 2`;
with `m ∈ {a, b}` only `need = 3` ever arises and (F†) suffices. This is why
§4.4 restricts to `m ∈ {a,b}` when `F = 2K + 1` exactly.

**A filler always exists** — the one hypothesis beyond (†)+(F†), and the branch
`_relay_room_to` currently rolls back on
([router.py:777](code/router.py:777)). Assume `κ_c ≥ 5` for every core. Then:
every core strictly between the donor and `target` on the relay path has
`free ≤ 2` (otherwise BFS would have chosen it as the nearer donor), so its
occupancy is `≥ κ − 2 ≥ 3`, and at most one of the two protected qubits lives
there (`q` and `q'` are in distinct cores whenever a relay is needed) — leaving
`≥ 2` eligible fillers. `target` itself is entered only when
`free(target) < need`, so its occupancy is `≥ κ − need + 1 ≥ 2`, again with at
most one protected qubit. Every published core has `κ ∈ {16, 25, 27, 81}`, so
the hypothesis is not a restriction in practice — but it is load-bearing and
belongs in the theorem statement.

**It terminates.** `relay_slack_to` runs at most `need − free(target) ≤ 2`
chains of `≤ diam` hops; the relocation is exactly `d_C(c(x), m) ≤ diam` hops
per operand; the intra-core SWAP chain is at most the core's diameter. No loop
can retry.

**Therefore Tier 2 always succeeds** — it executes `g` and returns a safe state.
There is no failure branch to fall through to.

### 4.4 Cost, and in what sense it is optimal

```
total(m) = d_C(a,m) + d_C(b,m) + relay(m)
relay(m) = 0                  if free(m) >= 2 + arrivals(m)
         = d_C(m, s)          otherwise, s = nearest core with free >= 3
```

`argmin` over `V(G_C)` costs `O(K)` with the precomputed `core_dist`.

- **Common case — `free(m) ≥ 3` for some `m ∈ {a,b}`:** `total = d`. The
  fallback hits the information-theoretic lower bound exactly. **This is the
  whole point of the 2-slot reserve.**
- **A "fat" core on a shortest path** (`free ≥ 4`, so both operands can land):
  also `total = d`, with the gate ending up in a core that has room to keep
  working — often a better layout for what follows.
- **Tight case (`F = 2K + 1`, `free(m) = 2`, donor at `a`):** `relay = d`, so
  `total = 2d`. This is optimal among plans that restore the occupancy vector:
  moving `q` from `a` to `b` shifts occupancy by `(−1 at a, +1 at b)`, so
  cancelling it needs one unit of qubit flow from `b` back to `a`, costing
  `d_C(a,b) = d`. Total `2d`.
- **The relay is exactly optimal, not greedy-approximate,** for the single
  deficit unit that `m ∈ {a,b}` produces: raising `free(m)` by 1 requires one
  unit of qubit flow from `m` to some core that can absorb it, costing
  `d_C(m,s)`; a chain through an intermediate donor with `free = 2` costs at
  least as much by the triangle inequality. Nearest-donor BFS is the optimum.

**Contrast with the weaker invariant.** The 2026-08-03 session analysed this
same procedure under the old precondition (`free ≥ 1` everywhere, total
`≥ K+1`) and found — verified against an exhaustive product-state BFS over
(position of `q`, position of `q'`, position of the slack token) — a worst case
of `3d − 2 + m₁`, because with a single slack unit the vacated slot always
reappears one hop *behind* the qubit while the next hop needs it one hop
*ahead*, forcing a 2-hop relay between consecutive payload hops. The 2-slot
reserve is precisely the condition that removes that relay: it buys the
difference between `~3d` and `d`.

The one honest caveat: `total(m)` is optimal *within the class of plans that
never breach the floor and end in a safe state*. A plan allowed to end unsafe
can occasionally do better on a single gate; it is then not the fallback's
successor's problem to fix, which is the trade being made deliberately.

---

## 5. Termination with success

> **Theorem.** Let `G_C` be connected, every core's intra-coupling graph
> connected with `κ_c ≥ 5`, every core-graph edge carry at least one inter-core
> link, and let (F†) hold and the initial layout satisfy (†). Then Safe-dSABRE
> routes every gate of the input DAG. It never aborts.

*Proof.* (†) holds initially and is preserved by every Tier-1 action (§3) and
by every Tier-2 transaction (§4.3), hence at every boundary. Every Tier-2 call
therefore has its precondition met, and by §4.3 it succeeds and retires its
target gate. Gates only leave the DAG, so the count of unrouted gates strictly
decreases at each Tier-2 call: at most `|G|` calls occur. Between consecutive
Tier-2 calls the main loop runs at most `deadlock_limit` non-progressing
iterations (after which Tier 2 fires by construction) plus progressing ones,
of which there are at most `|G|` in total. Hence the loop runs at most
`|G|·(deadlock_limit + 2)` iterations and ends with an empty DAG. ∎

Three changes to the loop are needed to make the theorem true of the code:

| today | Safe-dSABRE |
|---|---|
| `DEADLOCK_BACKUP_FAILED` when `_backup_plan` returns False | cannot occur — Tier 2 has no failure branch |
| `DEADLOCK_BACKUP_EXHAUSTED` at `max_backup_attempts` | cap raised to `|G|`; each activation retires a gate, so it is never reached |
| `ITERATION_LIMIT` / `NO_ACTIONS_NO_FALLBACK` / `ALL_CANDIDATES_REJECTED` abort | switch to **safe-drain mode**: stop heuristic search and retire remaining gates one at a time with Tier 2 |

Safe-drain also gives a worst-case cost bound worth stating in the paper: a
circuit drained entirely by Tier 2 costs at most `2·diam(G_C)` EPRs per remote
gate, so `EPR ≤ 2·diam(G_C)·|G_remote|` — the first non-vacuous upper bound
dSABRE has had.

---

## 6. Mechanical preconditions (two small hardenings)

The theorem needs *every* hop to be executable, so two structural gaps in
`_apply_teleport` must close. Neither affects the published architectures —
both B-grid and H-grid cores are `m×m` grids with **no articulation points**,
verified — but both bite on heavy-hex tiles, where the ring ports 1 and 25
*are* cut vertices.

1. **Staging path must exist.** `n_s` is chosen by each caller as the neighbour
   of the source port nearest to `q` in the *unrestricted* intra graph
   ([router.py:230](code/router.py:230), and again at `:784`, `:874`, `:976`),
   but the staging path is then computed inside `_apply_teleport` on
   `intra − {port}` ([router.py:486](code/router.py:486)). If `n_s` lands in
   a different component the hop fails. Fix: choose `n_s` among the port's
   neighbours **inside `q`'s own component of `intra − {port}`**. That set is
   never empty — every component of `intra − {port}` is adjacent to `port`,
   since `intra` is connected. (On the 27-qubit tile the severed component is
   the single leaf 0 or 26, which is itself a port neighbour, so today's code
   happens to be safe there; the fix makes it safe in general.)
2. **A qubit already sitting on the source port** is rejected outright
   ([router.py:479](code/router.py:479)), yet it is physically the easiest case
   — `_force_make_room` already teleports straight off the port
   ([router.py:684](code/router.py:684)). Fix: take that path instead of
   failing.

With these, "`free(dst) ≥ 1` ⇒ the hop executes" is unconditional for any
connected intra topology, and the theorem needs no 2-connectivity caveat.

---

## 7. Feasibility on the published architectures

`F ≥ 2K + 1` holds everywhere, with large margin:

| suite | K | κ | P | n | F | 2K+1 | diam |
|---|---|---|---|---|---|---|---|
| 25q B-grid 2×2, 4×4 | 4 | 16 | 64 | 25 | 39 | 9 | 2 |
| 36q B-grid 2×2, 4×4 | 4 | 16 | 64 | 36 | 28 | 9 | 2 |
| 64q H-grid 2×3, 4×4 | 6 | 16 | 96 | 64 | 32 | 13 | 3 |
| 80q H-grid 2×5, 4×4 | 10 | 16 | 160 | 80 | 80 | 21 | 5 |
| 100q H-grid 2×3, 5×5 | 6 | 25 | 150 | 100 | 50 | 13 | 3 |
| 200q H-grid 4×3, 5×5 | 12 | 25 | 300 | 200 | 100 | 25 | 5 |
| 360q H-grid 2×3, 9×9 | 6 | 81 | 486 | 360 | 126 | 13 | 3 |

**The initial layouts already satisfy (†).** `adaptive_corner_count`
([layout.py:216](code/layout.py:216)) reserves `k` corners per core and
`sabre_locked_boundary_layout` removes them from the graph SabreLayout sees,
so at most `κ − k` qubits land in any core and every core starts with
`free ≥ k`. Measured `k`, and the resulting fill:

| suite | 25q | 36q | 64q | 80q | 100q | 200q | 360q |
|---|---|---|---|---|---|---|---|
| `k` | 4 | 4 | **2** | 4 | 4 | 4 | 4 |
| usable fill | 52.1 % | 75.0 % | 76.2 % | 66.7 % | 79.4 % | 79.4 % | 77.9 % |

`k ≥ 2` everywhere, so **(†) holds at `t = 0` on every published suite with no
change at all.** Measured per-core free vectors confirm it: 64q gives
`[2,2,5,2,5,16]`, `[14,5,2,7,2,2]`, `[3,4,5,5,9,6]` (floor exactly `k = 2`);
360q gives `[24,16,19,25,15,27]`, `[5,4,23,5,29,60]`, `[18,10,20,29,17,32]`
for `qft` (floor 4, 15 in the best seed).

### 7.1 So does the initial layout need revising? Mostly no — see 360q

**No: the initial layout is the healthiest state in the entire run.** Traced
`qft_360` (13 300 CX, `H-grid 2×3 9×9`, `K = 6`, `κ = 81`, `P = 486`,
`n = 360`, `F = 126`), production config, SabreLayout seed 0, dSE:

| point in the run | free vector | min | Σ |
|---|---|---|---|
| initial layout → pass 1 in | `[24, 16, 19, 25, 15, 27]` | **15** | 126 |
| pass 1 out → **pass 2 in** | `[68, 2, 2, 22, 20, 12]` | **2** | 126 |
| pass 2, 1st recovery | `[70, 3, 0, 22, 19, 12]` | **0** | 126 |
| pass 2, 62nd recovery → abort | `[72, 0, 0, 51, 0, 3]` | **0** | 126 |

Pass 1 *succeeds* (604 EPR, no failure log) and still leaves the chip with core
0 hoarding 68 of the 126 free slots and two cores down to 2. Pass 2 then aborts
`DEADLOCK_BACKUP_FAILED` at iteration 20 124 with 4 535 operations unrouted —
budget was 50 000 — and **all 62 recoveries ran with at least one zero-free
core**. The stuck gate straddles cores 4 and 2 in every one of them; core 2 is
already at 0 at the first recovery and never recovers, and by the last both its
cores are at 0, so no operand has anywhere to go.

**All three SabreLayout seeds, for contrast:**

| seed | initial free (min) | pass 1 | pass 1 out (min) | pass 2 | recoveries **with a 0-free core** |
|---|---|---|---|---|---|
| 0 | `[24,16,19,25,15,27]` (**15**) | 604 ✓ | `[68,2,2,22,20,12]` (2) | 837 **ABORT** | **62 / 62** |
| 1 | `[5,4,23,5,29,60]` (4) | 729 ✓ | `[37,12,12,40,5,20]` (5) | 925 ✓ | **19 / 19** |
| 2 | `[18,10,20,29,17,32]` (10) | 559 ✓ | `[34,24,23,9,15,21]` (9) | 754 ✓ | **2 / 2** |

Two readings, both awkward for a layout-side fix:

- **The seed with the healthiest start is the one that aborts.** Seed 0 begins
  with 15 free in its poorest core — the best of the three — and is the only
  failure. Seed 1 starts at 4 and completes. Initial min-free does not predict
  the abort; if anything it is anti-correlated here.
- **83 of 83 deadlock recoveries, across all six passes, ran with at least one
  core at zero free.** Not most — all of them. The precondition every recovery
  mechanism in the router assumes is violated 100 % of the time it is invoked.

(Seed 0 also produces the best pass-1 EPR of the three, 604. Best-of-3 keeps it
and discards the aborted pass, so this failure never reaches a results table —
the masking the complexity session flagged.)

Three things this settles:

1. **The initial layout starts with 15 free in its poorest core — over seven
   times the reserve (†) asks for.** No layout policy could have started
   healthier, and the two seeds that start *worse* both complete. Revising the
   layout would not have prevented this abort.
2. **Σ free is 126 at every one of those four rows.** Nothing is consumed;
   routing *redistributes* slack until it is all in one corner of the chip.
   That is a legality problem, not a budget problem, and it can only be fixed
   where the moves are chosen — Tier 1 (§3).
3. **Passes 2 and 3 never see `sabre_locked_boundary_layout` at all.**
   `run_sabre_passes` feeds pass 2 the *routed output* of pass 1
   ([layout.py:570](code/layout.py:570)). So even a perfect initial-layout
   policy governs one of three passes. Under Safe-dSABRE this stops mattering,
   because pass 1 now *exits* safe — which is the point of making (†) an
   invariant rather than a starting condition.

### 7.2 The three layout changes still worth making

Guards, not a redesign. None changes any current suite's `k`.

1. **Floor `k` at 2, not 1.** `adaptive_corner_count`
   ([layout.py:242](code/layout.py:242)) sets `k_floor = 1 if P − n ≥ K else 0`.
   Every published suite gets `k ≥ 2` from the fill rule anyway, so (†) holds
   today *by coincidence, not by construction*. Raise the floor to 2 guarded by
   `P − n ≥ 2K + 1` — which is (F†) — and refuse safe mode below it.
2. **Fix the completion step.** Logical qubits SabreLayout did not place are
   assigned to a random free physical drawn from `arch.data_qubits`
   ([layout.py:297–303](code/layout.py:297)) — the *whole chip*, reserved
   corners included, with no per-core cap. That path can put a core below its
   reserve. It did not fire on any measured suite (every observed floor equals
   `k` exactly), but it is the one place the reserve is established by luck.
   Restrict the fill to slots that keep `free(c) ≥ 2`.
3. **Check (or repair) at `route()` entry**, not only at layout time — because
   of point 3 above. A violation should raise in safe mode; optionally, a
   `_make_layout_safe()` that teleports qubits out of over-full cores costs a
   handful of EPRs and lets an externally supplied layout be accepted rather
   than rejected.

### 7.3 A separate, quality-only observation

On the small 360q circuits SabreLayout returns free vectors like
`[81, 29, 4, 4, 4, 4]` — one core **entirely empty**, four sitting at exactly
the reserve. `bv`, `dj`, `vqe_su2` and `wstate` all do this on at least one
seed. It is a perfectly safe state under (†), and it is a poor starting
distribution: the coupling map SabreLayout minimises over treats an inter-core
link as an ordinary edge, so it packs qubits into as few cores as it can. That
is the same axis as the `freeslot-frontier` finding (central cores want more
slack than edge cores) and is orthogonal to safety — worth its own experiment,
not a fix to fold into this one.

---

## 8. Measured cost of the floor

Instrumented run of the production router (dSE, 3 SabreLayout seeds ×
fwd→bwd→fwd) on the 64q suite, recording for every teleport the main loop
executed the destination core's free count immediately before arrival, and for
every scoring iteration whether *any* candidate would have survived a floor of
τ (`starved` ⇒ Tier 2 fires):

| circuit | EPR | teleports | dst free < 2 | **dst free < 3** | starved @ τ=2 | **starved @ τ=3** | min free reached |
|---|---|---|---|---|---|---|---|
| ae | 141 | 2 447 | 0.3 % | 23.5 % | 0.00 % | 0.00 % | 0 |
| ghz | 6 | 60 | 0.0 % | 25.0 % | 0.00 % | 0.00 % | 1 |
| graphstate | 16 | 206 | 0.0 % | 1.9 % | 0.00 % | 0.00 % | 1 |
| qft | 192 | 6 262 | 0.5 % | 16.1 % | 0.00 % | 0.81 % | 0 |
| qnn | 497 | 6 305 | 0.1 % | 8.2 % | 0.00 % | 0.00 % | 0 |
| qaoa | 517 | 6 309 | 0.1 % | 6.5 % | 0.00 % | 0.00 % | 0 |
| qpeexact | 204 | 4 895 | 0.6 % | 24.8 % | 0.00 % | 0.14 % | 0 |
| random | 712 | 7 237 | 0.0 % | 14.4 % | 0.00 % | 0.01 % | 1 |
| multiplier | 1 523 | 21 526 | 0.1 % | 16.9 % | 0.00 % | 0.00 % | 0 |

All 9 canonical 64q circuits; EPR counts match the current published suite.

Reading:

- **The current router really does drive cores to zero** — `min free = 0` on
  six of nine circuits, on a chip with 32 spare slots. This is the drift that
  killed `qft_360`, present (harmlessly, so far) at 64q too.
- **The floor removes 2–25 % of the moves currently chosen**, which is what
  `CAPACITY_SAFE_RECOVERY.md` objected to when it rejected a global capacity
  invariant. But rejecting a candidate is not rejecting a move: the router
  falls through the score-sorted list.
- **The list empties on at most 0.81 % of iterations** (τ=3), and never at
  τ=2. So Tier 2 — the expensive, guaranteed path — is invoked on well under
  1 % of iterations. That is the number the earlier objection was missing.

**Which of these two columns predicts the EPR cost? The rejection one.**
Measured after the fact (§10.3): the safe-mode regression tracks
`dst free < 3` almost monotonically — `graphstate` at 1.9 % is unchanged,
`ghz`/`ae`/`qpeexact` at 23–25 % regress 43–67 % — while the starvation column
predicts nothing. Reading "rarely empties" as "small disruption" was wrong:
taking the second-best candidate 6–25 % of the time compounds. §10.4 is the
consequence.

---

## 9. What was implemented

Everything behind `config.safe_mode = False`, so published numbers stay
reproducible bit-for-bit.

| # | change | where | status |
|---|---|---|---|
| 1 | `safe_mode: bool = False`, `core_reserve: int = 2` | `config.py` | done |
| 2 | Validate (F†) and repair the entry layout to (†) | `router.py` `route()`, `_make_layout_safe` | done |
| 3 | Component-aware staging retry when the port is a cut vertex | `router.py` `_apply_teleport` | done, safe-mode-gated¹ |
| 4 | `_relay_slack_to(target, need, protect)` — donor floor (`reserve+1`) split from the target requirement | `router.py` | done |
| 5 | `_plan_meeting_core` + `_safe_route_gate` (§4.2) | `router.py` | done |
| 6 | Candidate floor `< 1` → `< core_reserve + 1` | `router.py` `_generate_candidates` | done |
| 7 | Loop: Tier 2 on an empty/rejected candidate list, on deadlock, and as `_safe_drain` in place of the `ITERATION_LIMIT` / backup-exhausted aborts | `router.py` `route()`, `_backup_plan` | done |
| 8 | `adaptive_corner_count(..., reserve=)` floors `k` at 2 when affordable | `layout.py` | done, **not committed**¹ |

¹ `code/layout.py` carries ~240 lines of unrelated uncommitted work (the
`k_floor = 1` rule itself, `sabre_completed_boundary_layout`,
`repack_to_budgets`, the dist-k completion helpers). Committing the file would
sweep that in under this change's message. The `reserve` parameter is inert
until a caller passes it — safe mode does not depend on it, because
`_make_layout_safe` repairs an unsafe entry layout at `route()` entry — so it
stays in the working tree with the rest.

¹ grid cores have no articulation points, so this path cannot fire on any
published architecture; gating it keeps default-mode output identical.

**Not implemented, and not needed:** the "teleport straight off the source
port" hardening. Under (†) the source core always has ≥1 free slot, so
`_evict` always clears the port and the on-port case never reaches the
rejection branch. It would only matter at `free = 0`, which the invariant
excludes.

`_force_make_room` was left in place rather than converted to an assertion —
it measures 0 calls in safe mode (§11), which is the same information without
the risk of crashing a long run.

**Verification.** `verify_router.py --suite 25q` (full SWAP/teleport traces,
54 passes): **ALL IDENTICAL**. The 64q verifier was still running at time of
writing.

New metrics: `safe_routes` (gates retired by the guaranteed transaction) and
`safe_route_failed` (attempts that hit one of its assertion paths — must stay
0 while the hypotheses hold).

---

## 10. Measured result — `qft_360`

Config `deadlock_limit=200, max_backup_attempts=200, max_iterations=50000`
(the production 360q settings), dSE, `H-grid 2×3 9×9`.

### 10.1 The isolated test — pass 2 from the state that aborts

Pass 1 is run with the **default** router in both arms, so both start from
bit-identically the same layout `[68,2,2,22,20,12]`. Only pass 2 differs.

| pass 2 | EPR | aborted | safe routes | `force_make_room` | min free reached | failure log |
|---|---|---|---|---|---|---|
| `safe_mode=False` | 837 | **True** | 0 | 124 | **0** | `DEADLOCK_BACKUP_FAILED` @ 20 124, 4 535 unrouted |
| `safe_mode=True` | 925 | **False** | 126 | **0** | **2** | *(empty)* |

Every prediction of §3–§5 shows up in that row:

- **the abort is gone**, with an empty failure log;
- **`safe_route_failed = 0`** — no assertion path in the guaranteed transaction
  ever fired, i.e. (†) and (F†) held at every one of its 126 invocations;
- **`force_make_room = 0`**, against 124 in the same pass without the
  invariant. The reactive room-making patch really is dead code under (†) — §3
  predicted this and it is the cleanest confirmation that the invariant holds;
- **min free = 2**, never 0 — and the exit vector `[23,3,2,81,14,3]` still sums
  to 126.

### 10.2 The protocol test — full fwd→bwd→fwd, all 3 seeds

| mode | seed | pass 1 | pass 2 | pass 3 | seed best |
|---|---|---|---|---|---|
| default | 0 | 604 | **837 ABORT** | 857 | 604 |
| default | 1 | 729 | 925 | 641 | 641 |
| default | 2 | 559 | 754 | 729 | **559** |
| safe | 0 | 676 | 921 | 1069 | **676** |
| safe | 1 | 891 | 873 | 943 | 891 |
| safe | 2 | 822 | 684 | 696 | 696 |

| | protocol EPR | aborts | safe routes | **failed** | `force_make_room` | min free |
|---|---|---|---|---|---|---|
| default | **559** | 1 | 0 | 0 | 222 | **0** |
| safe | **676** (+20.9 %) | **0** | 887 | **0** | **0** | **1** |

Min free of 1 in safe mode is the *transient* dip §4.3 allows inside a
transaction (a qubit arriving at an intermediate core before it leaves again);
it is never 0, and every iteration boundary is at ≥ 2.

### 10.3 The 64q suite — and what actually drives the cost

Standard harness, 9 circuits × 3 SabreLayout seeds × fwd→bwd→fwd. The
`dst free < 3` column is from §8: the share of the teleports the *default*
router chose that the safe-mode floor rejects.

| circuit | `dst free < 3` | base EPR | safe EPR | **Δ** | safe routes | `force_make_room` base → safe |
|---|---|---|---|---|---|---|
| graphstate | 1.9 % | 16 | 16 | **0.0 %** | 1 | 0 → 0 |
| qaoa | 6.5 % | 517 | 542 | +4.8 % | 5 | 0 → 0 |
| qnn | 8.2 % | 497 | 530 | +6.6 % | 4 | 2 → 0 |
| random | 14.4 % | 712 | 768 | +7.9 % | 19 | 0 → 0 |
| qft | 16.1 % | 192 | 191 | **−0.5 %** | 4 | 0 → 0 |
| multiplier | 16.9 % | 1 523 | 1 783 | +17.1 % | 49 | 4 → 0 |
| ae | 23.5 % | 141 | 231 | +63.8 % | 14 | 3 → 0 |
| qpeexact | 24.8 % | 204 | 291 | +42.6 % | 25 | 12 → 0 |
| ghz | 25.0 % | 6 | 10 | +66.7 % | 3 | 0 → 0 |
| **total** | | **3 808** | **4 362** | **+14.5 %** (gmean +20.8 %) | 120 | 21 → **0** |

No aborts in either arm; `safe_route_failed = 0` everywhere; `force_make_room`
is 0 in safe mode on every circuit.

**This corrects the reading in §8, and a hypothesis about `qft_360`.** The
cost is *not* a deadlock-scheduling artifact. At 64q the guaranteed transaction
is barely used — 1–49 activations per circuit across nine `route()` calls —
and the regression is still +20.8 % gmean. It comes from Tier 1: the floor
rejects the highest-scoring candidate and the router takes the second-best,
repeatedly.

The `dst free < 3` column predicts it almost perfectly (monotone except `qft`):
the three worst regressions — `ghz` +66.7 %, `ae` +63.8 %, `qpeexact` +42.6 % —
are exactly the three circuits whose chosen moves most often land on a core
with fewer than 3 free, and the circuit that barely notices, `graphstate`
at 1.9 %, is unchanged. §8 read the *starvation* column (≤ 0.81 %) and inferred
small disruption; the *rejection* column was the predictive one. Two suites now
agree on the magnitude: +20.8 % gmean at 64q, +20.9 % at 360q.

### 10.4 Splitting the Tier-1 floor from the Tier-2 reserve

The floor is doing two different jobs, and only one of them needs to be this
strict:

- **Tier 2 needs `free ≥ 2` at its entry**, or its plan is not the `d`-optimal
  one (an intermediate core can hit 0 and strand the mover mid-path). This is
  what `core_reserve = 2` is for, and §10.1–10.2 show it working.
- **Tier 1 does not need to enforce it every iteration.** It only has to avoid
  driving a core to 0 — a floor of `free(dst) ≥ 2`, post-state ≥ 1, which §8
  measures as rejecting **0.0–0.6 %** of chosen moves instead of 1.9–25 %.

Implemented as `config.tier1_floor` (`None` = `core_reserve + 1`, the strict
setting; `2` = the split setting). Tier 1 uses the weaker floor and
`_safe_route_gate` calls `_make_layout_safe` on entry to restore `free ≥ 2`
before it plans.

**The termination theorem survives the split.** Tier 1 now maintains only
`free(c) ≥ 1`, so the argument has to be redone at Tier 2's entry, and it goes
through:

- *A donor exists.* Under (F†) `F ≥ 2K + 1`, so if every core held ≤ 2 the
  total would be ≤ 2K — some core holds ≥ 3.
- *There is enough slack for all short cores at once.* With deficit
  `D = Σ_short (2 − free)` and surplus `S = Σ (free − 2)⁺`, `S − D = F − 2K ≥ 1`,
  so `S > D` always. `_make_layout_safe` fixes at least one short core per
  sweep and never creates a new one (a donor drops 3 → 2, not below), so it
  terminates in ≤ K sweeps.
- *Every relay hop is legal.* A filler moves into a core that currently holds
  the slack (≥ 3 at the head, ≥ 2 after receiving it); intermediates return to
  their entry count; no core drops below 1.
- *Fillers exist.* Intermediates on a relay path have `free ≤ 2` (else BFS
  would have chosen them as the nearer donor), so occupancy ≥ κ − 2 ≥ 3.

After `_make_layout_safe` the state satisfies (†) exactly as before, so §4.3
applies unchanged and the transaction still cannot fail. `_force_make_room`
also still never fires, because Tier 1's floor of 2 keeps every core at ≥ 1.

### 10.5 Measured: split vs strict

**64q suite** (9 circuits, 3 layouts, fwd→bwd→fwd, best-of-3):

| circuit | base | strict | Δ | **split** | **Δ** |
|---|---|---|---|---|---|
| ae | 141 | 231 | +63.8 % | **153** | **+8.5 %** |
| ghz | 6 | 10 | +66.7 % | **6** | **0.0 %** |
| graphstate | 16 | 16 | 0.0 % | **16** | **0.0 %** |
| qft | 192 | 191 | −0.5 % | 233 | +21.4 % |
| qnn | 497 | 530 | +6.6 % | **497** | **0.0 %** |
| random | 712 | 768 | +7.9 % | **712** | **0.0 %** |
| qpeexact | 204 | 291 | +42.6 % | **229** | **+12.3 %** |
| qaoa | 517 | 542 | +4.8 % | 544 | +5.2 % |
| multiplier | 1 523 | 1 783 | +17.1 % | **1 421** | **−6.7 %** |
| **total** | **3 808** | 4 362 | +14.5 % | **3 811** | **+0.1 %** |
| **gmean** | — | — | **+20.8 %** | — | **+4.2 %** |

| | base | strict | split |
|---|---|---|---|
| Tier-2 activations | 0 | 124 | **9** |
| `safe_route_failed` | 0 | 0 | **0** |
| `force_make_room` | 21 | 0 | **0** |
| main-loop iterations | 69 155 | 78 873 | **65 721** |

**The split recovers almost all of it: +20.8 % → +4.2 % gmean, and the totals
are within 0.1 %.** It also uses *fewer* iterations than the default router
(65 721 vs 69 155) and invokes the guaranteed transaction 9 times instead of
124 — the guarantee is now something the router almost never has to fall back
on, rather than a path it lives in.

Per-circuit variance stays large in both directions (`multiplier` −6.7 %,
`qft` +21.4 %) and should not be over-read: a single different choice cascades
through fwd→bwd→fwd and best-of-3, the same mechanism that swung `ae`
202 → 141 when only `_backup_plan` changed. The gmean over nine circuits is
the number that means something.

Two things the cost buys that the EPR column does not show: the abort is gone,
and the 360q protocol no longer depends on best-of-3 masking a failed seed —
default mode reports 559 only because seed 2 happened to survive.

## 11. `deadlock_limit` and predicting the iteration count

### 11.1 Were the two arms started from the same layout?

Yes, and it was controlled for:

- **§10.1** ran pass 1 **once, with the default router**, and handed both arms
  the same `fwd_final`. One variable.
- **§10.2 / §10.3** call `sabre_locked_boundary_layout(qc, dag, arch, seed=0)`,
  which depends only on the circuit and architecture, not on `config`.
  `adaptive_corner_count`'s new `reserve` parameter defaults to 1, so the
  layouts are byte-identical to before this change.
- `_make_layout_safe` never fired: entry layouts already have min free ≥ 2
  (64q `[2,2,5,2,5,16]`; 360q min 15 / 4 / 10), so it returns having spent
  nothing.

The caveat is the protocol's, not the test's: **passes 2 and 3 diverge**, since
each starts from the previous pass's routed output. Only pass 1 is matched in
§10.2. That is what §10.1 exists to control.

### 11.2 In safe mode, `deadlock_limit` is a cost knob — and it should be small

`deadlock_limit` swept over the 64q suite with `safe_mode=True` (8 circuits,
3 layouts, fwd→bwd→fwd for EPR; pass 1 alone for iteration counts):

| L | total EPR | vs default (gmean) | pass-1 iterations | bound | bound / actual | safe routes |
|---|---|---|---|---|---|---|
| *default router, L=100* | *2 285* | — | *13 436* | — | — | *0* |
| **10** | **2 573** | **+21.1 %** | **13 542** | 218 537 | **16.1×** | 79 |
| 25 | 2 579 | +21.3 % | 14 469 | 516 542 | 35.7× | 70 |
| 50 | 2 579 | +21.3 % | 16 194 | 1 013 217 | 62.6× | 70 |
| 100 | 2 579 | +21.3 % | 19 644 | 2 006 567 | 102.1× | 70 |
| 200 | 2 579 | +21.3 % | 26 544 | 3 993 267 | 150.4× | 70 |

**EPR is flat in L; iterations are linear in it.** This closes the question
§10.3 opened: deadlock scheduling contributes essentially nothing to the
+21 %. It is entirely Tier-1 substitution.

The practical consequence is the opposite of the default-mode intuition. In
default mode a high limit is protective — recovery is a gamble that can fail,
so you avoid triggering it. In safe mode recovery always succeeds, so there is
no reason to let the heuristic thrash first: **L = 10 gives the same EPR as
L = 200, matches the default router's own iteration count (13 542 vs 13 436),
and tightens the worst-case bound 20×.** `deadlock_limit_for(arch,
safe_mode=True)` returns 10 for this reason; the architectural derivation
(`4·diam_C·(diam_intra+1)`, giving 56 / 84 / 204 against the hand-tuned
50 / 100 / 200) is kept for default mode.

### 11.3 Reporting iterations-to-completion

Two figures, both live, both now in `metrics` (`iterations`,
`iterations_bound`):

**A hard bound, valid at every step** — `General_dSABRE_Router.iterations_bound`:

```
I_remaining  ≤  R_2q · (deadlock_limit + 1)
```

Each iteration either retires an operation or increments the no-progress
counter; after `L` of the latter the guaranteed transaction fires and retires
a gate. So no gate absorbs more than `L + 1` iterations. Only 2-qubit gates
count — 1q gates and already-adjacent intra-core 2q gates drain inside the
iteration that finds them. In default mode there is no such bound at all;
`max_iterations` is a budget the route can hit and abort on.

**An estimate, because the bound is loose** — measured 16–150× overall, and
18–200× per circuit. A tight *a-priori* estimate from `(n, |G_2q|)` alone is
not available, and the data says why: iterations/`|G_2q|` ranges **0.5–5.5**
across the 64q suite at fixed `n = 64` (qnn 0.51, qft 0.78, ae 1.07, random
1.60, qpeexact 2.87, ghz 5.46). What drives it is how many gates are *remote
under the current layout*, i.e. circuit locality, not gate count. So pair the
bound with a self-correcting extrapolation from the observed rate —
`iterations_estimate(iterations_so_far, retired_2q, remaining_2q)`:

```
I_remaining  ≈  I_so_far · R_2q / (G_2q − R_2q)
```

Choosing L = 10 helps here too: it is the same knob that makes the hard bound
16× loose instead of 102×.

## 12. Transaction accounting

New metrics: `snapshots`, `rollbacks`, `wdag_rebuilds`, and (under
`config.profile_transactions`) `snapshot_s`, `rollback_s`. `wdag_rebuild_s` is
timed unconditionally — it is rare and dominant when it happens. Counters live
on the router, not in `metrics`: a rollback restores every scalar metric, so a
counter kept there would undo its own increment and always read 0.

**64q suite, whole run (9 circuits × 3 layouts × 3 passes = 81 `route()` calls
per condition):**

| condition | snapshots | **rollbacks** | wdag rebuilds | wdag_s | snapshot_s | rollback_s | wall | txn share |
|---|---|---|---|---|---|---|---|---|
| base | 15 800 | **0** | 29 | 0.25 s | 0.04 s | 0.00 s | 349 s | **0.08 %** |
| safe_strict | 26 244 | **0** | 124 | 1.16 s | 0.06 s | 0.00 s | 354 s | 0.34 % |
| safe_split | 13 330 | **0** | 9 | 0.09 s | 0.04 s | 0.00 s | 339 s | **0.04 %** |

Three findings:

1. **Rollbacks are essentially never taken.** Zero across all 47 374 snapshots
   of the 64q suite; across every suite measured, the only nonzero count is
   **one** rollback, in `qft_100` under the default router. The atomic-transaction
   machinery built in the 2026-08-03/05 sessions (`_apply_teleport`'s checked
   rollback, `_route_gate_transaction`) is a safety net whose failure path the
   published suites do not exercise. Verified rather than assumed: a
   deliberately invalid `TeleportAction` increments the counter, so the zero is
   real, not an unwired probe.
2. **The transaction machinery costs under 0.4 % of compile time**, and 0.04 %
   in the split configuration. Snapshot creation — two dict copies on every
   candidate teleport, the hottest path — totals 0.04 s over a 349 s run. The
   expensive operation is the O(N) `_rebuild_wdag` a checkpoint restore needs,
   and there is exactly one per deadlock recovery: 0.25 s / 29 in base,
   1.16 s / 124 under the strict floor, 0.09 s / 9 under the split.
3. **`wdag_rebuilds == backup_activations` exactly**, in every condition — a
   useful consistency check that no recovery path skips its checkpoint restore.

The strict floor's cost shows up here too: 66 % more snapshots and 4× the
checkpoint-restore work than base, because it drives the router into recovery
4× as often. The split configuration is *cheaper than the default router* on
every one of these counters.

## 13. Does safe mode always terminate with success?

**Yes, under the hypotheses of §5 — and every one of them is now checked at
`route()` entry**, because a guarantee whose hypotheses go unchecked is not a
guarantee. Violating each of the five raises a `ValueError` naming the problem
rather than failing somewhere inside the transaction:

| hypothesis | check | verified by |
|---|---|---|
| deadlock recovery enabled | `enable_deadlock_recovery` — the guaranteed transaction is reached *through* the deadlock path | rejected ✓ |
| core graph connected | `nx.is_connected(core_graph)` — otherwise a distance lookup on an unreachable core raises `KeyError` | rejected ✓ |
| every core's coupling graph connected | `nx.is_connected(intra[c])` — otherwise a qubit cannot always reach a comm port | (structurally true of all built architectures) |
| `κ_c ≥ core_reserve + 3` | a relay must always find a filler that is not one of the gate's two operands (§4.3) | rejected ✓ |
| `F ≥ core_reserve·K + 1` (F†) | a donor core must always exist | rejected ✓ |
| entry layout satisfies (†) | repaired by `_make_layout_safe`, or rejected if it cannot be | exercised ✓ |

Given those, §5's argument holds: (†) is preserved by every Tier-1 action and
restored by every Tier-2 transaction, so every Tier-2 call meets its
precondition and succeeds; each retires a gate; the gate count strictly
decreases; every former abort path (`ITERATION_LIMIT`, backup exhausted, backup
failed, no candidates) now routes into `_safe_drain` instead.

**Empirically, across every run measured — 64q, 100q so far, 360q, heavy-hex
ring and star, and the deliberately-broken layouts below — `safe_route_failed`
is 0.** Not one guaranteed transaction has ever taken a rollback path.

### 13.1 The two paths the suites never reach, tested directly

**`_make_layout_safe` on a genuinely unsafe entry layout.** Every suite starts
at `free ≥ 2`, so this code had never fired. Forcing 64 qubits onto the 64q
architecture as `[16,16,16,10,4,2]` per core — entry free vector
`[0, 0, 0, 6, 12, 14]`, **three cores completely packed**:

| | EPR | aborted | safe routes | failed | `force_make_room` | exit free |
|---|---|---|---|---|---|---|
| base | 284 | False | — | — | 12 | `[13,4,8,3,3,1]` |
| safe_strict | 283 | False | 5 | **0** | 3 | `[16,3,4,4,3,2]` |
| safe_split | 281 | False | 1 | **0** | 3 | `[11,3,5,6,1,6]` |

All three route it; the safe arms repair the layout first (8 and 6 relay hops)
and exit satisfying their invariant.

**This refines the `force_make_room = 0` claim.** It is 0 *once (†) holds* —
which is the claim §3 makes. Repairing an entry layout that violates (†) runs
teleports from a state with 0-free cores, and those legitimately use
`_force_make_room`: 3 calls here, all inside `_make_layout_safe`. Every run
starting from a valid layout still measures exactly 0.

**Heavy-hex cores, where comm ports *are* cut vertices.** Grid cores have no
articulation points, so no suite exercises the staging retry of §6. On the
27-qubit tile, core 0's ports are 1 and 25 — both articulation points:

| topology | | EPR | aborted | failed | `force_make_room` | min free |
|---|---|---|---|---|---|---|
| ring | base | 189 | False | — | 0 | 2 |
| ring | safe_split | **169** | False | **0** | **0** | 2 |
| star | base | 233 | False | — | 7 | 2 |
| star | safe_split | **186** | False | **0** | **0** | 2 |

Safe mode is *better* on both, and the star — whose hub is a genuine capacity
bottleneck — is where the invariant helps most (−20 %).

### 13.2 What the guarantee still does not cover

- **It is a guarantee about routing, not about cost.** `_safe_drain` bounds
  EPR at `2·diam(G_C)` per remote gate; nothing bounds how bad that is
  relative to a good solution.
- **Architectures below the feasibility line are refused, not degraded.** The
  25-qubit 3-core `F=2<K=3` stress case cannot satisfy (†) at all; safe mode
  raises rather than silently falling back. Those keep the default router.
- **`core_reserve = 1` is now rejected too.** `_safe_route_gate`'s plan relays
  only at the meeting core, which is sound only for `reserve ≥ 2` (§2.1); at
  `reserve = 1` an intermediate core reaches 0 as the mover arrives and can
  strand it. A reserve-1 guarantee needs a relay before *every* hop — a
  different procedure, not a parameter of this one — so the config raises
  rather than silently offering a broken guarantee.

## 14. Open risks

1. **EPR regresses — measured, +20.8 % gmean at 64q and +20.9 % at 360q**
   (§10.3). This was the flagged risk and it materialised, roughly 3× what §8
   suggested. It is Tier 1 substituting second-best moves, not Tier 2 overhead:
   at 64q the guaranteed transaction fires 1–49 times per circuit and the
   regression is undiminished. §10.4 gives the variant that should recover most
   of it, untested.
2. **`free ≥ 3` is a uniform reserve, and the free-slot ablation found the real
   abort frontier is not uniform** (`freeslot-frontier-and-monolithic-completion`:
   central cores want more slack than edge cores). The answer here is the
   separation of concerns — the hard floor of 2 is for *safety* and is uniform
   because the proof needs it uniform; the *quality* shaping stays where it is,
   in `cap_penalty`/`capacity_threshold`, which can still be made
   degree-weighted independently.
3. **(F†) excludes genuinely tight architectures.** The 25-qubit 3-core
   "F=2<K=3" stress case named in `adaptive_corner_count`'s docstring cannot
   satisfy (†) at all. Safe mode must refuse it explicitly rather than silently
   degrade — those architectures keep today's best-effort router, and the
   paper's claim becomes "guaranteed when `P ≥ n + 2K + 1`", which every
   evaluated architecture satisfies.
4. **Tier 2 changes the layout more aggressively than `_backup_plan` did**
   (it relays slack and retires the gate outright). The `ae` 202 → 141 swing
   showed how far a single recovery's downstream effect can propagate through
   fwd→bwd→fwd. Expect per-circuit noise larger than the mechanism's own
   direct cost.
5. **The `route()` state is per-instance, not thread-safe** (unchanged from
   `4ea5d6f`); Tier 2 adds more per-route state, so the caveat gets no better.
