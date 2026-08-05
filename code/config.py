from dataclasses import dataclass


@dataclass
class HardwareConfig:
    """Routing hyperparameters and cost model for a distributed quantum processor.

    Cost model
    ----------
    A routing solution's total cost is:
        cost = n_local_swaps * cost_local_swap
             + n_teleports   * (cost_teleport + cost_teleport_per_hop * hops)

    All default values are tuned for the B-grid 2x2 4x4 architecture (25q/36q).
    For larger circuits (64q+) increase deadlock_limit, max_backup_attempts, and
    max_iterations proportionally.
    """

    # ── Operation costs ────────────────────────────────────────────────────────
    cost_local_swap: float = 3.0
    # Base EPR pairs consumed per teleportation (distance-independent).
    cost_teleport: float = 10.0
    # Additional EPR pairs per inter-core hop (set to 0 for flat cost model).
    cost_teleport_per_hop: float = 0.0

    # ── Heuristic scoring weights ──────────────────────────────────────────────
    # Weight of the extended-set lookahead term in teleport candidate scoring.
    weight_extended: float = 0.25
    # Number of gates in the extended lookahead window beyond the front layer.
    lookahead_size: int = 20
    # Per-gate exponential decay applied to extended-layer gates (1.0 = flat).
    lookahead_decay: float = 0.9
    # Soft score discount for candidates that continue teleporting the gate the
    # PREVIOUS iteration chose to advance (0.0 = off, matches all prior results).
    # 2026-08-03 trace of the 64q suite found the router re-scores every
    # competing inter-core gate from scratch each iteration with no memory of
    # which one it was mid-move on: 90.8% of gates needing >=2 teleport hops had
    # another gate's teleport served in between, and 35.6% saw their own
    # qubit-pair distance actually INCREASE while parked (98.6% of those from a
    # DIFFERENT gate's eviction side effect, not their own). commit_bonus makes
    # the currently-committed gate's candidates cheaper by this amount so it
    # tends to win ties/near-ties against competing gates, without making it
    # unconditionally win against a genuinely better move elsewhere.
    commit_bonus: float = 0.0
    # Hard variant of the same idea: while the gate committed via commit_bonus'
    # bookkeeping still has at least one legal teleport candidate, competing
    # gates' candidates are not even generated -- only the committed gate's
    # own next hop can be chosen.  If the committed gate has NO legal move
    # this iteration (e.g. every neighbouring core is full), the lock releases
    # for that iteration only and normal all-gates scoring applies -- a
    # temporarily-stuck lock falls back to congestion relief rather than
    # deadlocking.  Independent of commit_bonus (default 0.0 = off; combining
    # a nonzero commit_bonus with hard_lock=True has no additional effect,
    # since no competitor ever enters the list for the bonus to out-rank).
    commit_hard_lock: bool = False
    # Extra score PER UNIT of a candidate's own gate's CURRENT qubit-pair
    # phys_dist (before this hop): 0.0 = off, matches all prior results.
    # Independent of commit_bonus/commit_hard_lock -- this prioritizes by how
    # close a gate already is to resolution, not by whichever gate moved last.
    # Motivation: the 2026-08-03 trace found that once given a turn, most
    # multi-hop gates resolve in ONE further hop (their distance was already
    # small the whole time they sat waiting) -- so a positive weight lets an
    # almost-finished gate's cheap last hop cut ahead of a farther gate's
    # expensive one, rather than queuing behind it in arrival/score order.
    cheapest_first_weight: float = 0.0
    # When True, `_evict` prefers a free slot that does not increase the
    # evicted qubit's OWN pending front-layer gate's phys_dist to its partner,
    # falling back to nearest-free-slot only among ties or when the evicted
    # qubit has no pending partner gate.  Off by default -- matches all prior
    # results.  Targets the eviction side effect directly: of the "distance
    # backslides" the 2026-08-03 trace measured on the 64q suite, 97-99% were
    # exactly this -- a DIFFERENT gate's teleport (or force_make_room) evicting
    # a bystander qubit toward whichever free slot was nearest to the vacated
    # comm port, with no regard for what that did to the bystander's own
    # pending gate.
    evict_distance_aware: bool = False

    # ── Capacity / congestion scoring ─────────────────────────────────────────
    # Penalty multiplier when a destination core has fewer free slots than threshold.
    cap_penalty: float = 15.0
    capacity_threshold: int = 3

    # ── Deadlock recovery ──────────────────────────────────────────────────────
    # Maximum routing iterations before declaring failure.
    max_iterations: int = 10000
    # Consecutive non-progress iterations before activating backup plan.
    deadlock_limit: int = 50
    # Maximum number of backup-plan activations before aborting.
    max_backup_attempts: int = 50
    # When True, _backup_plan's cross-core hops use _relay_room_to (BFS relay
    # of a genuine free-slot surplus along the core graph) to guarantee every
    # core keeps >=1 free qubit before each forced hop, instead of the default
    # greedy loop's `_force_make_room`, which only looks at DIRECT neighbours
    # of the destination core and gives up (silently making no progress that
    # backup_plan call) if none of them has room. False (default) matches all
    # prior results. See General_dSABRE_Router._relay_room_to's docstring for
    # the termination/invariant argument -- provided every core has >=1 free
    # and the total exceeds the core count by >=1 when backup_plan is first
    # invoked, this mode is guaranteed to make progress without ever dropping
    # a core to 0 free.
    backup_relay_mode: bool = False

    # ── Mechanism ablation toggles ────────────────────────────────────────────
    # Disable to ablate each mechanism for the §3 contribution analysis.
    enable_deadlock_recovery: bool = True

    # ── Diagnostics ───────────────────────────────────────────────────────────
    # When True, every SWAP and teleport is appended to metrics["trace"].
    trace_routing: bool = False
