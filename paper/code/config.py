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

    # ── Capacity / congestion scoring ─────────────────────────────────────────
    # Penalty multiplier when a destination core has fewer free slots than threshold.
    cap_penalty: float = 15.0
    capacity_threshold: int = 3
    # Bonus per inter-core hop that moves a qubit closer to its target core.
    hop_gain: float = 5.0

    # ── Proactive congestion relief ────────────────────────────────────────────
    # Score bonus for teleports that relieve a congested core.
    relief_bonus: float = 8.0
    # How many gates ahead to scan when estimating core demand.
    demand_lookahead: int = 5
    # Minimum demand count that triggers congestion relief.
    demand_threshold: int = 1
    # Maximum free slots in the congested core that allow triggering relief.
    congestion_threshold: int = 1
    # Minimum free slots required in the receiving core to accept a relief move.
    relief_space_req: int = 3
    # Weight on victim's next-use depth: higher → prefer moving qubits used furthest in future.
    relief_depth_weight: float = 0.5
    # Weight on busyness gradient between congested and relief core.
    relief_gradient_weight: float = 1.0

    # ── Deadlock recovery ──────────────────────────────────────────────────────
    # Maximum routing iterations before declaring failure.
    max_iterations: int = 10000
    # Consecutive non-progress iterations before activating backup plan.
    deadlock_limit: int = 50
    # Maximum number of backup-plan activations before aborting.
    max_backup_attempts: int = 50

    # ── Mechanism ablation toggles ────────────────────────────────────────────
    # Disable to ablate each mechanism for the §3 contribution analysis.
    enable_congestion_relief: bool = True
    enable_deadlock_recovery: bool = True

    # ── Diagnostics ───────────────────────────────────────────────────────────
    # When True, every SWAP and teleport is appended to metrics["trace"].
    trace_routing: bool = False
