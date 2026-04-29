"""
Profiling utilities for dsabre.

  python profiler.py            # sweep several architecture sizes
  python profiler.py --full     # larger sweep including routing pass timing
"""
from __future__ import annotations

import argparse
import time
import sys
from typing import List, Tuple


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def profile_architecture(r: int, s: int, m: int,
                         builder_name: str = "basic") -> dict:
    """
    Build an architecture, measure wall-clock time and estimated memory.

    Returns a dict with timing and memory breakdown.
    """
    from architecture import (
        build_basic_grid_architecture,
        build_b_grid_architecture,
        build_h_grid_architecture,
    )

    builders = {
        "basic": build_basic_grid_architecture,
        "b":     build_b_grid_architecture,
        "h":     build_h_grid_architecture,
    }
    build_fn = builders[builder_name]

    t0 = time.perf_counter()
    arch = build_fn(r, s, m)
    build_time = time.perf_counter() - t0

    mem = arch.memory_report()
    mem["build_time_ms"] = build_time * 1000
    mem["builder"] = builder_name
    mem["r"] = r
    mem["s"] = s
    mem["m"] = m
    return mem


def print_architecture_sweep(configs: List[Tuple[int, int, int]],
                              builder: str = "basic") -> None:
    header = (
        f"{'r×s×m':<12} {'qubits':>7} {'cores':>6} "
        f"{'build(ms)':>10} {'phys_dist':>10} "
        f"{'intra_dist':>11} {'total_est':>11}"
    )
    print(header)
    print("-" * len(header))
    for r, s, m in configs:
        d = profile_architecture(r, s, m, builder)
        label = f"{r}×{s}×{m}²"
        print(
            f"{label:<12} {d['num_qubits']:>7} {d['num_cores']:>6} "
            f"{d['build_time_ms']:>10.1f} "
            f"{_fmt_bytes(d['phys_dist_bytes_est']):>10} "
            f"{_fmt_bytes(d['intra_dist_bytes_est']):>11} "
            f"{_fmt_bytes(d['total_bytes_est']):>11}"
        )


def profile_routing(circuit_path: str,
                    r: int = 2, s: int = 2, m: int = 4,
                    router_name: str = "dsabre") -> dict:
    """
    Load a QASM circuit, build the architecture, run one routing pass, and
    return combined timing and metric data.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_dag
    except ImportError:
        raise RuntimeError("Qiskit is required for profile_routing")

    from architecture import build_basic_grid_architecture
    from config import HardwareConfig
    from router import General_dSABRE_Router
    from burst_router import BurstDSABRE
    import random

    t_load = time.perf_counter()
    qc  = QuantumCircuit.from_qasm_file(circuit_path)
    dag = circuit_to_dag(qc)
    load_time = time.perf_counter() - t_load

    t_arch = time.perf_counter()
    arch = build_basic_grid_architecture(r, s, m)
    arch_time = time.perf_counter() - t_arch

    phys = list(arch.data_qubits)
    rng = random.Random(42)
    rng.shuffle(phys)
    layout = {lq: p for lq, p in zip(dag.qubits, phys[:qc.num_qubits])}

    config = HardwareConfig()
    if router_name == "burst":
        router = BurstDSABRE(arch, config)
    else:
        router = General_dSABRE_Router(arch, config)

    t_route = time.perf_counter()
    metrics, _ = router.route(dag, layout)
    route_time = time.perf_counter() - t_route

    return {
        "circuit":      circuit_path,
        "n_qubits":     qc.num_qubits,
        "n_gates":      dag.count_ops().get("cx", 0),
        "router":       router_name,
        "arch":         f"{r}×{s}×{m}²",
        "load_ms":      load_time  * 1000,
        "arch_build_ms":arch_time  * 1000,
        "route_ms":     route_time * 1000,
        **{k: v for k, v in metrics.items()
           if k not in ("trace", "failure_log")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dsabre profiler")
    parser.add_argument("--full", action="store_true",
                        help="Include larger architecture sizes")
    parser.add_argument("--builder", default="basic",
                        choices=["basic", "b", "h"])
    args = parser.parse_args()

    configs = [
        (1, 1, 4),
        (2, 2, 4),
        (2, 2, 8),
        (4, 4, 4),
    ]
    if args.full:
        configs += [
            (4, 4, 8),
            (8, 8, 4),
            (4, 4, 16),
        ]

    print(f"\nArchitecture build time & estimated distance-table memory "
          f"(builder={args.builder})\n")
    print_architecture_sweep(configs, builder=args.builder)

    print("\nNote: phys_dist scales as O(n²) in both time and space.")
    print("For n > 1000 qubits, consider restricting lookups to intra_dist +")
    print("core_dist and computing cross-core paths lazily.\n")
