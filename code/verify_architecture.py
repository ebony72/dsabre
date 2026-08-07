"""Verify the default hierarchical phys_dist reproduces the dense one exactly.

Compares the composed distance against `architecture.py`'s all-pairs Dijkstra
for EVERY ordered pair of physical qubits, on every architecture the paper
uses, and reports the setup-time and space difference.

Usage:  python3 verify_architecture.py
"""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture import (build_b_grid_architecture, build_h_grid_architecture,
                          build_heavy_hex_architecture)
import _baseline_architecture as _base

_BASE = {build_b_grid_architecture: _base.build_b_grid_architecture,
         build_h_grid_architecture: _base.build_h_grid_architecture,
         build_heavy_hex_architecture: _base.build_heavy_hex_architecture}

CASES = [
    ("B-grid 2x2 4x4   (25q/36q)", build_b_grid_architecture, dict(r=2, s=2, m=4)),
    ("H-grid 2x3 4x4   (64q)",     build_h_grid_architecture, dict(r=2, s=3, m=4)),
    ("H-grid 2x3 5x5   (100q)",    build_h_grid_architecture, dict(r=2, s=3, m=5)),
    ("H-grid 4x3 5x5   (200q)",    build_h_grid_architecture, dict(r=4, s=3, m=5)),
    ("H-grid 2x3 9x9   (360q)",    build_h_grid_architecture, dict(r=2, s=3, m=9)),
    ("heavy-hex ring x4",          build_heavy_hex_architecture,
     dict(num_cores=4, core_graph="ring")),
    ("heavy-hex star x4",          build_heavy_hex_architecture,
     dict(num_cores=4, core_graph="star")),
]


def run(label, builder, kw):
    t0 = time.perf_counter()
    dense = _BASE[builder](**kw)
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = builder(**kw)
    t_fast = time.perf_counter() - t0

    qubits = list(dense.Gr.nodes())
    P = len(qubits)
    bad = 0
    first = None
    for p in qubits:
        row = dense.phys_dist.get(p, {})
        frow = fast.phys_dist.get(p, {})
        for q in qubits:
            a = row.get(q, 999)
            b = frow.get(q, 999)
            if a != b:
                bad += 1
                if first is None:
                    first = (p, q, a, b)

    dm = dense.memory_report()
    fm = fast.memory_report()
    print(f"{label:<28} P={P:>4}  pairs={P*P:>7}  mismatches={bad:<6} "
          f"setup {t_dense:6.3f}s -> {t_fast:6.3f}s ({t_dense/max(t_fast,1e-9):4.1f}x)  "
          f"tables {dm['total_bytes']/1024:8.0f}KB -> {fm['total_bytes']/1024:7.0f}KB "
          f"({dm['total_bytes']/max(fm['total_bytes'],1):4.1f}x)  "
          f"cached={fm['phys_dist_cached']}", flush=True)
    if first:
        print(f"    ! first mismatch: d({first[0]},{first[1]}) "
              f"dense={first[2]} hierarchical={first[3]}", flush=True)
    return bad


if __name__ == "__main__":
    total = 0
    for label, builder, kw in CASES:
        total += run(label, builder, kw)
    print()
    if total:
        print(f"FAILED: {total} mismatched pairs")
        sys.exit(1)
    print("PASS: hierarchical d_phys is exact on every pair of every architecture.")
