"""Where the benchmark circuits live.

Every driver used to hardcode ``~/Documents/telesabre/circuits``, a directory
that exists only on the author's machine and is shared with several sibling
projects.  That made the repository unreproducible for anyone else: the QASM
files ship in ``benchmark_circuits/`` but nothing looked there.

``circuits_path()`` resolves a suite directory (or a file inside one) against,
in order:

1. ``$DSABRE_CIRCUITS``, if set — an explicit override;
2. ``<repo>/benchmark_circuits`` — the 45 circuits every published table was
   computed from, with provenance and CX counts in its own README;
3. ``~/Documents/telesabre/circuits`` — the author's shared MQT Bench tree,
   which still holds suites the paper does not cite (``qasm_80``,
   ``qasm_fill_sweep``, and the families excluded from the curated copy).

The first root that actually contains the requested path wins, so a reader
gets the curated circuits and the author's exploratory suites keep resolving
to the shared tree.  The 27 MQT Bench files common to roots 2 and 3 were
verified byte-identical on 2026-08-19, so the order is not observable in any
published number.

Generators that *write* circuits (``gen_deep_qnn.py``, ``gen_80q_suite.py``,
``gen_new_64q_circuits.py``, ``gen_regular_cz.py``) deliberately do not use
this helper: they target the shared tree by design, and regenerating a suite
circuit in place is exactly the mistake ``../benchmark_circuits/README.md`` warns about.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: Candidate roots, most specific first.  ``None`` entries are dropped.
ROOTS = [
    os.environ.get("DSABRE_CIRCUITS"),
    os.path.join(_REPO, "benchmark_circuits"),
    os.path.expanduser("~/Documents/telesabre/circuits"),
]
ROOTS = [r for r in ROOTS if r]


def circuits_path(*parts):
    """Return the path to a circuit suite directory, or a file inside one.

    ``circuits_path("qasm_25")`` -> the 25-qubit suite directory.
    ``circuits_path("qasm_25", "qft_...qasm")`` -> one circuit in it.
    ``circuits_path()`` -> the resolved circuits root.

    Falls back to the last root when nothing exists, so a caller that is about
    to create the path still gets a sensible answer instead of an exception.
    """
    rel = os.path.join(*parts) if parts else ""
    for root in ROOTS:
        candidate = os.path.join(root, rel) if rel else root
        if os.path.exists(candidate):
            return candidate
    return os.path.join(ROOTS[-1], rel) if rel else ROOTS[-1]


def describe():
    """One line per candidate root and whether it exists — for diagnostics."""
    return "\n".join(
        "  %s  %s" % ("present" if os.path.isdir(r) else "absent ", r)
        for r in ROOTS
    )


if __name__ == "__main__":
    print("circuit search path:")
    print(describe())
    print("\nresolved suites:")
    for n in (25, 36, 64, 100, 200, 360):
        p = circuits_path("qasm_%d" % n)
        n_files = len([f for f in os.listdir(p) if f.endswith(".qasm")]) \
            if os.path.isdir(p) else 0
        print("  qasm_%-4d %2d circuits  %s" % (n, n_files, p))
