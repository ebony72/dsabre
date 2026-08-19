"""
ablate_telegate.py — does a single-gate telegate help \\dSABRE{} on the 64q suite?

Section VI of the manuscript claims that *scoring* a telegate is easy -- drop
c_cap, keep d_prep, make the progress term the retired gate -- and that the
hard part is amortising one entangled state over a burst.  This driver
measures the easy half so the claim rests on data.

Protocol is the main table's, so the `off` row must reproduce
`results/results_64q.json`'s dSE column exactly:
  9 circuits, H-grid 2x3 of 4x4, `default_layouts` (SabreLayout with adaptive
  per-core corner reservation, 3 seeds), fwd->bwd->fwd, best forward pass by
  EPR over the three layouts.  EPR = teledata + telegate, one pair per
  operation -- the same accounting the paper applies to \\TeleSABRE{}.

Variants
--------
  off       telegate disabled -- the published router, as an anchor.
  nat       the Section VI score verbatim: c_cap dropped, d_prep kept
            (both sides), progress = the gate's current d_phys.
  nodst     as `nat` but charging only the SOURCE-side staging, i.e. the
            most literal reading of "only the port staging its two link
            qubits need"; the far operand's walk to its own endpoint is free.
  bN        `nat` plus a flat score offset N -- how much a telegate must beat
            the best teleport by before it is taken.  b -> infinity recovers
            `off` exactly, which is itself a check on the plumbing.
  amX       `nat` plus X * the extended-set gain a teledata hop of the same
            qubit would have earned -- pricing the amortisation the telegate
            declines rather than discouraging it by a constant.
  ctrl      `nat` restricted to cat-entangling qargs[0] (the CX control), so
            no local Hadamard rewrite of the target is assumed.
  prog0.5   `nat` with the progress term halved.
  sel       selective, not discouraging: a telegate is offered only for gates
            whose cat-entangled operand would gain nothing in the extended
            set by relocating -- i.e. only where teledata has no future to
            amortise over.  sel_b4 adds a bias of 4 on top.
  tgmax     telegate whenever legal (bias -1000).  Not a proposal: it prices
            the primitive on its own, since its EPR count is the number of
            gate executions that had to cross a core boundary.

Suites
------
`--suite 64q` is the paper's H-grid (96 physical for 64 logical, 33% spare).
`--suite 64q_c33` and `--suite 64q_b243` run the SAME nine circuits on chips
tight enough that capacity binds (81 and 72 physical), which is the regime
the dropped c_cap term is supposed to matter in.  Both fall below safe mode's
feasibility line, so they run in score-only mode -- see `base_config`.

Usage:
    python3 ablate_telegate.py                       # everything
    python3 ablate_telegate.py --variants off,nat,b8
    python3 ablate_telegate.py --circuits ae,qft --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ablate_common import (RESULTS_DIR, SUITES, SUITE_CIRCUITS, default_layouts,
                           gmean, load_circuit, meta, run_protocol_ex, save_json)
from telegate_router import dSABRE_Telegate, TelegateConfig

# Recovery budgets are derived per architecture (deadlock_limit_for /
# iterations_bound), exactly as `benchmark.py` does -- no hand-tuned constant.
_BASE = dict(deadlock_limit=None, max_backup_attempts=None, max_iterations=None)

VARIANTS = {
    "off":     dict(telegate=False),
    "nat":     dict(telegate=True),
    "nodst":   dict(telegate=True, telegate_dst_weight=0.0),
    "b2":      dict(telegate=True, telegate_bias=2.0),
    "b4":      dict(telegate=True, telegate_bias=4.0),
    "b6":      dict(telegate=True, telegate_bias=6.0),
    "b8":      dict(telegate=True, telegate_bias=8.0),
    "b12":     dict(telegate=True, telegate_bias=12.0),
    "b16":     dict(telegate=True, telegate_bias=16.0),
    "am0.25":  dict(telegate=True, telegate_amort_weight=0.25),
    "am0.5":   dict(telegate=True, telegate_amort_weight=0.5),
    "am1":     dict(telegate=True, telegate_amort_weight=1.0),
    "am2":     dict(telegate=True, telegate_amort_weight=2.0),
    "ctrl":    dict(telegate=True, telegate_control_only=True),
    "prog0.5": dict(telegate=True, telegate_progress=0.5),
    # Selective rather than discouraging: offer a telegate only for gates
    # whose cat-entangled operand would gain nothing in the extended set by
    # relocating -- the gates a teledata hop has no future to amortise over.
    "sel":     dict(telegate=True, telegate_require_no_lookahead=True),
    "sel_b4":  dict(telegate=True, telegate_require_no_lookahead=True,
                    telegate_bias=4.0),
    # Telegate whenever one is legal.  Not a proposal -- it measures the cost
    # of the primitive on its own: EPR here is the number of gate executions
    # that had to cross a core boundary, which is what a telegate-only router
    # would pay and what teledata's amortisation is competing against.
    "tgmax":   dict(telegate=True, telegate_bias=-1000.0),
}

# Architectures too tight for safe mode's feasibility line
# (P - n >= core_reserve*K + 1).  There `route()` raises rather than
# degrading, so these suites are run in score-only mode with the explicit
# recovery budgets `ablate_common` already uses for them -- which is what
# keeps these rows comparable with the other ablation tables, and is the only
# reason the budgets are spelled out here.  (Until 2026-08-17 they were also
# load-bearing: `max_backup_attempts=None` raised TypeError from route()'s
# cap test outside safe mode.  It now means "do not cap", as for
# `max_iterations`, so the Nones would run -- they would just route these
# suites under different budgets than the published rows.)
_SCORE_ONLY = dict(deadlock_limit=100, max_backup_attempts=100,
                   max_iterations=20000, safe_mode=False)


def base_config(suite: str, arch) -> dict:
    """`_BASE`, unless this architecture cannot satisfy safe mode."""
    n = SUITES[suite]["n_qubits"]
    free = len(arch.data_qubits) - n
    if free >= 2 * arch.num_cores + 1:
        return dict(_BASE)
    return dict(_SCORE_ONLY)


def run_one(arch, cfg, dag, rev_dag, layouts):
    """Best-of-layouts fwd->bwd->fwd, with telegate counters summed over
    EVERY pass (the reported metrics dict only covers the winning forward
    pass, but a telegate taken in the backward pass still shapes the layout
    pass 3 starts from, so it must be visible)."""
    best, total_s = None, 0.0
    tg_all = td_all = 0
    fi_all = fa_all = 0
    for L in layouts:
        r = dSABRE_Telegate(arch, cfg)
        t0 = time.perf_counter()
        m, info = run_protocol_ex(r, dag, rev_dag, L, "fbf")
        total_s += time.perf_counter() - t0
        tg_all += r._tg_all_telegates
        td_all += r._tg_all_teledata
        fi_all += r._tg_all_front_inter
        fa_all += r._tg_all_front_adj
        if m is not None and (best is None or m["eprs"] < best["eprs"]):
            best = m
    if best is None:
        return dict(aborted=True, time_s=round(total_s, 2))
    return dict(
        aborted=False,
        eprs=best["eprs"], ls=best["ls"],
        telegates=best.get("telegates", 0),
        teledata=best.get("teledata", 0),
        time_s=round(total_s, 2),
        backup_activations=best.get("backup_activations", 0),
        safe_routes=best.get("safe_routes", 0),
        # Whole-protocol totals (all passes, all layouts).
        all_telegates=tg_all, all_teledata=td_all,
        front_inter=fi_all, front_adj=fa_all,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="64q")
    ap.add_argument("--circuits", default=None)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--layouts", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s = SUITES[args.suite]
    arch = s["arch"]
    # `64q_c33` / `64q_b243` are the same nine 64-qubit circuits on tighter
    # chips, so they inherit the 64q circuit list.
    circuits = (args.circuits.split(",") if args.circuits
                else SUITE_CIRCUITS.get(args.suite, SUITE_CIRCUITS["64q"]))
    vnames = (args.variants.split(",") if args.variants else list(VARIANTS))
    out = args.out or os.path.join(RESULTS_DIR, f"telegate_{args.suite}.json")
    base = base_config(args.suite, arch)

    payload = dict(
        meta=meta("telegate macro-action (single gate, one hop)", args.suite,
                  circuits=circuits, variants=vnames,
                  routed_by="dSE + telegate (telegate_router.dSABRE_Telegate)",
                  protocol=f"default_layouts x{args.layouts}, fwd-bwd-fwd, "
                           f"best forward pass by EPR",
                  epr_accounting="1 EPR pair per teledata hop AND per telegate",
                  mode=("safe" if base.get("safe_mode", True) else "score-only"),
                  budgets={k: base[k] for k in
                           ("deadlock_limit", "max_backup_attempts",
                            "max_iterations")}),
        results={},
    )

    print(f"suite {args.suite}: {len(circuits)} circuits x {len(vnames)} variants",
          flush=True)
    hdr = f"{'circuit':11s} {'variant':8s} {'eprs':>6s} {'ls':>6s} {'tg':>5s} {'td':>5s} {'adj%':>6s} {'t_s':>7s}"
    for cname in circuits:
        qc, dag, rev_dag, ncx = load_circuit(args.suite, cname)
        layouts = default_layouts(qc, dag, arch, s["n_qubits"], seed=0,
                                  n_seeds=args.layouts)
        print(f"\n{hdr}\n{'-'*len(hdr)}", flush=True)
        for v in vnames:
            cfg = TelegateConfig(**base, **VARIANTS[v])
            row = run_one(arch, cfg, dag, rev_dag, layouts)
            payload["results"].setdefault(cname, {})[v] = row
            save_json(out, payload)                    # partial runs are usable
            if row["aborted"]:
                print(f"{cname:11s} {v:8s} {'ABORT':>6s}", flush=True)
                continue
            fi, fa = row["front_inter"], row["front_adj"]
            print(f"{cname:11s} {v:8s} {row['eprs']:6d} {row['ls']:6d} "
                  f"{row['telegates']:5d} {row['teledata']:5d} "
                  f"{(100.0*fa/fi if fi else 0):5.1f}% {row['time_s']:7.1f}",
                  flush=True)

    # ── Summary: geometric mean of the per-circuit EPR ratio against `off` ──
    print(f"\n{'='*64}\nEPR geometric mean vs `off` ({len(circuits)} circuits)\n{'='*64}",
          flush=True)
    base = {c: payload["results"][c]["off"] for c in circuits
            if "off" in payload["results"].get(c, {})}
    summary = {}
    for v in vnames:
        ratios, eprs, tg = [], [], 0
        ok = True
        for c in circuits:
            row = payload["results"].get(c, {}).get(v)
            if row is None or row["aborted"] or c not in base or base[c]["aborted"]:
                ok = False
                continue
            ratios.append(row["eprs"] / base[c]["eprs"])
            eprs.append(row["eprs"])
            tg += row["all_telegates"]
        if not ratios:
            continue
        g = gmean(ratios)
        summary[v] = dict(gmean_ratio=round(g, 4),
                          pct=round(100 * (g - 1), 2),
                          total_eprs=sum(eprs),
                          telegates_all_passes=tg,
                          complete=ok)
        print(f"  {v:8s} {100*(g-1):+7.2f}%   total EPR {sum(eprs):6d}   "
              f"telegates(all passes) {tg:6d}", flush=True)
    payload["summary"] = summary
    save_json(out, payload)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
