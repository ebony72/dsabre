"""Instrument TeleSABRE's decision stream.

Answers two questions from its own debug output:
  Q1  In iterations where the (post-drain) front still contains a same-core 2Q
      gate -- dSABRE's F_intra, the case where dSABRE is *forbidden* to
      teleport -- how often does TeleSABRE apply a teleport/telegate anyway?
  Q2  What is the energy separation between SWAP and TELEPORT candidates?

Streams stdout so large circuits do not need to be buffered.
"""
import re, os, sys, json, subprocess, statistics as st
from circuit_paths import circuits_path

TS   = os.path.expanduser("~/Documents/telesabre/telesabre")
DEV  = os.path.expanduser("~/Documents/telesabre/devices/%s.json")
CIRC = circuits_path("qasm_%d/%s_nativegates_ibm_qiskit_opt3_%d.qasm")

RE_ANSI  = re.compile(r"\x1b\[[0-9;]*m")
RE_FRONT = re.compile(r"\(\s*\d+\):\s*Virt:\s*((?:\s*\d+)+?)\s*-\s*"
                      r"Phys:\s*((?:\s*\d+)+?)\s*-\s*Cores:\s*((?:\s*\d+)+)\s*$")
RE_ENER  = re.compile(r"Type:\s*(\w+).*?Energy:\s*(-?[\d.]+)")
RE_APPL  = re.compile(r"Applied operation:\s*(\w+)\(([^)]*)\)")
RE_ITER  = re.compile(r"Iteration\s+(\d+)")


def run(circuit, device, seed=1, timeout=1800):
    cfg = {"config": {
        "name": "probe", "seed": seed, "energy_type": "extended-set",
        "usage_penalties_reset_interval": 5, "optimize_initial": True,
        "initial_layout_type": "hungarian", "teleport_bonus": 100,
        "telegate_bonus": 100, "safety_valve_iters": 100, "extended_set_size": 20,
        "extended_set_factor": 0.05, "inter_core_edge_weight": 2,
        "full_core_penalty": 10, "max_solving_deadlock_iterations": 1000,
        "gate_usage_penalty": 0.0, "swap_usage_penalty": 0.002,
        "teledata_usage_penaly": 0.005, "telegate_usage_penalty": 0.005,
        "init_layout_hun_min_free_gate": 5, "init_layout_hun_min_free_qubit": 4,
        "enable_passing_core_emptying_teleport_possibility": False,
        "max_iterations": 200000, "save_report": False,
        "required_successes": 1, "max_attempts": 1}}
    cfgp = "/tmp/_ts_probe_cfg.json"
    json.dump(cfg, open(cfgp, "w"))

    st_ = {"iters": 0, "intra_present": 0, "intra_and_comm": 0,
           "comm_total": 0, "teleports": 0, "telegates": 0, "splits_local_pair": 0,
           "energy": {"SWAP": [], "TELEPORT": [], "TELEGATE": []},
           "examples": []}
    # front: list of (virt1, virt2, phys1, phys2, core1, core2) for gates still
    # unexecutable after the drain phase.  core1==core2 is exactly dSABRE's F_intra.
    state = {"front": [], "it": -1, "applied": None, "args": (), "open": False}

    def close_iteration():
        if not state["open"]:
            return
        intra = [g for g in state["front"] if g[4] == g[5]]
        if intra:
            st_["intra_present"] += 1
        if state["applied"] in ("Teleport", "Telegate"):
            st_["comm_total"] += 1
            if state["applied"] == "Telegate":
                st_["telegates"] += 1
            if intra:
                st_["intra_and_comm"] += 1
        # STRONG FORM: a Teleport moves the qubit sitting on args[0].  Since a
        # logical qubit can be in at most one front gate, if that physical qubit
        # is an operand of a same-core front gate then this teleport splits a
        # pair that was about to become executable locally.
        if state["applied"] == "Teleport" and state["args"]:
            st_["teleports"] += 1
            src = state["args"][0]
            for (v1, v2, p1, p2, c1, c2) in intra:
                if src in (p1, p2):
                    st_["splits_local_pair"] += 1
                    if len(st_["examples"]) < 6:
                        moved, stay = ((v1, p1), (v2, p2)) if src == p1 else ((v2, p2), (v1, p1))
                        st_["examples"].append(
                            {"it": state["it"], "gate": (v1, v2), "core": c1,
                             "moved_virt": moved[0], "moved_phys": moved[1],
                             "partner_virt": stay[0], "partner_phys": stay[1]})
                    break
        state["open"] = False

    p = subprocess.Popen([TS, cfgp, device, circuit], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        for raw in p.stdout:
            line = RE_ANSI.sub("", raw)
            m = RE_ITER.search(line)
            if m:
                close_iteration()          # close the previous one first
                state.update(front=[], it=int(m.group(1)), applied=None, open=True)
                st_["iters"] += 1
                continue
            m = RE_FRONT.search(line)
            if m:
                v, ph, c = (m.group(i).split() for i in (1, 2, 3))
                if len(v) == 2 and len(ph) == 2 and len(c) == 2:
                    state["front"].append((int(v[0]), int(v[1]), int(ph[0]),
                                           int(ph[1]), int(c[0]), int(c[1])))
                continue
            m = RE_ENER.search(line)
            if m and m.group(1) in st_["energy"]:
                st_["energy"][m.group(1)].append(float(m.group(2)))
                continue
            m = RE_APPL.search(line)
            if m:
                state["applied"] = m.group(1)
                state["args"] = tuple(int(x) for x in m.group(2).replace(",", " ").split())
        close_iteration()                  # and the last
    finally:
        p.stdout.close()
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
    return st_


if __name__ == "__main__":
    n = int(sys.argv[1]); dev = sys.argv[2]; names = sys.argv[3].split(",")
    agg = {"SWAP": [], "TELEPORT": [], "TELEGATE": []}
    rows = []
    for nm in names:
        c = CIRC % (n, nm, n)
        if not os.path.exists(c):
            print(f"  skip {nm} (missing)", flush=True); continue
        r = run(c, DEV % dev)
        for k in agg:
            agg[k] += r["energy"][k]
        rows.append((nm, r))
        pct = 100.0 * r['splits_local_pair'] / r['teleports'] if r['teleports'] else 0.0
        print(f"  {nm:11s} iters={r['iters']:6d}  intra-front={r['intra_present']:6d}  "
              f"comm={r['comm_total']:5d}  while-intra={r['intra_and_comm']:5d}  "
              f"|  teledata={r['teleports']:5d}  telegate={r['telegates']:5d} "
              f"({100.0*r['telegates']/r['comm_total'] if r['comm_total'] else 0:.1f}% of comm)"
              f"  split-local-pair={r['splits_local_pair']:3d} ({pct:.1f}%)", flush=True)
        for ex in r["examples"][:2]:
            print(f"      e.g. it {ex['it']}: front gate ({ex['gate'][0]},{ex['gate'][1]}) both in "
                  f"core {ex['core']} -- teleported q{ex['moved_virt']} (p{ex['moved_phys']}) away "
                  f"from partner q{ex['partner_virt']} (p{ex['partner_phys']})", flush=True)

    print("\n=== energy by op type (pooled) ===", flush=True)
    for k, v in agg.items():
        if v:
            print(f"  {k:9s} n={len(v):7d}  min {min(v):9.2f}  median {st.median(v):9.2f}  max {max(v):9.2f}")
    if agg["SWAP"] and agg["TELEPORT"]:
        print(f"\n  max(TELEPORT) < min(SWAP)?  {max(agg['TELEPORT']) < min(agg['SWAP'])}")
    tot_i = sum(r['intra_present'] for _, r in rows)
    tot_c = sum(r['intra_and_comm'] for _, r in rows)
    tot_a = sum(r['comm_total'] for _, r in rows)
    tot_t = sum(r['teleports'] for _, r in rows)
    tot_g = sum(r['telegates'] for _, r in rows)
    tot_s = sum(r['splits_local_pair'] for _, r in rows)
    print(f"\n  WEAK  form -- comm ops issued while some same-core front gate was pending: "
          f"{(100.0*tot_c/tot_a if tot_a else 0):.1f}%  ({tot_c}/{tot_a})")
    print(f"  STRONG form -- teleports that moved an OPERAND of such a gate away from its "
          f"partner: {(100.0*tot_s/tot_t if tot_t else 0):.1f}%  ({tot_s}/{tot_t} teleports)")
    print(f"  TELEGATE  -- applied {tot_g} of {tot_a} communication ops "
          f"({(100.0*tot_g/tot_a if tot_a else 0):.1f}%); dSABRE's action set cannot express these")

    out = {"suite": n, "device": dev,
           "per_circuit": {nm: {k: r[k] for k in
                                ("iters", "intra_present", "comm_total", "intra_and_comm",
                                 "teleports", "telegates", "splits_local_pair")}
                           for nm, r in rows},
           "examples": {nm: r["examples"] for nm, r in rows},
           "energy": {k: ({"n": len(v), "min": min(v), "median": st.median(v), "max": max(v)}
                          if v else None) for k, v in agg.items()},
           "totals": {"iters_with_intra_front": tot_i, "comm_while_intra": tot_c,
                      "comm_total": tot_a, "teleports": tot_t, "telegates": tot_g,
                      "teleports_splitting_local_pair": tot_s}}
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", f"telesabre_action_probe_{n}q.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\n  wrote {dest}")
