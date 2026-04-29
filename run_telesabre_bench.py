"""
Benchmark TeleSABRE on the same 6 MQT-Bench 25-qubit circuits and same
2×2 grid of 4×4-core architecture used for the BurstDSABRE evaluation.

All TeleSABRE code is inlined here (copied from telesabre_py1.py) so that
the module-level `router.run(...)` in the original file does NOT fire on
import.

Metrics reported:
  EPR_best  — best teleport count over NUM_SEEDS trials (= EPR pairs consumed)
  SWAPs     — corresponding intra-core SWAP count
  remain    — unscheduled gates (0 = success)
"""

import json, heapq, re, random, glob, os, io, contextlib
from copy import deepcopy

# ── Architecture description ───────────────────────────────────────────────
# 4 cores in a 2×2 grid, each a 4×4 qubit grid.
# Matches build_b_grid_architecture(r=2, s=2, m=4) in dsabre/architecture.py.
# Inter-core links use LOCAL qubit IDs within each core (0–15).
_INTRA_4x4 = [
    (0,1),(1,2),(2,3),
    (4,5),(5,6),(6,7),
    (8,9),(9,10),(10,11),
    (12,13),(13,14),(14,15),
    (0,4),(4,8),(8,12),
    (1,5),(5,9),(9,13),
    (2,6),(6,10),(10,14),
    (3,7),(7,11),(11,15),
]
_DEVICE_DICT = {
    "name": "b_grid_2_2_4_4",
    "num_cores": 4,
    "cores": [{"id": c, "num_qubits": 16, "couplings": list(_INTRA_4x4)}
              for c in range(4)],
    "inter_core_links": [
        {"core0": 0, "qubit0":  7, "core1": 1, "qubit1":  4},
        {"core0": 0, "qubit0": 13, "core1": 2, "qubit1":  1},
        {"core0": 1, "qubit0": 14, "core1": 3, "qubit1":  2},
        {"core0": 2, "qubit0": 11, "core1": 3, "qubit1":  8},
    ],
}

# ── TeleSABRE data structures (inlined from telesabre_py1.py) ──────────────

class Gate:
    def __init__(self, idx, name, qubits):
        self.idx = idx; self.name = name; self.qubits = qubits
    def __repr__(self): return f"Gate({self.idx},{self.name},{self.qubits})"


class Device:
    def __init__(self, d):
        self.num_cores        = d["num_cores"]
        self.cores            = d["cores"]
        self.inter_core_links = d["inter_core_links"]
        self.core_offsets = []
        offset = 0
        for c in self.cores:
            self.core_offsets.append(offset)
            offset += c["num_qubits"]
        self.num_qubits = offset
        self.qubit_core = {}
        for ci, c in enumerate(self.cores):
            for lq in range(c["num_qubits"]):
                self.qubit_core[self.core_offsets[ci] + lq] = ci
        self.couplings = set()
        for ci, c in enumerate(self.cores):
            off = self.core_offsets[ci]
            for (a, b) in c["couplings"]:
                self.couplings.add((off+a, off+b))
                self.couplings.add((off+b, off+a))
        self.comm_qubits = set()
        self.inter_links  = []
        for link in self.inter_core_links:
            c0 = link["core0"]; q0 = self.core_offsets[c0] + link["qubit0"]
            c1 = link["core1"]; q1 = self.core_offsets[c1] + link["qubit1"]
            self.comm_qubits.update([q0, q1])
            self.inter_links.append((q0, q1))
        INF = 10**9
        N   = self.num_qubits
        D   = [[INF]*N for _ in range(N)]
        for i in range(N): D[i][i] = 0
        for (a, b) in self.couplings:
            D[a][b] = 1
        for k in range(N):
            for i in range(N):
                if D[i][k] == INF: continue
                for j in range(N):
                    if D[i][k] + D[k][j] < D[i][j]:
                        D[i][j] = D[i][k] + D[k][j]
        self.D = D

    def core_qubits(self, core_id):
        off = self.core_offsets[core_id]
        return list(range(off, off + self.cores[core_id]["num_qubits"]))


class Layout:
    def __init__(self, num_virt, num_phys):
        self.v2p = [-1]*num_virt; self.p2v = [-1]*num_phys
    def assign(self, v, p): self.v2p[v] = p; self.p2v[p] = v
    def swap(self, p1, p2):
        v1, v2 = self.p2v[p1], self.p2v[p2]
        self.p2v[p1] = v2; self.p2v[p2] = v1
        if v1 != -1: self.v2p[v1] = p2
        if v2 != -1: self.v2p[v2] = p1
    def teleport(self, p_src, p_cs, p_cd):
        v = self.p2v[p_src]
        self.p2v[p_src] = -1
        if v != -1: self.v2p[v] = -1
        vc = self.p2v[p_cs]
        if vc != -1: self.p2v[p_cs] = -1; self.v2p[vc] = -1
        old = self.p2v[p_cd]
        if old != -1: self.v2p[old] = -1
        self.p2v[p_cd] = v
        if v != -1: self.v2p[v] = p_cd
    def copy(self):
        l = Layout(len(self.v2p), len(self.p2v))
        l.v2p = self.v2p[:]; l.p2v = self.p2v[:]
        return l


def parse_qasm(path):
    gates = []; num_qubits = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"qreg\s+\w+\[(\d+)\]", line)
            if m: num_qubits = int(m.group(1)); continue
            m = re.match(r"(cx)\s+\w+\[(\d+)\],\w+\[(\d+)\]", line)
            if m: gates.append(Gate(len(gates), m.group(1), [int(m.group(2)), int(m.group(3))])); continue
            m = re.match(r"(rz|rx|ry|p|u1|u2|u3)\(.*?\)\s+\w+\[(\d+)\]", line)
            if m: gates.append(Gate(len(gates), m.group(1), [int(m.group(2))])); continue
            m = re.match(r"(sx|x|h|s|sdg|t|tdg|id)\s+\w+\[(\d+)\]", line)
            if m: gates.append(Gate(len(gates), m.group(1), [int(m.group(2))]))
    return num_qubits, gates


def initial_layout(num_virt, device, gates, seed=0):
    random.seed(seed)
    layout = Layout(num_virt, device.num_qubits)
    core_free = [list(device.core_qubits(ci)) for ci in range(device.num_cores)]
    for c in core_free: random.shuffle(c)
    core_ptr = [0]*device.num_cores
    assigned = [False]*num_virt
    cur_core = 0
    for g in gates:
        if len(g.qubits) == 2:
            v1, v2 = g.qubits
            if not assigned[v1] and not assigned[v2]:
                c = cur_core % device.num_cores
                if core_ptr[c]+1 < len(core_free[c]):
                    p1 = core_free[c][core_ptr[c]]; core_ptr[c] += 1
                    p2 = core_free[c][core_ptr[c]]; core_ptr[c] += 1
                    layout.assign(v1, p1); layout.assign(v2, p2)
                    assigned[v1] = assigned[v2] = True
                    cur_core += 1
    ci = 0
    for v in range(num_virt):
        if not assigned[v]:
            while ci < device.num_cores and core_ptr[ci] >= len(core_free[ci]):
                ci += 1
            if ci < device.num_cores:
                layout.assign(v, core_free[ci][core_ptr[ci]])
                core_ptr[ci] += 1
    return layout


def _nearest_free(pcomm, layout, device):
    cid = device.qubit_core[pcomm]; best = 10**9
    for pq in device.core_qubits(cid):
        if layout.p2v[pq] == -1: best = min(best, device.D[pcomm][pq])
    return best

def _full_penalty(cid, layout, device, thr=1, pen=10):
    free = sum(1 for pq in device.core_qubits(cid) if layout.p2v[pq] == -1)
    return pen if free <= thr else 0

def _gate_energy(g, layout, device):
    if len(g.qubits) < 2: return 0.0
    v1, v2 = g.qubits; p1, p2 = layout.v2p[v1], layout.v2p[v2]
    if p1 < 0 or p2 < 0: return 0.0
    c1, c2 = device.qubit_core[p1], device.qubit_core[p2]
    if c1 == c2: return float(device.D[p1][p2])
    INF = 10**9
    nodes = set([p1, p2]) | device.comm_qubits
    dist = {n: INF for n in nodes}; dist[p1] = 0
    pq = [(0, p1)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v in nodes:
            if v == u: continue
            w = device.D[u][v] if device.D[u][v] < INF else INF
            if w == INF: continue
            if v in device.comm_qubits:
                w = max(0, w-1) + (_nearest_free(v, layout, device)/2.0) + _full_penalty(device.qubit_core[v], layout, device)
            nd = d + w
            if nd < dist[v]: dist[v] = nd; heapq.heappush(pq, (nd, v))
    for (lq0, lq1) in device.inter_links:
        if lq0 in dist and lq1 in dist:
            for src, dst in [(lq0,lq1),(lq1,lq0)]:
                nd = dist[src]+1
                if nd < dist.get(dst, INF): dist[dst] = nd
    return float(dist.get(p2, INF))

def _total_energy(front, ext, layout, device, k=0.05):
    F = sum(_gate_energy(g, layout, device) for g in front)
    H = sum(_gate_energy(g, layout, device) for g in ext)
    return F/max(len(front),1) + k*H/max(len(ext),1)

def _front_layer(gates, executed):
    blocked = {}
    for g in gates:
        if not executed[g.idx]:
            for v in g.qubits:
                if v not in blocked: blocked[v] = g
    front = []; seen = set()
    for g in gates:
        if not executed[g.idx] and g.idx not in seen:
            if all(blocked.get(v, g) is g for v in g.qubits):
                front.append(g); seen.add(g.idx)
    return front

def _extended_set(gates, executed, front, size=20):
    fids = {g.idx for g in front}; ext = []
    for g in gates:
        if not executed[g.idx] and g.idx not in fids and len(g.qubits)==2:
            ext.append(g)
            if len(ext) >= size: break
    return ext

def _cand_swaps(front, layout, device):
    rel = set()
    for g in front:
        for v in g.qubits:
            p = layout.v2p[v]
            if p >= 0:
                rel.add(p)
                for (a,b) in device.couplings:
                    if a==p: rel.add(b)
                    if b==p: rel.add(a)
    out = set()
    for p in rel:
        for (a,b) in device.couplings:
            if a==p or b==p: out.add((min(a,b), max(a,b)))
    return list(out)

def _cand_teles(front, layout, device):
    teles = []
    for g in front:
        if len(g.qubits) < 2: continue
        v1, v2 = g.qubits; p1, p2 = layout.v2p[v1], layout.v2p[v2]
        if p1<0 or p2<0: continue
        c1, c2 = device.qubit_core[p1], device.qubit_core[p2]
        if c1==c2: continue
        for (lq0, lq1) in device.inter_links:
            lc0, lc1 = device.qubit_core[lq0], device.qubit_core[lq1]
            for src, cs, cd, sc, dc in [(p1,lq0,lq1,c1,c2),(p1,lq1,lq0,c1,c2),
                                         (p2,lq0,lq1,c2,c1),(p2,lq1,lq0,c2,c1)]:
                if device.qubit_core[cs]==sc and device.qubit_core[cd]==dc:
                    if layout.p2v[cs]==-1 and layout.p2v[cd]==-1:
                        teles.append((src, cs, cd))
    return teles


def route_one(circuit_path, device, teleport_bonus=100, k=0.05,
              extended_size=20, max_iter=100000, seed=0):
    """Return (teleports, swaps, remaining_gates)."""
    num_virt, gates = parse_qasm(circuit_path)
    layout   = initial_layout(num_virt, device, gates, seed=seed)
    executed = [False]*len(gates)
    ops      = []
    coupled  = {(min(a,b), max(a,b)) for (a,b) in device.couplings}

    for iteration in range(max_iter):
        # execute ready gates
        progress = True
        while progress:
            progress = False
            front = _front_layer(gates, executed)
            for g in front:
                if len(g.qubits)==1:
                    executed[g.idx]=True; ops.append(("GATE",g)); progress=True
                elif len(g.qubits)==2:
                    p1, p2 = layout.v2p[g.qubits[0]], layout.v2p[g.qubits[1]]
                    if (min(p1,p2),max(p1,p2)) in coupled:
                        executed[g.idx]=True; ops.append(("GATE",g)); progress=True
        front = _front_layer(gates, executed)
        if not front: break

        ext = _extended_set(gates, executed, front, extended_size)
        best_score = float("inf"); best_op = None

        for (pa,pb) in _cand_swaps(front, layout, device):
            trial = layout.copy(); trial.swap(pa, pb)
            s = _total_energy(front, ext, trial, device, k)
            if s < best_score: best_score=s; best_op=("SWAP",pa,pb)

        for (ps,pcs,pcd) in _cand_teles(front, layout, device):
            trial = layout.copy(); trial.teleport(ps, pcs, pcd)
            s = _total_energy(front, ext, trial, device, k) - teleport_bonus
            if s < best_score: best_score=s; best_op=("TELE",ps,pcs,pcd)

        if best_op is None: break
        if best_op[0]=="SWAP":
            layout.swap(best_op[1], best_op[2]); ops.append(best_op)
        else:
            layout.teleport(best_op[1], best_op[2], best_op[3]); ops.append(best_op)

    remaining = sum(1 for e in executed if not e)
    teleports = sum(1 for op in ops if op[0]=="TELE")
    swaps     = sum(1 for op in ops if op[0]=="SWAP")
    return teleports, swaps, remaining


# ── Benchmark runner ───────────────────────────────────────────────────────

CIRCUIT_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_25")
NUM_SEEDS   = 3
DEVICE      = Device(_DEVICE_DICT)


def main():
    qasm_files = sorted(glob.glob(os.path.join(CIRCUIT_DIR, "*.qasm")))
    if not qasm_files:
        print(f"No .qasm files in {CIRCUIT_DIR}"); return

    print(f"{'circuit':<12}  {'q':>3}  {'TS_EPR_best':>11}  {'TS_SWAPs':>8}  {'ok':>4}")
    print("-" * 46)

    rows = []
    for qf in qasm_files:
        cname = os.path.basename(qf).replace("_nativegates_ibm_qiskit_opt3_25.qasm","")
        num_virt, _ = parse_qasm(qf)

        best_tele = best_swap = None
        for seed in range(NUM_SEEDS):
            buf = io.StringIO()
            try:
                t, s, r = route_one(qf, DEVICE, seed=seed)
            except Exception as e:
                print(f"  ERROR {cname} seed={seed}: {e}"); continue
            ok = (r == 0)
            if ok and (best_tele is None or t < best_tele):
                best_tele, best_swap = t, s

        if best_tele is None:
            print(f"{cname:<12}  {num_virt:>3}  {'FAIL':>11}  {'':>8}  {'':>4}")
        else:
            print(f"{cname:<12}  {num_virt:>3}  {best_tele:>11}  {best_swap:>8}  {'yes':>4}")
        rows.append((cname, num_virt, best_tele, best_swap))

    return rows


if __name__ == "__main__":
    main()
