"""
Distributed quantum processor architecture definitions.

A DistributedArchitecture consists of multiple cores, each a 2-D grid of
physical qubits.  Cores are connected by a small number of inter-core links
whose endpoints are the communication (comm) qubits.  All precomputed distance
tables are built at construction time for O(1) lookup during routing.
"""

import networkx as nx
from typing import Dict, List, Tuple


class DistributedArchitecture:
    """Multi-core quantum processor topology with precomputed routing tables."""

    def __init__(self,
                 intra_graphs: Dict[int, nx.Graph],
                 inter_links: List[Tuple[int, int]]):
        """
        Parameters
        ----------
        intra_graphs : {core_id: nx.Graph}
            Per-core connectivity graphs.  Node IDs must be globally unique
            across all cores.
        inter_links : [(p_src, p_dst), ...]
            Physical inter-core link endpoints.  Each pair (u, v) means u and v
            are in different cores and share one EPR-pair communication channel.
        """
        self.intra = intra_graphs
        self.num_cores = len(self.intra)
        self.inter_core_links = inter_links

        # Combined full-chip graph (intra edges weight=1, inter edges weight=10).
        self.Gr = nx.Graph()
        self.data_qubits: List[int] = []
        self.comm_qubits: set = set()
        self.qubit_to_core: Dict[int, int] = {}

        for core_id, graph in self.intra.items():
            self.Gr.add_edges_from(graph.edges(data=True))
            for node in graph.nodes():
                self.data_qubits.append(node)
                self.qubit_to_core[node] = core_id

        self.core_graph = nx.Graph()
        self.core_graph.add_nodes_from(self.intra.keys())
        for u, v in self.inter_core_links:
            self.Gr.add_edge(u, v, weight=10)
            self.comm_qubits.update([u, v])
            c_u, c_v = self.core_of(u), self.core_of(v)
            if c_u != c_v:
                self.core_graph.add_edge(c_u, c_v)

        # Precomputed tables — all O(1) at routing time.
        self.core_dist = dict(nx.all_pairs_shortest_path_length(self.core_graph))
        self.core_path = dict(nx.all_pairs_shortest_path(self.core_graph))

        self.intra_dist: Dict[int, Dict[int, Dict[int, int]]] = {}
        self.intra_path: Dict[int, Dict[int, Dict[int, list]]] = {}
        for core_id, g in self.intra.items():
            self.intra_dist[core_id] = dict(nx.all_pairs_shortest_path_length(g))
            self.intra_path[core_id] = dict(nx.all_pairs_shortest_path(g))

        self._core_comm_ports: Dict[int, List[int]] = {c: [] for c in self.intra}
        for p in self.comm_qubits:
            self._core_comm_ports[self.core_of(p)].append(p)

        self._core_qubits_list: Dict[int, List[int]] = {
            core_id: list(g.nodes()) for core_id, g in self.intra.items()
        }

        self._inter_links_between: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for c0 in range(self.num_cores):
            for c1 in range(self.num_cores):
                self._inter_links_between[(c0, c1)] = []
        for u, v in self.inter_core_links:
            cu, cv = self.core_of(u), self.core_of(v)
            self._inter_links_between[(cu, cv)].append((u, v))
            self._inter_links_between[(cv, cu)].append((v, u))

        # Full-chip Dijkstra: intra edges weight 1, inter edges weight 10.
        # O(n²) space; enables O(1) cross-core distance lookup in hot paths.
        self.phys_dist: Dict[int, Dict[int, int]] = dict(
            nx.all_pairs_dijkstra_path_length(self.Gr)
        )

    def core_of(self, p: int) -> int:
        return self.qubit_to_core[p]

    def core_qubits(self, core_id: int) -> List[int]:
        return self._core_qubits_list[core_id]

    def inter_links_between(self, c0: int, c1: int) -> List[Tuple[int, int]]:
        return self._inter_links_between.get((c0, c1), [])

    def memory_report(self) -> dict:
        """Estimate RAM footprint of the precomputed distance tables (bytes)."""
        _ENTRY = 56  # one Python (int, int) dict entry, conservative lower bound
        phys_n  = sum(len(v) for v in self.phys_dist.values())
        intra_n = sum(len(v) for d in self.intra_dist.values() for v in d.values())
        core_n  = sum(len(v) for v in self.core_dist.values())
        return {
            "num_qubits":  len(self.data_qubits),
            "num_cores":   self.num_cores,
            "total_bytes": (phys_n + intra_n + core_n) * _ENTRY,
        }


# ── Architecture builders ──────────────────────────────────────────────────────

def build_b_grid_architecture(r: int, s: int, m: int) -> DistributedArchitecture:
    """B-grid: R×S array of M×M cores with staggered inter-core links.

    Inter-core links alternate between two rows (horizontal neighbours) and two
    columns (vertical neighbours) to balance communication port placement.
    Used for 25q and 36q benchmarks with r=s=2, m=4 (64 physical qubits).
    """
    qpc = m * m

    def _gid(cr, cs, lr, ls):
        return (cr * s + cs) * qpc + (lr * m + ls)

    intra_graphs, inter_links = {}, []
    for cr in range(r):
        for cs in range(s):
            sg = nx.Graph()
            for lr in range(m):
                for ls in range(m):
                    u = _gid(cr, cs, lr, ls)
                    sg.add_node(u)
                    if ls + 1 < m: sg.add_edge(u, _gid(cr, cs, lr, ls + 1), weight=1)
                    if lr + 1 < m: sg.add_edge(u, _gid(cr, cs, lr + 1, ls), weight=1)
            intra_graphs[cr * s + cs] = sg

            horiz_row = (m // 2 - 1) if cr % 2 == 0 else (m // 2)
            if cs + 1 < s:
                inter_links.append((_gid(cr, cs, horiz_row, m - 1),
                                    _gid(cr, cs + 1, horiz_row, 0)))

            vert_col = (m // 2 - 1) if cs % 2 == 0 else (m // 2)
            if cr + 1 < r:
                inter_links.append((_gid(cr, cs, m - 1, vert_col),
                                    _gid(cr + 1, cs, 0, vert_col)))

    return DistributedArchitecture(intra_graphs, inter_links)


# IBM Falcon r5.11 27-qubit heavy-hex coupling map (ibmq_montreal / ibm_hanoi
# family).  28 edges, six degree-1 leaves, diameter 12 — an irregular,
# non-grid core topology used to test architecture independence.
IBM_HEAVY_HEX_27 = [
    (0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (4, 7), (5, 8), (6, 7), (7, 10),
    (8, 9), (8, 11), (10, 12), (11, 14), (12, 13), (12, 15), (13, 14),
    (14, 16), (15, 18), (16, 19), (17, 18), (18, 21), (19, 20), (19, 22),
    (21, 23), (22, 25), (23, 24), (24, 25), (25, 26),
]


def _heavy_hex_ports(tile: nx.Graph) -> Tuple[int, int]:
    """The two most distant non-leaf qubits of a heavy-hex tile.

    Comm ports must not be degree-1 leaves: a leaf port is a dead end, and the
    layout's corner reservation already claims leaves as escape slots.  Picking
    the farthest-apart pair spreads the two ports to opposite ends of the tile,
    mirroring the boundary-port placement of the grid architectures.
    """
    dist = dict(nx.all_pairs_shortest_path_length(tile))
    cands = sorted(n for n in tile if tile.degree(n) >= 2)
    return max(((u, v) for i, u in enumerate(cands) for v in cands[i + 1:]),
               key=lambda uv: (dist[uv[0]][uv[1]], -uv[0], -uv[1]))


def build_heavy_hex_architecture(num_cores: int = 4) -> DistributedArchitecture:
    """Ring of IBM 27-qubit heavy-hex cores (4 cores = 108 physical qubits).

    Each core is a Falcon-family heavy-hex tile rather than a 2-D grid, and the
    cores form a cycle with one inter-core link per adjacent pair (4 links at
    num_cores=4, matching the B-grid's link budget).  Core c reaches c-1 through
    its low port and c+1 through its high port.
    """
    qpc = len(set(sum(map(list, IBM_HEAVY_HEX_27), [])))  # 27

    intra_graphs, ports = {}, {}
    for c in range(num_cores):
        sg = nx.Graph()
        sg.add_nodes_from(c * qpc + n for n in range(qpc))
        sg.add_edges_from((c * qpc + u, c * qpc + v, {"weight": 1})
                          for u, v in IBM_HEAVY_HEX_27)
        intra_graphs[c] = sg
        lo, hi = _heavy_hex_ports(nx.Graph(IBM_HEAVY_HEX_27))
        ports[c] = (c * qpc + lo, c * qpc + hi)

    inter_links = [(ports[c][1], ports[(c + 1) % num_cores][0])
                   for c in range(num_cores)]
    if num_cores == 2:                      # a 2-cycle would duplicate the link
        inter_links = inter_links[:1]

    return DistributedArchitecture(intra_graphs, inter_links)


def build_h_grid_architecture(r: int, s: int, m: int) -> DistributedArchitecture:
    """H-grid: R×S array of M×M cores with skewed vertical inter-core links.

    Horizontal links alternate rows (same as B-grid); vertical links use
    column offsets that differ at the top and bottom of each column pair,
    giving the 'H' cross-bar pattern.
    Used for 64q benchmarks with r=2, s=3, m=4 (96 physical qubits).
    """
    qpc = m * m

    def _gid(cr, cs, lr, ls):
        return (cr * s + cs) * qpc + (lr * m + ls)

    intra_graphs, inter_links = {}, []
    for cr in range(r):
        for cs in range(s):
            sg = nx.Graph()
            for lr in range(m):
                for ls in range(m):
                    u = _gid(cr, cs, lr, ls)
                    sg.add_node(u)
                    if ls + 1 < m: sg.add_edge(u, _gid(cr, cs, lr, ls + 1), weight=1)
                    if lr + 1 < m: sg.add_edge(u, _gid(cr, cs, lr + 1, ls), weight=1)
            intra_graphs[cr * s + cs] = sg

            if cs + 1 < s:
                horiz_row = (m // 2 - 1) if cr % 2 == 0 else (m // 2)
                inter_links.append((_gid(cr, cs, horiz_row, m - 1),
                                    _gid(cr, cs + 1, horiz_row, 0)))

            if cr + 1 < r:
                col_top    = (m // 2)     if cs > 0       else (m // 2 - 1)
                col_bottom = (m // 2 - 1) if cs < s - 1   else (m // 2)
                inter_links.append((_gid(cr, cs, m - 1, col_top),
                                    _gid(cr + 1, cs, 0, col_bottom)))

    return DistributedArchitecture(intra_graphs, inter_links)
