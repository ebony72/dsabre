import networkx as nx
from typing import Dict, List, Tuple


class DistributedArchitecture:
    """通用分布式量子计算机架构"""
    def __init__(self,
                 intra_graphs: Dict[int, nx.Graph],
                 inter_links: List[Tuple[int, int]]):
        """
        intra_graphs: 映射 core_id -> 该核心内部连接图 (节点ID必须全局唯一)
        inter_links: 表示物理硬件核间连接的元组列表 (p_src, p_dst)
        """
        self.intra = intra_graphs
        self.num_cores = len(self.intra)
        self.inter_core_links = inter_links

        self.Gr = nx.Graph()
        self.data_qubits = []
        self.comm_qubits = set()
        self.qubit_to_core = {}

        # 注册核内拓扑
        for core_id, graph in self.intra.items():
            self.Gr.add_edges_from(graph.edges(data=True))
            for node in graph.nodes():
                self.data_qubits.append(node)
                self.qubit_to_core[node] = core_id

        # 注册核间拓扑与 Core Graph
        self.core_graph = nx.Graph()
        self.core_graph.add_nodes_from(self.intra.keys())

        for u, v in self.inter_core_links:
            self.Gr.add_edge(u, v, weight=10)
            self.comm_qubits.update([u, v])
            c_u, c_v = self.core_of(u), self.core_of(v)
            if c_u != c_v:
                self.core_graph.add_edge(c_u, c_v)

        # 预计算核间距离
        self.core_dist = dict(nx.all_pairs_shortest_path_length(self.core_graph))
        self.core_path = dict(nx.all_pairs_shortest_path(self.core_graph))

        self.intra_dist: Dict[int, Dict[int, Dict[int, int]]] = {}
        for core_id, g in self.intra.items():
            self.intra_dist[core_id] = dict(nx.all_pairs_shortest_path_length(g))

        self.intra_path: Dict[int, Dict[int, Dict[int, list]]] = {}
        for core_id, g in self.intra.items():
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

        # Dijkstra over full Gr (intra edges weight=1, inter edges weight=10).
        # O(n^2) space; enables O(1) cross-core distance lookup in routing hot paths.
        self.phys_dist: Dict[int, Dict[int, int]] = dict(
            nx.all_pairs_dijkstra_path_length(self.Gr)
        )

    def memory_report(self) -> dict:
        """
        Estimate memory footprint of the precomputed distance tables.

        Python dict-of-dicts entries cost roughly:
          - outer dict key + pointer: ~56 bytes each
          - inner dict key + int value: ~56 bytes each
        These are conservative lower bounds; actual RSS will be higher due to
        dict hash-table load-factor padding.
        """
        _ENTRY_BYTES = 56   # one Python (int key, int value) pair in a dict

        def _count(table):
            return sum(len(v) for v in table.values())

        phys_n   = _count(self.phys_dist)
        intra_n  = sum(_count(d) for d in self.intra_dist.values())
        core_n   = _count(self.core_dist)

        return {
            "num_qubits":          len(self.data_qubits),
            "num_comm_ports":      len(self.comm_qubits),
            "num_cores":           self.num_cores,
            "phys_dist_entries":   phys_n,
            "phys_dist_bytes_est": phys_n   * _ENTRY_BYTES,
            "intra_dist_entries":  intra_n,
            "intra_dist_bytes_est":intra_n  * _ENTRY_BYTES,
            "core_dist_entries":   core_n,
            "core_dist_bytes_est": core_n   * _ENTRY_BYTES,
            "total_bytes_est":     (phys_n + intra_n + core_n) * _ENTRY_BYTES,
        }

    def core_of(self, p: int) -> int:
        return self.qubit_to_core[p]

    def core_qubits(self, core_id: int) -> List[int]:
        return self._core_qubits_list[core_id]

    def inter_links_between(self, c0: int, c1: int) -> List[Tuple[int, int]]:
        return self._inter_links_between.get((c0, c1), [])


# --- 辅助函数：快速生成以前的网格架构 ---
def build_basic_grid_architecture(r: int, s: int, m: int) -> DistributedArchitecture:
    """构建 R x S 个核心，每个核心为 M x M 物理量子比特的网格架构"""
    qubits_per_core = m * m

    def _gid(cr, cs, lr, ls):
        return (cr * s + cs) * qubits_per_core + (lr * m + ls)

    intra_graphs = {}
    inter_links = []
    mid = m // 2

    for cr in range(r):
        for cs in range(s):
            core_id = cr * s + cs
            sg = nx.Graph()
            for lr in range(m):
                for ls in range(m):
                    u = _gid(cr, cs, lr, ls)
                    sg.add_node(u)
                    if ls + 1 < m:
                        v = _gid(cr, cs, lr, ls + 1)
                        sg.add_edge(u, v, weight=1)
                    if lr + 1 < m:
                        v = _gid(cr, cs, lr + 1, ls)
                        sg.add_edge(u, v, weight=1)
            intra_graphs[core_id] = sg

            if cs + 1 < s:
                u = _gid(cr, cs, mid, m - 1)
                v = _gid(cr, cs + 1, mid, 0)
                inter_links.append((u, v))
            if cr + 1 < r:
                u = _gid(cr, cs, m - 1, mid)
                v = _gid(cr + 1, cs, 0, mid)
                inter_links.append((u, v))

    return DistributedArchitecture(intra_graphs, inter_links)


def build_b_grid_architecture(r: int, s: int, m: int) -> DistributedArchitecture:
    """构建 R x S 个核心，每个核心为 M x M 物理量子比特的网格架构 (使用交错通信端口)"""
    qubits_per_core = m * m

    def _gid(cr, cs, lr, ls):
        return (cr * s + cs) * qubits_per_core + (lr * m + ls)

    intra_graphs = {}
    inter_links = []

    for cr in range(r):
        for cs in range(s):
            core_id = cr * s + cs
            sg = nx.Graph()
            for lr in range(m):
                for ls in range(m):
                    u = _gid(cr, cs, lr, ls)
                    sg.add_node(u)
                    if ls + 1 < m:
                        v = _gid(cr, cs, lr, ls + 1)
                        sg.add_edge(u, v, weight=1)
                    if lr + 1 < m:
                        v = _gid(cr, cs, lr + 1, ls)
                        sg.add_edge(u, v, weight=1)
            intra_graphs[core_id] = sg

            horiz_row = (m // 2 - 1) if cr % 2 == 0 else (m // 2)
            if cs + 1 < s:
                u = _gid(cr, cs, horiz_row, m - 1)
                v = _gid(cr, cs + 1, horiz_row, 0)
                inter_links.append((u, v))

            vert_col = (m // 2 - 1) if cs % 2 == 0 else (m // 2)
            if cr + 1 < r:
                u = _gid(cr, cs, m - 1, vert_col)
                v = _gid(cr + 1, cs, 0, vert_col)
                inter_links.append((u, v))

    return DistributedArchitecture(intra_graphs, inter_links)


def build_h_grid_architecture(r: int, s: int, m: int) -> DistributedArchitecture:
    """
    Builds an R x S grid of M x M cores using the 'H-Grid' staggered
    inter-core connection topology as depicted in the H_grid_2_3_4_4 diagram.
    """
    qubits_per_core = m * m

    def _gid(cr, cs, lr, ls):
        return (cr * s + cs) * qubits_per_core + (lr * m + ls)

    intra_graphs = {}
    inter_links = []

    for cr in range(r):
        for cs in range(s):
            core_id = cr * s + cs
            sg = nx.Graph()
            for lr in range(m):
                for ls in range(m):
                    u = _gid(cr, cs, lr, ls)
                    sg.add_node(u)
                    if ls + 1 < m:
                        v = _gid(cr, cs, lr, ls + 1)
                        sg.add_edge(u, v, weight=1)
                    if lr + 1 < m:
                        v = _gid(cr, cs, lr + 1, ls)
                        sg.add_edge(u, v, weight=1)
            intra_graphs[core_id] = sg

            # 1. Horizontal Links (Alternating Rows)
            if cs + 1 < s:
                horiz_row = (m // 2 - 1) if cr % 2 == 0 else (m // 2)
                u = _gid(cr, cs, horiz_row, m - 1)
                v = _gid(cr, cs + 1, horiz_row, 0)
                inter_links.append((u, v))

            # 2. Vertical Links (Skewed / Staggered Columns)
            if cr + 1 < r:
                col_top = (m // 2) if cs > 0 else (m // 2 - 1)
                col_bottom = (m // 2 - 1) if cs < s - 1 else (m // 2)
                u = _gid(cr, cs, m - 1, col_top)
                v = _gid(cr + 1, cs, 0, col_bottom)
                inter_links.append((u, v))

    return DistributedArchitecture(intra_graphs, inter_links)
