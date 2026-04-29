from dataclasses import dataclass
from typing import Any

@dataclass
class TeleportAction:
    __slots__ = ['node', 'virt', 'p_src', 'n_s', 'p_comm_src', 'p_comm_dst', 'src_core', 'next_core', 'tgt_core', 'score']
    node: Any
    virt: Any
    p_src: int
    n_s: int
    p_comm_src: int
    p_comm_dst: int
    src_core: int
    next_core: int
    tgt_core: int
    score: float