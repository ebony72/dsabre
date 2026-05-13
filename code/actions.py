from dataclasses import dataclass
from typing import Any


@dataclass
class TeleportAction:
    """A single inter-core teleportation move considered by the router.

    Fields
    ------
    node        : DAG gate that triggered this candidate (None for proactive moves)
    virt        : logical qubit being teleported
    p_src       : current physical location of virt
    n_s         : intra-core staging qubit adjacent to the comm port
    p_comm_src  : source-side communication qubit (inter-core link endpoint)
    p_comm_dst  : destination-side communication qubit
    src_core    : core virt is moving FROM
    next_core   : core virt is moving TO (one hop)
    tgt_core    : final destination core (may equal next_core or be further)
    score       : heuristic cost — lower is better; best candidate has score[0]
    """
    __slots__ = [
        'node', 'virt', 'p_src', 'n_s',
        'p_comm_src', 'p_comm_dst',
        'src_core', 'next_core', 'tgt_core',
        'score',
    ]
    node:       Any
    virt:       Any
    p_src:      int
    n_s:        int
    p_comm_src: int
    p_comm_dst: int
    src_core:   int
    next_core:  int
    tgt_core:   int
    score:      float
