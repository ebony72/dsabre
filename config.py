from dataclasses import dataclass

@dataclass
class HardwareConfig:
    """针对特定分布式硬件的可配置参数和开销指标"""
    # 基础操作开销
    cost_local_swap: float = 3.0
    cost_teleport: float = 10.0      # base EPR cost (independent of distance)
    cost_teleport_per_hop: float = 0.0  # extra cost per core-hop; 0 = flat model
    cost_cat_comm: float = 10.0
    
    # 启发式评估参数
    weight_extended: float = 0.5
    lookahead_size: int = 20
    capacity_threshold: int = 3
    cap_penalty: float = 15.0
    hop_gain: float = 5.0
    cat_resolve_reward: float = 20.0
    
    # 主动拥塞缓解机制 (Mechanism 3) 的微调参数
    relief_bonus: float = 8.0          # 缓解瓶颈的得分奖励
    demand_lookahead: int = 5          # 向前看几步来计算需求
    demand_threshold: int = 1          # 触发警告的最小需求量
    congestion_threshold: int = 1      # 触发警告的最大空闲槽位数量
    relief_space_req: int = 3          # 接收核心必须具备的最小空闲槽位
    
    lookahead_decay: float = 0.9    # per-gate exponential decay for extended-layer scoring

    # Diagnostic
    trace_routing: bool = False     # record swap/teleport trace in metrics["trace"]

    # 避免死锁的控制参数
    max_iterations: int = 10000
    deadlock_limit: int = 50
    max_backup_attempts: int = 50