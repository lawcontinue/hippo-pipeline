"""
shard.py — 分片元数据模型

定义 Pipeline Parallelism 的分片信息：
- 每台设备只加载 model.layers[start:end]
- 通过 device_rank 和 world_size 确定角色
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShardMetadata:
    """描述模型的一个分片（分配给单台设备的层范围）"""

    model_id: str          # 模型 ID（如 "Qwen/Qwen2.5-14B"）
    start_layer: int       # 起始层（含）
    end_layer: int         # 结束层（不含）
    n_layers: int          # 模型总层数
    device_rank: int       # 本设备排名 (0, 1, ...)
    world_size: int        # 总设备数

    @property
    def is_first(self) -> bool:
        return self.device_rank == 0

    @property
    def is_last(self) -> bool:
        return self.device_rank == self.world_size - 1

    @property
    def n_local_layers(self) -> int:
        return self.end_layer - self.start_layer

    def __str__(self) -> str:
        return (
            f"Shard(model={self.model_id}, "
            f"rank={self.device_rank}/{self.world_size}, "
            f"layers=[{self.start_layer}:{self.end_layer}] "
            f"of {self.n_layers})"
        )


def split_model(model_id: str, n_layers: int, world_size: int) -> list[ShardMetadata]:
    """
    将模型均匀切分为 world_size 个分片。
    
    借鉴 Exo 的 ring_memory_weighted_partitioning_strategy，
    当前实现为均匀切分，未来可按设备内存权重调整。
    """
    shards = []
    layers_per_device = n_layers // world_size
    remainder = n_layers % world_size

    start = 0
    for rank in range(world_size):
        # 前 remainder 个设备多分一层
        extra = 1 if rank < remainder else 0
        end = start + layers_per_device + extra
        shards.append(ShardMetadata(
            model_id=model_id,
            start_layer=start,
            end_layer=end,
            n_layers=n_layers,
            device_rank=rank,
            world_size=world_size,
        ))
        start = end

    return shards


def memory_weighted_split(
    model_id: str,
    n_layers: int,
    device_memories: list[float],
) -> list[ShardMetadata]:
    """
    按设备可用内存权重分配层数。
    
    Args:
        model_id: 模型 ID
        n_layers: 模型总层数
        device_memories: 每台设备的可用内存 (GB)，按 rank 顺序
    
    Returns:
        每台设备的 ShardMetadata 列表
    """
    world_size = len(device_memories)
    total_mem = sum(device_memories)
    
    shards = []
    start = 0
    for rank, mem in enumerate(device_memories):
        # 按内存比例分配层数（至少 1 层）
        ratio = mem / total_mem
        if rank < world_size - 1:
            n = max(1, round(n_layers * ratio))
        else:
            # 最后一个设备取剩余所有层
            n = n_layers - start
        
        end = start + n
        shards.append(ShardMetadata(
            model_id=model_id,
            start_layer=start,
            end_layer=end,
            n_layers=n_layers,
            device_rank=rank,
            world_size=world_size,
        ))
        start = end

    return shards
