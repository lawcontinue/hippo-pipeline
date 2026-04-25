#!/usr/bin/env python3
"""
shard_loader.py — 直接从 safetensors 加载指定层，不加载全模型

解决 16GB 机器无法加载 7.5GB 模型的问题：
- 只读取 model.layers[start:end] 的权重
- 使用 safetensors 直接 mmap 指定 tensor
- 内存占用 = 分片大小（~3.75GB），不是全模型（7.5GB）
"""

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

try:
    from shard import ShardMetadata
except ImportError:
    from .shard import ShardMetadata


def _get_cache_dir(model_id: str) -> Path:
    """获取 HuggingFace 缓存目录"""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    # 转换 model_id 为目录名: org/model -> models--org--model
    dir_name = "models--" + model_id.replace("/", "--")
    return cache / dir_name


def _get_snapshot_dir(cache_dir: Path) -> Path:
    """获取 snapshot 目录"""
    refs = cache_dir / "refs" / "main"
    if refs.exists():
        commit = refs.read_text().strip()
        return cache_dir / "snapshots" / commit
    # fallback: 用最新的 snapshot
    snapshots = cache_dir / "snapshots"
    if snapshots.exists():
        dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime)
        if dirs:
            return dirs[-1]
    raise FileNotFoundError(f"No snapshot found in {cache_dir}")


def load_shard_weights(
    model_id: str,
    shard: ShardMetadata,
    verbose: bool = True,
) -> dict:
    """
    从 safetensors 直接加载指定层的权重。
    
    Args:
        model_id: HuggingFace 模型 ID
        shard: 分片元数据
        verbose: 是否打印进度
    
    Returns:
        dict: {tensor_name: mx.array} 只包含指定层的权重
    """
    from safetensors import safe_open
    
    cache_dir = _get_cache_dir(model_id)
    snap_dir = _get_snapshot_dir(cache_dir)
    
    if not snap_dir.exists():
        raise FileNotFoundError(f"Model not found: {snap_dir}")
    
    # 读取 config.json 获取层数和 hidden_size
    config_path = snap_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        n_layers = config.get("num_hidden_layers", config.get("n_layers", 0))
        hidden_size = config.get("hidden_size", config.get("d_model", 0))
        if verbose:
            print(f"   模型配置: {n_layers} 层, hidden_size={hidden_size}")
    else:
        if verbose:
            print(f"   ⚠️ 未找到 config.json，使用 shard 配置")
        n_layers = shard.n_layers
        hidden_size = 0
    
    # 找到所有 safetensors 文件
    safetensor_files = sorted(snap_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files in {snap_dir}")
    
    if verbose:
        print(f"   找到 {len(safetensor_files)} 个 safetensors 文件")
    
    # 构建需要加载的层名前缀
    start = shard.start_layer
    end = shard.end_layer
    layer_patterns = []
    for i in range(start, end):
        # 常见的层命名模式
        layer_patterns.append(f"model.layers.{i}.")
        layer_patterns.append(f"language_model.model.layers.{i}.")
        layer_patterns.append(f"model.h.{i}.")
        layer_patterns.append(f"transformer.h.{i}.")
    
    # 根据设备角色决定加载哪些全局权重
    if shard.device_rank == 0:
        # Rank 0 需要 embedding
        global_patterns = [
            "language_model.model.embed_tokens",
            "model.embed_tokens", "model.wte",
        ]
    else:
        # Rank 1 需要 lm_head 和 final_norm
        global_patterns = [
            "language_model.model.norm",
            "language_model.lm_head",
            "model.norm", "model.ln_f",
            "lm_head",
        ]
    
    # 从 safetensors 加载权重
    weights = {}
    loaded_layers = set()
    total_size = 0
    t0 = time.time()
    
    for sf_path in safetensor_files:
        # 使用 mlx 的 safetensors 加载（支持 bfloat16）
        weights_in_file = mx.load(str(sf_path))
        for key, tensor in weights_in_file.items():
                # 检查是否是需要加载的层
                should_load = False
                
                # 检查层前缀
                for pattern in layer_patterns:
                    if pattern in key:
                        should_load = True
                        # 提取层号
                        parts = key.split(".")
                        for pi, p in enumerate(parts):
                            if p == "layers" and pi + 1 < len(parts):
                                try:
                                    layer_idx = int(parts[pi + 1])
                                    loaded_layers.add(layer_idx)
                                except ValueError:
                                    pass
                        break
                
                # 检查全局权重
                if not should_load:
                    for pattern in global_patterns:
                        if pattern in key:
                            should_load = True
                            break
                
                if should_load:
                    weights[key] = tensor
                    total_size += tensor.nbytes
    
    t1 = time.time()
    
    if verbose:
        loaded_range = sorted(loaded_layers) if loaded_layers else []
        print(f"   加载了 {len(weights)} 个 tensor")
        print(f"   加载层: {loaded_range[:5]}...{loaded_range[-3:] if len(loaded_range) > 5 else ''}")
        print(f"   大小: {total_size / 1024**3:.2f} GB")
        print(f"   耗时: {t1-t0:.1f}s")
    
    return weights


def load_tokenizer(model_id: str):
    """只加载 tokenizer（不加载模型权重）"""
    from transformers import AutoTokenizer
    cache_dir = _get_cache_dir(model_id)
    snap_dir = _get_snapshot_dir(cache_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(str(snap_dir))
    return tokenizer


if __name__ == "__main__":
    # 测试：加载 Rank 0 的前 24 层
    shard = ShardMetadata(
        model_id='mlx-community/gemma-3-12b-it-qat-4bit',
        start_layer=0,
        end_layer=24,
        n_layers=48,
        device_rank=0,
        world_size=2,
    )
    
    print("=== 测试分片加载 ===")
    weights = load_shard_weights(shard.model_id, shard)
    print(f"\n前 10 个 tensor:")
    for k in sorted(weights.keys())[:10]:
        print(f"  {k}: {weights[k].shape}")
