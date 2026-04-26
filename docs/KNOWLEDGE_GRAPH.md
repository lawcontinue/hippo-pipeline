# Hippo Pipeline 技术名词与知识图谱

**版本**: v1.0 | **更新日期**: 2026-04-20 | **维护者**: 忒弥斯 🔮

---

## 核心技术名词

### 分布式推理

| 术语 | 英文 | 定义 | 关键参数 |
|------|------|------|---------|
| **Pipeline Parallelism** | Pipeline Parallelism | 将模型按层切分到多台设备，数据像流水线一样经过每台设备 | Hippo: 2 × 24 层 |
| **Tensor Parallelism** | Tensor Parallelism | 将单个算子（如矩阵乘法）切分到多台设备并行计算 | 需要 NCCL/RCCL |
| **Data Parallelism** | Data Parallelism | 多台设备各自持有完整模型，处理不同 batch | 常用于训练 |
| **Expert Parallelism** | Expert Parallelism (EP) | MoE 模型中，不同专家分布在不同设备 | DeepEP 实现 |
| **Tensor Split** | tensor_split | llama.cpp 中控制各设备分到的层比例 | Hippo: 50/50 |

### Hippo 架构

| 术语 | 定义 | 文件 |
|------|------|------|
| **ShardMetadata** | 分片元数据（model_id, start/end_layer, rank） | `shard.py` |
| **TensorSender/Receiver** | TCP tensor 传输层（魔数 0xHPPT） | `tcp_transport.py` |
| **ShardLoader** | 按层范围加载 safetensors 权重 | `shard_loader.py` |
| **Forward Layer** | Gemma3 单层前向传播（6 norm + RoPE + GELU） | `sharded_inference.py` |
| **Overlapped Pipeline** | 连接+加载并行优化 | `overlapped_pipeline.py` |
| **Bidirectional Pipeline** | 双向流水线（两请求并行） | `bidirectional_pipeline.py` |
| **KV Cache SSD** | SSD 持久化 KV 缓存（读 1.75 GB/s） | `kv_cache.py` |

### Gemma3 架构特殊性

| 特性 | 标准模型 | Gemma3 | 来源 |
|------|---------|--------|------|
| **Norm 层数** | 2 | **6** | input_layernorm + q_norm + k_norm + post_attn + pre_ffn + post_ffn |
| **残差方式** | Pre-norm | **Post-norm** | 先计算→norm→加残差 |
| **Q/K Norm** | 无 | **有** | attention 内部 RMSNorm |
| **激活函数** | SiLU/GeLU | **GELU** | `nn.gelu_approx` |
| **Embedding 缩放** | 无 | **有** | `h *= sqrt(hidden_size)` |
| **RMSNorm 增益** | `*weight` | **`*(1.0+weight)`** | `mx.fast.rms_norm` |
| **Attention 缩放** | `HEAD_DIM^-0.5` | **`query_pre_attn_scalar^-0.5`** | = 256^-0.5 |
| **Attention 模式** | 全局 | **交替 local/global** | sliding_window_pattern=6 |
| **RoPE base** | 10000 | **10000(local)/1000000(global)** | 每 6 层全局 |

### DeepSeek 开源项目

| 项目 | 核心能力 | 性能 | Hippo 借鉴 |
|------|---------|------|-----------|
| **DeepEP** | MoE all-to-all 通信 | NVLink 153GB/s, RDMA 58GB/s | A1 通信-计算重叠 |
| **DualPipe** | 双向流水线并行 | 训练吞吐 +100% | A2 双向流水线 |
| **DeepGEMM** | FP8 矩阵乘法 | 1550 TFLOPS, ~300 行 | A4 简洁工程 |
| **3FS** | 分布式文件系统 | 40 GiB/s 读吞吐 | A3 SSD KV Cache |

---

## 知识图谱

```
┌─────────────────────────────────────────────────────┐
│                    Hippo Pipeline                      │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐    TCP/Thunderbolt    ┌──────────┐ │
│  │   Rank 0     │ ──────────────────→   │  Rank 1  │ │
│  │ 192.168.1.10 │ ←──────────────────  │192.168.11│ │
│  │  Layers 0-23 │    logits (1MB)       │ L24-47   │ │
│  │  3.57 GB     │                       │ 3.57 GB  │ │
│  └──────────────┘                       └──────────┘ │
│                                                       │
│  Forward Layer (Gemma3):                              │
│  ┌─────────────────────────────────────────────────┐ │
│  │ input_layernorm → QKV → Q/K Norm → RoPE         │ │
│  │ → Attention → o_proj → post_attn_norm → Residual │ │
│  │ pre_ffn_norm → Gate/Up/Down(GELU)                │ │
│  │ → post_ffn_norm → Residual                        │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  Optimizations:                                       │
│  ┌────────────┐ ┌────────────┐ ┌─────────────┐      │
│  │A1: Overlap │ │A2: Bidir   │ │A3: SSD KV   │      │
│  │Connect+Load│ │2 requests  │ │1.75 GB/s    │      │
│  └────────────┘ └────────────┘ └─────────────┘      │
└─────────────────────────────────────────────────────┘

         ↓ 启发来源 ↓

┌─────────────────────────────────────────────────────┐
│              DeepSeek 开源四项目                       │
├─────────────────────────────────────────────────────┤
│  DeepEP ──→ 通信-计算重叠 (hook-based)                │
│  DualPipe ──→ 双向流水线 (参数 2× 代价)                │
│  DeepGEMM ──→ 简洁工程 (~300 行核心)                  │
│  3FS ──→ SSD KV Cache (40 GiB/s)                    │
└─────────────────────────────────────────────────────┘

         ↓ 调试方法论 ↓

┌─────────────────────────────────────────────────────┐
│              model-forward-validator                  │
├─────────────────────────────────────────────────────┤
│  1. 读 config.json (永远先于写代码)                    │
│  2. 读源码 1:1 对照                                    │
│  3. A/B 逐层对比 (diff_mean < 0.1 = pass)             │
│  4. 子操作隔离 (embed→norm→linear→attn→mlp)          │
│                                                       │
│  常见坑:                                              │
│  • 架构参数推断错误 → 必须读配置                       │
│  • Norm 层数缺失 → 对照源码数清楚                     │
│  • RMSNorm 增益不同 → 检查 weight vs 1+weight         │
│  • 缺少 RoPE → 位置编码是必须的                       │
│  • 激活函数不同 → SiLU vs GELU vs GeLU                │
└─────────────────────────────────────────────────────┘
```

---

## 硬件配置

| 设备 | IP | 用户名 | 角色 | 内存 | GPU |
|------|-----|--------|------|------|-----|
| Mac Mini M4 #1 | 192.168.1.10 | deepsearch | Rank 0 | 16 GB | Apple M4 |
| Mac Mini M4 #2 | 192.168.1.11 | finance | Rank 1 | 16 GB | Apple M4 |

**网络**: Wi-Fi (Thunderbolt 直连 1ms 可用)
**SSH**: `ssh finance@192.168.1.11`（ed25519 密钥认证）
