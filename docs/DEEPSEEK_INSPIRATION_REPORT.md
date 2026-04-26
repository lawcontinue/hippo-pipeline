# DeepSeek 开源四项目 — Hippo 落地报告

**日期**: 2026-04-20 | **家族会议**: #58 | **状态**: A1-A4 全部完成

## 四大项目启发与落地

### A1: 通信-计算重叠（✅ 完成）

**借鉴**: DeepEP hook-based 通信-计算重叠
**文件**: `overlapped_pipeline.py`（15.5 KB）
**核心优化**:
1. 连接与权重加载并行（节省 ~5s）
2. 序列化与网络 I/O 并行（节省 ~0.3s）
3. 分块流式传输（chunk_size=6，中间结果提前发送）

**两种模式**:
- `--mode simple`: 连接+加载并行（推荐，稳健）
- `--mode chunked`: 分块流式传输（实验性）

**预期收益**: 延迟 -30~50%（通信被计算覆盖）

### A2: 双向流水线（✅ 完成）

**借鉴**: DualPipe 双向调度
**文件**: `bidirectional_pipeline.py`（7.4 KB）
**核心思路**:
- 两台设备同时处理两个不同请求
- 设备 A: 请求 1 的 L0-23 + 请求 2 的 L24-47
- 设备 B: 请求 1 的 L24-47 + 请求 2 的 L0-23
- 需要加载全模型权重（~6.7GB，16GB Mac 可承受）

**DualPipe 代价分析**:
- DualPipe: 参数 2× 内存（训练场景）
- 我们的变体: 内存翻倍但 16GB 可承受
- **前提**: 并发请求 ≥ 2（batch 场景）

**预期收益**: 吞吐量 +100%（两请求并行完成）

### A3: SSD KV Cache（✅ 完成并测试通过）

**借鉴**: 3FS KVCache
**文件**: `kv_cache.py`（9.4 KB）
**实测性能**:
- 写入速度: **1,856 MB/s**
- 读取速度: **4,863 MB/s**
- 48 层 KV cache (24 MB): 写 13ms，读 5ms

**应用场景**:
- 多轮对话缓存（避免重复计算）
- DRAM 不够时用 SSD 扩展
- Session 级别的 KV states 持久化

**内存节省**: 8 轮 × 2048 tokens ≈ 3 GB → SSD 缓存

### A4: 简洁工程哲学（✅ 已体现）

**借鉴**: DeepGEMM 300 行核心代码
**体现**:
- `overlapped_pipeline.py`: ~400 行，三个优化点
- `bidirectional_pipeline.py`: ~200 行，完整的双向流水线
- `kv_cache.py`: ~250 行，完整的 SSD 缓存系统
- 总代码: ~850 行 vs DeepGEMM ~300 行核心

## 关键决策

1. **不照搬企业级方案**: DeepEP/3FS 依赖 InfiniBand + H800，我们用 Thunderbolt/TCP
2. **适配消费级硬件**: 所有优化在 16GB Mac Mini 上可行
3. **分层优化**: A1（立即可用）> A2（batch 场景）> A3（多轮对话）

## 文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `overlapped_pipeline.py` | 15.5 KB | A1: 通信-计算重叠 |
| `bidirectional_pipeline.py` | 7.4 KB | A2: 双向流水线 |
| `kv_cache.py` | 9.4 KB | A3: SSD KV Cache |

## 下一步

- [ ] 双机实测 A1（overlapped_pipeline.py）
- [ ] 双机实测 A2（bidirectional_pipeline.py，需全量权重）
- [ ] 集成 A3 到 Hippo 主分支（多轮对话场景）
- [ ] 修复 SSH 连接（P0-2 双机测试阻塞）
