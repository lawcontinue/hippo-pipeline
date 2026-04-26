# Crit 独立核验报告 — Hippo Pipeline Parallelism

**审查者**: Crit ⚖️
**日期**: 2026-04-19 21:47
**审查范围**: hippo/pipeline/ 全部 4 个模块 + 架构决策
**评分**: B+ (82/100) — 框架正确，有 2 个 P0 + 3 个 P1 问题

---

## P0 问题（必须修复）

### P0-1: distributed_model.py 加载了完整模型再切片 — 违背核心设计目标

**文件**: `distributed_model.py` 第 51-56 行

```python
# 当前实现
model, _ = mlx_load(self.model_path)        # ← 加载了完整模型！
layers = get_layers(inner)
local_layers = layers[start:end]             # ← 然后切片
```

**问题**: `mlx_load()` 会把整个模型的所有层权重都加载到内存中。切片后虽然只保留部分层的引用，但**原始层的内存不会被释放**（Python GC 不保证即时回收，MLX 的 tensor 可能被缓存）。

**这意味着**: 对于 15.5GB 的 Qwen3.6-35B，`mlx_load()` 仍然会消耗 ~15GB 内存，然后切片到 ~7GB，但峰值已经 OOM 了。

**与 Exo 的区别**: Exo 的 `slice_transformer_blocks()` 是在加载**之前**就告诉加载器只读需要的层。具体来说，Exo 使用 `adapter.slice_transformer_blocks(start, end)` 修改模型结构，然后 `mx.eval()` 时只物化需要的层。

**修复方案**:
1. 使用 `mlx_lm` 的 `load_config` 先读取 config.json
2. 手动构建只包含 `start:end` 层的模型
3. 只加载这些层的 safetensors 权重
4. 或者：加载后立即删除不需要的层 + `gc.collect()` + `mx.synchronize()`

### P0-2: 跨机器通信依赖 MPI — 未验证可行性

**发现**: MLX 的 `ring` 后端只支持单机多进程。跨机器需要 `mpi` 后端 + `mpi4py`。

**问题**:
1. `mpi4py` 需要 MPI 库（如 OpenMPI），macOS 上安装复杂
2. MPI 通过 Thunderbolt 通信的性能未验证
3. Exo 使用的是 `mx.distributed.init()` 的默认后端（可能是 ring + 自定义发现）

**修复方案**:
1. 验证 `mpi4py` 在 macOS 上的安装可行性
2. 或者：不走 MLX distributed，用自定义 TCP/gRPC 传 tensor（更可控）
3. 或者：使用 `nccl` 后端（需要确认 macOS 支持）

---

## P1 问题（建议修复）

### P1-1: PipelineLastLayer 的 all_gather 在 Pipeline 模式下不正确

**文件**: `pipeline_layers.py` 第 62-72 行

```python
# 当前实现
gathered = mx.distributed.all_gather(output, group=self.group)
return gathered[-output.shape[0]:]  # 取最后一个 rank 的输出
```

**问题**: 在 Pipeline Parallelism 中，只有最后一个 rank（`world_size - 1`）会产生最终 logits。其他 rank 的 `output` 是中间 hidden states，all_gather 它们没有意义。

**Exo 的做法**: 只有非 prefill 模式才 all_gather，而且取的是 `[-output.shape[0]:]`（即最后一个 rank 的部分）。但这在 decode 阶段才需要。

**修复**: 参照 Exo，添加 `is_prefill` 标志，prefill 时跳过 all_gather。

### P1-2: 没有 KV Cache 分片策略

**问题**: 自回归生成时，KV Cache 是关键。当前实现没有处理 KV Cache 在分布式环境下的管理。

- 每个 rank 只缓存自己负责的层的 KV
- 但 attention 计算需要所有层的 KV 吗？不是——每层只用自己的 KV
- 所以 KV Cache 天然可以按层分片，当前不需要额外处理

**结论**: 不是 bug，但需要在集成测试中验证。

### P1-3: tokenizer 加载方式错误

**文件**: `distributed_model.py` 第 49 行

```python
self.tokenizer = mlx_load(self.model_path, tokenizer_config={})
```

`mlx_load` 返回 `(model, tokenizer)` 元组，不是单独的 tokenizer。这行会抛异常。

**修复**: 
```python
_, self.tokenizer = mlx_load(self.model_path)
```
或者用 `mlx_lm.tokenizer_utils` 单独加载 tokenizer。

---

## P2 问题（可以后续优化）

### P2-1: 没有 embedding 和 lm_head 的处理

在 Pipeline 中：
- **Embedding**（token → hidden）: 只在 rank 0 执行
- **lm_head**（hidden → logits）: 只在最后一个 rank 执行

当前实现只处理了中间层，没有处理 embedding 和 lm_head 的分配。

### P2-2: 没有错误恢复机制

如果一个 rank 崩溃，其他 rank 会永远等待 recv。需要超时机制。

### P2-3: memory_weighted_split 边界情况

如果一台设备内存为 0，`ratio = 0`，`round(n_layers * 0) = 0`，但 `max(1, ...)` 会强制分配 1 层。这可能导致内存不足的设备被分配层。

---

## 架构决策核验

| 决策 | Crit 评价 | 备注 |
|------|----------|------|
| 使用 MLX 而非 llama.cpp | ✅ 正确 | MLX 支持精确分层加载，GGUF 不支持 |
| Pipeline Parallelism | ✅ 正确 | 适合我们的带宽高、延迟低场景 |
| ring 后端 | ⚠️ 需验证 | 跨机器可能需要 MPI |
| 先加载全模型再切片 | ❌ **错误** | 峰值内存仍然是全模型大小 |
| safetensors 格式 | ✅ 正确 | 精确加载必需 |

---

## 评分明细

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 架构设计 | 20 | 25 | Pipeline 方向正确，但加载策略有 P0 问题 |
| 代码质量 | 18 | 20 | 清晰、有文档，但 distributed_model 有 bug |
| 测试覆盖 | 12 | 20 | 只有单元测试，没有集成/E2E 测试 |
| 可行性 | 15 | 20 | MPI 可行性未验证，内存加载有 P0 |
| 文档 | 17 | 15 | 良好的 docstring 和注释 |
| **总分** | **82** | **100** | **B+** |

---

## 修复优先级

1. **P0-1** (今天): 修改 distributed_model.py，加载后立即释放不需要的层
2. **P0-2** (明天): 验证 mpi4py 或实现自定义 tensor 传输
3. **P1-3** (今天): 修复 tokenizer 加载方式

**Crit 结论**: 框架方向正确，但核心的"只加载分配层"还没有真正实现。当前实现是"加载全模型再切片"，峰值内存仍然是全模型大小。**必须修复 P0-1 才能验证分布式推理可行性。**

---

_Crit 签名_: ⚖️ "框架正确，但魔鬼在细节。加载策略是成败关键。"
