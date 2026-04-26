# P0-1 批量处理优化实施计划

**批准日期**: 2026-04-21 17:31
**批准人**: 哥哥
**负责人**: Code 💻
**状态**: ⏳ 实施中

---

## 🎯 优化目标

**当前性能**: 8.29 tok/s
**目标性能**: 42 tok/s
**预期提升**: **5x**

---

## 📊 技术方案

### 核心思路

**当前实现**（1 token/step）:
```
R0: 生成 token_N → 发送 hidden_N → 等待 R1
R1: 接收 hidden_N → fwd_N → lm_head_N → 发送 logits_N → 等待 R0
```

**优化实现**（8 tokens/step）:
```
R0: 生成 token_{N..N+7} → 批量发送 hidden_{N..N+7} → 等待 R1
R1: 批量接收 hidden_{N..N+7} → 批量 fwd_{N..N+7} → 批量 lm_head_{N..N+7} → 发送 logits_{N..N+7} → 等待 R0
```

**关键优势**:
- R1 fwd 可以利用 GPU 并行（23 层 × 8 tokens）
- lm_head 可以利用 GPU 并行（vocab_size × 8 tokens）
- 减少 asyncio 开销（1 次 vs 8 次）

---

## 🔧 实施步骤

### Step 1: 修改 R0 发送逻辑（30 分钟）

**文件**: `sharded_inference.py`

**当前代码**（rank0_generate）:
```python
for step in range(1, max_tokens):
    # 生成 1 个 token
    token_id = sample_from_topk(...)
    generated_tokens.append(token_id)

    # 发送 1 个 hidden state
    await sender.send(hidden_states, rank=0)

    # 接收 1 个 logits
    frame = await receiver.recv(timeout=300)
```

**优化代码**:
```python
BATCH_SIZE = 8  # 可配置：2/4/8

for step in range(1, max_tokens, BATCH_SIZE):
    # 生成 BATCH_SIZE 个 tokens
    batch_tokens = []
    batch_hidden = []
    for i in range(BATCH_SIZE):
        if step + i >= max_tokens:
            break
        token_id = sample_from_topk(...)
        batch_tokens.append(token_id)
        batch_hidden.append(hidden_states)

    # 批量发送 BATCH_SIZE 个 hidden states
    batch_hidden_tensor = mx.concatenate(batch_hidden, axis=1)  # (1, BATCH_SIZE, HIDDEN_SIZE)
    await sender.send(batch_hidden_tensor, rank=0)

    # 批量接收 BATCH_SIZE 个 logits
    frame = await receiver.recv(timeout=300)
    batch_logits = frame_to_mlx(frame)  # (1, BATCH_SIZE, vocab_size)

    # 批量采样
    for i, logits in enumerate(batch_logits):
        token_id = sample_from_topk(logits)
        batch_tokens[i] = token_id
```

---

### Step 2: 修改 R1 接收逻辑（30 分钟）

**文件**: `sharded_inference.py`

**当前代码**（rank1_serve）:
```python
while True:
    # 接收 1 个 hidden state
    frame = await pending_recv_task
    hidden_states = frame_to_mlx(frame)  # (1, 1, HIDDEN_SIZE)

    # Fwd 1 个 token
    t_fwd = time.time()
    h = await asyncio.to_thread(_do_forward, hidden_states, offset)
    t_fwd = time.time() - t_fwd

    # lm_head 1 个 token
    t_lm = time.time()
    logits = await asyncio.to_thread(_do_lm_head, h)
    t_lm = time.time() - t_lm

    # 发送 1 个 logits
    await sender.send(logits, rank=1)
```

**优化代码**:
```python
while True:
    # 接收 BATCH_SIZE 个 hidden states
    frame = await pending_recv_task
    hidden_states = frame_to_mlx(frame)  # (1, BATCH_SIZE, HIDDEN_SIZE)
    batch_size = hidden_states.shape[1]

    # 批量 fwd BATCH_SIZE 个 tokens
    t_fwd = time.time()
    h = await asyncio.to_thread(_do_forward_batch, hidden_states, offset)
    t_fwd = time.time() - t_fwd

    # 批量 lm_head BATCH_SIZE 个 tokens
    t_lm = time.time()
    batch_logits = await asyncio.to_thread(_do_lm_head_batch, h)
    t_lm = time.time() - t_lm

    # 批量发送 BATCH_SIZE 个 logits
    await sender.send(batch_logits, rank=1)

def _do_forward_batch(hidden_states, offset):
    """批量前向传播（R1 layers 24-47）"""
    h = hidden_states  # (1, BATCH_SIZE, HIDDEN_SIZE)
    batch_size = h.shape[1]

    for i in range(shard.start_layer, shard.end_layer):
        h = forward_layer(h, i, weights, kv_cache=kv_cache_r1, offset=offset)

    mx.eval(h)
    mx.synchronize()
    return h

def _do_lm_head_batch(hidden_states):
    """批量 lm_head + top-k 压缩"""
    batch_size = hidden_states.shape[1]

    # RMS norm
    last_hidden = hidden_states  # (1, BATCH_SIZE, HIDDEN_SIZE)
    last_hidden = rms_norm(last_hidden, norm_w)

    # quantized_matmul（批量）
    logits = mx.quantized_matmul(last_hidden, lm_w, lm_s, lm_b, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(logits)

    # 批量 top-k 压缩
    batch_topk_indices = []
    batch_topk_values = []

    for i in range(batch_size):
        top_idx, top_val = compress_logits_topk(logits[:, i:i+1, :])
        batch_topk_indices.append(top_idx)
        batch_topk_values.append(top_val)

    # 打包
    packed = mx.concatenate([
        mx.concatenate(batch_topk_indices, axis=1).astype(mx.float32),
        mx.concatenate(batch_topk_values, axis=1).astype(mx.float32)
    ], axis=-1)

    mx.eval(packed)
    return packed
```

---

### Step 3: 测试不同 batch size（30 分钟）

**测试配置**:
- BATCH_SIZE = 2
- BATCH_SIZE = 4
- BATCH_SIZE = 8

**测试脚本**:
```bash
# R0
python3 sharded_inference.py --rank 0 --host 192.168.1.11 --batch-size 2

# R1
python3 sharded_inference.py --rank 1 --host 0.0.0.0 --batch-size 2
```

**性能指标**:
- tok/s
- Step 时间
- 内存使用

---

### Step 4: 性能对比和调优（30 分钟）

**对比基准**:
- 当前实现（batch=1）: 8.29 tok/s
- 目标（batch=8）: 42 tok/s

**调优方向**:
1. 如果 batch=8 内存不足: 降低到 4
2. 如果 batch=8 没有加速: 检查实现
3. 如果 batch=8 加速明显: 尝试 16

---

## 🎯 成功标准

**定量指标**:
- ✅ tok/s > 30（至少 3.6x 提升）
- ✅ 内存 < 8GB（R1 内存限制）
- ✅ 功能正确（生成文本一致）

**定性指标**:
- ✅ 代码可读性
- ✅ 错误处理
- ✅ 向后兼容（支持 batch_size=1）

---

## 📊 预期时间表

| 步骤 | 预计时间 | 状态 |
|------|---------|------|
| Step 1: 修改 R0 发送逻辑 | 30 分钟 | ⏳ 待开始 |
| Step 2: 修改 R1 接收逻辑 | 30 分钟 | ⏳ 待开始 |
| Step 3: 测试不同 batch size | 30 分钟 | ⏳ 待开始 |
| Step 4: 性能对比和调优 | 30 分钟 | ⏳ 待开始 |
| **总计** | **2 小时** | - |

---

## ⚠️ 风险和缓解

**风险 1**: GPU 内存不足（batch=8）
- **缓解**: 动态调整 batch_size（8 → 4 → 2）

**风险 2**: KV cache 实现复杂（batch 场景）
- **缓解**: 先实现简化版（每次重建 cache）

**风险 3**: 性能提升不如预期（< 2x）
- **缓解**: 深入分析瓶颈，可能需要 P1-1 优化

---

## 📝 验收标准

**Crit 条件**:
1. ✅ 功能正确性（生成文本与 baseline 一致）
2. ✅ 性能提升（tok/s > 30）
3. ✅ 内存可控（< 8GB）
4. ✅ 代码质量（可读性、错误处理）

**家族验收**:
- Code: 实现正确性
- 雅典娜: 性能数据
- Crit: 风险评估

---

_Code签名_: 💻 立即开始实施！
_哥哥批准_: 2026-04-21 17:31
