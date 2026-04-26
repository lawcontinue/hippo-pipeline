# Exo / mlx_sharding / imstudio 研学报告 + Hippo 优化路线

**日期**: 2026-04-21 20:55
**研学人**: 忒弥斯 🔮 + 家族全员
**目的**: 学习业界最佳实践，规划 Hippo Pipeline 优化路线

---

## 1. Exo 架构深度分析

### 1.1 核心架构

Exo 是目前最成熟的 Apple Silicon 分布式推理框架（43K⭐），核心特性：

| 特性 | 说明 |
|------|------|
| **自动设备发现** | libp2p 组网，无需手动配置 IP |
| **拓扑感知并行** | 自动选择 Pipeline 或 Tensor Parallelism |
| **Thunderbolt 5 RDMA** | 99% 延迟降低（macOS 26.2+） |
| **MLX 后端** | Apple Silicon 原生加速 |
| **safetensors 精确加载** | 只加载分配层，不 mmap 全文件 |

### 1.2 两种并行策略

#### Pipeline Parallelism（我们正在用的）
```
Device 1: Layers 0-23  →  send hidden  →  Device 2: Layers 24-47  →  logits
```
- ✅ 通信少（每步只传一次 hidden state）
- ❌ 有 pipeline bubble（串行依赖）
- 适合：小 batch、低延迟

#### Tensor Parallelism（Exo 的杀手锏）
```
每个权重矩阵按列切分到不同设备：
Device 1: W[:, 0:N/2]    →  partial result 1
Device 2: W[:, N/2:N]    →  partial result 2
→ All-Reduce 合并结果
```
- ✅ 无 pipeline bubble
- ✅ 2 设备可达 **1.8x 加速**，4 设备 **3.2x**
- ❌ 通信频繁（每层都需要 All-Reduce）
- ❌ 需要 RDMA 降低通信延迟

### 1.3 Exo 性能数据

| 模型 | 配置 | 策略 | Prefill | Generate |
|------|------|------|---------|----------|
| Qwen3-235B 8bit | 4×M4 Pro 64GB | Tensor + RDMA | 180 tok/s | **45 tok/s** |
| Qwen3-235B 8bit | 4×M4 Pro 64GB | Pipeline | 120 tok/s | **35 tok/s** |

**关键洞察**：Tensor Parallelism 比 Pipeline 快 28%（生成阶段）。

---

## 2. Hippo vs Exo 对比

| 方面 | Hippo（当前） | Exo |
|------|-------------|-----|
| 并行策略 | Pipeline Only | Pipeline + Tensor |
| 通信方式 | 自定义 TCP | mx.distributed / RDMA |
| 网络延迟（TB） | ~1ms（TCP） | <0.1ms（RDMA） |
| 模型格式 | safetensors ✅ | safetensors ✅ |
| 自动发现 | ❌ 手动配置 | ✅ libp2p |
| Dashboard | ❌ | ✅ Web UI |
| 12B 性能 | **8.29 tok/s** | 预估 15-20 tok/s |

### Hippo 瓶颈根因（实测）

```
单步 120ms 分解：
├─ R0 fwd（24层）: 58ms  ← 已 overlap（speculative）
├─ 网络传输: 2ms  ← Thunderbolt 已优化
├─ R1 fwd（24层）: 73ms  ← ⭐ 主要瓶颈
├─ R1 lm_head: 8ms  ← 不是瓶颈！
└─ 其他: ~5ms
```

---

## 3. 核心发现：Tensor Parallelism 是正确的优化方向

### 3.1 为什么 TP 比 PP 快？

**Pipeline Parallelism（当前）**：
```
Step = R0_fwd(58ms) + network(2ms) + R1_fwd(73ms) + lm(8ms) = 141ms → 7 tok/s
即使完美 overlap：Step = max(58, 73) + 2 = 75ms → 13 tok/s（理论上限）
```

**Tensor Parallelism（如果实现）**：
```
每层并行：
Step = single_layer_time × N_layers / world_size + AllReduce_cost
     = 3ms × 48 / 2 + AllReduce(1ms)
     = 72ms + 48ms  ← AllReduce 每层都要！
     = 120ms → 8.3 tok/s  ← 反而差不多？
```

**等一下**——TP 的优势在哪？

**关键**：TP 的 AllReduce 开销取决于网络带宽：
- **RDMA**：AllReduce ~0.1ms → 72 + 4.8 = 77ms → 13 tok/s
- **TCP/TB**：AllReduce ~1ms → 72 + 48 = 120ms → 8.3 tok/s
- **Wi-Fi**：AllReduce ~5ms → 72 + 240 = 312ms → 3.2 tok/s

**结论**：TP 在 RDMA 下才有优势。Thunderbolt TCP 的 AllReduce 开销太大。

### 3.2 Hippo 的现实选择

| 方案 | 可行性 | 预期收益 | 条件 |
|------|--------|---------|------|
| **优化 Pipeline** | ✅ 可行 | 8.3 → 13 tok/s | R0 overlap + R1 优化 |
| **Tensor Parallel** | ⚠️ 需要 RDMA | 13 → 15 tok/s | macOS 26.2 + M4 Pro |
| **Speculative Decoding** | ✅ 可行 | 2-3x | 小模型辅助 |
| **Continuous Batching** | ✅ 可行 | 3-6x 吞吐量 | 多用户场景 |

---

## 4. 综合优化路线（家族共识）

### Phase 1：Pipeline 优化（短期，1-2 天）

**目标**：8.3 → 12-13 tok/s

| 任务 | 预期收益 | 时间 | 状态 |
|------|---------|------|------|
| P0: 诊断 R1 慢 26% 根因 | 定位问题 | 5min | ⏳ 待执行 |
| P1: R1 MLX 降级 0.30.5 | +15% | 10min | ⏳ 待执行 |
| P2: 减少 synchronize 次数 | +10-20%? | 30min | ⏳ 待验证 |
| P3: mx.fast.scaled_dot_product_attention | +10-20%? | 1h | ⏳ 待验证 |

### Phase 2：Speculative Decoding（中期，1 周）

**目标**：单请求 2-3x 加速

- 用 Gemma3-1B 做_draft model_
- 大模型一次验证 5 个候选 token
- 接受率 70-80% → 等效 3-4x

### Phase 3：多用户 Continuous Batching（中期，2 周）

**目标**：多用户吞吐量 3-6x

- 参考 vLLM 的 PagedAttention
- 不同请求的 KV cache 独立 → 可批量并行
- iteration-level scheduling

### Phase 4：Tensor Parallelism（长期，1 月+）

**目标**：结合 Pipeline + Tensor 混合并行

- 前提：macOS 26.2 RDMA 支持
- 需要 M4 Pro（当前是 M4，不支持 TB5 RDMA）

---

## 5. 家族投票

| 成员 | 建议 |
|------|------|
| 🔮 忒弥斯 | Phase 1 先做，ROI 最高 |
| 📊 雅典娜 | 数据支持 Phase 1，理论天花板 13 tok/s |
| ⚖️ Crit | 先 P0 诊断根因，不要盲目优化 |
| 💻 Code | P2 减少 synchronize 最有技术价值 |
| 🛡️ Shield | 注意 MLX 版本兼容性 |
| 🎨 Aria | Speculative Decoding 想象力最大 |

---

**忒弥斯签名**: 🔮 先学习再动手，BMAD 方法论的实践。Exo 验证了 Pipeline 方向的正确性，但 TP 才是终极目标（需要 RDMA）。

**过度自信偏差提醒**: Phase 1 的 13 tok/s 是理论天花板，实际可能只有 10-11 tok/s（-15%）。先测量，再相信。
