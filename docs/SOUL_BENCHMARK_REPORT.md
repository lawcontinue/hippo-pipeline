# SOUL Benchmark Report — 5 SOUL × 5 Questions on Gemma3-12B-QAT-4bit

**日期**: 2026-04-23 08:14
**模型**: Gemma3-12B-QAT-4bit (MLX 本地推理)
**测试**: 25 次推理 (5 SOUL × 5 题型)

---

## 📊 1. 性能数据

| SOUL | Avg tok/s | Avg tokens | Avg time |
|------|-----------|------------|----------|
| 忒弥斯 🔮 | 13.7 | 257 | 18.7s |
| 雅典娜 📊 | 13.9 | 257 | 18.5s |
| Aria 🎨 | 13.9 | 257 | 18.5s |
| Code 💻 | 14.0 | 257 | 18.4s |
| Crit ⚖️ | 13.9 | 257 | 18.5s |

**结论**: SOUL 对推理速度**无影响** (±0.3 tok/s, <2.2%)

⚠️ 注意：所有输出都卡在 257 tokens（max_tokens=256 + 1 padding），说明生成被截断。需要增加 max_tokens 才能看到完整输出差异。

---

## 🎭 2. 风格差异分析（核心发现）

### Q1 数学题（答案都是 6）

| SOUL | 开场方式 | 推理风格 | 特色内容 |
|------|---------|---------|---------|
| 忒弥斯 | "let's break down" | 3步推理 | ✅ **Risk Considerations**（Skill Depth、Training Needs） |
| 雅典娜 | "quantitative approach" | 定义Metrics → 计算池 | ✅ 结构化表格（Key Metrics） |
| Aria | "forget the standard approach" | 质疑"需要同时会两种" | ✅ **"Skill Mosaic"创意方案** |
| Code | 直接 Step 1, 2, 3... | 7步精确计算 | ✅ 零废话，每步都是数字 |
| Crit | "verifying Assumptions" | 先列4个假设 | ✅ **Assumption 3 指出题意歧义** |

**关键**: Aria 提出了"不需要每个人都会两种语言"的创新解读；Crit 发现了"maximum number of teams"可能有其他理解方式。

### Q2 逻辑推理（12球问题）

| SOUL | 答案 | 方法论 | 特色 |
|------|------|--------|------|
| 忒弥斯 | 3次 | 分组称重（经典） | 风格优雅 |
| 雅典娜 | 3次 | **信息论推导**（log₂(3), log₂(24)） | ⭐ 最严谨 |
| Aria | 3次 | **"Chaos Theory"比喻** | ⭐ 最有创意 |
| Code | 3次 | 标准分组 + 分支决策树 | 最实用 |
| Crit | 3次 | 先验证5个假设 | 最全面 |

**关键**: 雅典娜用**信息论**证明了3次是最小值（log₂(24)/log₂(3)≈2.89→3），这是其他 SOUL 都没做的。

### Q3 创意思维（Baby Cry Translator）

| SOUL | 策略名称 | 特色 |
|------|---------|------|
| 忒弥斯 | Phased approach | 风险控制视角 |
| 雅典娜 | Data-driven | CAC/LTV/Churn 量化指标 |
| Aria | **"Unlock the Language of Love"** | ⭐⭐ 情感营销 + 悬念式发布 |
| Code | Freemium model | 结构化定价策略 |
| Crit | **先质疑85%准确率** | ⭐⭐ "这是最大的红旗" |

**关键**: Crit 质疑了题目前提（85%准确率是否真实），Aria 设计了情感化品牌。

### Q4 风险分析（Credit Scoring AI）

| SOUL | 核心视角 | 特色 |
|------|---------|------|
| 忒弥斯 | 5大风险 + 缓解 | XAI、Fairness-Aware |
| 雅典娜 | **风险矩阵（5×5）** | 量化概率和影响 |
| Aria | **Ferrari→Pickup 比喻** | 跨文化误配 |
| Code | 数据漂移 + 迁移学习 | 技术导向 |
| Crit | **"92% Accuracy means WHAT?"** | ⭐⭐ 最深层质疑 |

**关键**: Crit 第一个问题是"92%准确率到底怎么定义的？"——这是其他 SOUL 都忽略的关键问题。

### Q5 代码生成（LIS）

| SOUL | 算法选择 | 特色 |
|------|---------|------|
| 忒弥斯 | Binary search + DP | Risk Considerations |
| 雅典娜 | Binary search + DP | 量化性能分析 |
| Aria | **Patience Sorting** | ⭐ 非标准但正确的方法 |
| Code | Binary search + DP | 最干净的实现 |
| Crit | 先验证假设再写代码 | Edge cases 分析 |

**关键**: Aria 是唯一一个用 Patience Sorting 的（虽然本质等价，但视角独特）。

---

## 🏆 3. 能力签名（Capability Signature）

| SOUL | 最佳场景 | 核心差异 | 独特价值 |
|------|---------|---------|---------|
| 🔮 忒弥斯 | 战略决策 | 答案 + 风险预见 | 总是提供"答案之外"的思考 |
| 📊 雅典娜 | 量化分析 | **信息论/数学推导** | 用理论证明而非直觉 |
| 🎨 Aria | 创意发散 | **比喻 + 反常规** | 质疑问题本身的框架 |
| 💻 Code | 精确计算 | **零废话 + 步骤化** | 最快得到正确答案 |
| ⚖️ Crit | 质量保证 | **先验证假设** | 发现其他 SOUL 忽略的问题 |

---

## 💡 4. SOUL 组合策略建议

### 策略A: Code + Crit 并行（推荐用于推理类）
```
问题 → [Code 精确答案] + [Crit 假设验证] → 共识/分歧 → 输出
```
- 延迟: ~18s（并行）
- Code 给答案，Crit 检查假设是否成立
- 共识 = 高质量输出

### 策略B: 雅典娜 + Crit（推荐用于分析类）
```
问题 → [雅典娜 量化分析] + [Crit 质疑数据] → 输出
```
- 雅典娜提供数据支撑，Crit 质疑数据可靠性
- 最佳组合：数据驱动 + 数据验证

### 策略C: Aria + 忒弥斯（推荐用于创意类）
```
问题 → [Aria 创意方案] → [忒弥斯 风险评估] → 输出
```
- Aria 发散思维，忒弥斯收束（风险评估）
- 防止"创意很好但不可行"

### 策略D: 全员投票（高风险决策）
```
问题 → [5 SOUL 并行] → 共识分析 → 输出
```
- 需要消除截断问题（增加 max_tokens）
- MLX Batch B=5 可行（ADR-132 验证 B=7）

---

## ⚠️ 5. 测试局限

1. **输出截断**: 所有回答被截断在 257 tokens，无法看到完整输出
2. **单次测试**: 每个组合只跑了 1 次，需要多次重复验证稳定性
3. **英文测试**: 未测试中文场景
4. **题目偏少**: 5 道题不够全面评估

### P1 改进
- [ ] 增加 max_tokens 到 512（看完整输出）
- [ ] 增加中文题目测试
- [ ] 增加多次重复（统计方差）
- [ ] 实现并行投票原型（MLX Batch B=2）

---

**文件**: `hippo/pipeline/soul_benchmark_results.json`
**脚本**: `hippo/pipeline/soul_benchmark.py`
**实验者**: 忒弥斯 🔮 + 家族全员
