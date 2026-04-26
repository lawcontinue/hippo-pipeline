# 双机 Pipeline 测试指南

## 概述

测试 Gemma3-12B-it 4bit 模型在双 Mac Mini 上的 Pipeline 分布式推理。

## 环境信息

- **模型**: Gemma3-12B-it 4bit (48 层，7.5GB)
- **Rank 0**: 本机 (192.168.1.10) - 前 24 层 (~3.75GB)
- **Rank 1**: 第二台 (192.168.1.11) - 后 24 层 (~3.75GB)

## 前置条件

### 第二台机器需要安装：

```bash
# 1. 升级 Python（需要 3.11+）
python3 --version  # 检查版本

# 2. 安装 MLX 和 mlx-lm
pip3 install --break-system-packages mlx mlx-lm

# 3. 测试 MLX
python3 -c "import mlx; print(mlx.__version__)"

# 4. 复制代码到第二台机器
scp -r hippo/pipeline deepsearch@192.168.1.11:~/workspace/
```

## 测试步骤

### 方案 A: 真正的双机测试（推荐）

**在第二台机器 (192.168.1.11) 上：**
```bash
cd ~/workspace/pipeline
python3 test_dual_rank.py --rank 1 --host 0.0.0.0
```

**在本机 (192.168.1.10) 上：**
```bash
cd /Users/deepsearch/.openclaw/workspace/hippo/pipeline
python3 test_dual_rank.py --rank 0 --host 192.168.1.11
```

### 方案 B: 单机模拟（用于调试）

在本机上开两个终端：

**终端 1 (Rank 1):**
```bash
cd /Users/deepsearch/.openclaw/workspace/hippo/pipeline
python3 test_dual_rank.py --rank 1 --host 127.0.0.1
```

**终端 2 (Rank 0):**
```bash
cd /Users/deepsearch/.openclaw/workspace/hippo/pipeline
python3 test_dual_rank.py --rank 0 --host 127.0.0.1
```

## 预期结果

- ✅ Rank 0: 加载 ~3.75GB，前向传播 ~100ms
- ✅ Rank 1: 加载 ~3.75GB，前向传播 ~100ms
- ✅ TCP 传输: ~50-100 MB/s
- ✅ 总延迟: < 1s

## 故障排查

### SSH 连接失败
```bash
# 在本机复制 SSH 公钥到第二台机器
ssh-copy-id deepsearch@192.168.1.11
```

### 第二台机器 Python 版本过低
```bash
# 使用 Homebrew 升级 Python
brew install python@3.13
```

### 内存不足
- 检查每台机器可用内存: `mx.metal.get_active_memory()`
- 如果 < 8GB，考虑使用更小的模型

## 下一步

1. ✅ TCP Transport 测试通过（单机）
2. ⏳ 双机 Pipeline 测试
3. ⏳ 端到端推理验证
4. ⏳ 性能基准测试（延迟、吞吐量）
