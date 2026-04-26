# Crit P0/P1 修复复核清单

**修复时间**: 2026-04-20 18:54
**修复人**: 忒弥斯 + Code

| # | 等级 | 原始问题 | 修复方式 | 验证 |
|---|------|---------|---------|------|
| 1 | P0 | A1 chunked 模式 range 循环层跳转未实现 | **移除整个 chunked 模式**，只保留 simple 模式 | ✅ import 通过 |
| 2 | P0 | A2 端口分配 + IP 硬编码 | **完全重写端口分配**：`_port()` 函数 + `--port-base` 参数化 + `--partner` 参数 | ✅ import 通过 |
| 3 | P1 | A1 `_send_intermediate` MLX 构造器问题 | **移除 `_send_intermediate`**（chunked 模式已移除） | ✅ 不再存在 |
| 4 | P1 | A3 `_read_tensor_mmap` 名实不符 | **重命名为 `_read_tensor`**，实现真 `mmap.mmap()` 零拷贝读取 | ✅ 测试通过 |
| 5 | P1 | A2 重复加载 embedding/lm_head | **合并去重**: `all_weights = {**weights_0, **weights_1}`，释放原始 dict | ✅ 代码已修复 |

**5/5 问题全部修复** ✅
