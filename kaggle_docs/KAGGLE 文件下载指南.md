# Kaggle 文件下载指南

## 📥 建议下载的文件（有助于解析错误）

### ✅ 必须下载的核心文件

| 文件名 | 大小 | 用途 | 重要性 |
|--------|------|------|--------|
| `execution.log` | ~50-200KB | **最重要的调试文件**，包含完整的执行流程、错误信息、Sandbox 路径、编译器输出 | ⭐⭐⭐⭐⭐ |
| `pipeline_log.jsonl` | ~10-50KB | **关键日志**，记录每个 pipeline 阶段的状态、工具调用、错误详情 | ⭐⭐⭐⭐⭐ |
| `session_log.jsonl` | ~5-20KB | **重要**，记录 AgentLoop 的状态持久化、工具执行历史 | ⭐⭐⭐⭐ |
| `audit_report.md` | ~10-30KB | **重要**，包含 M1/M2/M3/M4不变量违规分析、安全审计报告 | ⭐⭐⭐⭐ |

### ✅ 建议下载的分析文件

| 文件名 | 大小 | 用途 | 重要性 |
|--------|------|------|--------|
| `debug_messages_longcat_9msg_3tool.json` | ~5-15KB | **有用**，特定调试会话的完整消息记录（仅在分析 CodeGen 工具调用时需要） | ⭐⭐⭐ |
| `cmd_*.log` | ~1-5KB 每个 | **有用**，单个命令的执行日志（仅在分析具体命令错误时需要） | ⭐⭐ |

### ❌ 不建议下载的文件

| 文件名 | 原因 |
|--------|------|
| `source.cu` | **临时文件**，每次编译都会重新生成，无分析价值 |
| `freq_event_probe` | **编译产物**（~1MB 二进制文件），占用空间大，无分析价值 |
| `freq_event_timed` | **编译产物**（~1MB 二进制文件），占用空间大，无分析价值 |
| `freq_probe` | **编译产物**（~1MB 二进制文件），占用空间大，无分析价值 |
| 其他 `debug_messages_*.json` | **冗余调试文件**，大部分是历史测试遗留，除非特定调试需要否则无需下载 |

---

## 📊 下载优先级建议

### 🔴 优先级 1（必须下载）- 用于快速诊断
```
execution.log
pipeline_log.jsonl
session_log.jsonl
audit_report.md
```
**总大小**: ~100-300KB  
**用途**: 90% 的错误诊断都可以基于这些文件完成

### 🟡 优先级 2（建议下载）- 用于深度分析
```
debug_messages_longcat_9msg_3tool.json（如果 CodeGen 失败）
cmd_*.log（如果有命令执行错误）
```
**总大小**: ~20-50KB  
**用途**: 分析特定工具调用失败、命令执行错误

### 🟢 优先级 3（无需下载）- 临时文件
```
source.cu
freq_event_probe
freq_event_timed
freq_probe
其他 debug_messages_*.json（除非特别指定）
```
**总大小**: ~3-5MB  
**原因**: 临时生成、占用空间大、分析价值低

---

## 🔍 错误分析场景与文件需求

### 场景 1: CodeGen 编译失败
**需要下载**:
- ✅ `execution.log`（查看编译错误详情）
- ✅ `pipeline_log.jsonl`（查看 CodeGen 阶段状态）
- ✅ `debug_messages_longcat_9msg_3tool.json`（查看 CodeGen 工具调用）

**无需下载**:
- ❌ `source.cu`（临时文件）
- ❌ 二进制文件

### 场景 2: Pipeline 阶段失败
**需要下载**:
- ✅ `pipeline_log.jsonl`（查看失败阶段）
- ✅ `session_log.jsonl`（查看状态转换）
- ✅ `audit_report.md`（查看不变量违规）

**无需下载**:
- ❌ 二进制文件
- ❌ 临时源文件

### 场景 3: 路径错误/文件位置错误
**需要下载**:
- ✅ `execution.log`（查看 Sandbox 路径配置）
- ✅ `pipeline_log.jsonl`（查看文件写入位置）

**无需下载**:
- ❌ 二进制文件（位置错误是代码问题，不是文件本身问题）

### 场景 4: 工具调用错误
**需要下载**:
- ✅ `execution.log`（查看工具执行日志）
- ✅ `session_log.jsonl`（查看工具调用历史）
- ✅ 相关的 `debug_messages_*.json`（查看具体工具调用消息）

---

## 📋 下载命令示例

### 方法 1: Kaggle Notebook 下载
```python
from kaggle_secrets import UserSecretsClient
import os

# 必须下载的核心文件
core_files = [
    "kaggle_results/execution.log",
    "kaggle_results/pipeline_log.jsonl",
    "kaggle_results/session_log.jsonl",
    "kaggle_results/audit_report.md",
]

# 选择性下载（根据错误类型）
optional_files = [
    "kaggle_results/debug_messages_longcat_9msg_3tool.json",
    # "kaggle_results/cmd_*.log",  # 按需下载
]

# 不要下载的文件
# - kaggle_results/source.cu
# - kaggle_results/freq_*
# - kaggle_results/probe_binary
```

### 方法 2: Kaggle API 下载
```bash
# 下载核心文件
kaggle kernels output <username>/<kernel-id> -p kaggle_results/execution.log
kaggle kernels output <username>/<kernel-id> -p kaggle_results/pipeline_log.jsonl
kaggle kernels output <username>/<kernel-id> -p kaggle_results/session_log.jsonl
kaggle kernels output <username>/<kernel-id> -p kaggle_results/audit_report.md

# 按需下载调试文件
# kaggle kernels output <username>/<kernel-id> -p kaggle_results/debug_messages_longcat_9msg_3tool.json
```

---

## 💡 文件分析技巧

### execution.log 关键信息
```bash
# 查看 Sandbox 路径配置
grep -n "Sandbox" execution.log | head -20

# 查看编译错误
grep -n "nvcc\|compilation\|error" execution.log

# 查看文件写入位置
grep -n "Writing\|Creating\|path" execution.log
```

### pipeline_log.jsonl 关键信息
```bash
# 查看每个阶段的状态
cat pipeline_log.jsonl | jq '.stage, .status, .error'

# 查看工具调用
cat pipeline_log.jsonl | jq 'select(.tool_calls > 0) | .tool_calls'
```

### session_log.jsonl 关键信息
```bash
# 查看工具执行历史
cat session_log.jsonl | jq '.payload | select(.tool_name != "__loop_state__")'
```

---

## 🎯 总结

**下载原则**:
1. ✅ **下载日志文件** - 包含错误信息、执行流程
2. ✅ **下载审计报告** - 包含不变量违规分析
3. ❌ **不下载编译产物** - 二进制文件占用空间大、无分析价值
4. ❌ **不下载临时文件** - source.cu 等文件会重新生成

**存储空间估算**:
- 核心日志文件：~100-300KB
- 完整调试文件：~500KB-1MB
- 包含二进制文件：~5-10MB（**不推荐**）

**最佳实践**:
- 优先下载核心日志文件
- 根据错误类型选择性下载调试文件
- 定期清理 kaggle_results 目录，只保留必要的日志
