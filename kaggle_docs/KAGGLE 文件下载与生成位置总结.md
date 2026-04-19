# Kaggle 文件下载与生成位置总结

**日期**: 2026-04-17  
**依据**: 实际代码，实事求是

---

## 📊 核心文件下载清单

### ✅ 必须下载的 6 个文件（总计 ~70 KB）

| # | 文件名 | Kaggle 路径 | 大小 | 生成代码 | 重要性 |
|---|--------|-------------|------|----------|--------|
| 1 | `execution.log` | `/kaggle/working/gpu_profiling_system/` | ~56 KB | [`main.py:26-31`](file:///e:/GPU_Profiling_System/src/main.py#L26-L31) | ⭐⭐⭐⭐⭐ |
| 2 | `pipeline_log.jsonl` | `/kaggle/working/gpu_profiling_system/.state/` | ~1 KB | [`pipeline.py:67`](file:///e:/GPU_Profiling_System/src/domain/pipeline.py#L67) | ⭐⭐⭐⭐⭐ |
| 3 | `session_log.jsonl` | `/kaggle/working/gpu_profiling_system/.state/` | ~3 KB | [`state_persist.py:18-21`](file:///e:/GPU_Profiling_System/src/infrastructure/state_persist.py#L18-L21) | ⭐⭐⭐⭐ |
| 4 | `audit_report.md` | `/kaggle/working/gpu_profiling_system/audit/` | ~0.25 KB | [`audit_report.py:222-238`](file:///e:/GPU_Profiling_System/src/application/audit_report.py#L222-L238) | ⭐⭐⭐⭐ |
| 5 | `audit_report.json` | `/kaggle/working/gpu_profiling_system/audit/` | ~10 KB | [`audit_report.py:222-238`](file:///e:/GPU_Profiling_System/src/application/audit_report.py#L222-L238) | ⭐⭐⭐ |
| 6 | `results.json` | `/kaggle/working/gpu_profiling_system/` | ~1 KB | [`main.py:401-446`](file:///e:/GPU_Profiling_System/src/main.py#L401-L446) | ⭐⭐⭐⭐⭐ |

**总计大小**: ~70-290 KB

---

## 🗺️ Kaggle 目录结构

```
/kaggle/working/gpu_profiling_system/
├── .state/
│   ├── session_log.jsonl          ← AgentLoop 状态日志
│   └── pipeline_log.jsonl         ← Pipeline 状态日志
├── audit/
│   ├── audit_report.md            ← 审计报告（Markdown）
│   └── audit_report.json          ← 审计报告（JSON）
├── execution.log                   ← 完整执行日志（stdout 捕获）
└── results.json                    ← 最终测量结果
```

---

## 🔍 文件生成时机

### 1. execution.log

**生成方式**: 
- ❌ **不是主动写入的文件**
- ✅ **stdout 输出被 Kaggle 自动捕获**

**代码**: [`main.py:26-31`](file:///e:/GPU_Profiling_System/src/main.py#L26-L31)
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,  # ← 输出到 stdout
)
```

**内容**:
- 所有 `print()` 输出
- 所有 `logging.info/warning/error()` 输出
- Sandbox 路径配置
- 编译错误详情

---

### 2. session_log.jsonl

**生成方式**: 
- ✅ **StatePersister 主动写入**
- ✅ **JSONL 格式（每行一个 JSON 对象）**

**代码**: [`state_persist.py:18-21`](file:///e:/GPU_Profiling_System/src/infrastructure/state_persist.py#L18-L21)
```python
class StatePersister:
    def __init__(self, log_dir: str, filename: str = "session_log.jsonl") -> None:
        self._log_path = os.path.join(log_dir, filename)
        os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)
```

**内容**:
- 工具执行日志
- 权限决策日志
- 错误日志
- 不变量违规日志

---

### 3. pipeline_log.jsonl

**生成方式**: 
- ✅ **Pipeline 初始化时创建**
- ✅ **JSONL 格式**

**代码**: [`pipeline.py:67`](file:///e:/GPU_Profiling_System/src/domain/pipeline.py#L67)
```python
class Pipeline:
    def __init__(self, stages: list[PipelineStep], ...) -> None:
        self._persister = StatePersister(log_dir=state_dir, filename="pipeline_log.jsonl")
```

**内容**:
- Pipeline 启动/完成日志
- Stage 开始/结束日志
- 重试日志
- Metric 反馈日志

---

### 4. audit_report.md / audit_report.json

**生成方式**: 
- ✅ **Pipeline 完成后主动写入**
- ✅ **同时生成 Markdown 和 JSON 格式**

**代码**: [`audit_report.py:222-238`](file:///e:/GPU_Profiling_System/src/application/audit_report.py#L222-L238)
```python
def save(self, output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON report
    json_path = os.path.join(output_dir, "audit_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Markdown report
    md_path = os.path.join(output_dir, "audit_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(self.to_markdown())
    
    return json_path, md_path
```

**调用时机**: [`main.py:354-357`](file:///e:/GPU_Profiling_System/src/main.py#L354-L357)
```python
audit_dir = os.path.join(output_dir, "audit")
os.makedirs(audit_dir, exist_ok=True)
json_path, md_path = audit.save(audit_dir)
```

**内容**:
- Stage 执行时间线
- Handoff 验证结果
- Circuit Breaker 状态
- 工具执行摘要
- P7 合规性审计
- 最终测量结果

---

### 5. results.json

**生成方式**: 
- ✅ **Pipeline 完成后主动写入**
- ✅ **JSON 格式**

**代码**: [`main.py:401-446`](file:///e:/GPU_Profiling_System/src/main.py#L401-L446)
```python
def _assemble_final_results(output_dir, hardware_results, pipeline_data, target_spec):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    
    # ... 组装测量结果 ...
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    
    return results_path
```

**调用时机**: [`main.py:318-329`](file:///e:/GPU_Profiling_System/src/main.py#L318-L329)

**内容**:
- 所有测量结果
- 交叉验证数据
- 证据文件列表
- 方法论说明
- 目标列表

---

## ❌ 不需要下载的文件

### 已清理的临时文件

| 文件 | 原因 |
|------|------|
| `source.cu` | 临时 CUDA 源文件，每次编译重新生成 |
| `freq_probe` | 编译产物（~1 MB），占用空间大 |
| `freq_event_probe` | 编译产物（~1 MB），占用空间大 |
| `freq_event_timed` | 编译产物（~1 MB），占用空间大 |
| `cmd_*.log` | 冗余命令日志 |
| `debug_messages_*.json` | 冗余调试文件 |

**清理工具**: [`cleanup.py`](file:///e:/GPU_Profiling_System/cleanup.py)

---

## 📥 Kaggle 下载方法

### 方法 1: Kaggle UI 下载

1. 打开 Kaggle Notebook
2. 点击 "Output" 标签
3. 导航到对应目录下载文件

### 方法 2: Kaggle API 下载

```bash
# 下载 execution.log
kaggle kernels output <username>/<kernel-id> -p execution.log

# 下载 .state 目录
kaggle kernels output <username>/<kernel-id> -p .state/session_log.jsonl
kaggle kernels output <username>/<kernel-id> -p .state/pipeline_log.jsonl

# 下载 audit 目录
kaggle kernels output <username>/<kernel-id> -p audit/audit_report.md
kaggle kernels output <username>/<kernel-id> -p audit/audit_report.json

# 下载 results.json
kaggle kernels output <username>/<kernel-id> -p results.json
```

### 方法 3: Notebook 内下载

```python
from IPython.display import FileLink

# 创建下载链接
FileLink('execution.log')
FileLink('.state/session_log.jsonl')
FileLink('.state/pipeline_log.jsonl')
FileLink('audit/audit_report.md')
FileLink('audit/audit_report.json')
FileLink('results.json')
```

---

## 🎯 下载优先级

### 🔴 优先级 1（必须下载）- 61 KB

```
execution.log          (56 KB)
pipeline_log.jsonl     (1 KB)
session_log.jsonl      (3 KB)
results.json           (1 KB)
```

**用途**: 90% 的错误诊断

### 🟡 优先级 2（建议下载）- 10 KB

```
audit_report.md        (0.25 KB)
audit_report.json      (10 KB)
```

**用途**: 深度分析 Pipeline 执行质量

### 🟢 优先级 3（无需下载）

```
source.cu              # 临时文件
freq_*                 # 编译产物
cmd_*.log              # 冗余日志
debug_messages_*.json  # 冗余调试
```

**原因**: 临时生成、占用空间大、无分析价值

---

## 📊 文件生成流程图

```
Pipeline 启动
    ↓
1. logging.basicConfig(stream=sys.stdout)
   → execution.log (stdout 捕获)
    ↓
2. StatePersister(log_dir=".state")
   → session_log.jsonl
   → pipeline_log.jsonl
    ↓
3. Pipeline 执行
   → 记录工具调用到 session_log.jsonl
   → 记录阶段状态到 pipeline_log.jsonl
   → print/logging 输出到 execution.log
    ↓
4. Pipeline 完成
   ↓
   PipelineAuditReport.save("audit/")
   → audit_report.md
   → audit_report.json
    ↓
   _assemble_final_results()
   → results.json
```

---

## ✅ 总结

### 核心要点

1. ✅ **6 个核心文件** 总计 ~70 KB，包含所有必要信息
2. ✅ **execution.log** 是 stdout 捕获，不是主动写入
3. ✅ **.state/** 目录包含 JSONL 状态日志
4. ✅ **audit/** 目录包含审计报告
5. ❌ **不要下载** 编译产物和临时文件（~3 MB）

### 下载清单（6 个文件）

- ✅ `execution.log` - 完整执行日志
- ✅ `pipeline_log.jsonl` - Pipeline 状态
- ✅ `session_log.jsonl` - AgentLoop 状态
- ✅ `audit_report.md` - 审计报告（Markdown）
- ✅ `audit_report.json` - 审计报告（JSON）
- ✅ `results.json` - 最终测量结果

### 参考资料

- 详细分析: [`KAGGLE 文件生成位置详解.md`](file:///e:/GPU_Profiling_System/KAGGLE 文件生成位置详解.md)
- 下载指南: [`KAGGLE 文件下载指南.md`](file:///e:/GPU_Profiling_System/KAGGLE 文件下载指南.md)
- 清理工具: [`cleanup.py`](file:///e:/GPU_Profiling_System/cleanup.py)

---

**分析人**: AI Assistant  
**分析日期**: 2026-04-17  
**分析依据**: 实际代码，实事求是
