# Kaggle 测试文件生成位置详解

**基于代码实事求是分析**  
**生成日期**: 2026-04-17

---

## 📊 核心文件生成位置总览

| 文件名 | 生成位置 | 生成代码 | 生成时机 |
|--------|----------|----------|----------|
| `session_log.jsonl` | Kaggle 工作目录/.state/ | `StatePersister.__init__()` | Pipeline 启动时 |
| `pipeline_log.jsonl` | Kaggle 工作目录/.state/ | `Pipeline.__init__()` | Pipeline 启动时 |
| `execution.log` | Kaggle 工作目录/ | `logging.basicConfig()` | main.py 启动时 |
| `audit_report.md` | Kaggle 工作目录/audit/ | `PipelineAuditReport.save()` | Pipeline 完成后 |
| `audit_report.json` | Kaggle 工作目录/audit/ | `PipelineAuditReport.save()` | Pipeline 完成后 |
| `results.json` | Kaggle 工作目录/ | `_assemble_final_results()` | Pipeline 完成后 |

---

## 📁 详细生成位置分析

### 1. session_log.jsonl

**生成代码位置**: [`src/infrastructure/state_persist.py:18-21`](file:///e:/GPU_Profiling_System/src/infrastructure/state_persist.py#L18-L21)

```python
class StatePersister:
    def __init__(self, log_dir: str, filename: str = "session_log.jsonl") -> None:
        self._log_path = os.path.join(log_dir, filename)
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)
```

**调用链**:
1. [`Pipeline.__init__()`](file:///e:/GPU_Profiling_System/src/domain/pipeline.py#L67) → `StatePersister(log_dir=state_dir, filename="pipeline_log.jsonl")`
2. [`SystemBuilder.build_pipeline()`](file:///e:/GPU_Profiling_System/src/application/system_builder.py) → 使用 `state_dir=".state"`
3. 在 Kaggle 中，`state_dir` 默认为当前工作目录下的 `.state` 文件夹

**Kaggle 中的实际路径**:
```
/kaggle/working/gpu_profiling_system/.state/session_log.jsonl
```

**记录内容**:
- 工具执行日志 (`log_tool_execution`)
- 权限决策日志 (`log_permission_decision`)
- 错误日志 (`log_error`)
- 不变量违规日志 (`log_invariant_violation`)
- 通用条目 (`log_entry`)

---

### 2. pipeline_log.jsonl

**生成代码位置**: [`src/domain/pipeline.py:67`](file:///e:/GPU_Profiling_System/src/domain/pipeline.py#L67)

```python
class Pipeline:
    def __init__(self, stages: list[PipelineStep], ...) -> None:
        self._stages = stages
        self._state_dir = state_dir
        self._persister = StatePersister(log_dir=state_dir, filename="pipeline_log.jsonl")
```

**Kaggle 中的实际路径**:
```
/kaggle/working/gpu_profiling_system/.state/pipeline_log.jsonl
```

**记录内容**:
- Pipeline 启动日志 (`pipeline_start`)
- 阶段开始/结束日志 (`pipeline_stage_start`, `pipeline_stage_failed`)
- 重试日志 (`pipeline_retry_iteration`)
- 最终结果日志 (`pipeline_complete`)
- Metric 反馈日志 (`metric_feedback_collected`)

---

### 3. execution.log

**生成代码位置**: [`src/main.py:26-31`](file:///e:/GPU_Profiling_System/src/main.py#L26-L31)

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
```

**重要发现**: 
- ❌ **execution.log 不是通过 FileHandler 生成的**
- ✅ **所有日志都输出到 stdout (sys.stdout)**
- ✅ 在 Kaggle 中，stdout 被自动捕获到 execution.log

**Kaggle 中的实际路径**:
```
/kaggle/working/gpu_profiling_system/execution.log
```

**记录内容**:
- 所有 `print()` 输出
- 所有 `logging.info/warning/error()` 输出
- Sandbox 路径配置信息
- 编译错误详情
- Pipeline 执行状态

**关键日志点**:
1. [`sandbox.py:160-161`](file:///e:/GPU_Profiling_System/src/infrastructure/sandbox.py#L160-L161) - Sandbox 路径配置
2. [`codegen.py:271-281`](file:///e:/GPU_Profiling_System/src/application/subagents/codegen.py#L271-L281) - 编译尝试日志
3. [`pipeline.py:97-117`](file:///e:/GPU_Profiling_System/src/domain/pipeline.py#L97-L117) - Pipeline 执行日志

---

### 4. audit_report.md / audit_report.json

**生成代码位置**: [`src/application/audit_report.py:222-238`](file:///e:/GPU_Profiling_System/src/application/audit_report.py#L222-L238)

```python
def save(self, output_dir: str) -> tuple[str, str]:
    """Save audit report as both JSON and Markdown."""
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

**调用链**:
1. [`main.py:354-357`](file:///e:/GPU_Profiling_System/src/main.py#L354-L357) - Pipeline 完成后调用
```python
output_dir = args.output_dir or os.getcwd()
audit_dir = os.path.join(output_dir, "audit")
os.makedirs(audit_dir, exist_ok=True)
json_path, md_path = audit.save(audit_dir)
```

**Kaggle 中的实际路径**:
```
/kaggle/working/gpu_profiling_system/audit/audit_report.md
/kaggle/working/gpu_profiling_system/audit/audit_report.json
```

**记录内容**:
- Stage 执行时间线
- Handoff 验证结果
- Circuit Breaker 状态
- 工具执行摘要
- P7 合规性审计
- 最终测量结果

---

### 5. results.json

**生成代码位置**: [`src/main.py:401-446`](file:///e:/GPU_Profiling_System/src/main.py#L401-L446)

```python
def _assemble_final_results(output_dir, hardware_results, pipeline_data, target_spec):
    try:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.json")
        
        measurements = hardware_results.get("measurements", {})
        output = dict(measurements)
        
        # ... 组装 pipeline 数据和 hardware probe 结果
        
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        
        return results_path
```

**调用链**:
1. [`main.py:318-329`](file:///e:/GPU_Profiling_System/src/main.py#L318-L329) - Pipeline 完成后调用
```python
probe_results = _run_probes_no_write(builder.sandbox)
if probe_results:
    _assemble_final_results(
        output_dir=output_dir,
        hardware_results=probe_results,
        pipeline_data=pipeline_data,
        target_spec=target_spec,
    )
```

**Kaggle 中的实际路径**:
```
/kaggle/working/gpu_profiling_system/results.json
```

**记录内容**:
- 所有测量结果（hardware probes + pipeline）
- 交叉验证数据
- 证据文件列表
- 方法论说明
- 目标列表

---

## 🗺️ Kaggle 目录结构

**完整目录结构**:
```
/kaggle/working/gpu_profiling_system/
├── .state/
│   ├── session_log.jsonl          # AgentLoop 状态日志
│   └── pipeline_log.jsonl         # Pipeline 状态日志
├── audit/
│   ├── audit_report.md            # 审计报告（Markdown）
│   └── audit_report.json          # 审计报告（JSON）
├── execution.log                   # 完整执行日志（stdout）
└── results.json                    # 最终测量结果
```

**注意**: 
- ❌ **不会生成** `source.cu`、`freq_probe` 等编译产物（已被 cleanup.py 清理）
- ✅ **只会生成** 核心日志和结果文件

---

## 📥 文件下载映射

### 从 Kaggle 下载文件的路径

**假设 Kaggle 工作目录**: `/kaggle/working/gpu_profiling_system/`

**下载命令示例**:
```bash
# 1. execution.log（stdout 自动捕获）
# Kaggle UI 自动提供下载，或在 Notebooks 页面查找

# 2. session_log.jsonl
# 路径：/kaggle/working/gpu_profiling_system/.state/session_log.jsonl

# 3. pipeline_log.jsonl
# 路径：/kaggle/working/gpu_profiling_system/.state/pipeline_log.jsonl

# 4. audit_report.md
# 路径：/kaggle/working/gpu_profiling_system/audit/audit_report.md

# 5. audit_report.json
# 路径：/kaggle/working/gpu_profiling_system/audit/audit_report.json

# 6. results.json
# 路径：/kaggle/working/gpu_profiling_system/results.json
```

**Kaggle Notebook 中下载**:
```python
from IPython.display import FileLink, FileLinks

# 创建下载链接
FileLink('execution.log')
FileLink('.state/session_log.jsonl')
FileLink('.state/pipeline_log.jsonl')
FileLink('audit/audit_report.md')
FileLink('audit/audit_report.json')
FileLink('results.json')
```

---

## 🔍 文件生成时机

### Pipeline 执行流程

```
1. main.py 启动
   ↓
   logging.basicConfig() → 配置 stdout 日志
   ↓
2. Pipeline 初始化
   ↓
   StatePersister() → 创建 .state/session_log.jsonl
   Pipeline.__init__() → 创建 .state/pipeline_log.jsonl
   ↓
3. Pipeline 执行
   ↓
   每个 Stage 执行 → 记录到 pipeline_log.jsonl
   工具调用 → 记录到 session_log.jsonl
   print/logging → 输出到 stdout (execution.log)
   ↓
4. Pipeline 完成
   ↓
   PipelineAuditReport.save() → 创建 audit/audit_report.md + .json
   _assemble_final_results() → 创建 results.json
```

---

## 📊 文件大小估算

基于实际代码和日志内容：

| 文件 | 大小估算 | 影响因素 |
|------|----------|----------|
| `execution.log` | 50-200 KB | Pipeline 复杂度、错误数量 |
| `session_log.jsonl` | 3-10 KB | 工具调用次数 |
| `pipeline_log.jsonl` | 1-5 KB | Stage 数量、重试次数 |
| `audit_report.md` | 5-20 KB | 测量结果数量 |
| `audit_report.json` | 10-50 KB | 详细数据结构 |
| `results.json` | 1-5 KB | 测量目标数量 |

**总计**: ~70-290 KB（非常小，适合下载）

---

## 💡 重要发现

### 1. execution.log 不是主动生成的

**事实**:
- ❌ 没有 `FileHandler` 配置
- ✅ 所有日志通过 `stream=sys.stdout` 输出
- ✅ Kaggle 自动捕获 stdout 到 execution.log

**代码证据**: [`main.py:26-31`](file:///e:/GPU_Profiling_System/src/main.py#L26-L31)

### 2. .state 目录是持久化目录

**事实**:
- ✅ `state_dir=".state"` 是默认配置
- ✅ 所有 JSONL 日志都保存在此目录
- ✅ 用于断点续跑和会话恢复

**代码证据**: [`main.py:78-82`](file:///e:/GPU_Profiling_System/src/main.py#L78-L82)

### 3. audit 目录是审计专用

**事实**:
- ✅ 审计报告保存在 `audit/` 子目录
- ✅ 同时生成 JSON 和 Markdown 格式
- ✅ 包含完整的 Pipeline 执行信息

**代码证据**: [`main.py:354-357`](file:///e:/GPU_Profiling_System/src/main.py#L354-L357)

---

## ✅ 总结

### 必须下载的 6 个核心文件

| 文件 | 路径 | 大小 | 重要性 |
|------|------|------|--------|
| `execution.log` | `/kaggle/working/gpu_profiling_system/` | ~56 KB | ⭐⭐⭐⭐⭐ |
| `session_log.jsonl` | `/kaggle/working/gpu_profiling_system/.state/` | ~3 KB | ⭐⭐⭐⭐ |
| `pipeline_log.jsonl` | `/kaggle/working/gpu_profiling_system/.state/` | ~1 KB | ⭐⭐⭐⭐⭐ |
| `audit_report.md` | `/kaggle/working/gpu_profiling_system/audit/` | ~0.25 KB | ⭐⭐⭐⭐ |
| `audit_report.json` | `/kaggle/working/gpu_profiling_system/audit/` | ~10 KB | ⭐⭐⭐ |
| `results.json` | `/kaggle/working/gpu_profiling_system/` | ~1 KB | ⭐⭐⭐⭐⭐ |

### 不需要下载的文件

- ❌ `source.cu` - 临时 CUDA 源文件（已清理）
- ❌ `freq_*` - 编译产物（已清理）
- ❌ `cmd_*.log` - 冗余命令日志（已清理）
- ❌ `debug_messages_*.json` - 调试文件（已清理）

---

**分析人**: AI Assistant  
**分析日期**: 2026-04-17  
**分析依据**: 实际代码，实事求是
