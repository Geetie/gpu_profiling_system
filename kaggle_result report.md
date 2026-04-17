## Kaggle 测试结果分析报告：CodeGen 输出文件位置错误与探测器文件位置错误

***

### 一、问题概述

在 `e:\GPU_Profiling_System\kaggle_results` 目录中发现以下文件位置错误：

1. **CodeGen 输出文件位置错误**：`source.cu` 出现在 `kaggle_results` 根目录，而非预期的 `probe_binary` 子目录或其他指定位置
2. **探测器二进制文件位置错误**：`freq_event_probe`、`freq_event_timed`、`freq_probe` 等文件出现在 `kaggle_results` 根目录，而非 `probe_binary` 子目录

***

### 二、日志分析结果

#### 2.1 execution.log 分析

**关键发现**：

- Line 131-133: Sandbox 检测到 Kaggle 环境，**直接使用项目根目录作为 sandbox**：
  ```
  [Sandbox] Kaggle environment detected (KAGGLE_KERNEL_RUN_TYPE=Interactive)
  [Sandbox] Using project root as sandbox: /kaggle/working/gpu_profiling_system
  ```
- Line 396-461: 硬件探测器在 fallback 模式下运行，编译的二进制文件直接写入 sandbox 根目录
- Line 401, 414, 428: `freq_probe` 等二进制文件在 `/kaggle/working/gpu_profiling_system` 目录编译和执行

#### 2.2 pipeline\_log.jsonl 分析

**CodeGen 阶段状态**：

- Line 5: `code_gen` 阶段开始
- Line 6: CodeGen 阶段失败（`status: failed`），执行了 4 次 tool call
- Line 7-8: 失败原因：`CodeGen compilation failed — check tool call results for errors`

**关键问题**：CodeGen 的编译工具调用返回错误，但探测器 fallback 代码绕过了正常的 sandbox 路径管理

#### 2.3 session\_log.jsonl 分析

该文件仅包含 `__loop_state__` 工具的持久化记录，**没有记录具体的文件写入操作**。这表明文件写入发生在 sandbox 层，而非通过 Agent 的工具调用。

#### 2.4 debug\_messages\_longcat\_\*.json 分析

从 `debug_messages_longcat_9msg_3tool.json` 可见：

- CodeGen 尝试调用 `compile_cuda` 工具
- 编译失败原因：`nvcc fatal: Don't know what to do with '3'` 和 `'5'`（架构参数格式错误）
- CodeGen 随后尝试 `read_file` 但返回空内容

***

### 三、代码审查结果

#### 3.1 CodeGen 文件写入逻辑 ([codegen.py](file://e:\GPU_Profiling_System\src\application\subagents\codegen.py))

**问题位置**：Line 263-267

```python
result = self._sandbox.run(
    source_code=source_code,
    command="nvcc",
    args=["-o", binary_name, "source.cu", f"-arch={arch}", "-O3"],
)
```

**预期行为**：二进制文件应编译到 `sandbox_root` 目录

**实际行为**：在 Kaggle 环境下，`sandbox_root = /kaggle/working/gpu_profiling_system`（项目根目录）

#### 3.2 Sandbox 路径配置 ([sandbox.py](file://e:\GPU_Profiling_System\src\infrastructure\sandbox.py))

**关键问题代码**：Line 125-129

```python
elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    project_root = os.getcwd()
    self._sandbox_root = project_root
    print(f"[Sandbox] Kaggle environment detected (KAGGLE_KERNEL_RUN_TYPE={os.environ['KAGGLE_KERNEL_RUN_TYPE']})")
    print(f"[Sandbox] Using project root as sandbox: {project_root}")
```

**设计缺陷**：Kaggle 环境下，sandbox 直接使用项目根目录，导致所有编译产物都写入项目根目录而非子目录

#### 3.3 探测器文件写入逻辑 ([clock\_measurement.py](file://e:\GPU_Profiling_System\src\infrastructure\probing\clock_measurement.py))

**问题位置 1**：Line 235-244

```python
compile_result = runner.run(
    source_code=source,
    command=nvcc,
    args=["-o", "freq_probe", "source.cu", f"-arch={arch}"],
    work_dir=work_dir,
)
if not compile_result.success:
    return None, None

binary = os.path.join(work_dir, "freq_probe")
```

**问题位置 2**：Line 396-405, 424-433

```python
# freq_event_probe
args=["-o", "freq_event_probe", "source.cu", f"-arch={arch}"],
binary = os.path.join(work_dir, "freq_event_probe")

# freq_event_timed
args=["-o", "freq_event_timed", "source.cu", f"-arch={arch}"],
timed_binary = os.path.join(work_dir, "freq_event_timed")
```

**根本问题**：

- 探测器编译时直接使用**硬编码的二进制文件名**（如 `"freq_probe"`）
- `work_dir` 是 sandbox 根目录（Kaggle 环境下 = 项目根目录）
- 编译产物直接写入 sandbox 根目录，**没有子目录隔离**

#### 3.4 compile\_and\_run 辅助函数 ([probe\_helpers.py](file://e:\GPU_Profiling_System\src\infrastructure\probing\probe_helpers.py))

**问题位置**：Line 49-64

```python
compile_args = ["-o", "probe_binary", "source.cu"] + safe_flags
compile_result = runner.run(
    source_code=source,
    command=nvcc,
    args=compile_args,
    work_dir=work_dir,
)

binary_path = os.path.join(work_dir or ".", "probe_binary")
```

**问题**：同样使用硬编码的 `"probe_binary"` 文件名，直接写入 sandbox 根目录

***

### 四、5 Why 根因分析

#### 问题 1：探测器文件为何出现在 kaggle\_results 根目录？

**1 Why**: 为什么 `freq_probe` 等文件出现在根目录？

- 因为编译命令将输出路径设置为 `work_dir/freq_probe`

**2 Why**: 为什么 `work_dir` 是根目录？

- 因为 `work_dir = sandbox_root`，而在 Kaggle 环境下 `sandbox_root = project_root`

**3 Why**: 为什么 Kaggle 环境下 sandbox 使用项目根目录？

- 因为 [sandbox.py](file://e:\GPU_Profiling_System\src\infrastructure\sandbox.py) Line 125-129 的硬编码逻辑：检测到 `KAGGLE_KERNEL_RUN_TYPE` 环境变量后直接使用 `os.getcwd()`

**4 Why**: 为什么没有使用子目录（如 `probe_binary`）？

- 因为代码设计假设 sandbox 本身提供隔离，所有编译产物都在 sandbox 根目录，不需要额外子目录

**5 Why (根本原因)**:

- **设计缺陷**：LocalSandbox 的隔离模型在 Kaggle 环境下退化为"项目根目录即 sandbox"，但探测器编译代码没有适配这种退化模式，仍然假设 sandbox 根目录是安全的隔离空间
- **路径管理缺失**：编译命令没有使用相对路径或子目录，直接写入 sandbox 根目录

#### 问题 2：CodeGen 为何将 CUDA 源代码写入错误位置？

**1 Why**: 为什么 `source.cu` 出现在根目录？

- 因为 `compile_cuda_handler` 将源代码写入 sandbox 根目录

**2 Why**: 为什么会写入 sandbox 根目录？

- 因为 [compile\_cuda\_handler](file://e:\GPU_Profiling_System\src\infrastructure\tools\compile_cuda.py#L14-L96) Line 80-85 调用 `runner.run(source_code=source, ..., work_dir=runner.sandbox_root)`

**3 Why**: 为什么 `sandbox_root` 是项目根目录？

- 同问题 1 的 2 Why 和 3 Why

**4 Why**: 为什么没有路径验证或子目录隔离？

- 因为代码假设 sandbox 提供完全隔离，sandbox 根目录就是安全的工作目录

**5 Why (根本原因)**:

- **环境感知缺失**：LocalSandbox 在 Kaggle 环境下没有创建真正的隔离子目录
- **路径拼接错误**：所有文件操作都基于 `sandbox_root`，但该变量在 Kaggle 环境下指向项目根目录

***

### 五、影响范围

#### 5.1 受影响的文件

根据日志和代码分析，以下文件会被错误地写入项目根目录：

**CodeGen 阶段**：

- `source.cu`
- `benchmark_{target}`（编译产物）

**探测器阶段**（fallback 模式）：

- `freq_probe`（时钟频率探测二进制）
- `freq_event_probe`（事件计时探测二进制）
- `freq_event_timed`（事件计时探测二进制）
- `probe_binary`（通用探测二进制）
- 其他探测器二进制文件

#### 5.2 受影响的 Kaggle 提交

- 文件污染：探测器二进制文件（每个约 1MB）会污染 Kaggle 工作目录
- 路径混乱：源代码和编译产物混在项目根目录，难以清理
- 可能的提交失败：Kaggle 对输出文件大小有限制，大型二进制文件可能导致提交失败

***

### 六、修复建议

#### 6.1 短期修复（Hotfix）

**修复位置 1**: [sandbox.py](file://e:\GPU_Profiling_System\src\infrastructure\sandbox.py) Line 123-132

```python
def __init__(self, config: SandboxConfig | None = None, sandbox_root: str | None = None) -> None:
    super().__init__(config or SandboxConfig())
    if sandbox_root is not None:
        self._sandbox_root = sandbox_root
    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        # FIX: Create a subdirectory in Kaggle environment
        project_root = os.getcwd()
        self._sandbox_root = os.path.join(project_root, ".kaggle_sandbox")
        os.makedirs(self._sandbox_root, exist_ok=True)
        print(f"[Sandbox] Kaggle environment detected (KAGGLE_KERNEL_RUN_TYPE={os.environ['KAGGLE_KERNEL_RUN_TYPE']})")
        print(f"[Sandbox] Using isolated sandbox: {self._sandbox_root}")
    else:
        self._sandbox_root = os.path.join(os.getcwd(), ".sandbox")
    os.makedirs(self._sandbox_root, exist_ok=True)
```

**修复位置 2**: [clock\_measurement.py](file://e:\GPU_Profiling_System\src\infrastructure\probing\clock_measurement.py) Line 235-244

```python
# FIX: Use a subdirectory for probe binaries
probe_binary_dir = os.path.join(work_dir, "probe_binaries")
os.makedirs(probe_binary_dir, exist_ok=True)

compile_result = runner.run(
    source_code=source,
    command=nvcc,
    args=["-o", os.path.join(probe_binary_dir, "freq_probe"), "source.cu", f"-arch={arch}"],
    work_dir=work_dir,
)
if not compile_result.success:
    return None, None

binary = os.path.join(probe_binary_dir, "freq_probe")
```

#### 6.2 中期修复（架构改进）

**建议 1**: 引入路径管理器类

创建 `src/infrastructure/path_manager.py`：

```python
class PathManager:
    """Manages file paths for compilation and execution."""
    
    def __init__(self, sandbox_root: str):
        self.sandbox_root = sandbox_root
        self.source_dir = os.path.join(sandbox_root, "src")
        self.binary_dir = os.path.join(sandbox_root, "bin")
        self.evidence_dir = os.path.join(sandbox_root, "evidence")
        
    def source_path(self, filename: str = "source.cu") -> str:
        os.makedirs(self.source_dir, exist_ok=True)
        return os.path.join(self.source_dir, filename)
    
    def binary_path(self, name: str) -> str:
        os.makedirs(self.binary_dir, exist_ok=True)
        return os.path.join(self.binary_dir, name)
    
    def evidence_path(self, probe_name: str) -> str:
        os.makedirs(self.evidence_dir, exist_ok=True)
        return os.path.join(self.evidence_dir, f"evidence_{probe_name}.json")
```

**建议 2**: 修改所有编译调用使用路径管理器

```python
# 在所有编译代码中
path_mgr = PathManager(sandbox_root)
compile_result = runner.run(
    source_code=source,
    command=nvcc,
    args=["-o", path_mgr.binary_path("freq_probe"), path_mgr.source_path(), f"-arch={arch}"],
)
```

#### 6.3 长期修复（环境隔离）

**建议**: 在 Kaggle 环境下使用 DockerSandbox 而非 LocalSandbox

修改 [sandbox.py](file://e:\GPU_Profiling_System\src\infrastructure\sandbox.py) 的工厂函数：

```python
def create_sandbox(config: SandboxConfig = None) -> SandboxRunner:
    """Factory function to create appropriate sandbox."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        # In Kaggle, prefer DockerSandbox if available
        if docker_available():
            return DockerSandbox(config)
        else:
            # Fallback with isolated subdirectory
            return LocalSandbox(config, sandbox_root=os.path.join(os.getcwd(), ".kaggle_sandbox"))
    elif docker_available():
        return DockerSandbox(config)
    else:
        return LocalSandbox(config)
```

***

### 七、验证计划

修复后需要验证：

1. **单元测试**：
   - 测试 `LocalSandbox` 在 Kaggle 环境下创建 `.kaggle_sandbox` 子目录
   - 测试编译产物出现在正确的子目录
2. **集成测试**：
   - 在 Kaggle 环境中运行完整 pipeline
   - 验证 `kaggle_results` 目录只包含预期的输出文件（`results.json`, `execution.log` 等）
   - 验证探测器二进制文件在 `probe_binary` 子目录
3. **回归测试**：
   - 确保修复不影响非 Kaggle 环境的正常工作
   - 验证 DockerSandbox 和 LocalSandbox 的行为一致性

***

### 八、总结

**根本原因**：

1. **Sandbox 路径配置缺陷**：Kaggle 环境下 LocalSandbox 直接使用项目根目录作为 sandbox 根目录，没有创建隔离子目录
2. **硬编码路径**：探测器编译代码使用硬编码的二进制文件名，直接写入 sandbox 根目录
3. **环境感知缺失**：代码没有区分"真正的 sandbox 隔离"和"退化模式下的路径 confinement"

**修复优先级**：

1. **高优先级**：修复 [sandbox.py](file://e:\GPU_Profiling_System\src\infrastructure\sandbox.py) 的 Kaggle 环境检测逻辑，创建隔离子目录
2. **中优先级**：修改探测器编译代码使用子目录管理二进制文件
3. **低优先级**：引入路径管理器类，统一路径管理逻辑

**预计修复时间**：

- Hotfix: 2-4 小时
- 架构改进：1-2 天
- 完整测试验证：1 天

## 📊 Kaggle 测试结果分析：CodeGen 文件位置错误根因报告

我已经完成了对最新 Kaggle 测试结果的深度分析。以下是**基于代码事实**的详细报告：

***

### 🔍 **问题概述**

从您提供的图片和 `kaggle_results` 目录结构可以看到：

1. **❌ CodeGen 输出错误**：`source.cu` 出现在 `kaggle_results` 根目录（应该在 `probe_binary` 子目录）
2. **❌ 探测器文件位置错误**：
   - `freq_event_probe`
   - `freq_event_timed`
   - `freq_probe`
   - `probe_binary`
   - `stream_event`
   - 这些文件都出现在根目录（应该在 `probe_binary` 子目录）

***

### 📋 **日志分析结果**

#### **1. execution.log 关键发现**

**Sandbox 路径配置** (Line 131-133):

```
[Sandbox] Kaggle environment detected (KAGGLE_KERNEL_RUN_TYPE=Interactive)
[Sandbox] Using project root as sandbox: /kaggle/working/gpu_profiling_system
```

**问题**：Kaggle 环境下，sandbox 直接使用**项目根目录**作为工作目录，导致所有编译产物都写入根目录。

**探测器编译** (Line 396-461):

```
[freq_probe] Compiling with nvcc: -o freq_probe source.cu -arch=sm_60
[freq_event_probe] Compiling with nvcc: -o freq_event_probe source.cu -arch=sm_60
[freq_event_timed] Compiling with nvcc: -o freq_event_timed source.cu -arch=sm_60
```

**问题**：探测器编译使用**硬编码的文件名**，直接写入 sandbox 根目录。

***

#### **2. pipeline\_log.jsonl 分析**

**CodeGen 阶段失败** (Line 5-8):

```json
{
  "stage": "code_gen",
  "status": "failed",
  "tool_calls": 4,
  "error": "CodeGen compilation failed — check tool call results for errors"
}
```

**关键发现**：

- CodeGen 尝试编译 CUDA 代码但失败
- 系统回退到 hardware probes fallback 模式
- Fallback 模式直接编译探测器到 sandbox 根目录

***

#### **3. debug\_messages\_longcat\_9msg\_3tool.json 分析**

**CodeGen 编译错误**:

```
nvcc fatal: Don't know what to do with '3'
nvcc fatal: Don't know what to do with '5'
```

**问题**：CodeGen 传递给 nvcc 的架构参数格式错误（应该是 `-arch=sm_60`，但可能传递了 `"3"` 或 `"5"` 这样的单独数字）

***

### 🔬 **代码审查结果**

#### **问题 1: Sandbox 路径配置错误**

**位置**: [`sandbox.py:125-129`](file:///e:/GPU_Profiling_System/src/infrastructure/sandbox.py#L125-L129)

```python
elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    project_root = os.getcwd()
    self._sandbox_root = project_root  # ❌ 直接使用项目根目录
    print(f"[Sandbox] Kaggle environment detected...")
    print(f"[Sandbox] Using project root as sandbox: {project_root}")
```

**影响**：

- 所有编译产物都写入 `/kaggle/working/gpu_profiling_system`
- 没有子目录隔离，污染项目根目录

***

#### **问题 2: 探测器编译路径错误**

**位置**: [`clock_measurement.py:235-244`](file:///e:/GPU_Profiling_System/src/infrastructure/probing/clock_measurement.py#L235-L244)

```python
compile_result = runner.run(
    source_code=source,
    command=nvcc,
    args=["-o", "freq_probe", "source.cu", f"-arch={arch}"],  # ❌ 硬编码文件名
    work_dir=work_dir,
)
if not compile_result.success:
    return None, None

binary = os.path.join(work_dir, "freq_probe")  # ❌ 直接拼接 sandbox 根目录
```

**问题**：

- 使用硬编码的 `"freq_probe"` 文件名
- `work_dir` 是 sandbox 根目录（Kaggle 环境下 = 项目根目录）
- 编译产物直接写入项目根目录

**同样的问题存在于**:

- Line 396-405: `freq_event_probe`
- Line 424-433: `freq_event_timed`

***

#### **问题 3: CodeGen 编译路径错误**

**位置**: [`codegen.py:263-267`](file:///e:/GPU_Profiling_System/src/application/subagents/codegen.py#L263-L267)

```python
result = self._sandbox.run(
    source_code=source,
    command="nvcc",
    args=["-o", binary_name, "source.cu", f"-arch={arch}", "-O3"],
)
```

**问题**：

- `binary_name` 可能是相对路径或直接是文件名
- 在 Kaggle 环境下，编译产物写入项目根目录

***

#### **问题 4: compile\_and\_run 辅助函数路径错误**

**位置**: [`probe_helpers.py:49-64`](file:///e:/GPU_Profiling_System/src/infrastructure/probing/probe_helpers.py#L49-L64)

```python
compile_args = ["-o", "probe_binary", "source.cu"] + safe_flags
compile_result = runner.run(
    source_code=source,
    command=nvcc,
    args=compile_args,
    work_dir=work_dir,
)

binary_path = os.path.join(work_dir or ".", "probe_binary")
```

**问题**：同样使用硬编码的 `"probe_binary"` 文件名

***

### 🎯 **5 Why 根因分析**

#### **问题 1: 为什么探测器文件出现在 kaggle\_results 根目录？**

**1 Why**: 为什么 `freq_probe` 等文件出现在根目录？

- 因为编译命令输出路径是 `work_dir/freq_probe`

**2 Why**: 为什么 `work_dir` 是根目录？

- 因为 `work_dir = sandbox_root`，而 Kaggle 环境下 `sandbox_root = project_root`

**3 Why**: 为什么 Kaggle 环境下 sandbox 使用项目根目录？

- 因为 [`sandbox.py:125-129`](file:///e:/GPU_Profiling_System/src/infrastructure/sandbox.py#L125-L129) 检测到 `KAGGLE_KERNEL_RUN_TYPE` 后直接使用 `os.getcwd()`

**4 Why**: 为什么没有使用子目录（如 `probe_binary`）？

- 因为代码设计假设 sandbox 本身提供隔离，所有编译产物都在 sandbox 根目录即可

**5 Why (根本原因)**:

- **设计缺陷**：LocalSandbox 在 Kaggle 环境下的隔离模型退化为"项目根目录即 sandbox"
- **路径管理缺失**：编译代码没有使用子目录，直接写入 sandbox 根目录

***

#### **问题 2: 为什么 CodeGen 将 source.cu 写入错误位置？**

**1 Why**: 为什么 `source.cu` 出现在根目录？

- 因为 `compile_cuda_handler` 将源代码写入 sandbox 根目录

**2 Why**: 为什么会写入 sandbox 根目录？

- 因为编译工具调用使用 `sandbox_root` 作为工作目录

**3 Why**: 为什么 `sandbox_root` 是项目根目录？

- 同问题 1 的 2 Why 和 3 Why

**4 Why**: 为什么没有路径验证或子目录隔离？

- 因为代码假设 sandbox 提供完全隔离，sandbox 根目录就是安全的工作目录

**5 Why (根本原因)**:

- **环境感知缺失**：LocalSandbox 在 Kaggle 环境下没有创建真正的隔离子目录
- **路径拼接错误**：所有文件操作都基于 `sandbox_root`，但该变量在 Kaggle 环境下指向项目根目录

***

### 🛠️ **修复建议（按优先级排序）**

#### **🔴 P0 - 紧急修复（Hotfix）**

**修复 1: Sandbox Kaggle 环境路径配置**

**位置**: [`sandbox.py:123-132`](file:///e:/GPU_Profiling_System/src/infrastructure/sandbox.py#L123-L132)

```python
def __init__(self, config: SandboxConfig | None = None, sandbox_root: str | None = None) -> None:
    super().__init__(config or SandboxConfig())
    if sandbox_root is not None:
        self._sandbox_root = sandbox_root
    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        # ✅ FIX: Create a subdirectory in Kaggle environment
        project_root = os.getcwd()
        self._sandbox_root = os.path.join(project_root, ".kaggle_sandbox")
        os.makedirs(self._sandbox_root, exist_ok=True)
        print(f"[Sandbox] Kaggle environment detected (KAGGLE_KERNEL_RUN_TYPE={os.environ['KAGGLE_KERNEL_RUN_TYPE']})")
        print(f"[Sandbox] Using isolated sandbox: {self._sandbox_root}")
    else:
        self._sandbox_root = os.path.join(os.getcwd(), ".sandbox")
    os.makedirs(self._sandbox_root, exist_ok=True)
```

**修复 2: 探测器编译使用子目录**

**位置**: [`clock_measurement.py:235-244`](file:///e:/GPU_Profiling_System/src/infrastructure/probing/clock_measurement.py#L235-L244)

```python
# ✅ FIX: Use a subdirectory for probe binaries
probe_binary_dir = os.path.join(work_dir, "probe_binaries")
os.makedirs(probe_binary_dir, exist_ok=True)

compile_result = runner.run(
    source_code=source,
    command=nvcc,
    args=["-o", os.path.join(probe_binary_dir, "freq_probe"), "source.cu", f"-arch={arch}"],
    work_dir=work_dir,
)
if not compile_result.success:
    return None, None

binary = os.path.join(probe_binary_dir, "freq_probe")
```

**需要同样修复的位置**:

- Line 396-405: `freq_event_probe`
- Line 424-433: `freq_event_timed`
- [`probe_helpers.py:49-64`](file:///e:/GPU_Profiling_System/src/infrastructure/probing/probe_helpers.py#L49-L64): `probe_binary`

***

#### **🟡 P1 - 中期修复（架构改进）**

**建议：引入路径管理器类**

创建 `src/infrastructure/path_manager.py`:

```python
class PathManager:
    """Manages file paths for compilation and execution."""
    
    def __init__(self, sandbox_root: str):
        self.sandbox_root = sandbox_root
        self.source_dir = os.path.join(sandbox_root, "src")
        self.binary_dir = os.path.join(sandbox_root, "bin")
        self.evidence_dir = os.path.join(sandbox_root, "evidence")
        
    def source_path(self, filename: str = "source.cu") -> str:
        os.makedirs(self.source_dir, exist_ok=True)
        return os.path.join(self.source_dir, filename)
    
    def binary_path(self, name: str) -> str:
        os.makedirs(self.binary_dir, exist_ok=True)
        return os.path.join(self.binary_dir, name)
    
    def evidence_path(self, probe_name: str) -> str:
        os.makedirs(self.evidence_dir, exist_ok=True)
        return os.path.join(self.evidence_dir, f"evidence_{probe_name}.json")
```

**使用示例**:

```python
# 在所有编译代码中
path_mgr = PathManager(sandbox_root)
compile_result = runner.run(
    source_code=source,
    command=nvcc,
    args=["-o", path_mgr.binary_path("freq_probe"), path_mgr.source_path(), f"-arch={arch}"],
)
```

***

#### **🟢 P2 - 长期修复（环境隔离）**

**建议：在 Kaggle 环境下使用 DockerSandbox**

修改 [`sandbox.py`](file:///e:/GPU_Profiling_System/src/infrastructure/sandbox.py) 的工厂函数:

```python
def create_sandbox(config: SandboxConfig = None) -> SandboxRunner:
    """Factory function to create appropriate sandbox."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        # ✅ In Kaggle, prefer DockerSandbox if available
        if docker_available():
            return DockerSandbox(config)
        else:
            # Fallback with isolated subdirectory
            return LocalSandbox(config, sandbox_root=os.path.join(os.getcwd(), ".kaggle_sandbox"))
    elif docker_available():
        return DockerSandbox(config)
    else:
        return LocalSandbox(config)
```

***

### 📊 **影响范围评估**

#### **受影响的文件**

**CodeGen 阶段**:

- `source.cu`
- `benchmark_{target}`（编译产物）

**探测器阶段（fallback 模式）**:

- `freq_probe`（时钟频率探测二进制，\~1MB）
- `freq_event_probe`（事件计时探测二进制，\~1MB）
- `freq_event_timed`（事件计时探测二进制，\~1MB）
- `probe_binary`（通用探测二进制，\~1MB）
- `stream_event`（流事件探测二进制，\~1MB）

#### **对 Kaggle 提交的影响**

- **文件污染**：探测器二进制文件（每个约 1MB）污染 Kaggle 工作目录
- **路径混乱**：源代码和编译产物混在项目根目录，难以清理
- **可能的提交失败**：Kaggle 对输出文件大小有限制（通常 20MB），大型二进制文件可能导致提交失败

***

### ✅ **总结**

**根本原因**（2 个核心问题）:

1. **Sandbox 路径配置缺陷**：Kaggle 环境下 LocalSandbox 直接使用项目根目录作为 sandbox 根目录，**没有创建隔离子目录**
2. **硬编码路径**：探测器编译代码使用硬编码的二进制文件名（如 `"freq_probe"`），直接写入 sandbox 根目录

**修复优先级**:

1. 🔴 **高优先级**：修复 [`sandbox.py`](file:///e:/GPU_Profiling_System/src/infrastructure/sandbox.py) 的 Kaggle 环境检测逻辑，创建 `.kaggle_sandbox` 子目录
2. 🟡 **中优先级**：修改探测器编译代码使用 `probe_binaries` 子目录管理二进制文件
3. 🟢 **低优先级**：引入路径管理器类，统一路径管理逻辑

**预计修复时间**:

- Hotfix: 2-4 小时
- 架构改进：1-2 天
- 完整测试验证：1 天

