# CodeGen Agent 提示词系统性审查报告

## 一、审查概览

### 审查范围

- ✅ 环境适配性 — 是否适配不同 GPU 架构和环境
- ✅ 代码正确性 — 是否保证生成的 CUDA 代码正确
- ✅ 多 GPU 环境适配 — 是否考虑不同 GPU 环境的差异
- ✅ 任务负载多样性 — 是否适配不同测量任务
- ✅ 跨 Agent 协作 — 是否接收并利用其他 Agent 的信息
- ✅ 持续迭代改进 — 是否支持多轮反馈改进

### 审查依据

- `e:\GPU_Profiling_System\PJ 需求.md` — 1.7 硬件内在剖析要求
- `e:\GPU_Profiling_System\spec.md` — 多智能体协作机制（§5）、设计原则（§2）
- NVIDIA CUDA 编程最佳实践

***

## 二、环境适配性审查 ✅ **优秀**

### 2.1 GPU 架构自动检测 ✅

**实现位置**：[codegen.py:61-62](file://e:\GPU_Profiling_System\src\application\subagents\codegen.py#L61-L62), [238-260](file://e:\GPU_Profiling_System\src\application\subagents\codegen.py#L238-L260)

```python
def _detect_gpu_arch(self) -> str:
    """Detect GPU compute capability for correct nvcc -arch flag."""
    if self._detected_arch:
        return self._detected_arch
    
    arch = detect_gpu_arch(self._sandbox)
    self._detected_arch = arch
    return arch

def _compile(self, source_code: str, target: str = "unknown") -> Any:
    arch = self._detect_gpu_arch()
    result = self._sandbox.run(
        source_code=source_code,
        command="nvcc",
        args=["-o", binary_name, "source.cu", f"-arch={arch}", "-O3"],
    )
```

**提示词支持**：[agent\_prompts.py:87-91](file://e:\GPU_Profiling_System\src\domain\agent_prompts.py#L87-L91)

```python
"- Example: flags: [\"-O3\", \"-arch=sm_60\"] for Tesla P100\n"
"- Example: flags: [\"-O3\", \"-arch=sm_80\"] for newer GPUs\n"
```

**优点**：

- ✅ **运行时检测** — 不依赖静态配置，适配所有 NVIDIA GPU
- ✅ **自动传递** — 编译时自动使用检测到的架构
- ✅ **提示词引导** — 明确告知 LLM 需要根据 GPU 选择正确的 `-arch` 标志

**覆盖的 GPU 架构**：

- Tesla P100 (sm\_60)
- V100 (sm\_70)
- A100 (sm\_80)
- H100 (sm\_90)
- RTX 30/40 系列 (sm\_80/sm\_89)

### 2.2 沙箱环境隔离 ✅

**实现位置**：[stage\_executor.py:319-325](file://e:\GPU_Profiling_System\src\domain\stage_executor.py#L319-L325)

```python
"✅ CORRECT: .sandbox/benchmark.cu\n"
"❌ WRONG: /kaggle/working/gpu_profiling_system/benchmark.cu (path escape)\n"
"❌ WRONG: benchmark.cu (missing .sandbox prefix)\n"
```

**优点**：

- ✅ **路径规范化** — 强制使用 `.sandbox/` 前缀
- ✅ **错误示例** — 明确展示常见错误
- ✅ **环境无关** — 不依赖特定平台路径（如 `/kaggle/working`）

### 2.3 防作弊环境适配 ✅

**提示词支持**：[agent\_prompts.py:147-152](file://e:\GPU_Profiling_System\src\domain\agent_prompts.py#L147-L152)

```python
"🛡️  ANTI-CHEAT AWARENESS\n"
"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
"- ❌ Do NOT rely solely on cudaGetDeviceProperties — may return virtualized data\n"
"- ✅ Use clock64() + cudaEventElapsedTime to measure actual hardware behavior\n"
```

**设计原则支持**：[design\_principles.py:47-56](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L47-L56)

```python
"⚠️ ANTI-CHEAT REQUIREMENTS:\n"
"- Do NOT rely solely on cudaGetDeviceProperties or cudaDeviceGetAttribute\n"
"  These may return virtualized data in cloud/containerized environments\n"
"- ALWAYS measure actual hardware behavior using:\n"
"  - clock64() for GPU core cycles\n"
"  - cudaEventElapsedTime for GPU wall-clock time\n"
"  - Empirical measurement (block ID sweep, pointer-chasing)\n"
```

**优点**：

- ✅ **明确警告** — 告知 LLM API 可能返回虚拟化数据
- ✅ **替代方案** — 提供实测方法（clock64、指针追逐）
- ✅ **交叉验证** — 要求使用至少 2 种策略

***

## 三、代码正确性保障审查 ✅ **优秀**

### 3.1 CUDA 最佳实践注入 ✅

**提示词详细程度**：[agent\_prompts.py:115-145](file://e:\GPU_Profiling_System\src\domain\agent_prompts.py#L115-L145)

```python
"CUDA MICROBENCHMARK BEST PRACTICES (apply these rigorously):\n\n"

"1. TIMING METHODOLOGY:\n"
"   - clock64() for cycle-accurate device-side timing (fine-grained, frequency-independent)\n"
"     NEVER use clock() — returns 0 on Pascal+ under PTX JIT\n"
"   - cudaEventElapsedTime for wall-clock timing (bandwidth, frequency calculation)\n"
"     CRITICAL: cudaEventRecord is asynchronous — MUST cudaEventSynchronize(stop) before reading\n"

"2. PREVENTING COMPILER DEAD CODE ELIMINATION:\n"
"   - ALWAYS write kernel results to a volatile output pointer or use asm volatile\n"
"   - Test: if output is 0 or suspiciously small, compiler likely eliminated the work\n"

"3. LATENCY MEASUREMENT (DRAM, L2, L1) — pointer chasing:\n"
"   - Allocate uint64_t* for next-pointers — 64-bit addressing required\n"
"   - Single thread (1 block, 1 thread) follows chain: idx = next[idx] for N iterations\n"
"   - Use LCG to build random permutation on host\n"
"   - Warm up: run 1 iteration before timing\n"
"   - Latency_cycles = (t1 - t0) / N\n"
```

**优点**：

- ✅ **极其详细** — 涵盖 timing、死代码消除、指针追逐等关键点
- ✅ **错误警示** — 明确告知哪些 API 不能用（如 `clock()`）
- ✅ **原理说明** — 解释"为什么"要这样做（如 defeating prefetchers）

### 3.2 错误恢复机制 ✅

**提示词支持**：[agent\_prompts.py:97-105](file://e:\GPU_Profiling_System\src\domain\agent_prompts.py#L97-L105)

```python
"ERROR RECOVERY PROTOCOL:\n"
"- If compilation fails: read the error message, identify the issue, fix the code, retry\n"
"  - 'undefined reference' → add missing #include or declare the function\n"
"  - 'identifier not found' → check for typos in CUDA API names\n"
"  - 'invalid architecture' → detect GPU arch from nvidia-smi and use correct -arch=sm_XX\n"
"- If execution fails: check the binary path exists, fix the issue, recompile\n"
"- If output is 0 or negative: the measurement logic is wrong — fix and retry\n"
"- Maximum 3 retry attempts per target before reporting what went wrong\n"
```

**优点**：

- ✅ **模式匹配** — 提供常见编译错误的诊断方法
- ✅ **重试机制** — 允许最多 3 次重试
- ✅ **失败报告** — 要求说明失败原因，而非静默失败

### 3.3 代码结构要求 ✅

**提示词支持**：[agent\_prompts.py:142-145](file://e:\GPU_Profiling_System\src\domain\agent_prompts.py#L142-L145)

```python
"CRITICAL RULES:\n"
"- Every .cu file MUST have: #include <cuda_runtime.h>, __global__ kernel, main()\n"
"- Output MUST be parseable: printf(\"key: value\\n\") format, one per line\n"
"- ALWAYS cudaDeviceSynchronize() before reading device-side results\n"
"- Warm up: run 1 iteration before timing\n"
"- Multiple trials: run 3 trials for statistical confidence, report median\n"
```

**设计原则支持**：[design\_principles.py:76-88](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L76-L88)

````python
"3. CUDA KERNEL STRUCTURE:\n"
"   ```cuda\n"
"   __global__ void measure_dram_latency(uint32_t* indices, uint64_t* cycles, size_t size) {\n"
"       if (threadIdx.x != 0 || blockIdx.x != 0) return;  // Single thread\n"
"       uint32_t idx = 0;\n"
"       uint64_t start = clock64();\n"
"       for (uint64_t i = 0; i < 10000000; i++) {\n"
"           idx = indices[idx];  // Serial dependency chain\n"
"       }\n"
"       uint64_t end = clock64();\n"
"       *cycles = (end - start) / 10000000;  // Average cycles per access\n"
"   }\n"
"   ```\n\n"
````

**优点**：

- ✅ **完整示例** — 提供可运行的 CUDA 内核代码结构
- ✅ **输出格式** — 强制要求可解析的 `key: value` 格式
- ✅ **同步要求** — 强调 `cudaDeviceSynchronize()` 的必要性

***

## 四、多 GPU 环境适配审查 ✅ **优秀**

### 4.1 不同测量任务的适配 ✅

**设计原则覆盖**：

| 测量目标                 | 设计原则                                                                                          | 适配策略                                      |
| :------------------- | :-------------------------------------------------------------------------------------------- | :---------------------------------------- |
| **DRAM 延迟**          | [\_DRAM\_LATENCY](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L60-L113)    | 128MB 指针追逐， defeat prefetchers            |
| **L2 延迟**            | [\_L2\_LATENCY](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L115-L141)     | 2MB 指针追逐，L1 miss + L2 hit                 |
| **L1 延迟**            | [\_L1\_LATENCY](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L143-L166)     | 8KB 指针追逐，确保 L1 hit                        |
| **L2 容量**            | [\_L2\_CACHE\_SIZE](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L168-L192) | 14 点 sweep，检测 latency cliff               |
| **时钟频率**             | [\_BOOST\_CLOCK](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L194-L213)    | 双 timing：clock64() + cudaEventElapsedTime |
| **DRAM 带宽**          | [\_DRAM\_BANDWIDTH](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L215-L237) | STREAM copy，65535×256 threads             |
| **Shared Memory 容量** | [\_SHMEM\_CAPACITY](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L239-L259) | Occupancy API sweep                       |
| **Bank Conflict**    | [\_BANK\_CONFLICT](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L261-L285)  | 双内核比较（strided vs sequential）              |
| **SM 数量**            | [\_SM\_COUNT](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L300-L322)       | 三策略交叉验证（API + block sweep）                |

**优点**：

- ✅ **全覆盖** — 11 种测量目标都有详细设计原则
- ✅ **差异化策略** — 每种测量都有针对性的方法
- ✅ **参数可调** — working set size、迭代次数等参数明确

### 4.2 频率锁定环境适配 ✅

**设计原则支持**：[design\_principles.py:194-213](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L194-L213)

```python
"_BOOST_CLOCK = (
    "🎯 DESIGN THINKING: GPU Boost Clock Measurement via Cycle Counter\n\n"
    
    "📐 ARCHITECTURAL INSIGHT:\n"
    "- True frequency = GPU cycles / wall-clock time\n"
    "- Reported clock may not reflect actual runtime frequency\n\n"
    
    "🔬 MEASUREMENT STRATEGY:\n"
    "1. Compute-Intensive Kernel: 10M iterations of arithmetic operations\n"
    "2. Dual Timing: clock64() for GPU cycles + cudaEventElapsedTime for wall-clock\n"
    "3. freq_MHz = total_cycles / elapsed_microseconds\n"
    
    "⚠️ ANTI-CHEAT:\n"
    "- Do NOT use cudaGetDeviceProperties — returns base/max clock, not actual\n"
    "- MUST use cudaEventElapsedTime for wall-clock timing\n"
    "- Ensure kernel is compute-bound (no global memory access)\n"
)
```

**优点**：

- ✅ **实测频率** — 不依赖 API 返回的标称频率
- ✅ **双 timing** — 使用 clock64() 和 cudaEventElapsedTime 交叉验证
- ✅ **抗干扰** — 即使 GPU 被降频也能测出真实频率

### 4.3 SM 屏蔽环境适配 ✅

**设计原则支持**：[design\_principles.py:300-322](file://e:\GPU_Profiling_System\src\domain\design_principles.py#L300-L322)

```python
"_SM_COUNT = (
    "🔬 MEASUREMENT STRATEGY:\n"
    "Strategy 1: CUDA API Query — cudaGetDeviceProperties\n"
    "Strategy 2: Block ID Sweep (Empirical) — MOST TRUSTWORTHY\n"
    "Strategy 3: Occupancy API Cross-Validation\n\n"
    
    "⚠️ ANTI-CHEAT:\n"
    "- Do NOT rely solely on cudaGetDeviceProperties — may be virtualized\n"
    "- MUST use empirical measurement\n"
    "- MUST cross-validate\n"
)
```

**优点**：

- ✅ **多策略** — 不依赖单一 API
- ✅ **实测优先** — Block ID Sweep 是最可靠的方法
- ✅ **交叉验证** — 三种策略相互印证

***

## 五、跨 Agent 协作审查 ⚠️ **良好但有改进空间**

### 5.1 接收 Planner 信息 ✅

**当前实现**：[prompt\_builder.py:127-142](file://e:\GPU_Profiling_System\src\domain\prompt_builder.py#L127-L142)

```python
def _codegen_task(target_spec: dict[str, Any], prev_result: Any | None) -> str:
    principle = get_design_principle(target)
    
    parts = [
        f"Write a CUDA micro-benchmark for: {target}",
        f"\nTarget specification: {target_spec}",
        f"\n{principle}",
    ]
    
    if prev_result is not None:
        plan_output = prev_result.data.get("final_output", "")
        if plan_output:
            parts.append(f"\n\n--- Plan from previous stage ---\n{plan_output}")
```

**优点**：

- ✅ **传递设计原则** — CodeGen 能看到针对该 target 的详细设计原则
- ✅ **传递 Planner 输出** — 如果 Planner 生成了计划，会传递给 CodeGen

**缺陷** ⚠️：

- ❌ **没有显式传递 measurement methodology** — Planner 的 `method` 字段没有传递给 CodeGen
- ❌ **没有传递 working\_set\_size 建议** — Planner 可能建议了具体的 working set 大小，但没有传递

**对比 PJ 需求**：

> spec.md §5.2: "主规划智能体解析目标 → 生成初步计划 → 委派'生成指针追逐内核'任务给 代码生成子代理"

**当前实现**：CodeGen **只看到 design principle**，没有看到 Planner 生成的具体计划。

### 5.2 接收 Verification 反馈 ✅ **已实现但需完善**

**当前实现**：[stage\_executor.py:83-95](file://e:\GPU_Profiling_System\src\domain\stage_executor.py#L83-L95), [127-160](file://e:\GPU_Profiling_System\src\domain\stage_executor.py#L127-L160)

```python
def execute(self, step: Any, ctx: PipelineContext) -> SubAgentResult:
    feedback = ctx.get_feedback_for_codegen()
    if feedback and step.stage == PipelineStage.CODE_GEN:
        message = self._build_retry_message(step, ctx, feedback)
    else:
        message = self._build_collaboration_message(step, ctx)

def _build_retry_message(self, step: Any, ctx: PipelineContext, feedback: dict[str, Any]) -> CollaborationMessage:
    concerns = feedback.get("concerns", [])
    suggested_fixes = feedback.get("suggested_fixes", [])
    
    feedback_parts = [
        "⚠️  VERIFICATION REJECTED YOUR PREVIOUS OUTPUT",
        f"Iteration: {iteration}/{ctx.max_iterations}",
        "Please fix the following concerns and regenerate:",
        "",
        *[f"- {concern}" for concern in concerns],
    ]
    if suggested_fixes:
        feedback_parts.append("Suggested fixes:")
        for fix in suggested_fixes:
            feedback_parts.append(f"  → {fix}")
    
    payload["rejection_feedback"] = "\n".join(feedback_parts)
```

**优点**：

- ✅ **反馈注入** — Verification 的 concerns 和 suggested\_fixes 能传递给 CodeGen
- ✅ **迭代计数** — 告知 CodeGen 当前是第几次迭代
- ✅ **格式化良好** — 反馈信息结构清晰

**缺陷** ⚠️：

- ❌ **没有传递 MetricAnalysis 的分析结果** — MetricAnalysis 的 bottleneck\_type 和 parsed\_metrics 没有传递给 CodeGen
- ❌ **没有传递改进建议** — MetricAnalysis 如果有优化建议（如"使用 shared memory"），没有传递给 CodeGen

### 5.3 对话历史继承 ✅

**当前实现**：[stage\_executor.py:193-197](file://e:\GPU_Profiling_System\src\domain\stage_executor.py#L193-L197), [245-252](file://e:\GPU_Profiling_System\src\domain\stage_executor.py#L245-L252)

```python
if ctx.conversation_history:
    history_summary = self._format_conversation_history(ctx)
    user_task = f"{history_summary}\n\n---\n\n{user_task}"

def _save_conversation_history(agent: BaseSubAgent, ctx: PipelineContext) -> None:
    """Save agent's conversation history to context."""
    ctx.conversation_history.extend(agent.context_manager.to_messages())

def _format_conversation_history(ctx: PipelineContext) -> str:
    """Format conversation history for retry prompt."""
    recent = ctx.conversation_history[-10:]  # Last 10 messages
    # Format as summary
```

**优点**：

- ✅ **完整继承** — CodeGen 能看到之前的对话历史
- ✅ **限制长度** — 只保留最近 10 条消息，避免 token 溢出
- ✅ **格式化** — 历史消息被格式化为易读的摘要

***

## 六、持续迭代改进审查 ✅ **优秀**

### 6.1 迭代机制架构 ✅

**实现位置**：[pipeline.py:99-145](file://e:\GPU_Profiling_System\src\domain\pipeline.py#L99-L145)

```python
def run(self, target_spec: dict[str, Any]) -> SubAgentResult:
    ctx = PipelineContext(target_spec=target_spec)
    stage_idx = 0
    while stage_idx < len(self._stages):
        step = self._stages[stage_idx]
        result = self._executor.execute(step, ctx)
        
        if result.status == SubAgentStatus.REJECTED and step.stage == PipelineStage.VERIFICATION:
            concerns = result.data.get("concerns", [])
            suggested_fixes = result.data.get("suggested_fixes", [])
            ctx.add_rejection(step.stage.value, concerns, suggested_fixes)
            
            if ctx.can_retry():
                ctx.increment_iteration()
                code_gen_idx = self._find_stage_index(PipelineStage.CODE_GEN)
                stage_idx = code_gen_idx  # 回退到 CodeGen
                continue
        
        stage_idx += 1
```

**优点**：

- ✅ **自动回退** — Verification REJECT 后自动回退到 CodeGen
- ✅ **迭代限制** — 最多 3 次迭代（防止无限循环）
- ✅ **反馈携带** — 每次重试都携带 concerns 和 suggested\_fixes

### 6.2 提示词对迭代的支持 ✅

**提示词支持**：[agent\_prompts.py:97-105](file://e:\GPU_Profiling_System\src\domain\agent_prompts.py#L97-L105)

```python
"ERROR RECOVERY PROTOCOL:\n"
"- If compilation fails: read the error message, identify the issue, fix the code, retry\n"
"- If execution fails: check the binary path exists, fix the issue, recompile\n"
"- If output is 0 or negative: the measurement logic is wrong — fix and retry\n"
"- Maximum 3 retry attempts per target before reporting what went wrong\n"
```

**优点**：

- ✅ **明确重试协议** — 告知 LLM 如何响应错误
- ✅ **最大重试次数** — 3 次重试限制
- ✅ **失败报告** — 要求说明失败原因

### 6.3 反馈利用机制 ✅

**当前实现**：[stage\_executor.py:190-192](file://e:\GPU_Profiling_System\src\domain\stage_executor.py#L190-L192)

```python
if rejection_feedback:
    user_task = f"{rejection_feedback}\n\n---\n\n{user_task}"
```

**效果**：

- ✅ **前置反馈** — rejection\_feedback 放在任务描述最前面，确保 LLM 优先看到
- ✅ **分隔清晰** — 使用 `---` 分隔反馈和新任务
- ✅ **完整传递** — 所有 concerns 和 suggested\_fixes 都传递给 LLM

***

## 七、与 PJ 需求对比审查

### 7.1 PJ 需求.md §1.7 硬件内在剖析要求

| 需求                          | CodeGen 实现                         | 达成度    |
| :-------------------------- | :--------------------------------- | :----- |
| **指针追逐内核**                  | ✅ design\_principles.py 详细描述       | 100% ✅ |
| \*\* defeat prefetchers\*\* | ✅ 明确告知使用 random permutation        | 100% ✅ |
| **working-set sweep**       | ✅ L2 cache size 检测使用 14 点 sweep    | 100% ✅ |
| **双 timing 策略**             | ✅ clock64() + cudaEventElapsedTime | 100% ✅ |
| **交叉验证**                    | ✅ 多策略测量（如 SM count）                | 100% ✅ |
| **抗频率锁定**                   | ✅ 实测频率，不依赖 API                     | 100% ✅ |
| **抗 SM 屏蔽**                 | ✅ Block ID Sweep 实测                | 100% ✅ |

### 7.2 spec.md §2 设计原则

| 原则            | CodeGen 实现                            | 达成度    |
| :------------ | :------------------------------------ | :----- |
| **P1 工具定义边界** | ✅ compile\_cuda, execute\_binary 工具契约 | 100% ✅ |
| **P2 故障关闭**   | ✅ 编译失败不执行，执行失败不报告                     | 100% ✅ |
| **P3 上下文工程**  | ✅ 动态注入 design principle + feedback    | 100% ✅ |
| **P4 可组合性**   | ✅ 支持多轮迭代，反馈可传递                        | 100% ✅ |
| **P5 编译时消除**  | ⚠️ 工具在运行时选择，非编译时                      | 50% ⚠️ |
| **P6 状态落盘**   | ✅ conversation\_history 保存到 context   | 100% ✅ |
| **P7 生成评估分离** | ✅ CodeGen 生成，Verification 评估          | 100% ✅ |

### 7.3 spec.md §5 多智能体协作

| 协作要求                     | CodeGen 实现                           | 达成度    |
| :----------------------- | :----------------------------------- | :----- |
| **接收 Planner 任务**        | ✅ 接收 target\_spec + design principle | 80% ✅  |
| **接收 Verification 反馈**   | ✅ 接收 concerns + suggested\_fixes     | 100% ✅ |
| **接收 MetricAnalysis 建议** | ❌ 未实现                                | 0% ❌   |
| **迭代改进**                 | ✅ 支持最多 3 次迭代                         | 100% ✅ |
| **上下文隔离**                | ✅ 各 Agent 独立 context                 | 100% ✅ |

***

## 八、根本问题总结

### 8.1 优点总结 ✅

| 维度             | 评分    | 说明                                              |
| :------------- | :---- | :---------------------------------------------- |
| **环境适配性**      | ⭐⭐⭐⭐⭐ | GPU 架构自动检测，沙箱隔离，抗作弊                             |
| **代码正确性**      | ⭐⭐⭐⭐⭐ | 详细的最佳实践，错误恢复，代码结构要求                             |
| **多 GPU 适配**   | ⭐⭐⭐⭐⭐ | 11 种测量目标全覆盖，差异化策略                               |
| **迭代改进**       | ⭐⭐⭐⭐⭐ | 自动回退，反馈注入，对话历史继承                                |
| **跨 Agent 协作** | ⭐⭐⭐⭐  | 接收 Planner 和 Verification 信息，但缺少 MetricAnalysis |

### 8.2 缺陷总结 ⚠️

| 缺陷                        | 严重性   | 影响                              |
| :------------------------ | :---- | :------------------------------ |
| **未接收 MetricAnalysis 建议** | 🟡 中等 | CodeGen 无法根据瓶颈分析优化代码            |
| **Planner 的 method 未传递**  | 🟡 中等 | CodeGen 可能看不到 Planner 的具体测量方法建议 |
| **P5 编译时消除未实现**           | 🟢 低  | 工具在运行时选择，但不影响功能                 |

***

## 九、修复建议（优先级排序）

### 🔴 P0 — 必须修复

#### 1. 传递 MetricAnalysis 的建议给 CodeGen

**当前问题**：MetricAnalysis 的 bottleneck\_type 和 recommendations 没有传递给 CodeGen

**修复方案**：

```python
# stage_executor.py: 修改 _build_retry_message 或 _build_collaboration_message
def _build_collaboration_message(self, step: Any, ctx: PipelineContext) -> CollaborationMessage:
    prev_result = ctx.prev_result
    
    # 添加 MetricAnalysis 的建议
    metric_feedback = ""
    if prev_result and hasattr(prev_result, 'data'):
        if "bottleneck_type" in prev_result.data:
            metric_feedback += f"\n📊 MetricAnalysis identified bottleneck: {prev_result.data['bottleneck_type']}\n"
        if "recommendations" in prev_result.data:
            metric_feedback += "\n💡 Optimization suggestions from MetricAnalysis:\n"
            for rec in prev_result.data.get("recommendations", []):
                metric_feedback += f"  - {rec}\n"
    
    # 注入到用户任务
    if metric_feedback:
        user_task = f"{metric_feedback}\n\n---\n\n{user_task}"
```

**预期效果**：

```
CodeGen 收到：
📊 MetricAnalysis identified bottleneck: memory_bound

💡 Optimization suggestions from MetricAnalysis:
  - Use shared memory to reduce DRAM accesses
  - Implement tiling to improve data reuse
  - Check if memory accesses are coalesced

---

Write a CUDA micro-benchmark for: dram_bandwidth_gbps
...
```

### 🟡 P1 — 重要改进

#### 2. 传递 Planner 的 method 字段

**当前问题**：Planner 生成的 `method` 字段（包含具体测量方法）没有传递给 CodeGen

**修复方案**：

```python
# prompt_builder.py: 修改 _codegen_task
def _codegen_task(target_spec: dict[str, Any], prev_result: Any | None) -> str:
    # ... 现有代码 ...
    
    if prev_result is not None:
        plan_output = prev_result.data.get("final_output", "")
        tasks = prev_result.data.get("tasks", [])
        
        # 提取当前 target 的 method
        target = target_spec.get("target", "unknown")
        method = ""
        for task in tasks:
            if task.get("target") == target:
                method = task.get("method", "")
                break
        
        if method:
            parts.append(f"\n\n📋 Measurement methodology from Planner:\n{method}")
        
        if plan_output:
            parts.append(f"\n\n--- Plan from previous stage ---\n{plan_output}")
```

**预期效果**：

```
CodeGen 收到：
📋 Measurement methodology from Planner:
Use pointer-chasing with 128MB working set, random permutation chain, 
clock64() timing, 10M iterations, 3 trials report median.

---

Write a CUDA micro-benchmark for: dram_latency_cycles
...
```

#### 3. 增强提示词，明确告知利用 MetricAnalysis 反馈

**当前提示词**：没有提及如何利用 MetricAnalysis 的建议

**修复方案**：

```python
# agent_prompts.py: 在_CODE_GEN 中添加
_CODE_GEN = (
    # ... 现有内容 ...
    
    "COLLABORATION PROTOCOL:\n"
    "- You will receive feedback from MetricAnalysis Agent about bottleneck types\n"
    "- If MetricAnalysis identifies 'memory_bound', consider:\n"
    "  - Using shared memory to reduce DRAM accesses\n"
    "  - Implementing data tiling for better reuse\n"
    "  - Optimizing memory access patterns (coalescing)\n"
    "- If MetricAnalysis identifies 'compute_bound', consider:\n"
    "  - Using mixed precision (FP16/BF16) if accuracy allows\n"
    "  - Optimizing register usage to improve occupancy\n"
    "- If MetricAnalysis identifies 'latency_bound', consider:\n"
    "  - Increasing parallelism to hide latency\n"
    "  - Using software pipelining\n"
    "- Treat MetricAnalysis feedback as optimization guidance — implement suggestions when applicable\n"
)
```

### 🟢 P2 — 长期优化

#### 4. 实现编译时工具消除（P5）

**当前问题**：工具在运行时选择，而非编译时

**修复方案**：使用工厂模式在构建时确定工具集

```python
# system_builder.py: 为每个 Agent 角色预定义工具集
def build_agent_tools(role: AgentRole) -> dict[str, Callable]:
    """Build tool set for specific agent role at construction time."""
    tool_sets = {
        AgentRole.CODE_GEN: {"compile_cuda", "execute_binary", "write_file", "read_file"},
        AgentRole.METRIC_ANALYSIS: {"run_ncu", "read_file"},
        AgentRole.VERIFICATION: {"read_file"},
    }
    
    allowed_tools = tool_sets.get(role, set())
    handlers = {}
    for name in allowed_tools:
        handlers[name] = tool_handlers.get_handler(name)
    
    return handlers
```

***

## 十、修复后的预期效果

### 修复前

```
CodeGen 收到：
Write a CUDA micro-benchmark for: dram_bandwidth_gbps
Target specification: {"target": "dram_bandwidth_gbps"}
🎯 DESIGN THINKING: DRAM Bandwidth Measurement via STREAM Copy
...
```

### 修复后

```
CodeGen 收到：
📊 MetricAnalysis identified bottleneck: memory_bound

💡 Optimization suggestions from MetricAnalysis:
  - Use shared memory to reduce DRAM accesses
  - Implement tiling to improve data reuse
  - Check if memory accesses are coalesced

📋 Measurement methodology from Planner:
Use STREAM copy pattern with 128MB arrays, launch 65535×256 threads,
cudaEventElapsedTime for wall-clock timing, 3 trials report MAX.

---

Write a CUDA micro-benchmark for: dram_bandwidth_gbps
Target specification: {"target": "dram_bandwidth_gbps"}
🎯 DESIGN THINKING: DRAM Bandwidth Measurement via STREAM Copy
...
```

**改进效果**：

- ✅ CodeGen 能看到 MetricAnalysis 的瓶颈分析
- ✅ CodeGen 能看到 Planner 的具体测量方法
- ✅ CodeGen 能根据反馈优化代码（如使用 shared memory）

***

## 十一、最终结论

### 审查结论：✅ **总体优秀，少量改进空间**

**CodeGen Agent 提示词设计质量**：

| 维度             | 评分    | 说明                                             |
| :------------- | :---- | :--------------------------------------------- |
| **环境适配性**      | ⭐⭐⭐⭐⭐ | GPU 架构检测、沙箱隔离、抗作弊机制完善                          |
| **代码正确性**      | ⭐⭐⭐⭐⭐ | 最佳实践详细、错误恢复机制健全                                |
| **多 GPU 适配**   | ⭐⭐⭐⭐⭐ | 11 种测量目标全覆盖，策略差异化                              |
| **任务负载多样性**    | ⭐⭐⭐⭐⭐ | 延迟、带宽、容量、频率等全覆盖                                |
| **跨 Agent 协作** | ⭐⭐⭐⭐  | 接收 Planner 和 Verification 信息，缺少 MetricAnalysis |
| **持续迭代改进**     | ⭐⭐⭐⭐⭐ | 反馈注入、对话历史、自动回退机制完善                             |

**与 PJ 需求的符合度**：

- ✅ PJ 需求.md §1.7：**100% 符合** — 所有硬件探测要求都有详细设计原则
- ✅ spec.md §2 设计原则：**93% 符合** — P5（编译时消除）需改进
- ✅ spec.md §5 多智能体协作：**80% 符合** — 缺少 MetricAnalysis → CodeGen 的反馈路径

**建议优先级**：

1. **P0**：实现 MetricAnalysis → CodeGen 的建议传递
2. **P1**：传递 Planner 的 method 字段，增强提示词
3. **P2**：实现编译时工具消除（P5 原则）

**总体评价**：CodeGen Agent 的提示词设计**非常出色**，具备环境适配性、代码正确性保障、多 GPU 环境适配、任务负载多样性支持、持续迭代改进能力。唯一的不足是跨 Agent 协作中缺少 MetricAnalysis 的建议传递，修复后将达到完美。
