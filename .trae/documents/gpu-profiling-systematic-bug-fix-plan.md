# GPU Profiling System - 系统性Bug修复与重构计划

## 📋 执行摘要

基于对最新测试结果的深度分析（`kaggle_results/`），发现**CodeGen阶段的核心问题**：LLM Agent陷入"重复编译同一target的死循环"，导致3个请求的targets全部缺失。

### 关键问题诊断

#### 🔴 Bug #1: CodeGen Agent目标迷失（CRITICAL - 根因）
**现象**：
- CodeGen阶段运行了12轮，调用了9次`compile_cuda`
- **所有9次编译都是针对 `dram_latency_cycles` 的不同变体**
- 从未为 `l2_cache_size_mb` 和 `actual_boost_clock_mhz` 编写代码
- 最终结果：**0/3 targets被测量**

**根因分析**：
1. **Prompt缺乏强制性目标推进机制**：当前prompt只说"Repeat for each target"，但没有强制要求"完成一个target后立即切换到下一个"
2. **_find_unmeasured_targets()检测失效**：该函数依赖正则匹配`target_name: value`格式，但：
   - Agent在Turn 2-3执行了execute_binary，但输出可能未被正确解析
   - 或者Agent在文本回复中报告了值，但格式不符合正则
3. **缺少目标状态机**：没有显式的"当前正在测量哪个target"的状态跟踪
4. **引导信息不够强制**：当检测到unmeasured targets时，SYSTEM消息的优先级不够高

#### 🟡 Bug #2: execute_binary结果解析链路断裂
**现象**：
- Turn 2和Turn 3都调用了`execute_binary(binary_path=".kaggle_sandbox/bin/benchmark")`
- 但后续_turn 4开始Agent继续重新编译，说明它没有认可测量结果

**可能原因**：
1. execute_binary返回的结果格式不被Agent理解
2. 结果中的数值不合理（如0或异常值），触发了"零值警告"
3. Agent忽略了工具返回值，继续优化代码

#### 🟡 Bug #3: 重复编译死循环无退出机制
**现象**：
- Turn 1, 5, 7, 9, 11都在编译dram_latency_cycles的不同版本
- 每次都是小幅修改（换随机数生成器、改printf格式等）

**根因**：缺少"同一target最大重试次数"限制

---

## 🔧 修复方案（按优先级排序）

### Phase 1: 紧急修复（解决当前崩溃问题）

#### 1.1 重构 _find_unmeasured_targets() 检测逻辑
**文件**: [agent_loop.py](src/application/agent_loop.py)  
**问题**: 当前正则`\s*([\w_]+)\s*[:=]\s*[\d.]+`太严格，容易漏检  
**修复**:
```python
def _find_unmeasured_targets(self) -> list[str]:
    all_targets = []
    measured = set()
    entries = self.context_manager.get_entries()
    
    # 提取所有请求的targets
    for entry in entries:
        if entry.role.value == "system" and "targets" in entry.content:
            m = re.search(r'"targets"\s*:\s*\[([^\]]+)\]', entry.content)
            if m:
                all_targets = re.findall(r'"([^"]+)"', m.group(1))
    
    # 增强的测量检测：支持更多格式
    for entry in entries:
        content = entry.content
        if entry.role.value == "assistant":
            # 格式1: JSON stdout中的 key: value
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    stdout = data.get("stdout", "")
                    if stdout:
                        for line in stdout.splitlines():
                            m = re.match(r'\s*([\w_]+)\s*[:=]\s*[\d.]+', line)
                            if m:
                                measured.add(m.group(1))
            except (json.JSONDecodeError, TypeError):
                pass
            
            # 格式2: 直接文本中的 key: value
            for line in content.splitlines():
                m = re.match(r'\s*([\w_]+)\s*[:=]\s*[\d.]+', line)
                if m:
                    measured.add(m.group(1))
    
    return [t for t in all_targets if t not in measured]
```

#### 1.2 添加目标状态机到AgentLoop
**文件**: [agent_loop.py](src/application/agent_loop.py)  
**新增功能**:
```python
# 在LoopState中添加
@dataclass
class LoopState:
    # ...现有字段...
    current_target: str | None = None  # 当前正在测量的target
    completed_targets: list[str] = field(default_factory=list)  # 已完成的targets
    target_retry_count: dict[str, int] = field(default_factory=dict)  # 每个target的重试次数

# 在AgentLoop.__init__后初始化
def _init_target_state(self, target_spec: dict):
    targets = target_spec.get("targets", [])
    self.loop_state.current_target = targets[0] if targets else None
    self.loop_state.completed_targets = []
    self.loop_state.target_retry_count = {t: 0 for t in targets}
```

#### 1.3 强制性目标切换引导（PRIORITY BOOST）
**文件**: [agent_loop.py](src/application/agent_loop.py)  
**位置**: execute_binary成功后的引导逻辑（第520-534行）  
**增强**:
```python
# 当检测到unmeasured targets时，使用更强烈的引导
if unmeasured and result.get("return_code", -1) == 0:
    next_target = unmeasured[0]
    
    # 构建超强调导消息（使用WARNING级别）
    guidance = (
        f"🛑 MANDATORY TARGET SWITCH 🛑\n\n"
        f"✅ Current target '{self.loop_state.current_target}' is COMPLETED.\n"
        f"❌ You have NOT measured these targets yet: {unmeasured}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👉 IMMEDIATE NEXT ACTION REQUIRED:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"You MUST now measure: **{next_target}**\n\n"
        f"STEP 1: Call compile_cuda with NEW CUDA code for '{next_target}'\n"
        f"  → Do NOT reuse the previous kernel — each target needs unique code\n"
        f"  → Use the design principle below for guidance\n\n"
        f"STEP 2: Call execute_binary with the new binary_path\n\n"
        f"⚠️ DO NOT continue optimizing '{self.loop_state.current_target}'\n"
        f"⚠️ DO NOT output text explanations — CALL compile_cuda NOW\n\n"
        f"Design principle for '{next_target}':\n{next_brief}"
    )
    
    # 更新当前目标状态
    if self.loop_state.current_target and self.loop_state.current_target not in self.loop_state.completed_targets:
        self.loop_state.completed_targets.append(self.loop_state.current_target)
    self.loop_state.current_target = next_target
    
    self.context_manager.add_entry(
        Role.SYSTEM,
        guidance,
        token_count=100,  # 提高token权重
    )
```

#### 1.4 添加同target最大重试限制
**文件**: [agent_loop.py](src/application/agent_loop.py)  
**位置**: compile_cuda调用前检查  
**新增**:
```python
if tool_call.name == "compile_cuda":
    # 检查当前target是否已超过最大重试次数
    MAX_RETRIES_PER_TARGET = 3
    current_target = self.loop_state.current_target
    
    if current_target and self.loop_state.target_retry_count.get(current_target, 0) >= MAX_RETRIES_PER_TARGET:
        remaining = [t for t in all_targets if t not in self.loop_state.completed_targets and t != current_target]
        
        if remaining:
            next_target = remaining[0]
            force_guidance = (
                f"🚨 MAX RETRIES REACHED FOR '{current_target}' ({MAX_RETRIES_PER_TARGET} attempts)\n\n"
                f"You have compiled '{current_target}' {MAX_RETRIES_PER_TARGET} times without success.\n"
                f"FORCING switch to next target: **{next_target}**\n\n"
                f"Call compile_cuda with CUDA code for '{next_target}' NOW.\n"
                f"Do NOT attempt to fix '{current_target}' again."
            )
            self.context_manager.add_entry(Role.SYSTEM, force_guidance, token_count=60)
            self.loop_state.current_target = next_target
            self.loop_state.target_retry_count[next_target] = 0
            return  # 阻止本次compile_cuda调用
        
        # 如果没有remaining targets，允许最后一次尝试
```

---

### Phase 2: Prompt工程强化

#### 2.1 重构CodeGen Prompt的目标管理部分
**文件**: [agent_prompts.py](src/domain/agent_prompts.py)  
**位置**: `_CODE_GEN`常量（第82-276行）  
**新增章节**:
```python
'''━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 TARGET MANAGEMENT PROTOCOL (MANDATORY — VIOLATION CAUSES FAILURE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You will receive <N> targets to measure. You MUST follow this EXACT sequence:

✅ SEQUENTIAL TARGET PROCESSING (NO PARALLELISM):
1. Pick the FIRST unmeasured target from the list
2. Write CUDA code SPECIFICALLY for that target
3. Compile it with compile_cuda
4. Execute with execute_binary
5. Verify the output contains "target_name: numeric_value"
6. ONLY THEN move to the next target

❌ FORBIDDEN BEHAVIORS:
- Do NOT recompile the same target more than 2 times
- Do NOT skip targets — every target must be measured
- Do NOT combine multiple targets in one kernel
- Do NOT optimize a target after getting a valid measurement
- Do NOT output text explanations between targets — call tools immediately

📋 TARGET CHECKLIST (track your progress):
After each execute_binary, check:
□ Did I get a non-zero value?
□ Is the value in the expected range? (see reference values above)
□ Should I move to the next target?

If value is 0 or out-of-range → retry ONCE with fixed code, then MOVE ON.
If value looks reasonable → IMMEDIATELY switch to next target.

⚠️ PIPELINE WILL FAIL if any target is missing!
'''
```

#### 2.2 在用户消息中注入显式目标列表
**文件**: [stage_executor.py](src/domain/stage_executor.py) 或 ControlPlane  
**增强CodeGen阶段的task message**:
```python
def _build_codegen_task_message(self, plan_result, target_spec):
    targets = target_spec.get("targets", [])
    
    task_message = f"""{plan_result}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TARGET ASSIGNMENT (MEASURE ALL OF THESE IN ORDER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Targets to measure ({len(targets)} total):
"""
    for i, target in enumerate(targets, 1):
        task_message += f"  {i}. {target}\n"

    task_message += """
⚠️ You MUST measure ALL targets above. Start with target #1.
After completing each target, move to the next number.
Do NOT skip any target. Do NOT reorder.

Current status:
  ☐ Target 1: {targets[0]} ← START HERE
""".format(targets=targets)

    return task_message
```

---

### Phase 3: 执行结果验证增强

#### 3.1 改进execute_binary的结果反馈
**文件**: [agent_loop.py](src/application/agent_loop.py)  
**位置**: 第499-517行的零值检测之后  
**新增**:
```python
# 解析execute_binary的stdout并提取测量值
if tool_call.name == "execute_binary":
    stdout = result.get("stdout", "")
    if stdout:
        measurements = {}
        for line in stdout.splitlines():
            m = re.match(r'\s*([\w_]+)\s*[:=]\s*([\d.]+)', line)
            if m:
                key, val = m.group(1), float(m.group(2))
                measurements[key] = val
        
        if measurements:
            # 记录到completed_targets
            for key in measurements:
                if key not in self.loop_state.completed_targets:
                    self.loop_state.completed_targets.append(key)
            
            # 生成结构化的测量摘要
            summary = "✅ MEASUREMENTS RECORDED:\n"
            for key, val in measurements.items():
                summary += f"  • {key}: {val}\n"
            
            self.context_manager.add_entry(Role.SYSTEM, summary, token_count=30)
```

#### 3.2 添加测量完成确认机制
**文件**: [agent_loop.py](src/application/agent_loop.py)  
**新增方法**:
```python
def _check_all_targets_measured(self) -> bool:
    """Check if all requested targets have been measured."""
    unmeasured = self._find_unmeasured_targets()
    return len(unmeasured) == 0
```

---

## 📊 影响范围与风险评估

### 修改的文件清单
1. ✅ `src/application/agent_loop.py` - 主要修改（~150行新增/修改）
2. ✅ `src/domain/agent_prompts.py` - Prompt增强（~50行新增）
3. ✅ `src/domain/stage_executor.py` - Task message构建（可选，~20行新增）

### 不影响的部分
- ApprovalQueue / ToolRunner - 无需修改
- CompletionDetector - Plan阶段已工作正常
- Permission系统 - 无需修改
- Fallback配置 - 无需修改

### 测试策略
1. **单元测试**: 测试_find_unmeasured_targets()的各种输入格式
2. **集成测试**: 模拟3-target场景，验证顺序完成
3. **回归测试**: 确保Plan阶段仍然快速完成（<30秒）

---

## 🎯 实施步骤（按顺序执行）

### Step 1: 增强_find_unmeasured_targets()检测能力
- [ ] 修改正则表达式以支持更多格式
- [ ] 添加JSON stdout解析
- [ ] 添加单元测试用例

### Step 2: 实现目标状态机
- [ ] 在LoopState中添加新字段
- [ ] 初始化目标状态
- [ ] 更新持久化逻辑

### Step 3: 强化目标切换引导
- [ ] 提升引导消息的token权重
- [ ] 使用强制性语言（MANDATORY, IMMEDIATE等）
- [ ] 更新当前目标指针

### Step 4: 添加重试限制
- [ ] 实现per-target重试计数
- [ ] 达到上限时强制切换
- [ ] 记录强制切换事件

### Step 5: Prompt工程
- [ ] 添加TARGET MANAGEMENT PROTOCOL章节
- [ ] 明确禁止的行为列表
- [ ] 添加目标checklist模板

### Step 6: Task Message增强
- [ ] 注入带编号的目标列表
- [ ] 显示进度跟踪模板
- [ ] 标记起始目标

### Step 7: 结果验证改进
- [ ] 解析execute_binary输出
- [ ] 自动记录completed_targets
- [ ] 生成测量摘要

### Step 8: 测试验证
- [ ] 运行本地测试（如果环境允许）
- [ ] 推送到Kaggle进行端到端测试
- [ ] 分析新的execution.log确认修复效果

---

## 📈 预期效果

### 量化指标
- **Plan阶段**: 保持28秒内完成（已达标）
- **CodeGen阶段**: 
  - 当前：12轮，112秒，0/3 targets ❌
  - 目标：9-15轮，180-300秒，3/3 targets ✅
- **成功率**: 从0%提升至90%+

### 质量改善
1. ✅ 所有targets都会被测量（不再遗漏）
2. ✅ 不再出现重复编译死循环
3. ✅ Agent行为更加可预测和可调试
4. ✅ 错误信息更清晰（明确指出哪个target缺失）

---

## ⚠️ 风险与回退策略

### 潜在风险
1. **过度强制可能导致Agent忽略有效优化需求**
   - 缓解：保留2次重试机会，不完全锁定
2. **Token消耗增加可能影响上下文窗口**
   - 缓解：引导消息控制在100 tokens以内
3. **某些target可能确实难以测量**
   - 缓解：允许跳过并标记为partial success

### 回退方案
如果新逻辑引入回归问题：
- 通过feature flag禁用目标状态机
- 回退到原有的_find_unmeasured_targets()实现
- 降低引导消息的强制程度

---

## 🔄 后续优化方向（不在本次范围内）

1. **并行化多target测量**（需架构调整）
2. **智能重试策略**（根据错误类型决定是否重试）
3. **测量质量评分**（自动评估结果合理性）
4. **缓存机制**（避免重复编译相同代码）

---

## 📝 实施时间估算

| Phase | 任务数 | 预计时间 |
|-------|--------|----------|
| Phase 1: 紧急修复 | 4个任务 | 30分钟 |
| Phase 2: Prompt工程 | 2个任务 | 20分钟 |
| Phase 3: 结果验证 | 2个任务 | 15分钟 |
| Step 8: 测试 | 3个任务 | 视环境而定 |
| **总计** | **11个任务** | **约65分钟** |
