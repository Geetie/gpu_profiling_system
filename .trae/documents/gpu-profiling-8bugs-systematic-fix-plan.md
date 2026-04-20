# GPU Profiling System - 8个Bug系统性修复计划

## 📋 概述

基于 `新 Kaggle 测试结果分析.md` 的深度分析，本计划将系统性修复导致Pipeline完全失败的8个bug。
**核心目标**: 修复Approval系统致命缺陷 → 解锁CodeGen阶段 → 提升整体成功率从0%到≥90%

---

## 🔴 Bug-001: ApprovalQueue.submit() 每次创建新请求（核心致命缺陷）

### 问题位置
[approval_queue.py:53-100](src/application/approval_queue.py#L53-L100)

### 根本原因
```python
def submit(self, tool_name, arguments, permissions, mode):
    request_id = f"{tool_name}_{uuid.uuid4().hex[:12]}"  # ❌ 每次生成新的UUID！
    ...
    request = ApprovalRequest(id=request_id, ...)
    self._requests[request_id] = request  # ❌ 存入字典（不复用）
    return request  # ❌ 返回PENDING状态的新请求
```

**缺陷链路**:
1. LLM调用compile_cuda → ToolRunner检测到需要审批
2. submit()生成全新UUID → 创建新请求 → 状态为PENDING
3. 抛出ApprovalRequiredError → agent_loop捕获异常
4. approval_callback返回True → 更新该请求为APPROVED
5. **重新调用tool_executor()** → 回到步骤1 → 再次创建新请求 → 无限循环♾️

### 修复方案
**采用方案A：基于工具名+参数哈希的去重机制**

修改 `submit()` 方法：
```python
import hashlib
import json

def submit(self, tool_name, arguments, permissions, mode):
    # 生成稳定的请求ID（基于工具名+参数签名）
    arg_hash = hashlib.md5(
        json.dumps(arguments, sort_keys=True).encode()
    ).hexdigest()[:12]
    request_id = f"{tool_name}_{arg_hash}"

    # 检查是否已有已批准的同类请求（关键修复）
    with self._lock:
        existing = self._requests.get(request_id)
        if existing and existing.status == ApprovalStatus.APPROVED:
            print(f"[ApprovalQueue] Reusing approved request: {request_id}")
            return existing  # ✅ 直接返回已批准的请求

        if mode == PermissionMode.CONSERVATIVE:
            request = ApprovalRequest(
                id=request_id,
                tool_name=tool_name,
                arguments=arguments,
                permissions=permissions,
                status=ApprovalStatus.AUTO_REJECTED,
                reason="Auto-rejected in CONSERVATIVE mode",
                responded_at=datetime.now(timezone.utc).isoformat(),
            )
            self._requests[request_id] = request
            self._persist_decision(request)
            return request

        request = ApprovalRequest(
            id=request_id,
            tool_name=tool_name,
            arguments=arguments,
            permissions=permissions,
        )
        self._requests[request_id] = request

    self._persister.log_entry(
        "approval_request",
        details={
            "id": request_id,
            "tool_name": tool_name,
            "permissions": permissions,
            "mode": mode.value,
        },
    )
    return request
```

### 预期效果
✅ 相同的compile_cuda调用只会在第一次需要审批，后续自动复用APPROVED状态
✅ 消除ApprovalRequiredError无限循环
✅ CodeGen阶段可以正常编译和执行CUDA代码

---

## 🔴 Bug-002: ToolRunner.execute() 不检查已有审批状态

### 问题位置
[tool_runner.py:84-99](src/application/tool_runner.py#L84-L99)

### 根本原因
```python
if needs_approval:
    request = self._approval_queue.submit(...)  # ❌ 总是提交新请求
    if request.status == ApprovalStatus.PENDING:
        raise ApprovalRequiredError(request)  # ❌ 新请求总是PENDING
```

**缺陷**: 在提交新请求前，未检查队列中是否已有相同工具+参数的已批准请求。

### 修复方案
在Step 3添加预检查逻辑：

```python
# Step 3: Check approval requirements with caching
needs_approval = False
if contract.requires_approval:
    for perm in contract.permissions:
        if self._permission_checker.requires_approval(perm):
            needs_approval = True
            break

if needs_approval:
    # ✅ 新增：先检查是否已有已批准的同类请求
    arg_hash = hashlib.md5(
        json.dumps(arguments, sort_keys=True).encode()
    ).hexdigest()[:12]
    expected_id = f"{tool_name}_{arg_hash}"
    
    existing_request = self._approval_queue.get_request(expected_id)
    if existing_request and existing_request.status == ApprovalStatus.APPROVED:
        print(f"[ToolRunner] Using cached approval for {tool_name}")
        # 跳过审批流程，直接执行
    else:
        request = self._approval_queue.submit(
            tool_name=tool_name,
            arguments=arguments,
            permissions=contract.permissions,
            mode=self._permission_checker.mode,
        )
        if request.status == ApprovalStatus.PENDING:
            raise ApprovalRequiredError(request)
        elif request.status in (ApprovalStatus.REJECTED, ApprovalStatus.AUTO_REJECTED):
            raise PermissionError(
                f"Tool '{tool_name}' approval {request.status.value}: "
                f"{request.reason}"
            )
```

**注意**: 需要在文件顶部添加 `import hashlib, json`

### 预期效果
✅ 与Bug-001配合形成双重保障
✅ 即使ApprovalQueue去重失败，ToolRunner也能避免重复提交

---

## 🔴 Bug-003: AgentLoop._execute_with_approval() 递归调用导致无限循环

### 问题位置
[agent_loop.py:937-969](src/application/agent_loop.py#L937-L969)

### 根本原因
```python
def _execute_with_approval(self, tool_call):
    try:
        return self._tool_executor(tool_call.name, tool_call.arguments)
    except ApprovalRequiredError as e:
        # ... 处理审批 ...
        if approved:
            return self._tool_executor(tool_call.name, tool_call.arguments)  # ❌ 递归调用
```

**缺陷**: 批准后直接重新调用 `_tool_executor()`，回到函数开头，如果Bug-001/Bug-002未修复会再次触发审批流程。

### 修复方案
添加审批循环保护机制 + 优化重试逻辑：

```python
def _execute_with_approval(self, tool_call: ToolCall) -> dict[str, Any]:
    if self._tool_executor is None:
        return {}

    max_approval_retries = 3  # ✅ 新增：最大重试次数
    retry_count = 0
    
    while retry_count < max_approval_retries:
        try:
            result = self._tool_executor(tool_call.name, tool_call.arguments)
            return result  # ✅ 成功执行，退出循环
            
        except ApprovalRequiredError as e:
            retry_count += 1
            self._emit(EventKind.APPROVAL_REQUEST, {
                "tool": tool_call.name,
                "request_id": e.request.id,
                "retry_count": retry_count,
            })

            if self._approval_callback is not None:
                approved = self._approval_callback(e.request)
            else:
                approved = False

            self._respond_to_approval_queue(e.request, approved)

            if approved:
                self._emit(EventKind.APPROVAL_GRANTED, {
                    "tool": tool_call.name,
                    "request_id": e.request.id,
                    "retry_count": retry_count,
                })
                # 继续循环，尝试再次执行（此时应该复用已批准的请求）
                continue
            else:
                self._emit(EventKind.APPROVAL_DENIED, {
                    "tool": tool_call.name,
                    "request_id": e.request.id,
                })
                raise PermissionError(
                    f"Tool '{tool_call.name}' approval denied"
                ) from e

    # ✅ 超过最大重试次数，快速失败
    raise RuntimeError(
        f"Approval loop detected for '{tool_call.name}' after {max_approval_retries} retries. "
        f"This indicates a systemic issue with the approval system."
    )
```

### 预期效果
✅ 即使出现审批问题也能快速失败（最多3次重试），避免无限循环浪费资源
✅ 配合Bug-001/Bug-002修复后，通常第1次就会成功，不会触发重试
✅ 提供清晰的错误诊断信息

---

## 🔴 Bug-04: HIGH_AUTONOMY模式下process:exec仍需审批

### 问题位置
[permission.py:42-47](src/domain/permission.py#L42-L47)

### 根本原因
```python
_ALWAYS_REQUIRES_APPROVAL: dict[PermissionMode, frozenset[str]] = {
    PermissionMode.CONSERVATIVE: frozenset({"file:write", "process:exec"}),
    PermissionMode.DEFAULT: frozenset({"file:write", "process:exec"}),
    PermissionMode.RELAXED: frozenset({"process:exec"}),
    PermissionMode.HIGH_AUTONOMY: frozenset({"process:exec"}),  # ❌ 即使高自主模式也需要审批！
}
```

**矛盾点**:
- [stage_executor.py:340-341](src/domain/stage_executor.py#L340-L341) 设置了自动批准回调: `lambda request: True`
- 但 `permission.py` 仍然要求 `process:exec` 需要审批
- 这导致即使意图是"完全自主"，仍然要走审批流程（虽然会被自动批准，但仍会产生开销）

### 修复方案
**推荐方案：HIGH_AUTONOMY模式下无需任何常规审批**

```python
_ALWAYS_REQUIRES_APPROVAL: dict[PermissionMode, frozenset[str]] = {
    PermissionMode.CONSERVATIVE: frozenset({"file:write", "process:exec"}),
    PermissionMode.DEFAULT: frozenset({"file:write", "process:exec"}),
    PermissionMode.RELAXED: frozenset({"process:exec"}),
    PermissionMode.HIGH_AUTONOMY: frozenset(),  # ✅ 高自主模式下无需任何审批
}
```

### 替代保守方案（如果担心安全性）
```python
# HIGH_AUTONOMY: 只对危险操作保留审批
PermissionMode.HIGH_AUTONOMY: frozenset({"process:destructive"}),  # 需要新增权限类别
```

### 预期效果
✅ 符合用户对"high_autonomy"的语义期望
✅ 减少不必要的审批开销（即使被自动批准也有性能损耗）
✅ 提升自动化Pipeline的可靠性
✅ 在Kaggle等自动化场景下完全无需人工干预

---

## 🟡 Bug-005: StageExecutor设置approval_callback但未验证有效性

### 问题位置
[stage_executor.py:340-341](src/domain/stage_executor.py#L340-L341)

### 根本原因
```python
if agent.permission_mode == PermissionMode.HIGH_AUTONOMY:
    loop.set_approval_callback(lambda request: True)  # ⚠️ 未验证callback是否正确工作
```

**缺陷**: 虽然设置了自动批准回调，但：
1. 未测试该回调是否能正确处理ApprovalRequest对象
2. 未验证approval_queue是否与tool_runner共享同一个实例
3. 缺乏日志记录，难以调试审批流程

### 修复方案
添加验证和日志记录：

```python
if agent.permission_mode == PermissionMode.HIGH_AUTONOMY:
    def auto_approve_callback(request) -> bool:
        """Auto-approve all requests in HIGH_AUTONOMY mode."""
        logger.info(
            "[StageExecutor] Auto-approving request: tool=%s, id=%s",
            getattr(request, 'tool_name', 'unknown'),
            getattr(request, 'id', 'unknown'),
        )
        return True
    
    loop.set_approval_callback(auto_approve_callback)
    logger.info(
        "[StageExecutor] Set auto-approve callback for %s stage (mode=%s)",
        step.stage.value,
        agent.permission_mode.value,
    )

    # ✅ 新增：验证审批流程连通性
    if self._sandbox and self._tool_handlers:
        test_result = loop._test_approval_flow()
        if not test_result.get("success"):
            logger.error(
                "[StageExecutor] CRITICAL: Approval flow test failed: %s",
                test_result.get("error"),
            )
```

同时在AgentLoop中添加测试方法：

```python
def _test_approval_flow(self) -> dict:
    """Test the approval flow end-to-end."""
    try:
        if self._approval_callback is None:
            return {"success": False, "error": "No approval callback set"}
        
        from src.application.approval_queue import ApprovalRequest
        test_request = ApprovalRequest(
            id="test_001",
            tool_name="test_tool",
            arguments={},
            permissions=["test"],
        )
        
        approved = self._approval_callback(test_request)
        if not approved:
            return {"success": False, "error": "Callback returned False"}
        
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 预期效果
✅ 在Pipeline开始前发现配置问题
✅ 提供清晰的错误诊断信息
✅ 增强可观测性，便于未来调试

---

## 🟡 Bug-006: Bank Conflict Fallback Source clock64() 返回0

### 问题位置
[bank_conflict.py:152-262](src/infrastructure/probing/bank_conflict.py#L152-262)

### 根本原因
Fallback CUDA source中使用`clock64()`进行计时：
```cuda
long long start = clock64();
// ... kernel execution ...
long long end = clock64();
long long cycles = end - start;  // ❌ 可能返回0
```

**可能原因**:
1. `clock64()`在某些GPU架构或驱动版本下返回不准确的值
2. 当kernel执行时间极短时，时钟精度不足导致差值为0
3. 编译器优化掉了计时循环（虽然有asm volatile barrier）

### 修复方案
**采用方案A：使用cudaEventElapsedTime替代clock64() + 保留clock64作为fallback**

修改 `_get_fallback_source()` 方法中的kernel代码：

```cuda
if (threadIdx.x == 0) {
    // --- Strided access (bank-conflicting) ---
    long long start_strided = 0, end_strided = 0;
    float strided_ms = 0;
    
    // Primary timing: cudaEventElapsedTime (more reliable)
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cudaEventRecord(start_event);
    start_strided = clock64();  // Backup timer
    long long sink1 = 0;
    #pragma unroll 1
    for (int iter = 0; iter < 2000; iter++) {
        int idx = (iter * 32) % size;
        sink1 += shmem[idx];
        shmem[idx] = (int)sink1 + 1;
    }
    asm volatile("" : "+l"(sink1) : : "memory");
    end_strided = clock64();  // Backup timer
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&strided_ms, start_event, stop_event);
    
    long long strided_cycles = end_strided - start_strided;
    
    // Use event-based timing if clock64 returns 0 or invalid value
    if (strided_cycles <= 0 && strided_ms > 0) {
        strided_cycles = (long long)(strided_ms * 1e6);  // Convert ms to cycles (approximate)
    }
    
    // ... Sequential access (similar pattern) ...
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    result_strided[0] = strided_cycles;
    result_sequential[0] = sequential_cycles;
}
```

### 预期效果
✅ Bank conflict测量可以在所有GPU上正常工作
✅ 提高目标测量覆盖率从77.8%到88.9%（7/9 → 8/9）
✅ 增强系统的鲁棒性

---

## 🟡 Bug-007: Bank Conflict 验证逻辑过于严格

### 问题位置
[bank_conflict.py:245-248](src/infrastructure/probing/bank_conflict.py#L245-248)

### 根本原因
```cuda
if (h_sequential <= 0 || h_strided <= 0) {
    printf("CUDA_ERROR: Invalid cycles strided=%lld sequential=%lld\n", h_strided, h_sequential);
    return 1;
}
```

**缺陷**: 当clock64()返回0或负数时直接报错退出，没有尝试恢复或使用备用计时方法。

### 修复方案
结合Bug-006一起修复，添加容错处理：

```cuda
if (h_sequential <= 0 || h_strided <= 0) {
    printf("WARNING: clock64() returned invalid values (strided=%lld, sequential=%lld)\n", 
           h_strided, h_sequential);
    printf("Attempting to use event-based timing fallback...\n");
    
    // If we have event-based measurements, use those instead
    if (event_strided > 0 && event_sequential > 0) {
        h_strided = event_strided;
        h_sequential = event_sequential;
        printf("Using event-based timing: strided=%lld, sequential=%lld\n", 
               h_strided, h_sequential);
    } else {
        printf("CUDA_ERROR: All timing methods failed\n");
        return 1;
    }
}
```

### 预期效果
✅ 当主计时方法失败时能优雅降级到备用方法
✅ 减少因边缘情况导致的测量失败
✅ 提供更详细的诊断信息

---

## 🟢 Bug-008: Plan阶段轮次控制效率低（CompletionDetector问题）

### 问题位置
[agent_loop.py](src/application/agent_loop.py) (CompletionDetector相关)

### 根本原因
Plan阶段运行了50轮才完成，耗时9分钟，占总体时间的61.5%。

**时间线分析**:
- 轮1-10: Planner初始生成任务分解
- 轮11-40: 重复输出相同的JSON（LLM未识别到已完成）
- 轮41-50: CompletionDetector最终判定完成

### 修复方案
**方案A: 强化CompletionDetection（推荐）**

在agent_loop.py中针对Plan阶段的特殊处理：

```python
# 在_inner_loop_step()中，检测Plan阶段完成条件
if tool_call is None:
    # No tool call — check for completion
    has_tools = len(self.tool_registry.list_tools()) > 0
    
    if self._completion_detector.is_completion(self._model_output):
        # ✅ 新增：对于Plan阶段，检查是否有有效的JSON输出
        current_stage = getattr(self, '_current_stage', None)
        if current_stage == "plan" or self._has_valid_plan_json():
            # Plan阶段：如果有有效JSON任务列表，允许更早完成
            consecutive_completions = getattr(self, '_consecutive_completion_count', 0) + 1
            self._consecutive_completion_count = consecutive_completions
            
            if consecutive_completions >= 2:  # 连续2次完成信号即可
                self._emit(EventKind.STOP, {"reason": "plan_completion_detected"})
                self.stop()
                return
        else:
            # 其他阶段保持原有逻辑
            ...
```

**方案B: 设置阶段性最大轮次（补充）**

在stage_executor.py或配置中添加：

```python
MAX_TURNS_PER_STAGE = {
    "plan": 20,      # Plan阶段最多20轮（从默认15调整）
    "code_gen": 50,  # CodeGen阶段保持50轮
    "metric_analysis": 30,
    "verification": 20,
}
```

### 预期效果
✅ Plan阶段段时间减少50-70%（从556秒降至~200秒）
✅ 整体Pipeline执行时间缩短30-40%（从903秒降至~600秒）
✅ 减少API调用成本（节省约30轮LLM调用）

---

## 📊 修复优先级与依赖关系

### 第一批（必须立即修复 - 解锁系统）⚡
| 顺序 | Bug ID | 文件 | 预计时间 | 依赖关系 |
|------|--------|------|----------|----------|
| 1 | **Bug-004** | permission.py | 10分钟 | 无（最简单，先改这个减少干扰） |
| 2 | **Bug-001** | approval_queue.py | 30分钟 | 无 |
| 3 | **Bug-002** | tool_runner.py | 20分钟 | 依赖Bug-001（配合使用） |
| 4 | **Bug-003** | agent_loop.py | 25分钟 | 依赖Bug-001/002 |

**小计**: ~85分钟（1.5小时）

### 第二批（重要功能修复 - 提升质量）🔧
| 顺序 | Bug ID | 文件 | 预计时间 | 依赖关系 |
|------|--------|------|----------|----------|
| 5 | **Bug-006** | bank_conflict.py | 45分钟 | 无 |
| 6 | **Bug-007** | bank_conflict.py | 15分钟 | 依赖Bug-006 |

**小计**: ~60分钟（1小时）

### 第三批（优化改进 - 提升体验）⚡
| 顺序 | Bug ID | 文件 | 预计时间 | 依赖关系 |
|------|--------|------|----------|----------|
| 7 | **Bug-005** | stage_executor.py + agent_loop.py | 30分钟 | 无 |
| 8 | **Bug-008** | agent_loop.py | 30分钟 | 无 |

**小计**: ~60分钟（1小时）

### 总计预估时间: ~3.5小时

---

## 🎯 成功指标（修复后预期）

| 指标 | 当前值 | 目标值 | 改进幅度 |
|------|--------|--------|----------|
| **Pipeline成功率** | **0%** | ≥90% | **+90%** |
| **目标测量率** | **77.8%** (7/9) | ≥88.8% (8/9) | **+11%** |
| **总执行时间** | **903秒** (15min) | ≤600秒 (10min) | **-33%** |
| **Plan阶段时间** | **556秒** | ≤200秒 | **-64%** |
| **CodeGen重试次数** | **7次**（全部失败） | ≤3次（成功） | **N/A** |

---

## 🧪 测试策略

### 单元测试（每个Bug修复后立即运行）
```bash
# 1. 测试ApprovalQueue去重机制
python -m pytest tests/test_approval_queue.py::test_submit_deduplication -v

# 2. 测试ToolRunner审批缓存
python -m pytest tests/test_tool_runner.py::test_approval_cache -v

# 3. 测试AgentLoop审批循环保护
python -m pytest tests/test_agent_loop.py::test_approval_loop_protection -v

# 4. 测试权限模式配置
python -m pytest tests/test_permission.py::test_high_autonomy_no_approval -v
```

### 集成测试（所有Bug修复后运行）
```bash
# 完整Pipeline测试（使用high_autonomy模式）
python -m src.main Profile GPU hardware \
  --pipeline \
  --target-spec config/target_spec.json \
  --mode high_autonomy \
  --max-turns 50 \
  --output kaggle_results/fixed_test_run

# 验证结果
python -m scripts.analyze_results kaggle_results/fixed_test_run
```

### 回归测试（针对之前修复的Bug #3/Bug #4）
构造以下场景验证：
```
场景A: compile_cuda(A) → execute_binary(A) → compile_cuda(B) → execute_binary(无参数)
预期: 应该auto-inject B而不是A

场景B: compile_cuda → execute_binary(空参数) → 系统应auto-inject binary_path
预期: P2 Harness正确注入最新编译的binary_path
```

---

## 📝 实施注意事项

### 关键风险点
1. **线程安全**: ApprovalQueue的修改必须保持线程安全（使用self._lock）
2. **向后兼容**: 修改不应破坏CONSERVATIVE模式的行为
3. **哈希碰撞**: 参数哈希使用MD5前12位，碰撞概率极低但非零（可接受）
4. **CUDA事件API**: bank_conflict修复需要确保cudaEvent在kernel执行前创建

### 代码规范
- 保持现有代码风格（遵循PEP 8）
- 所有新增代码必须有类型注解
- 添加适当的日志输出（使用现有的logger或print）
- 不添加注释（除非用户明确要求）

### 验证清单
- [ ] 每个Bug修复后单独测试通过
- [ ] Approval系统端到端测试通过（submit→approve→reuse）
- [ ] HIGH_AUTONOMY模式下compile_cuda无需审批即可执行
- [ ] Bank conflict测量不再返回"Invalid cycles"错误
- [ ] Plan阶段在≤20轮内完成
- [ ] 完整Pipeline运行成功率达到≥90%

---

## 🚀 执行步骤总结

### Phase 1: 致命缺陷修复（预计1.5小时）
1. ✏️ 修改 `permission.py` - Bug-004 (10min)
2. ✏️ 修改 `approval_queue.py` - Bug-001 (30min)
3. ✏️ 修改 `tool_runner.py` - Bug-002 (20min)
4. ✏️ 修改 `agent_loop.py` - Bug-003 (25min)
5. 🧪 运行单元测试验证Phase 1

### Phase 2: 功能修复（预计1小时）
6. ✏️ 修改 `bank_conflict.py` - Bug-006 + Bug-007 (60min)
7. 🧪 运行Bank Conflict单元测试

### Phase 3: 优化改进（预计1小时）
8. ✏️ 修改 `stage_executor.py` + `agent_loop.py` - Bug-005 (30min)
9. ✏️ 修改 `agent_loop.py` - Bug-008 (30min)
10. 🧪 运行集成测试验证全部修复

### Phase 4: 最终验证（预计30分钟）
11. 🏃 运行完整Kaggle Pipeline测试
12. 📊 对比前后结果，确认改善幅度
13. 📝 记录基线数据用于未来对比

**总计: 约3.5-4小时**
