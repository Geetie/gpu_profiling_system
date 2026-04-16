Verification 验收 Agent 系统性审查报告
一、核心问题
审查目标：Verification Agent 在 REJECT 后，AgentLoop 是否能正常运作，CodeGen 是否能根据 MetricAnalysis 的反馈进行持续迭代。

关键发现 🔴
当前架构存在严重的设计缺陷：

Verification REJECT 后无迭代机制
AgentLoop 在 REJECT 后会直接终止
CodeGen 无法接收 Verification 的反馈进行迭代
二、详细分析
2.1 Verification Agent 的 REJECT 逻辑
代码位置：verification.py:88-250

Python



def _rule_review(...) -> dict[str, Any]:    # ...检查逻辑...        # Final determination    status = SubAgentStatus.    SUCCESS if accepted else     SubAgentStatus.REJECTED        return {        "status": status,        "findings": findings,        "concerns": concerns,        "accepted": accepted,    }
行为：

✅ Verification 正确生成 REJECTED 状态
✅ 包含详细的 concerns 列表说明拒绝原因
✅ 通过 SubAgentResult 返回给 Pipeline
2.2 Pipeline 对 REJECT 的处理
代码位置：pipeline.py:109-124

Python



def run(self, target_spec: dict[str, Any]) -> SubAgentResult:    for step in self._stages:        # ...                result = self._executor.        execute(step, ctx)                if result.is_failed():            result = self.            _handle_failure            (step, result, ctx,             stage_duration)            if result.is_failed            () and not self.            _can_continue_with_p            artial(step, result)            :                return result                  # 🔴 直接终止                 Pipeline
问题：

🔴 Pipeline 遇到 REJECT 后立即终止
🔴 没有重试或回退机制
🔴 CodeGen 无法收到反馈
2.3 StageExecutor 的 REJECT 处理
代码位置：stage_executor.py:78-94

Python



def execute(self, step: Any, ctx: PipelineContext) -> SubAgentResult:    for attempt in range(1 +     step.retry_on_failure):        # ...        last_result = self.        _run_with_agent_loop        (step, message, ctx)                print(f"[StageExecutor]         Attempt status:         {last_result.status.        value}")        if last_result.        is_success():            break        if last_result.status         == SubAgentStatus.        REJECTED:            break  # 🔴 REJECT             时立即跳出重试循环
问题：

🔴 REJECT 被视为"不可重试"状态
🔴 即使 retry_on_failure > 0，REJECT 也会跳过重试
🔴 没有将 Verification 的 concerns 传递给 CodeGen
2.4 AgentLoop 的行为
代码位置：agent_loop.py:154-260

AgentLoop 在单个 Stage 内部负责：

✅ 多轮对话迭代（max_turns=20）
✅ Tool 调用和结果收集
✅ 失败模式追踪（InvariantTracker）
但关键问题：

🔴 AgentLoop 只在单个 Stage 内部运行
🔴 无法跨 Stage 传递反馈
🔴 Verification 的 REJECT 不会触发 CodeGen 的 AgentLoop 重新运行
三、架构缺陷总结
3.1 数据流断裂
PlainText



当前流程：CodeGen → MetricAnalysis → Verification(REJECT) → ❌ Pipeline 终止期望流程：CodeGen → MetricAnalysis → Verification(REJECT)     ↓    └──────→ CodeGen(接收反馈) →     重新生成 → MetricAnalysis →     Verification(ACCEPT)
3.2 缺失的机制
机制	当前状态	需要性
REJECT 后回退到 CodeGen	❌ 不存在	🔴 关键
concerns 反馈传递	❌ 不存在	🔴 关键
CodeGen 迭代计数器	❌ 不存在	🟡 重要
最大迭代次数限制	❌ 不存在	🟡 重要
迭代历史追踪	❌ 不存在	🟡 重要
四、修复建议
4.1 Pipeline 层修复（关键）
修改 pipeline.py：

Python



class Pipeline:    def run(self, target_spec:     dict[str, Any]) ->     SubAgentResult:        ctx = PipelineContext        (target_spec=target_spec        )                # 新增：迭代追踪        iteration_count = 0        max_iterations = 3                for step in self.        _stages:            while True:  # 新            增：Stage 迭代循环                result = self.                _executor.                execute(step,                 ctx)                                if result.                is_success():                    ctx.update                    (step.                    stage,                     result)                    break                                if result.                status ==                 SubAgentStatus.                REJECTED:                    # 新增：检查                    是否可以回退                    if self.                    _can_retry_w                    ith_feedback                    (step, ctx,                     iteration_co                    unt,                     max_iteratio                    ns):                        iteratio                        n_count                         += 1                        self.                        _inject_                        feedback                        (step,                         ctx,                         result)                        continue                          # 重试                        当前                         Stage                    else:                        return                         result                          # 无法重                        试，终止                                return result                  # 其他失败情况                return ctx.final_result
4.2 StageExecutor 层修复
修改 stage_executor.py：

Python



def execute(self, step: Any, ctx: PipelineContext) -> SubAgentResult:    # 新增：处理 REJECT 反馈    if ctx.prev_result and ctx.    prev_result.status ==     SubAgentStatus.REJECTED:        concerns = ctx.        prev_result.data.get        ("concerns", [])        message = self.        _build_retry_message        (step, ctx, concerns)    else:        message = self.        _build_collaboration_mes        sage(step, ctx)        return self.    _run_with_agent_loop(step,     message, ctx)
4.3 PipelineContext 增强
修改 pipeline_context.py：

Python



@dataclassclass PipelineContext:    prev_result:     SubAgentResult | None = None    prev_stage: PipelineStage |     None = None    code_gen_data: dict[str,     Any] | None = None    target_spec: dict[str, Any]     = field    (default_factory=dict)        # 新增：迭代追踪    iteration_count: int = 0    max_iterations: int = 3    rejection_history: list    [dict] = field    (default_factory=list)        def add_rejection(self,     stage: str, concerns: list    [str]) -> None:        self.rejection_history.        append({            "stage": stage,            "concerns":             concerns,            "iteration": self.            iteration_count,        })
4.4 Verification Agent 增强
修改 verification.py：

Python



def _review(...) -> dict[str, Any]:    review = self._llm_review    (...)        # 新增：明确标识是否可以修复    review["can_be_fixed"] =     self._assess_fixability    (review["concerns"])    review["suggested_fixes"] =     self._suggest_fixes(review    ["concerns"])        return review
五、当前架构的实际行为
5.1 成功路径
PlainText



CodeGen (SUCCESS)   → MetricAnalysis (SUCCESS)   → Verification (SUCCESS,   accepted=True)  → Pipeline 完成 ✅
5.2 失败路径（当前）
PlainText



CodeGen (SUCCESS)   → MetricAnalysis (SUCCESS)   → Verification (REJECTED,   concerns=["..."])  → Pipeline 终止 ❌  → CodeGen 无法收到反馈
5.3 期望的失败路径
PlainText



CodeGen (SUCCESS)   → MetricAnalysis (SUCCESS)   → Verification (REJECTED,   concerns=["..."],   suggested_fixes=["..."])  → Pipeline 回退到 CodeGen  → CodeGen (根据 feedback 修复)  → MetricAnalysis (重新分析)  → Verification (ACCEPT)  → Pipeline 完成 ✅
六、总结
6.1 核心问题
问题	严重性	影响
REJECT 后无迭代机制	🔴 严重	系统无法自我修复
AgentLoop 无法跨 Stage	🟡 中等	单 Stage 内可迭代，但无法触发上游
反馈数据流断裂	🔴 严重	CodeGen 不知道如何修复
6.2 修复优先级
P0（立即）：Pipeline 增加 REJECT 回退逻辑
P1（短期）：StageExecutor 支持反馈注入
P2（中期）：PipelineContext 增加迭代追踪
P3（长期）：Verification 提供可修复性评估
6.3 当前系统状态
结论：当前系统不具备Verification REJECT 后的迭代能力。一旦 Verification 拒绝，Pipeline 立即终止，CodeGen 无法接收反馈进行修复。

这是一个架构级缺陷，需要在 Pipeline 层进行结构性修复。