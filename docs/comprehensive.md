# **Comprehensive Agent System Architecture Analysis**

## **1. Data Transmission Protocols, Formats, and Standards**

### **1.1 Core Data Structures**

#### **SubAgentResult - Primary Inter-Agent Communication Object**

```
@dataclass
class SubAgentResult:
    agent_role: AgentRole          # Agent identity (PLANNER, CODE_GEN, etc.)
    status: SubAgentStatus         # PENDING, SUCCESS, FAILED, REJECTED
    data: dict[str, Any]           # Stage-specific output data
    artifacts: list[str]           # File paths, binary locations
    error: str | None              # Error message if failed
    context_fingerprint: str       # SHA-256 hash for P7 audit trail
    metadata: dict[str, Any]       # Timing, target info, etc.
```

**Key Design Principle**: Agents communicate **ONLY** through `SubAgentResult` objects — no shared mutable state.

#### **CollaborationMessage - Inter-Agent Request/Response**

```
@dataclass
class CollaborationMessage:
    sender: AgentRole
    receiver: AgentRole
    message_type: str
    payload: dict[str, Any]
    timestamp: str
```

#### **PipelineContext - Mutable State Accumulator**

```
@dataclass
class PipelineContext:
    prev_result: SubAgentResult | None
    prev_stage: PipelineStage | None
    code_gen_data: dict[str, Any] | None
    target_spec: dict[str, Any]
    conversation_history: list[dict[str, str]]
    key_measurements: dict[str, Any]      # L1: High-priority memory
    binary_paths: list[str]
    stage_summaries: dict[str, str]       # L2: Medium-priority memory
    error_patterns: list[str]
```

### **1.2 Data Flow Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE ORCHESTRATOR                     │
│  Plan → CodeGen → MetricAnalysis → Verification             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PipelineContext (Mutable State)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ L0 Permanent│  │ L1 High     │  │ L2 Medium           │ │
│  │ Architecture│  │ Measurements│  │ Stage summaries     │ │
│  │ Target spec │  │ Binary paths│  │ Error patterns      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              AgentLoop (Per-Stage Execution)                 │
│  ContextManager → ModelCaller → ToolParser → ToolRunner     │
└─────────────────────────────────────────────────────────────┘
```

***

## **2. Data Flow Patterns**

### **2.1 Request/Response Cycles**

#### **Stage-to-Stage Flow**

1. **Plan Stage** → outputs `SubAgentResult` with `data.tasks` (list of targets)
2. **CodeGen Stage** ← receives Plan result via `PipelineContext.prev_result`
3. **CodeGen Stage** → outputs `SubAgentResult` with `data.measurements` and `data.tool_results`
4. **MetricAnalysis Stage** ← receives CodeGen data via `payload["codegen_data"]`
5. **MetricAnalysis Stage** → outputs `SubAgentResult` with `data.parsed_metrics` and `data.measurements`
6. **Verification Stage** ← receives all previous data via `PipelineContext`

#### **AgentLoop Inner Cycle**

```
Turn 1: ContextManager (system + user prompts) → ModelCaller → ToolCall
Turn 2: ContextManager (add tool result) → ModelCaller → ToolCall
Turn 3: ContextManager (add tool result) → ModelCaller → Completion
```

### **2.2 Event-Driven Communications**

#### **EventBus Pattern**

```
class EventBus:
    def publish(self, event: LoopEvent) -> None
    def subscribe(self, kind: EventKind, handler: Callable) -> None
```

**Event Types**:

- `TURN_START` / `TURN_END`
- `TOOL_CALL` / `TOOL_RESULT`
- `COMPLETION_DETECTED`
- `STALL_RECOVERED`

### **2.3 Asynchronous Data Exchange**

#### **State Persister (P6)**

All operations are logged asynchronously:

```
class StatePersister:
    def log_tool_execution(self, tool_name, inputs, status) -> None
    def log_entry(self, action, details, result_data=None) -> None
```

***

## **3. Data Validation, Transformation, and Enrichment**

### **3.1 Schema Validation Pipeline**

```
Tool Input → SchemaValidator.validate(input_schema) → Coerced Data
Tool Output → SchemaValidator.validate(output_schema) → Coerced Data
```

**Validation Rules**:

- P2 (Fail-Closed): Unregistered tools raise `KeyError`
- Type coercion: strings → ints/floats where possible
- Required field checking

### **3.2 Context Transformation**

#### **ContextManager Compression Strategy**

```
Phase 1: Remove DISPOSABLE entries (old guidance, short responses)
Phase 2: Summarize LOW priority entries (Control Plane, long responses)
Phase 3: Summarize MEDIUM priority entries (error messages)
Phase 4: Remove oldest MEDIUM entries if still over budget
```

#### **Priority Classification**

**Priority**

**Content Type**

**Action**

PERMANENT

Architecture info, target spec

Never removed

HIGH

Successful tool outputs, measurements

Preserved

MEDIUM

Error messages, LLM responses

Summarized

LOW

Control Plane, design principles

Aggressively compressed

DISPOSABLE

Old guidance, short responses

Removed first

### **3.3 Data Enrichment**

#### **PipelineContext.bubble\_codegen\_data()**

Propagates CodeGen measurements into downstream results:

```
def bubble_codegen_data(self, result: SubAgentResult) -> SubAgentResult:
    if "measurements" in self.code_gen_data:
        existing = result.data.get("measurements", {})
        for k, v in self.code_gen_data["measurements"].items():
            if k not in existing:
                existing[k] = v
        result.data["measurements"] = existing
    return result
```

***

## **4. Security Measures**

### **4.1 P7 (Generation/Evaluation Separation)**

```
class P7ViolationError(Exception):
    """Raised when generation and verification contexts are improperly shared."""
​
def execute(self, message: CollaborationMessage) -> SubAgentResult:
    if self.role == AgentRole.VERIFICATION:
        existing = self.context_manager.get_entries()
        if len(existing) > 0:
            raise P7ViolationError("Verification agent context must be empty")
```

### **4.2 Permission System**

```
PermissionMode.DEFAULT → All tools require approval
PermissionMode.HIGH_AUTONOMY → Auto-approve all tools
PermissionMode.LOCKDOWN → All tools blocked
```

### **4.3 Approval Queue**

```
class ApprovalQueue:
    def submit(self, tool_name, arguments, permissions, mode) -> ApprovalRequest
    def get_request(self, request_id) -> ApprovalRequest | None
```

**Approval Status**: PENDING → APPROVED / REJECTED / AUTO\_REJECTED

### **4.4 Context Fingerprinting**

```
def compute_fingerprint(self, context_manager: ContextManager) -> str:
    entries = context_manager.get_entries()
    content = "|".join(f"{e.role.value}:{e.content}" for e in entries)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
```

***

## **5. Error Handling, Retry Logic, and Fault Tolerance**

### **5.1 Stage-Level Retry**

```
for attempt in range(1 + step.retry_on_failure):
    last_result = self._run_with_agent_loop(step, message, ctx)
    if last_result.is_success():
        break
    if last_result.status == SubAgentStatus.REJECTED:
        break
```

### **5.2 AgentLoop Fault Tolerance**

#### **Stall Detection (C-01 FIX)**

```
if self.loop_state.consecutive_no_tool_calls >= 3:
    # Trigger stall recovery
    self.loop_state.stall_recovery_triggered = True
    self._inject_recovery_guidance()
```

#### **Compile Attempt Limiting (P0 FIX #3)**

```
compile_attempts: dict[str, int] = field(default_factory=dict)
MAX_COMPILE_ATTEMPTS_PER_TARGET: int = 3
```

#### **Time Budget Control (T5 FIX)**

```
MAX_TARGET_TIME_BUDGET = 240.0  # seconds per target
MAX_CODE_GEN_TOTAL = 600.0      # total CodeGen stage limit
GLOBAL_HARD_TIMEOUT = 1500.0    # 25 minutes absolute maximum
```

### **5.3 Circuit Breaker**

```
class CircuitBreaker:
    def record_failure(self, stage: str, error: str) -> None
    def should_trip(self) -> bool
    def get_state(self) -> CircuitState  # CLOSED, OPEN, HALF_OPEN
```

***

## **6. Performance Characteristics**

### **6.1 Latency Analysis**

**Operation**

**Typical Latency**

**Bottleneck**

LLM API Call

10-30s

Network I/O

CUDA Compilation

5-15s

CPU/GPU

Binary Execution

1-5s

GPU

NCU Profiling

30-60s

GPU/Tool

Context Compression

<100ms

CPU

### **6.2 Throughput**

- **Max Turns Per Stage**: 15-50 (configurable)
- **Concurrent Targets**: 1 (sequential execution)
- **LLM Requests Per Minute**: \~2-4 (limited by API latency)

### **6.3 Bandwidth Utilization**

- **Context Size**: 16,000 tokens max (configurable)
- **Tool Result Size**: \~1-5KB per execution
- **Log File Growth**: \~10-50KB per stage

***

## **7. Bottlenecks, Inefficiencies, and Points of Failure**

### **7.1 Critical Bottlenecks**

#### **Bottleneck 1: Sequential Target Processing**

```
Current: Process targets one-by-one (A → B → C → ...)
Problem: 8 targets × 240s = 1920s (32 minutes) > Global timeout
Solution: Parallel target compilation, batch NCU profiling
```

#### **Bottleneck 2: LLM API Latency**

```
Current: Synchronous API calls, 10-30s per turn
Problem: 50 turns × 20s = 1000s (16.7 minutes) just for API calls
Solution: Async API calls, response streaming, caching
```

#### **Bottleneck 3: Context Window Limit**

```
Current: 16K tokens, aggressive compression
Problem: Loss of critical information during compression
Solution: Hierarchical context, external memory (RAG)
```

### **7.2 Data Transmission Issues**

#### **Issue 1: Missing Measurements Field**

```
# MetricAnalysis returns:
data = {"parsed_metrics": metrics}  # ❌ Missing "measurements"
​
# StageExecutor expects:
data = {"measurements": metrics}    # ✅ Correct field name
```

#### **Issue 2: Incomplete Tool Support**

```
# Prompt tells LLM to use read_file:
"Use read_file to load measurement output files"
​
# But _call_tool only implements run_ncu:
if tool_name == "run_ncu": ...      # ❌ No read_file handler
```

#### **Issue 3: NCU Metric Name Mismatch**

```
# Agent requests:
"dram__throughput"                   # ❌ Simplified name
​
# NCU expects:
"dram__throughput.avg.pct_of_peak_sustained_elapsed"  # ✅ Full name
```

### **7.3 Points of Failure**

**Failure Point**

**Impact**

**Mitigation**

LLM API timeout

Stage failure

Retry with exponential backoff

CUDA compilation error

Target skip

Fallback to next target

NCU permission denied

MetricAnalysis failure

Hardware probes fallback

Context overflow

Information loss

Priority-based compression

Global timeout

Pipeline termination

Per-target time budgets

***

## **8. Recommendations**

### **8.1 Immediate Fixes**

1. **Fix measurements field propagation** - Ensure all stages return `measurements` key
2. **Implement read\_file tool handler** - Support file reading in MetricAnalysis
3. **Fix NCU metric names** - Use full metric names with `.avg.pct_of_peak_sustained_elapsed`
4. **Add compile\_attempts reset** - Reset counter when switching targets

### **8.2 Architecture Improvements**

1. **Parallel Target Processing** - Compile multiple targets concurrently
2. **Async LLM Calls** - Non-blocking API requests with callback handling
3. **Hierarchical Context** - Separate permanent, session, and turn-level context
4. **Result Caching** - Cache successful compilations and measurements
5. **Streaming Responses** - Process LLM output as it arrives

### **8.3 Monitoring Enhancements**

1. **Latency Metrics** - Track per-stage, per-turn timing
2. **Token Usage** - Monitor context compression effectiveness
3. **Error Patterns** - Classify and alert on recurring failures
4. **Success Rates** - Track target completion rates by type

***

## **Summary**

The Agent system uses a well-structured pipeline architecture with clear separation of concerns. However, several critical data transmission issues prevent successful operation:

1. **Field Name Mismatch**: `parsed_metrics` vs `measurements`
2. **Missing Tool Implementation**: `read_file` documented but not implemented
3. **Incorrect NCU Metric Names**: Simplified names that NCU doesn't recognize
4. **State Management Bugs**: `compile_attempts` not reset on target switch

These issues cause the MetricAnalysis stage to fail repeatedly, resulting in only 0-1 out of 8 targets being successfully measured.

I have completed the comprehensive analysis of the Agent system architecture, focusing on data transmission mechanisms between Agents. The analysis identified critical issues in the data flow that are preventing successful test completion.

Key findings include:

1. **Data Transmission Protocol**: Agents communicate exclusively through `SubAgentResult` objects with a structured format including `agent_role`, `status`, `data`, `artifacts`, `error`, and `context_fingerprint`
2. **Data Flow Patterns**: The pipeline follows Plan → CodeGen → MetricAnalysis → Verification with a mutable `PipelineContext` maintaining state between stages
3. **Critical Bugs Found**:
   - Field name mismatch: `parsed_metrics` vs `measurements`
   - Missing `read_file` tool implementation despite being documented in prompts
   - Incorrect NCU metric names causing profiling failures
   - `compile_attempts` counter not reset when switching targets
4. **Security Measures**: P7 separation (generation/evaluation isolation), permission system with approval queue, and context fingerprinting for audit trails
5. **Performance Bottlenecks**: Sequential target processing, synchronous LLM API calls, and context window limitations

The analysis provides detailed technical documentation of the architecture with specific recommendations for fixing the identified issues to achieve successful test completion.
