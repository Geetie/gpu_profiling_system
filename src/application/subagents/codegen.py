"""Code Generation Agent — writes CUDA micro-benchmark kernels.

Per spec.md P1/P5/P7 and PJ Requirement §1.7.4:
ALL CUDA C++ source code is generated exclusively by LLM.
No hardcoded templates, no skeleton fallbacks, no runtime code generation.
The agent receives design methodology from design_principles.py and writes
complete CUDA C++ source based on those principles.

Workflow:
1. Receives design methodology from prompt (see _build_system_prompt in subagent.py)
2. LLM writes complete CUDA C++ source implementing the design
3. compile_cuda tool compiles the source
4. execute_binary tool runs the compiled binary
5. Agent parses the numeric output and reports the measured value

Architecture Detection:
- Automatically detects GPU compute capability via cudaDeviceGetAttribute
- Passes correct -arch=sm_XX flag to nvcc for compilation
- Supports all NVIDIA GPU architectures (sm_35 to sm_90+)

GPUFeatureDB Integration (Phase 2 Enhancement):
------------------------------------------------
Added comprehensive DEBUG-level logging to track the complete integration chain:
- Module import success/failure
- Database initialization status
- GPU detection results (name, compute capability)
- Measurement parameter retrieval per target
- Context injection confirmation
- Graceful degradation on errors

This logging enables debugging of issues like:
- GPUFeatureDB not being called at all
- Silent failures in feature detection
- Incorrect parameter retrieval
- Missing context injection

COMPLIANCE NOTES:
- spec.md P1: Tool Definition Boundaries — No unregistered operations
- spec.md P5: Compile-time elimination — No runtime fallback to hardcoded code
- spec.md P7: Generation-Evaluation Separation — CodeGen only generates, does not evaluate
- PJ §1.7.4: Micro-benchmark validity — Proxy generates appropriate CUDA kernels
"""
from __future__ import annotations

import logging
from typing import Any

from src.application.context import ContextManager, Role
from src.domain.permission import PermissionMode
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    SubAgentResult,
    SubAgentStatus,
)
from src.domain.tool_contract import ToolRegistry
from src.infrastructure.probing.arch_detection import detect_gpu_arch
from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner

# Configure module-level logger for CodeGen agent and GPUFeatureDB integration
logger = logging.getLogger(__name__)


class CodeGenAgent(BaseSubAgent):
    """Generates, compiles, and executes CUDA micro-benchmark kernels.

    Uses LLM to generate CUDA source code from design principles.
    Per spec.md P1/P5/P7 compliance:
    - NO hardcoded templates or skeleton code
    - NO fallback to Python-generated CUDA source
    - LLM is the SOLE author of all CUDA C++ code
    - If LLM unavailable, raises RuntimeError (graceful failure)
    """

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        tool_registry: ToolRegistry | None = None,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 16000,
        sandbox: SandboxRunner | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.CODE_GEN,
            context_manager=context_manager or ContextManager(max_tokens=max_tokens),
            tool_registry=tool_registry or ToolRegistry(),
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )
        self._sandbox = sandbox or LocalSandbox(SandboxConfig())
        self._detected_arch: str | None = None

    def _process(self, message: CollaborationMessage) -> SubAgentResult:
        """Generate a CUDA micro-benchmark based on the task description."""
        task = message.payload.get("task", {})
        target = task.get("target", "unknown")
        category = task.get("category", "unknown")
        method = task.get("method", "custom micro-benchmark")

        # Bug fix: Detect GPU architecture before compilation
        # This ensures CodeGen knows the correct architecture to use
        detected_arch = self._detect_gpu_arch()
        print(f"[CodeGen] Detected GPU architecture: {detected_arch}")

        # Add architecture info to context so model knows the correct arch
        self.context_manager.add_entry(
            Role.SYSTEM,
            f"🔧 Detected GPU architecture: {detected_arch}\n"
            f"IMPORTANT: Use `-arch={detected_arch}` in compile_cuda flags.\n"
            f"NEVER use `-arch=sm_0` or `-arch=sm_50`.\n"
            f"Use the detected architecture {detected_arch} exactly.",
            token_count=50,
        )

        # GPUFeatureDB Integration (P0 Priority): Inject architecture-specific measurement parameters
        # This eliminates hardcoded sm_60 logic and auto-adapts to different GPUs
        #
        # Phase 2 Enhancement: Added comprehensive DEBUG-level logging for:
        # - Module import tracking
        # - Database initialization status
        # - GPU detection results (name, compute capability)
        # - Measurement parameter retrieval per target
        # - Context injection confirmation
        # - Graceful degradation on errors
        logger.debug("[CodeGen] 🔄 Attempting to initialize GPUFeatureDB...")

        try:
            from src.infrastructure.gpu_feature_db import GPUFeatureDB

            logger.debug("[CodeGen] ✅ GPUFeatureDB module imported successfully")
            logger.info("[CodeGen] 🔄 Initializing GPUFeatureDB instance...")

            gpu_db = GPUFeatureDB()
            logger.info("[CodeGen] ✅ GPUFeatureDB initialized successfully")

            logger.debug(f"[CodeGen] 🔍 Detecting GPU features for architecture: {detected_arch}")
            gpu_specs = gpu_db.detect_and_get_features()

            if gpu_specs:
                # Log successful GPU detection with full details
                logger.info(
                    f"[CodeGen] 📊 GPUFeatureDB detected: {gpu_specs.name} "
                    f"({gpu_specs.compute_capability})"
                )
                logger.debug(
                    f"[CodeGen] 📋 Full GPU specifications:\n"
                    f"  • Name: {gpu_specs.name}\n"
                    f"  • Compute Capability: {gpu_specs.compute_capability}\n"
                    f"  • Memory: {gpu_specs.memory_size_gb}GB {gpu_specs.memory_type}\n"
                    f"  • Memory Bandwidth: {gpu_specs.memory_bandwidth_gbps:.0f} GB/s\n"
                    f"  • SM Count: {gpu_specs.sm_count}\n"
                    f"  • L2 Cache: {gpu_specs.l2_cache_size_kb}KB\n"
                    f"  • Clock: {gpu_specs.base_clock_mhz}-{gpu_specs.boost_clock_mhz} MHz\n"
                    f"  • Shared Memory/Block: {gpu_specs.shared_memory_per_block_kb}KB\n"
                    f"  • Registers/Thread: {gpu_specs.register_count_per_thread}\n"
                    f"  • Warp Size: {gpu_specs.warp_size}\n"
                    f"  • Max Threads/SM: {gpu_specs.max_threads_per_sm}"
                )

                # Get target-specific optimal parameters
                logger.debug(
                    f"[CodeGen] 📏 Retrieving measurement params for target '{target}' "
                    f"on architecture '{detected_arch}'..."
                )
                measure_params = gpu_db.get_measurement_params(target, detected_arch)

                # Log retrieved measurement parameters
                logger.info(
                    f"[CodeGen] 📏 Measurement params for '{target}':\n"
                    f"  • Working set: {measure_params.get('working_set_mb', 'N/A')}MB\n"
                    f"  • Expected range: {measure_params.get('expected_range', 'N/A')}\n"
                    f"  • Method: {measure_params.get('method', 'N/A')}\n"
                    f"  • Notes: {measure_params.get('notes', 'N/A')}"
                )
                logger.debug(
                    f"[CodeGen] 🔧 Full measurement parameters dict: {measure_params}"
                )

                # Build comprehensive GPU context for LLM
                gpu_context_parts = [
                    f"📊 **GPU Feature Database** — Architecture-Specific Parameters\n",
                    f"Detected GPU: {gpu_specs.name} ({gpu_specs.compute_capability})\n",
                    f"Memory: {gpu_specs.memory_size_gb}GB {gpu_specs.memory_type}, "
                    f"{gpu_specs.memory_bandwidth_gbps:.0f} GB/s bandwidth\n",
                    f"SMs: {gpu_specs.sm_count}, L2 Cache: {gpu_specs.l2_cache_size_kb}KB, "
                    f"Clock: {gpu_specs.base_clock_mhz}-{gpu_specs.boost_clock_mhz} MHz\n",
                    f"\n📏 **Recommended Measurement Parameters for '{target}':**\n",
                ]

                # Add target-specific params
                if "working_set_mb" in measure_params:
                    gpu_context_parts.append(
                        f"  • Working set: {measure_params['working_set_mb']}MB "
                        f"(must exceed L2 cache)\n"
                    )
                    logger.debug(
                        f"[CodeGen] ➕ Added working_set_mb={measure_params['working_set_mb']}MB to context"
                    )
                if "expected_range" in measure_params:
                    gpu_context_parts.append(
                        f"  • Expected value range: {measure_params['expected_range']}\n"
                    )
                    logger.debug(
                        f"[CodeGen] ➕ Added expected_range={measure_params['expected_range']} to context"
                    )
                if "method" in measure_params:
                    gpu_context_parts.append(
                        f"  • Recommended method: {measure_params['method']}\n"
                    )
                    logger.debug(
                        f"[CodeGen] ➕ Added method={measure_params['method']} to context"
                    )
                if "notes" in measure_params:
                    gpu_context_parts.append(
                        f"  • Notes: {measure_params['notes']}\n"
                    )
                    logger.debug(
                        f"[CodeGen] ➕ Added notes to context"
                    )

                # Add general architecture guidance
                gpu_context_parts.extend([
                    f"\n⚠️ **Critical Constraints:**\n",
                    f"  • Max shared memory/block: {gpu_specs.shared_memory_per_block_kb}KB\n",
                    f"  • Max registers/thread: {gpu_specs.register_count_per_thread}\n",
                    f"  • Warp size: {gpu_specs.warp_size}, Max threads/SM: {gpu_specs.max_threads_per_sm}\n",
                    f"\n✅ Use these parameters to generate ACCURATE micro-benchmarks.\n",
                ])
                
                # T5 ENHANCEMENT: Add L2 cache-specific algorithm guidance
                if "l2_cache" in target.lower():
                    l2_guidance = (
                        f"\n🔧 **L2 CACHE MEASUREMENT - CRITICAL ALGORITHM CORRECTION:**\n\n"

                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "❌ WRONG APPROACHES (DO NOT DO THIS!)\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

                        "**Mistake #1: Linear Sweep with Large Range**\n"
                        f"  ❌ Testing 1-100MB with 1MB steps (22 test points)\n"
                        "  ❌ This is too coarse to detect the cliff accurately\n"
                        "  ❌ Takes too long (>60 seconds)\n\n"

                        "**Mistake #2: Returning Max Test Size as Answer**\n"
                        '  ❌ Outputting "l2_cache_size_mb: 100.0" when max tested was 100MB\n'
                        "  ❌ The cliff point should be DETECTED, not assumed!\n\n"

                        "**Mistake #3: Using cudaEvent* inside __global__ function**\n"
                        "  ❌ Causes compilation error: calling __host__ function from __global__\n"
                        "  ❌ Use clock64() for kernel-internal timing instead\n\n"

                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "✅ CORRECT ALGORITHM: Binary Search with Cliff Detection\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

                        f"**Step 1: Define FOCUSED search range around expected value**\n"
                        f"  Expected L2 for this GPU: ~{gpu_specs.l2_cache_size_kb/1024:.1f}MB\n"
                        "  Search range = [expected×0.25, expected×0.5, expected, expected×2, expected×4]\n"
                        f"  Example (P100 with ~4MB): [1, 2, 4, 8, 16] MB\n\n"

                        "**Step 2: Measure access time at each point**\n"
                        "```cuda\n"
                        "__global__ void measure_access_time(uint32_t* data, size_t size, uint64_t* cycles) {\n"
                        "    if (threadIdx.x != 0 || blockIdx.x != 0) return;\n"
                        "    volatile uint64_t start = clock64();\n"
                        "    // Pointer-chasing access pattern (stride=32 bytes to bypass L1)\n"
                        "    uint32_t idx = 0;\n"
                        "    #pragma unroll 1\n"
                        "    for (uint64_t i = 0; i < 500000; i++) {\n"
                        "        idx = data[idx % (size / sizeof(uint32_t))];\n"
                        "        if (idx >= size / sizeof(uint32_t)) idx = 0;\n"
                        "    }\n"
                        "    volatile uint64_t end = clock64();\n"
                        "    *cycles = (end - start) / 500000;  // Average cycles per access\n"
                        "}\n"
                        "```\n\n"

                        "**Step 3: Detect the CLIFF (where latency jumps >3x)**\n"
                        "```cpp\n"
                        "// In main():\n"
                        "float latencies[5];\n"
                        "size_t sizes_mb[] = {1, 2, 4, 8, 16};  // Focused search points\n"
                        "\n"
                        "for (int i = 0; i < 5; i++) {\n"
                        "    size_t array_size = (sizes_mb[i] * 1024 * 1024) / sizeof(uint32_t);\n"
                        "    // ... allocate and initialize array ...\n"
                        "    measure_access_time<<<1, 1>>>(d_data, array_size, d_cycles);\n"
                        "    cudaMemcpy(&latencies[i], d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);\n"
                        "}\n"
                        "\n"
                        "// Find cliff:\n"
                        "float detected_cliff = -1;\n"
                        "for (int i = 1; i < 5; i++) {\n"
                        "    float ratio = latencies[i] / latencies[i-1];\n"
                        "    printf(\"Size=%dMB Latency=%.1f Ratio=%.2f\\n\", sizes_mb[i], latencies[i], ratio);\n"
                        "    if (ratio > 3.0 && detected_cliff < 0) {\n"
                        "        detected_cliff = sizes_mb[i-1];  // Cliff is BEFORE the jump!\n"
                        "        break;\n"
                        "    }\n"
                        "}\n"
                        "```\n\n"

                        "**Step 4: Fine-tune with Binary Search (optional, for higher precision)**\n"
                        "```cpp\n"
                        "if (detected_cliff > 0) {\n"
                        "    float low = detected_cliff * 0.7f;\n"
                        "    float high = detected_cliff * 1.3f;\n"
                        "    while (high - low > 0.1f) {  // 0.1 MB precision\n"
                        "        float mid = (low + high) / 2.0f;\n"
                        "        // Measure latency at mid...\n"
                        "        float latency_at_mid = measure(mid);\n"
                        "        float latency_at_low = measure(low);\n"
                        "        if (latency_at_mid / latency_at_low > 2.0) {\n"
                        "            high = mid;  // Cliff is below mid\n"
                        "        } else {\n"
                        "            low = mid;   // Cliff is above mid\n"
                        "        }\n"
                        "    }\n"
                        "    detected_cliff = low;  // Final answer\n"
                        "}\n"
                        "```\n\n"

                        "⚠️ **CRITICAL OUTPUT FORMAT REQUIREMENT:**\n"
                        '  Print EXACTLY: "l2_cache_size_mb: X.X"\n'
                        "  Where X.X is the DETECTED CLIFF POINT in MB\n"
                        "  NOT the maximum tested size!\n"
                        "  Example correct output: \"l2_cache_size_mb: 4.0\" (for P100)\n"
                        "  Example wrong output: \"l2_cache_size_mb: 100.0\" (this will FAIL verification)\n\n"

                        f"📏 **Expected L2 Cache Sizes by GPU Architecture:**\n"
                        "  • Tesla K40/K80: ~1.5 MB\n"
                        "  • Tesla M40/M60/P100: ~4 MB\n"
                        "  • Tesla V100: ~6 MB (Volta)\n"
                        "  • Tesla T4: ~4 MB (Turing)\n"
                        "  • Tesla A100: ~40 MB (Ampere)\n"
                        "  • H100: ~50 MB (Hopper)\n"
                        f"  • Your GPU ({gpu_specs.name}): ~{gpu_specs.l2_cache_size_kb/1024:.1f}MB\n\n"

                        "💡 **Performance Target:**\n"
                        "  • Total execution time: <30 seconds\n"
                        "  • Number of test points: ≤10 (focused search)\n"
                        "  • Precision: ±0.5 MB (sufficient for L2 detection)\n"
                    )
                    gpu_context_parts.append(l2_guidance)

                    logger.info(
                        f"[CodeGen] 🔧 Added L2 cache CRITICAL algorithm correction with Binary Search "
                        f"({len(l2_guidance)} chars)"
                    )

                # P1-2 FIX: Clock measurement formula correction (prevent unit conversion errors)
                if "clock" in target.lower() or "boost" in target.lower() or "frequency" in target.lower():
                    clock_guidance = (
                        f"\n⏱️ **BOOST CLOCK MEASUREMENT - CRITICAL FORMULA CORRECTION:**\n\n"

                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "❌ COMMON MISTAKES THAT CAUSE 100x ERRORS!\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

                        "**Mistake #1: Confusing milliseconds and seconds**\n"
                        "  ❌ Using elapsed_ms directly without dividing by 1000\n"
                        "  ❌ Result: clock frequency appears 1000x too LOW\n"
                        '  Example wrong output: "actual_boost_clock_mhz: 10.0" (should be ~1400)\n'
                        "  ✅ Fix: elapsed_sec = elapsed_ms / 1000.0f\n\n"

                        "**Mistake #2: Forgetting to convert cycles to MHz**\n"
                        "  ❌ Using raw cycle count as the answer\n"
                        "  ❌ Result: value is in billions, not MHz\n"
                        "  ✅ Fix: freq_mhz = gpu_cycles / elapsed_sec / 1e6f\n\n"

                        "**Mistake #3: Using wall-clock time instead of GPU time**\n"
                        "  ❌ Measuring CPU time around kernel launch (includes overhead)\n"
                        "  ❌ Result: inaccurate due to launch latency, synchronization delays\n"
                        "  ✅ Fix: Use clock64() inside kernel + cudaEventElapsedTime() for total time\n\n"

                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "✅ CORRECT ALGORITHM & FORMULA\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

                        "**Method: clock64() + cudaEventElapsedTime()**\n\n"

                        "**Step 1: Kernel that measures GPU cycles**\n"
                        "```cuda\n"
                        "__global__ void measure_clock_kernel(volatile uint64_t* out_cycles) {\n"
                        "    // Ensure single-thread execution for accurate timing\n"
                        "    if (threadIdx.x != 0 || blockIdx.x != 0) return;\n\n"
                        "    volatile uint64_t start = clock64();\n\n"
                        "    // Do meaningful work to prevent compiler optimization\n"
                        "    float accumulator = 1.0f;\n"
                        "    #pragma unroll 32\n"
                        "    for (int i = 0; i < 100000; i++) {\n"
                        "        accumulator = accumulator * 1.000001f + 0.000001f;\n"
                        "        __syncthreads();  // Prevent loop unrolling optimization\n"
                        "    }\n\n"
                        "    volatile uint64_t end = clock64();\n"
                        "    *out_cycles = (end - start);  // Total GPU clock cycles\n"
                        "}\n"
                        "```\n\n"

                        "**Step 2: Host code to calculate frequency**\n"
                        "```cpp\n"
                        "// In main():\n"
                        "uint64_t* d_cycles;\n"
                        "cudaMalloc(&d_cycles, sizeof(uint64_t));\n\n"
                        "// Measure execution time using CUDA events\n"
                        "cudaEvent_t evt_start, evt_stop;\n"
                        "cudaEventCreate(&evt_start);\n"
                        "cudaEventCreate(&evt_stop);\n\n"
                        "cudaEventRecord(evt_start);  // Start timing\n"
                        "measure_clock_kernel<<<1, 1>>>(d_cycles);  // Launch kernel\n"
                        "cudaDeviceSynchronize();  // Wait for completion\n"
                        "cudaEventRecord(evt_stop);   // Stop timing\n"
                        "cudaEventSynchronize(evt_stop);  // Ensure event recorded\n\n"
                        "// Get elapsed time in MILLISECONDS\n"
                        "float elapsed_ms = 0.0f;\n"
                        "cudaEventElapsedTime(&elapsed_ms, evt_start, evt_stop);\n\n"
                        "// Get GPU cycle count from device\n"
                        "uint64_t host_cycles = 0;\n"
                        "cudaMemcpy(&host_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);\n\n"
                        "// ⚠️ CRITICAL: Convert to correct units\n"
                        "float elapsed_sec = elapsed_ms / 1000.0f;  // ms → seconds\n"
                        "float clock_freq_mhz = (float)host_cycles / elapsed_sec / 1e6f;  // Hz → MHz\n\n"
                        "// Output result\n"
                        'printf("actual_boost_clock_mhz: %.1f\\n", clock_freq_mhz);\n'
                        "\n"
                        "// Cleanup\n"
                        "cudaFree(d_cycles);\n"
                        "cudaEventDestroy(evt_start);\n"
                        "cudaEventDestroy(evt_stop);\n"
                        "```\n\n"

                        "**Formula Summary:**\n"
                        "```\n"
                        "freq_MHz = (gpu_clock_cycles) / (elapsed_seconds) / 1,000,000\n"
                        "\n"
                        "Where:\n"
                        "  • gpu_clock_cycles = from clock64() in kernel\n"
                        "  • elapsed_seconds = from cudaEventElapsedTime() / 1000\n"
                        "  • 1e6 = converts Hz to MHz\n"
                        "```\n\n"

                        f"📏 **Expected Boost Clock Ranges by GPU Architecture:**\n"
                        "  • Tesla K40/K80: 745-875 MHz\n"
                        "  • Tesla M40/M60/P100: 1250-1480 MHz (base: 1329)\n"
                        "  • Tesla V100: 1380-1530 MHz (Volta)\n"
                        "  • Tesla T4: 1110-1590 MHz (Turing)\n"
                        "  • Tesla A100: 1215-1410 MHz (Ampere)\n"
                        "  • H100: 1830-2500 MHz (Hopper)\n"
                        f"  • Your GPU ({gpu_specs.name}): {gpu_specs.boost_clock_mhz} MHz (expected range)\n\n"

                        "⚠️ **SANITY CHECK - If your output is outside this range, CHECK YOUR FORMULA!**\n\n"
                        "**Common error symptoms:**\n"
                        "  • <100 MHz → You forgot to divide by 1000 (ms→s) or by 1e6 (Hz→MHz)\n"
                        "  • >5000 MHz → You divided too many times or used wrong units\n"
                        "  • Exactly 10.0 MHz → Classic unit confusion (ms vs s AND missing 1e6 factor)\n\n"

                        "**Debug tips:**\n"
                        '  printf("DEBUG: cycles=%llu, elapsed_ms=%.2f, elapsed_sec=%.4f\\n",\n'
                        "         host_cycles, elapsed_ms, elapsed_sec);\n"
                        '  printf("DEBUG: raw_freq_hz=%.1f, final_freq_mhz=%.1f\\n",\n'
                        "         host_cycles/elapsed_sec, clock_freq_mhz);\n"
                    )
                    gpu_context_parts.append(clock_guidance)

                    logger.info(
                        f"[CodeGen] 🔧 Added Clock measurement formula correction "
                        f"({len(clock_guidance)} chars)"
                    )

                gpu_context = "".join(gpu_context_parts)

                # Inject into context manager
                logger.debug(
                    f"[CodeGen] 💉 Injecting GPUFeatureDB context ({len(gpu_context)} chars) "
                    f"into context_manager..."
                )
                self.context_manager.add_entry(
                    Role.SYSTEM,
                    gpu_context,
                    token_count=150,  # Generous token budget for rich context
                )

                logger.info(
                    f"[CodeGen] ✅ GPUFeatureDB context injected successfully\n"
                    f"  • Target: {target}\n"
                    f"  • GPU: {gpu_specs.name}\n"
                    f"  • Context length: {len(gpu_context)} chars\n"
                    f"  • Token budget: 150 tokens"
                )
            else:
                # GPU detection failed but no exception raised
                logger.warning(
                    "[CodeGen] ⚠️ GPUFeatureDB.detect_and_get_features() returned None\n"
                    "  This usually means:\n"
                    "  1. No NVIDIA GPU detected in the system\n"
                    "  2. CUDA runtime not available\n"
                    "  3. GPU driver issues\n"
                    "  → Continuing with fallback defaults (no architecture-specific params)"
                )
                print(f"[GPUFeatureDB] ⚠️ Could not detect GPU specs, using fallback defaults")

        except ImportError as e:
            # Module import failed — GPUFeatureDB not available
            logger.error(
                f"[CodeGen] ❌ FAILED to import GPUFeatureDB module: {e}\n"
                f"  Error type: ImportError\n"
                f"  This means src/infrastructure/gpu_feature_db.py does not exist or has syntax errors.\n"
                f"  → System will continue WITHOUT GPUFeatureDB data (graceful degradation)"
            )
            print(f"[GPUFeatureDB] ❌ Import error (non-fatal): {e}")

        except Exception as e:
            # Any other error during GPUFeatureDB integration
            logger.warning(
                f"[CodeGen] ⚠️ GPUFeatureDB integration encountered an error: {e}\n"
                f"  Exception type: {type(e).__name__}\n"
                f"  Error message: {str(e)[:200]}\n"
                f"  → This is NON-FATAL: system will continue without GPUFeatureDB data\n"
                f"  → Micro-benchmarks will use generic/fallback parameters instead"
            )
            # Log full traceback at DEBUG level for detailed debugging
            logger.debug(
                f"[CodeGen] 🔍 GPUFeatureDB error details (full traceback):\n",
                exc_info=True
            )
            print(f"[GPUFeatureDB] ❌ Integration error (non-fatal): {e}")
            # Non-fatal: continue without GPUFeatureDB data

        self.context_manager.add_entry(
            Role.USER,
            f"Generate a CUDA micro-benchmark for target '{target}' "
            f"(category: {category}, method: {method})",
            token_count=20,
        )

        max_compile_retries = 3
        compile_retry = 0
        source_code = None
        compile_result = None

        while compile_retry < max_compile_retries:
            try:
                source_code = self._generate_kernel(target, category, method)
            except RuntimeError as e:
                return SubAgentResult(
                    agent_role=self.role,
                    status=SubAgentStatus.FAILED,
                    error=str(e),
                )

            compile_result = self._compile(source_code, target=target)
            if compile_result.success:
                break

            compile_retry += 1
            if compile_retry < max_compile_retries:
                self.context_manager.add_entry(
                    Role.SYSTEM,
                    f"⚠️ Compilation failed (attempt {compile_retry}/{max_compile_retries}). "
                    f"Please fix the code.\nError:\n{compile_result.stderr[:1000]}",
                    token_count=100,
                )

        if not compile_result or not compile_result.success:
            return SubAgentResult(
                agent_role=self.role,
                status=SubAgentStatus.FAILED,
                error=f"Compilation failed after {max_compile_retries} attempts: {compile_result.stderr if compile_result else 'No result'}",
            )

        exec_result = self._execute(compile_result.artifacts, target=target)
        if not exec_result.success:
            return SubAgentResult(
                agent_role=self.role,
                status=SubAgentStatus.FAILED,
                error=f"Execution failed: {exec_result.stderr}",
            )

        binary_path = compile_result.artifacts.get("binary", "")
        source_path = compile_result.artifacts.get("source", "./source.cu")

        result = SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "target": target,
                "category": category,
                "raw_output": exec_result.stdout,
                "compile_output": compile_result.stdout,
                "binary_path": binary_path,
                "source_path": source_path,
                "detected_arch": self._detected_arch,
                "tool_results": [
                    {
                        "tool": "compile_cuda",
                        "status": "success",
                        "success": True,
                        "binary_path": binary_path,
                        "source_path": source_path,
                        "output": compile_result.stdout,
                        "arch": self._detected_arch,
                    },
                    {
                        "tool": "execute_binary",
                        "status": "success",
                        "stdout": exec_result.stdout,
                        "return_code": exec_result.return_code,
                    }
                ],
            },
            artifacts=list(compile_result.artifacts.values()),
        )

        return result

    def _generate_kernel(self, target: str, category: str, method: str) -> str:
        """Generate CUDA kernel source code exclusively via LLM.

        Per spec.md compliance requirements:
        - P1 (Tool Definition Boundaries): All operations must use pre-registered tools
        - P5 (Compile-time elimination): No runtime fallback to hardcoded code
        - P7 (Generation-Evaluation Separation): Single agent must NOT both generate AND evaluate
        - PJ Requirement §1.7.4: Micro-benchmark validity — proxy MUST generate appropriate CUDA kernels

        Design principles from design_principles.py are injected into the LLM context
        as methodology guidance. The LLM writes complete CUDA C++ source code based on these
        principles. NO hardcoded templates exist in this code path.

        If LLM is unavailable, FAILS GRACEFULLY with RuntimeError — no silent fallback.
        """
        from src.domain.design_principles import get_design_principle

        principle = get_design_principle(target)

        if self._model_caller is not None:
            messages = self.context_manager.to_messages()
            try:
                result = self._model_caller(messages)
                self._persister.log_entry(
                    action="llm_code_generation_success",
                    details={
                        "target": target,
                        "source_length": len(result),
                        "generation_method": "llm",
                        "principle_used": True,
                    },
                )
                return result
            except Exception as e:
                self._persister.log_entry(
                    action="llm_call_failed",
                    details={"error": str(e), "target": target, "principle_length": len(principle)},
                )
                raise RuntimeError(
                    f"LLM code generation failed for target '{target}': {e}. "
                    f"Per spec.md P1/P5/P7, CodeGen cannot fall back to hardcoded CUDA code."
                ) from e

        self._persister.log_entry(
            action="no_llm_configured",
            details={
                "target": target,
                "category": category,
                "method": method,
                "error": "No model_caller configured",
            },
        )
        raise RuntimeError(
            f"No LLM configured for CodeGen agent. "
            f"Per spec.md P1 (Tool Definition Boundaries), P5 (Compile-time elimination), "
            f"P7 (Generation-Evaluation Separation), and PJ §1.7.4 (Micro-benchmark validity), "
            f"ALL CUDA C++ source code must be generated by LLM. "
            f"No hardcoded fallback is permitted. "
            f"Target: {target}, Category: {category}, Method: {method}"
        )

    def _detect_gpu_arch(self) -> str:
        """Detect GPU compute capability for correct nvcc -arch flag.

        Delegates to the unified arch_detection module for consistent behavior
        across all probing components.

        Returns:
            Architecture string like 'sm_60', 'sm_80', etc.
        """
        if self._detected_arch:
            return self._detected_arch

        arch = detect_gpu_arch(self._sandbox)
        self._detected_arch = arch

        self._persister.log_entry(
            action="arch_detection",
            details={"method": "unified_detection", "arch": arch},
        )
        return arch

    def _compile(self, source_code: str, target: str = "unknown") -> Any:
        """Compile CUDA source code in the sandbox with correct architecture.

        Automatically detects GPU architecture and passes -arch=sm_XX to nvcc.
        This fixes the compilation error on Tesla P100 (sm_60) and other GPUs.

        Uses fixed 'benchmark' as binary name to match compile_cuda_handler output.
        This ensures _already_executed_binary position checking works correctly
        across both Pipeline and non-Pipeline modes.
        """
        arch = self._detect_gpu_arch()
        # P0 FIX: Use target-specific binary name to prevent overwriting
        safe_target = str(target).replace(" ", "_").replace("-", "_").lower()
        binary_name = f"benchmark_{safe_target}" if target and target != "unknown" else "benchmark"

        import os
        source_dir = os.path.join(self._sandbox.sandbox_root, "src")
        binary_dir = os.path.join(self._sandbox.sandbox_root, "bin")
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(binary_dir, exist_ok=True)

        self._persister.log_entry(
            action="compile_attempt",
            details={
                "source_length": len(source_code),
                "command": "nvcc",
                "arch": arch,
                "binary_name": binary_name,
                "target": target,
                "source_dir": source_dir,
                "binary_dir": binary_dir,
            },
        )

        source_path = os.path.join(source_dir, "source.cu")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source_code)

        result = self._sandbox.run(
            source_code=None,
            command="nvcc",
            args=["-o", os.path.join(binary_dir, binary_name), "source.cu", f"-arch={arch}", "-O3", "-Wno-deprecated-gpu-targets"],
            work_dir=source_dir,
        )

        if result.success:
            result.artifacts["source"] = source_path
            result.artifacts["binary"] = os.path.join(binary_dir, binary_name)

        self._persister.log_entry(
            action="compile_result",
            details={
                "success": result.success,
                "arch": arch,
                "binary_name": binary_name,
                "artifacts": list(result.artifacts.keys()) if hasattr(result, 'artifacts') else [],
                "stderr": result.stderr[:500] if result.stderr else "",
            },
        )
        return result

    def _execute(self, artifacts: dict, target: str = "unknown") -> Any:
        """Execute the compiled binary in the sandbox.

        Automatically detects the compiled binary file in the sandbox directory.
        Prioritizes target-specific binary names to avoid multi-target conflicts.
        """
        import os

        source_path = artifacts.get("source", "./source.cu")
        binary_dir = source_path.rsplit("/", 1)[0] if "/" in source_path else "."

        safe_target = target.replace(" ", "_").replace("-", "_").replace(".", "_")
        target_binary = f"benchmark_{safe_target}"

        possible_binary_names = [
            target_binary,
            "benchmark",
            "unknown_benchmark",
            "gpu_benchmark",
            "cuda_benchmark",
        ]

        binary_files = []
        try:
            if os.path.exists(binary_dir):
                for filename in os.listdir(binary_dir):
                    file_path = os.path.join(binary_dir, filename)
                    if os.path.isfile(file_path) and os.access(file_path, os.X_OK):
                        binary_files.append(filename)
        except Exception as e:
            self._persister.log_entry(
                action="execute_error",
                details={"error": f"Failed to list directory: {e}"},
            )

        all_binary_names = possible_binary_names + binary_files
        all_binary_names = list(set(all_binary_names))

        self._persister.log_entry(
            action="execute_attempt",
            details={
                "possible_binaries": all_binary_names,
                "work_dir": binary_dir,
            },
        )

        for binary_name in all_binary_names:
            result = self._sandbox.run(
                command=f"./{binary_name}",
                args=[],
                work_dir=binary_dir,
            )

            if result.success:
                self._persister.log_entry(
                    action="execute_result",
                    details={
                        "success": True,
                        "binary": binary_name,
                        "return_code": result.return_code,
                    },
                )
                return result
            else:
                self._persister.log_entry(
                    action="execute_attempt_failed",
                    details={
                        "binary": binary_name,
                        "error": result.stderr,
                        "return_code": result.return_code,
                    },
                )

        error_msg = f"No executable binary found in {binary_dir}. Tried: {all_binary_names}"
        self._persister.log_entry(
            action="execute_error",
            details={"error": error_msg},
        )
        return type('obj', (object,), {
            'stdout': '',
            'stderr': error_msg,
            'return_code': -1,
            'success': False
        })()
