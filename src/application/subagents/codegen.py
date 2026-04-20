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
                        f"\n🔧 **L2 CACHE MEASUREMENT - SPECIAL ALGORITHM GUIDANCE:**\n\n"
                        f"The L2 cache size is tricky to measure accurately. "
                        f"Use this PROVEN algorithm:\n\n"
                        f"**Method: Binary Search with Cliff Detection**\n"
                        f"1. Create an array of size N (start with N=16MB)\n"
                        f"2. Access every K-th element (K=16 or 32 bytes)\n"
                        f"3. Measure average access time (use clock64() for GPU timing)\n"
                        f"4. Double the array size and repeat (16→32→64→...→256MB)\n"
                        f"5. Plot access time vs array size\n"
                        f"6. Find the 'cliff' where time jumps 3-5x → that's L2 size!\n\n"

                        f"🚨 **CRITICAL: CUDA HOST/DEVICE FUNCTION RULES:**\n"
                        f"❌ **FORBIDDEN in __global__ or __device__ functions:**\n"
                        f"   • cudaEventCreate() / cudaEventDestroy() - HOST ONLY!\n"
                        f"   • cudaEventRecord() / cudaEventSynchronize() - HOST ONLY!\n"
                        f"   • cudaEventElapsedTime() - HOST ONLY!\n"
                        f"   • cudaMalloc() / cudaFree() - HOST ONLY!\n"
                        f"   • cudaMemcpy() - HOST ONLY!\n"
                        f"   • printf() - HOST ONLY (use device printf sparingly)\n\n"
                        f"✅ **ALLOWED in __global__ or __device__ functions:**\n"
                        f"   • clock64() - GPU cycle counter for timing\n"
                        f"   • __threadfence() - memory fence\n"
                        f"   • __syncthreads() - thread synchronization\n"
                        f"   • atomicAdd() / atomicExch() - atomic operations\n"
                        f"   • Standard C/C++ operators and math functions\n\n"
                        f"**Correct Architecture Pattern:**\n"
                        f"```cuda\n"
                        f"__global__ void kernel(...) {{\n"
                        f"    // Use clock64() for timing INSIDE kernel\n"
                        f"    uint64_t start = clock64();\n"
                        f"    // ... do work ...\n"
                        f"    uint64_t end = clock64();\n"
                        f"    *result = end - start;\n"
                        f"}}\n\n"
                        f"int main() {{\n"
                        f"    // Use cudaEvent* for timing OUTSIDE kernel (in main/host code)\n"
                        f"    cudaEvent_t start, stop;\n"
                        f"    cudaEventCreate(&start);\n"
                        f"    cudaEventCreate(&stop);\n"
                        f"    cudaEventRecord(start);\n"
                        f"    kernel<<<...>>>(...);\n"
                        f"    cudaDeviceSynchronize();\n"
                        f"    cudaEventRecord(stop);\n"
                        f"    cudaEventSynchronize(stop);\n"
                        f"    float ms = 0;\n"
                        f"    cudaEventElapsedTime(&ms, start, stop);\n"
                        f"}}\n"
                        f"```\n\n"

                        f"**Key Implementation Details:**\n"
                        f"• Use __restrict__ keyword for pointers (prevent compiler optimization)\n"
                        f"• Use volatile reads to prevent caching effects\n"
                        f"• Warm up loop: Run 1000 iterations before timing\n"
                        f"• Timing loop: Average over 10,000 iterations\n"
                        f"• Expected L2 cliff at: ~{gpu_specs.l2_cache_size_kb/1024:.1f}MB for this GPU\n\n"
                        f"⚠️ COMMON PITFALLS TO AVOID:\n"
                        f"• Don't use too small working set (<1MB) - fits in L1/L2\n"
                        f"• Don't use random access patterns - use stride=cache_line_size\n"
                        f"• Don't forget to call cudaDeviceSynchronize() before reading results\n"
                        f"• Don't use printf inside timing loop (I/O overhead)\n"
                        f"• **NEVER call host functions from device code - causes compilation error!**\n\n"
                        f"💡 This algorithm should complete in <30 seconds total.\n"
                        f"If your kernel takes >60s, there's likely a bug!\n"
                    )
                    gpu_context_parts.append(l2_guidance)

                    logger.info(
                        f"[CodeGen] 🔧 Added L2 cache-specific algorithm guidance with HOST/DEVICE rules ({len(l2_guidance)} chars)"
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
