"""Compile CUDA source code via nvcc.

This tool compiles CUDA code submitted by the agent and returns the path to the
compiled binary, along with any compiler output or errors.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.infrastructure.sandbox import SandboxRunner


def compile_cuda_handler(
    arguments: dict[str, Any],
    sandbox: SandboxRunner | None = None,
) -> dict[str, Any]:
    source = arguments.get("source", "")
    flags = arguments.get("flags", [])

    if not source:
        return {
            "status": "error",
            "success": False,
            "output": "",
            "errors": "No source code provided",
            "binary_path": "",
        }

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        return {
            "status": "error",
            "success": False,
            "output": "",
            "errors": "nvcc not found in PATH",
            "binary_path": "",
        }

    target_name = arguments.get("target", "")
    target_name = _normalize_target_name(target_name)

    # HARNESS: Fix printf format to match target name (minimal patching)
    # This only fixes the output format, NOT the measurement logic
    patched_source = _fix_printf_format(source, target_name)

    # HARNESS: Add runtime clamp [0,100] for pct_of_peak metric outputs
    # NOTE: Disabled clamp to allow Verification to see actual computed values
    # and provide meaningful feedback to CodeGen for iterative improvement
    # patched_source = _clamp_pct_of_peak_output(patched_source, target_name)

    # HARNESS: Add missing includes if LLM forgot them
    patched_source = _ensure_includes(patched_source)

    # HARNESS: Anti-cheat architectural enforcement (P0) - MUST RUN BEFORE enum name replacement!
    # Detect forbidden API usage that would fail in evaluation environment
    anti_cheat_warnings = []
    quality_warnings = []
    quality_errors = []
    if target_name:
        anti_cheat_warnings = _detect_anti_cheat_violations(patched_source, target_name)
        quality_warnings, quality_errors = _detect_kernel_quality_issues(patched_source, target_name)

    # HARNESS: Fix undefined CUDA device attribute enum names
    # NOTE: This must run AFTER anti-cheat and quality detection
    patched_source = _fix_cuda_enum_names(patched_source)

    output_hash = hashlib.md5(patched_source.encode()).hexdigest()[:8]
    temp_cu_path = f"/tmp/{output_hash}.cu"
    binary_path = f"/workspace/.sandbox/bin/benchmark_{output_hash}"

    os.makedirs(os.path.dirname(binary_path), exist_ok=True)

    try:
        with open(temp_cu_path, "w", encoding="utf-8") as f:
            f.write(patched_source)

        cmd = [nvcc_path, temp_cu_path, "-o", binary_path]
        cmd.extend(["-w"])
        if flags:
            cmd.extend(flags)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        try:
            os.remove(temp_cu_path)
        except OSError:
            pass

        if result.returncode == 0:
            response = {
                "status": "ok",
                "success": True,
                "output": result.stdout,
                "errors": "",
                "binary_path": binary_path,
            }

            # HARNESS: Anti-cheat architectural enforcement (P0)
            # Results were already computed BEFORE enum name replacement
            if anti_cheat_warnings:
                if response.get("status") == "ok":
                    response["status"] = "success_with_warning"
                existing_warning = response.get("warning", "")
                ac_msg = "\n".join([f"  🚫 {w}" for w in anti_cheat_warnings])
                response["anti_cheat_warnings"] = anti_cheat_warnings
                response["warning"] = (
                    f"{existing_warning}\nANTI-CHEAT VIOLATIONS DETECTED:\n{ac_msg}" if existing_warning
                    else f"ANTI-CHEAT VIOLATIONS DETECTED:\n{ac_msg}"
                )
                print(f"[compile_cuda] HARNESS: {len(anti_cheat_warnings)} anti-cheat violation(s) for target '{target_name}'")

            # HARNESS: Detect if LLM generated wrong measurement code
            # Return a WARNING (not error) so AgentLoop can guide the LLM to fix it
            if target_name and _detect_wrong_measurement(patched_source, target_name):
                response["status"] = "success_with_warning"
                response["warning"] = (
                    f"WARNING: Your code appears to measure SM count but the target is '{target_name}'. "
                    f"This is likely the WRONG measurement approach. "
                    f"You need to write code that specifically measures '{target_name}'. "
                    f"Please review the target definition and generate appropriate code."
                )
                print(f"[compile_cuda] HARNESS: Detected wrong measurement code for target '{target_name}'")

            # HARNESS: Detect kernel quality issues that reduce throughput
            # Results were already computed BEFORE enum name replacement
            if quality_errors:
                # CRITICAL: Block compilation if there are errors
                error_msg = "\n".join([f"  ❌ {e}" for e in quality_errors])
                print(f"[compile_cuda] HARNESS: BLOCKING compilation due to {len(quality_errors)} critical error(s)")
                return {
                    "status": "error",
                    "success": False,
                    "output": "",
                    "errors": f"KERNEL QUALITY BLOCKED:\n{error_msg}\n\nPlease fix these critical issues:\n{error_msg}",
                    "binary_path": "",
                }
            if quality_warnings:
                if response.get("status") == "ok":
                    response["status"] = "success_with_warning"
                existing_warning = response.get("warning", "")
                quality_msg = "\n".join([f"  ⚠️ {w}" for w in quality_warnings])
                response["kernel_quality_warnings"] = quality_warnings
                response["warning"] = (
                    f"{existing_warning}\n{quality_msg}" if existing_warning
                    else f"KERNEL QUALITY ISSUES DETECTED:\n{quality_msg}"
                )
                print(f"[compile_cuda] HARNESS: {len(quality_warnings)} kernel quality issue(s) detected for target '{target_name}'")

            return response
        else:
            return {
                "status": "error",
                "success": False,
                "output": result.stdout,
                "errors": result.stderr,
                "binary_path": "",
            }
    except subprocess.TimeoutExpired:
        try:
            os.remove(temp_cu_path)
        except OSError:
            pass
        return {
            "status": "error",
            "success": False,
            "output": "",
            "errors": "Compilation timed out after 60 seconds",
            "binary_path": "",
        }
    except Exception as e:
        try:
            os.remove(temp_cu_path)
        except OSError:
            pass
        return {
            "status": "error",
            "success": False,
            "output": "",
            "errors": f"Compilation failed: {str(e)}",
            "binary_path": "",
        }


def _normalize_target_name(name: str) -> str:
    if not name:
        return name
    typo_map = {
        'sm_coount': 'sm_count',
        'bytes_rread': 'bytes_read',
        'bytes_wwrite': 'bytes_write',
        'attriibute': 'attribute',
        'deevice__attribute': 'device__attribute',
        'deevice': 'device',
        'commpute': 'compute',
        'countt': 'count',
        'pper_second': 'per_second',
        'sustainedd': 'sustained',
        'launch__sm_countt': 'launch__sm_count',
        'dram__bytes_rread': 'dram__bytes_read',
        'dram__bytes_wwrite': 'dram__bytes_write',
        'device__attriibute': 'device__attribute',
        'sm_coountt': 'sm_count',
        'launch__sm_coount': 'launch__sm_count',
        'dram__bytes_rread.sum.pper_second': 'dram__bytes_read.sum.per_second',
        'dram__bytes_wwrite.sum.pper_second': 'dram__bytes_write.sum.per_second',
        'device__attribute_fb__bus_width': 'device__attribute_fb_bus_width',
        'fb__bus_width': 'fb_bus_width',
        'gpu__commpute_memory_throughput': 'gpu__compute_memory_throughput',
        'dram__bytess_read': 'dram__bytes_read',
        'dram__bytess_read.sum.per_second': 'dram__bytes_read.sum.per_second',
        'sm__throoughput': 'sm__throughput',
        'sm__throoughput.avg.pct_of_peak_sustained_elapsed': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__throughput.avvg': 'sm__throughput.avg',
        'sm__throughput.avvg.pct_of_peak_sustained_elapsed': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'gpu__commpute_memory_throughput.avg.pct_of_peak_sustained_elapsed': 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
        'pct_of_peak_sustainned_elapsed': 'pct_of_peak_sustained_elapsed',
        'pct_of_peaksustained_elapsed': 'pct_of_peak_sustained_elapsed',
    }
    result = name
    for typo, correct in typo_map.items():
        result = result.replace(typo, correct)
    import re as _re
    result = _re.sub(r'(.)\1{2,}', r'\1\1', result)
    return result


def _detect_wrong_measurement(source: str, target_name: str) -> bool:
    """Detect if LLM generated SM count code for a non-SM-count target.

    Returns True if the code appears to measure the wrong metric.
    The LLM should be warned to regenerate, NOT silently patched.

    IMPORTANT: For pct_of_peak_sustained_elapsed metrics, using
    cudaDeviceGetAttribute (including cudaDevAttrMultiProcessorCount)
    is CORRECT and REQUIRED — these are needed to compute peak values.
    Only flag as wrong if the code ONLY queries SM count without
    computing the actual percentage.
    """
    if target_name == "launch__sm_count":
        return False

    if "pct_of_peak_sustained_elapsed" in target_name:
        return False

    is_sm_count_code = (
        "cudaDevAttrMultiProcessorCount" in source or
        ("cudaDeviceGetAttribute" in source and "sm_count" in source.lower()) or
        ("cudaDeviceGetAttribute" in source and "MultiProcessorCount" in source)
    )

    return is_sm_count_code


def _fix_printf_format(source: str, target_name: str) -> str:
    """Fix printf format to match the target name.

    Only fixes the output format string, NOT the measurement logic.
    This ensures the measurement can be correctly parsed from stdout.
    Uses conservative replacement: only replaces when the old format
    is a simple identifier-like string (no spaces, no special chars).
    """
    if not source or not target_name:
        return source

    lines = source.split('\n')
    new_lines = []
    changed = False
    for line in lines:
        m = re.match(r'(\s*printf\(")([a-zA-Z_][\w.]*)(\s*:\s*%[^"]*\\n")', line)
        if m:
            indent_prefix = m.group(1)
            old_format = m.group(2)
            rest = m.group(3)
            normalized_old = _normalize_target_name(old_format)
            if normalized_old != target_name and old_format != target_name:
                if not changed:
                    print(f"[compile_cuda] HARNESS: Fixing printf format from '{old_format}' to '{target_name}'")
                line = indent_prefix + target_name + rest
                changed = True
        new_lines.append(line)
    return '\n'.join(new_lines)


def _fix_cuda_enum_names(source: str) -> str:
    """Fix undefined CUDA device attribute enum names in source code.

    Some CUDA versions don't have all enum names. Replace them with
    numeric casts that work on all CUDA versions.
    """
    enum_replacements = {
        "cudaDevAttrMemoryBusWidth": "(enum cudaDeviceAttr)37",
        "cudaDevAttrGlobalMemoryBusWidth": "(enum cudaDeviceAttr)37",
        "cudaDevAttrMemoryClockRate": "(enum cudaDeviceAttr)36",
        "cudaDevAttrClockRate": "(enum cudaDeviceAttr)13",
        "cudaDevAttrMultiProcessorCount": "(enum cudaDeviceAttr)16",
    }
    for old_name, new_value in enum_replacements.items():
        if old_name in source:
            source = source.replace(old_name, new_value)
    return source


def _clamp_pct_of_peak_output(source: str, target_name: str) -> str:
    """Harness-level safety clamp for pct_of_peak_sustained_elapsed metric outputs.

    DESIGN RATIONALE (Harness Engineering — systematic, not brute-force):

    The LLM generates CUDA code that computes pct_of_peak_sustained_elapsed.
    The prompt provides verified formulas and cudaDeviceGetAttribute calls
    for correct peak calculation. However, LLMs sometimes produce incorrect
    calculations resulting in absurd values like 1,106,194.72% or -169.3%.

    This is a DEFENSE-IN-DEPTH harness mechanism (layer 3 of 3):

    Layer 1 (Prompt): Verified formulas with cudaDeviceGetAttribute calls
    Layer 2 (Compile-time): This function adds runtime clamp [0, 100]
    Layer 3 (Extraction): StageExecutor clamps out-of-range extracted values

    The clamp is inserted BEFORE the printf statement as a simple if-guard.
    This does NOT modify the kernel computation logic — it only adds a
    safety clamp on the output value. If the LLM follows instructions
    correctly, the clamp never triggers.

    This is NOT brute-force replacement (which would force 0.0) — it
    preserves the LLM's computed value when it's within valid range.
    """
    if "pct_of_peak_sustained_elapsed" not in target_name:
        return source

    lines = source.split('\n')
    new_lines = []
    patched = False

    target_short = target_name.replace(".avg.pct_of_peak_sustained_elapsed", "")
    format_patterns = ['%.2f', '%.1f', '%.4f', '%.6f', '%f', '%d', '%g', '%e']

    for line in lines:
        stripped = line.strip()
        should_patch = False

        if 'printf' not in stripped:
            new_lines.append(line)
            continue

        has_target = target_name in stripped or target_short in stripped
        has_pct = 'pct_of_peak' in stripped or 'throughput' in stripped.lower()

        if has_target or has_pct:
            for fmt in format_patterns:
                if fmt in stripped:
                    should_patch = True
                    break

        if should_patch and not patched:
            indent = line[:len(line) - len(line.lstrip())]
            var_match = re.search(r'printf\s*\(\s*"[^"]*"\s*,\s*([a-zA-Z_]\w*)\s*\)', stripped)
            if var_match:
                var_name = var_match.group(1)
                clamp_lines = [
                    f'{indent}if ({var_name} < 0.0) {var_name} = 0.0;',
                    f'{indent}if ({var_name} > 100.0) {var_name} = 100.0;',
                ]
                for cl in clamp_lines:
                    new_lines.append(cl)
                if not patched:
                    print(f"[compile_cuda] HARNESS: Added value clamp [0,100] for pct_of_peak metric '{target_name}'")
                patched = True
            else:
                if not patched:
                    print(f"[compile_cuda] HARNESS: Could not extract variable name for clamping in '{target_name}'")
                patched = True
            new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)


def _ensure_includes(source: str) -> str:
    """Ensure critical includes are present in the CUDA source.

    Only adds missing includes, does not modify any other code.
    """
    required_includes = [
        ("#include <cuda_runtime.h>", "cuda_runtime.h"),
        ("#include <cstdio>", "cstdio"),
        ("#include <cstdint>", "cstdint"),
        ("#include <cstdlib>", "cstdlib"),
    ]

    missing = []
    for include_line, header in required_includes:
        if header not in source:
            missing.append(include_line)

    if missing:
        include_block = "\n".join(missing) + "\n"
        if "#include" in source:
            first_include = source.find("#include")
            source = source[:first_include] + include_block + source[first_include:]
        else:
            source = include_block + source

    # Check for std::sort usage without <algorithm>
    if "std::sort" in source and "<algorithm>" not in source:
        source = "#include <algorithm>\n" + source

    return source


def _detect_kernel_quality_issues(source: str, target_name: str) -> tuple[list[str], list[str]]:
    """Detect common kernel quality issues that reduce throughput metrics.

    Returns a tuple of (warnings, errors) for issues found.
    This is a READ-ONLY analysis — it does NOT modify the source code.
    The warnings/errors are returned to the AgentLoop to guide LLM improvement.
    Errors are CRITICAL issues that should block compilation.
    """
    warnings = []
    errors = []
    is_compute_target = "sm__throughput" in target_name
    is_compute_mem_target = "gpu__compute_memory_throughput" in target_name

    if not (is_compute_target or is_compute_mem_target):
        return warnings, errors

    has_double = "double" in source
    has_float = "float" in source and "double" not in source
    has_volatile = "volatile" in source
    has_warmup = False
    has_pragma_unroll = "#pragma unroll" in source
    has_asm_barrier = 'asm volatile' in source or "asm volatile" in source
    has_global_read_in_loop = False
    has_global_write_in_loop = False
    has_clock64 = "clock64()" in source
    has_input_read_before_fma = False
    uses_clock_rate_attr = "cudaDevAttrClockRate" in source
    uses_sm_count = "sm_count" in source.lower()
    grid_size_ok = False
    has_pct_calculation = False

    print(f"[compile_cuda] DEBUG: target={target_name}, is_compute={is_compute_target}, is_mem={is_compute_mem_target}")
    print(f"[compile_cuda] DEBUG: has_clock64={has_clock64}, uses_clock_rate_attr={uses_clock_rate_attr}")

    lines = source.split('\n')
    in_kernel = False
    in_loop = 0
    loop_depth = 0

    for line in lines:
        stripped = line.strip()
        if '__global__' in stripped or '__device__' in stripped:
            in_kernel = True
            continue
        if in_kernel and '{' in stripped:
            loop_depth += 1
        if in_kernel and '}' in stripped:
            loop_depth -= 1
            if loop_depth <= 0:
                in_kernel = False
                continue

        if 'for' in stripped or 'while' in stripped:
            in_loop += 1
        if in_loop > 0 and (stripped.endswith('}') or stripped == '}'):
            in_loop -= 1

        if '<<<' in line and '>>>' in line:
            import re as _re
            grid_pattern = r'<<<\s*([^)]+),\s*\d+\s*>>>'
            match = _re.search(grid_pattern, line)
            if match:
                grid_expr = match.group(1).strip()
                sm_patterns = [
                    r'\bsm_count\b', r'\bsm\s*\*', r'\*\s*sm\b',
                    r'\bnumsms\b', r'\bnum_sm\b', r'\bsmcount\b',
                    r'device.*sm', r'sm.*device',
                    r'occupancy.*sm', r'sm.*occupancy',
                    r'\*\s*\d+\s*\)', r'\)\s*\*\s*\d+',
                ]
                for pattern in sm_patterns:
                    if _re.search(pattern, grid_expr, _re.IGNORECASE):
                        grid_size_ok = True
                        break

        if in_loop > 0 and in_kernel:
            if ('[' in stripped or 'input[' in stripped or
                'data[' in stripped or 'array[' in stripped):
                has_global_read_in_loop = True
                if 'input[' in stripped and '=' in stripped:
                    has_input_read_before_fma = True
            if ('output[' in stripped or 'result[' in stripped or
                '=' in stripped and ']' in stripped):
                if 'output' in stripped or 'sink' not in stripped:
                    has_global_write_in_loop = True

    warmup_patterns = [
        'warmup', 'WARMUP', 'warm_up', 'dummy',
        '<<<', 'kernel<<<',
    ]
    for i, line in enumerate(lines):
        for pattern in warmup_patterns:
            if pattern in line.lower():
                if i < len(lines) - 1:
                    next_lines = '\n'.join(lines[i:min(i+5, len(lines))])
                    if '<<<' in next_lines or 'cudaEvent' in next_lines or 'kernel' in next_lines.lower():
                        has_warmup = True
                        break
        if has_warmup:
            break

    if is_compute_target:
        if not has_clock64:
            warnings.append(
                "KERNEL QUALITY ERROR: No clock64() call detected in sm__throughput kernel. "
                "Without clock64(), you cannot measure actual running frequency. "
                "cudaDevAttrClockRate reports BASE clock, not BOOST → pct > 100% → clamped to 100%! "
                "THIS IS MANDATORY: Add clock64() inside the kernel:\n"
                "  uint64_t start_cycle = clock64();\n"
                "  // ... FMA loop ...\n"
                "  uint64_t end_cycle = clock64();\n"
                "  if (threadIdx.x == 0 && blockIdx.x == 0) *cycle_out = end_cycle - start_cycle;\n"
                "Then in host code:\n"
                "  double actual_freq_mhz = (double)h_cycle_count / (elapsed_ms * 1000.0);\n"
                "  double peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2;\n"
                "DO NOT use cudaDevAttrClockRate for peak_flops!"
            )
            # CRITICAL: Force rejection if clock64() is missing for sm__throughput
            errors.append("COMPILATION BLOCKED: clock64() is MANDATORY for sm__throughput kernel")
        if uses_clock_rate_attr and is_compute_target:
            warnings.append(
                "KERNEL QUALITY ERROR: cudaDevAttrClockRate detected in sm__throughput kernel. "
                "This reports the BASE clock, not the actual BOOST clock the GPU runs at! "
                "Using it for peak_flops will UNDERESTIMATE peak → pct > 100% → clamped to 100%! "
                "INSTEAD: Use clock64() inside the kernel to measure actual running frequency, "
                "then compute actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0). "
                "REPLACE: peak_flops = sm_count * fp64_per_sm * (clock_khz/1000.0) * 1e6 * 2 "
                "WITH: peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2"
            )
            # CRITICAL: Force rejection if cudaDevAttrClockRate is used for sm__throughput
            errors.append("COMPILATION BLOCKED: cudaDevAttrClockRate is FORBIDDEN for sm__throughput kernel")
        if has_float and not has_double:
            warnings.append(
                "KERNEL QUALITY WARNING: Using float instead of double for sm__throughput. "
                "Double-precision FMA achieves HIGHER SM utilization on modern GPUs. "
                "Change ALL float to double in the compute loop."
            )
        if not has_volatile:
            warnings.append(
                "KERNEL QUALITY WARNING: Missing volatile qualifier on sink/output pointer. "
                "The compiler may optimize away your computation (dead-code elimination). "
                "Use 'volatile double* sink' or add 'asm volatile(\"\" : \"+d\"(result) : : \"memory\");' after the loop."
            )
        if has_global_read_in_loop:
            warnings.append(
                "KERNEL QUALITY WARNING: Global memory read INSIDE the timed compute loop. "
                "This makes the kernel memory-bound instead of compute-bound. "
                "Move all memory reads BEFORE the loop, use register variables only inside the loop."
            )
        if has_global_write_in_loop:
            warnings.append(
                "KERNEL QUALITY WARNING: Global memory write INSIDE the timed compute loop. "
                "This adds memory latency and reduces sm__throughput. "
                "Only write to volatile sink AFTER the loop completes."
            )
        if not has_warmup:
            warnings.append(
                "KERNEL QUALITY WARNING: No WARMUP kernel detected. "
                "The first kernel launch runs at lower clock speed. "
                "Run the kernel ONCE before starting cudaEventRecord timing."
            )
        if not has_pragma_unroll:
            warnings.append(
                "KERNEL QUALITY WARNING: No #pragma unroll before the FMA loop. "
                "The compiler may unroll unpredictably or optimize away iterations. "
                "Add '#pragma unroll 1' before the for loop to ensure consistent behavior."
            )

    if is_compute_mem_target:
        if not has_input_read_before_fma:
            errors.append(
                "KERNEL QUALITY ERROR: FMA chain does NOT use value read from input[i]. "
                "Register-only FMA does NOT stress memory → 0.09% throughput! "
                "WRONG: val = val * 1.0001f + 0.001f where val is register-only. "
                "RIGHT: val = input[i]; then val = val * 1.0001f + 0.001f; then output[i] = val; "
                "COMPILATION BLOCKED: Must read from input array to stress GPU memory!"
            )
        if not has_volatile:
            warnings.append(
                "KERNEL QUALITY WARNING: Missing volatile on output pointer. "
                "Compiler will eliminate writes (dead-code elimination), causing near-zero throughput. "
                "Use 'volatile float* output' for the output buffer."
            )
        if not has_warmup:
            warnings.append(
                "KERNEL QUALITY WARNING: No WARMUP kernel detected. "
                "First run measures lower throughput due to GPU power ramping. "
                "Run kernel once before timing."
            )

    if is_compute_target or is_compute_mem_target:
        if not grid_size_ok:
            errors.append(
                "KERNEL QUALITY ERROR: Grid size may be too small! "
                "You MUST use sm_count*4 blocks to saturate the GPU. "
                "Example: kernel<<<sm_count*4, 256>>>() "
                "Small grid sizes (< sm_count) will result in poor SM utilization! "
                "COMPILATION BLOCKED: Grid size must be sm_count*4 or larger!"
            )
        if not uses_sm_count:
            errors.append(
                "KERNEL QUALITY ERROR: sm_count not detected in code. "
                "You MUST query the SM count and use it to calculate grid size. "
                "Example: int sm_count; cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0); "
                "COMPILATION BLOCKED: Must use sm_count for proper GPU utilization!"
            )

    for line in lines:
        stripped = line.strip()
        if "peak_flops" in stripped.lower() or "peak_bw" in stripped.lower():
            has_pct_calculation = True
        if ("achieved" in stripped.lower() and "/" in stripped) or (" / peak" in stripped.lower()):
            has_pct_calculation = True
        if "* 100" in stripped or "*100.0" in stripped or "* 100.0" in stripped:
            has_pct_calculation = True

    if is_compute_target and not has_pct_calculation:
        errors.append(
            "KERNEL QUALITY ERROR: sm__throughput kernel does NOT compute percentage correctly! "
            "You MUST compute: pct = (achieved_flops / peak_flops) * 100.0 "
            "Where peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2 "
            "And achieved_flops = total_fma_ops / elapsed_seconds "
            "COMPILATION BLOCKED: Must implement proper percentage calculation!"
        )
    
    if is_compute_mem_target and not has_pct_calculation:
        errors.append(
            "KERNEL QUALITY ERROR: gpu__compute_memory_throughput kernel does NOT compute percentage correctly! "
            "You MUST compute: pct = (achieved_bw / peak_bw) * 100.0 "
            "Where peak_bw = (mem_clock_khz/1000.0) * 1e6 * (bus_width_bits/8) * 2 / 1e9 "
            "And achieved_bw = (2.0 * buffer_size_bytes) / elapsed_seconds / 1e9 "
            "COMPILATION BLOCKED: Must implement proper percentage calculation!"
        )

    return warnings, errors


def _detect_anti_cheat_violations(source: str, target_name: str) -> list[str]:
    """Detect anti-cheat violations in generated CUDA code.

    This is ARCHITECTURAL enforcement per spec.md 6.3:
    - Frequency lock detection: code must not rely on static frequency lookup
    - SM masking: code must not rely solely on cudaGetDeviceProperties
    - API interception: code must not depend on cudaGetDeviceProperties

    Returns list of violation messages. These are returned as warnings
    to the AgentLoop so the LLM can be guided to fix them.
    """
    violations = []

    if "cudaGetDeviceProperties" in source:
        violations.append(
            "ANTI-CHEAT VIOLATION: cudaGetDeviceProperties detected. "
            "This API may be INTERCEPTED or return VIRTUALIZED data in evaluation. "
            "You MUST use empirical measurement instead: "
            "clock64()+cudaEventElapsedTime for frequency, "
            "pointer-chasing for latency, "
            "occupancy API for SM count."
        )

    if "cudaDeviceGetAttribute" in source:
        empirical_fallbacks = [
            "clock64()" in source,
            "cudaEventElapsedTime" in source,
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in source,
        ]
        if not any(empirical_fallbacks):
            violations.append(
                "ANTI-CHEAT VIOLATION: cudaDeviceGetAttribute used WITHOUT empirical cross-validation. "
                "This API may return virtualized values. "
                "Add empirical measurement to cross-validate the API result."
            )

    if target_name and "clock" in target_name.lower() and "mhz" in target_name.lower():
        if "nvidia-smi" in source:
            violations.append(
                "ANTI-CHEAT VIOLATION: Using nvidia-smi for clock measurement. "
                "nvidia-smi reports LOCKED frequency, not actual running frequency. "
                "Use clock64()+cudaEventElapsedTime dual-timing to measure actual frequency."
            )

    if target_name and "latency" in target_name.lower():
        has_pointer_chasing = (
            "chase" in source.lower() or
            "pointer" in source.lower() or
            ("next" in source.lower() and "offset" in source.lower())
        )
        if not has_pointer_chasing and "clock64()" in source:
            violations.append(
                "ANTI-CHEAT WARNING: Latency measurement without pointer-chasing pattern. "
                "Hardware prefetcher may hide true latency. "
                "Use random pointer-chasing to bypass prefetcher."
            )

    if target_name and "cache_size" in target_name.lower():
        has_sweep = "sweep" in source.lower() or "range" in source.lower()
        if not has_sweep:
            violations.append(
                "ANTI-CHEAT WARNING: Cache size measurement without working-set sweep. "
                "Need to test multiple working-set sizes to detect the latency cliff. "
                "Implement a sweep from small to large working sets."
            )

    return violations
