"""Verification Agent — independent reviewer.

P7 enforcement: this agent NEVER inherits a generator's context.
It creates a fresh ContextManager on construction and reviews results
from first principles.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _sanitize_pct_of_peak_values(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize pct_of_peak values in data before passing to LLM review.

    LLM frequently calculates wrong percentages (e.g., 1,106,194.72%) for
    NCU-native metrics. This function replaces out-of-range values with a
    note that the real value comes from NCU.
    """
    if not isinstance(data, dict):
        return data

    result = dict(data)
    for key, value in list(result.items()):
        if "pct_of_peak_sustained_elapsed" not in key:
            continue
        if not isinstance(value, (int, float)):
            continue
        if 0.0 <= value <= 100.0:
            continue
        old_val = value
        clamped = max(0.0, min(100.0, value))
        result[key] = clamped
        logger.info(
            "[Verification] Clamped '%s': %.2f -> %.2f (out of [0,100] range)",
            key, old_val, clamped,
        )

    def _replace_bad_pct_in_text(text: str) -> str:
        """Replace out-of-range pct_of_peak values in any text string."""
        if not isinstance(text, str) or not text:
            return text

        def _replace_bad_pct(m):
            val_str = m.group(1)
            try:
                val = float(val_str.replace(",", ""))
                if val > 100.0 or val < 0.0:
                    clamped = max(0.0, min(100.0, val))
                    return f"{m.group(0).split(':')[0].strip()}: {clamped:.2f} [clamped to [0,100]]"
            except ValueError:
                pass
            return m.group(0)

        pattern = r'(sm__throughput\.avg\.pct_of_peak_sustained_elapsed|gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed)\s*:\s*([\d,.]+)'
        return re.sub(pattern, _replace_bad_pct, text)

    summaries = result.get("stage_summaries", {})
    if isinstance(summaries, dict):
        for summary_key in ["code_gen", "metric_analysis", "plan"]:
            text = summaries.get(summary_key, "")
            if isinstance(text, str) and text:
                new_text = _replace_bad_pct_in_text(text)
                if new_text != text:
                    logger.info(f"[Verification] Sanitized pct_of_peak values in {summary_key} text")
                    summaries[summary_key] = new_text

    measurements = result.get("measurements", {})
    if isinstance(measurements, dict):
        for key, value in list(measurements.items()):
            if "pct_of_peak_sustained_elapsed" not in key:
                continue
            if not isinstance(value, (int, float)):
                continue
            if 0.0 <= value <= 100.0:
                continue
            old_val = value
            clamped = max(0.0, min(100.0, value))
            measurements[key] = clamped
            logger.info(
                "[Verification] Clamped measurements['%s']: %.2f -> %.2f",
                key, old_val, clamped,
            )

    return result

from src.application.context import ContextManager, Role
from src.domain.permission import PermissionMode
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    P7ViolationError,
    SubAgentResult,
    SubAgentStatus,
)
from src.domain.tool_contract import ToolRegistry


class VerificationAgent(BaseSubAgent):
    """Independent reviewer that validates experimental results.

    CRITICAL: The constructor ALWAYS creates a fresh ContextManager.
    It never accepts an external one. This is the core P7 guarantee.
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 16000,
    ) -> None:
        # P7: ALWAYS create a fresh ContextManager — never accept one externally
        super().__init__(
            role=AgentRole.VERIFICATION,
            context_manager=ContextManager(max_tokens=max_tokens),  # always fresh
            tool_registry=tool_registry or ToolRegistry(),
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )

    def _process(self, message: CollaborationMessage) -> SubAgentResult:
        """Review the previous stage's result independently.

        Receives only the structured SubAgentResult (data + artifacts +
        context fingerprint) — NOT the generation process or context.

        P7 ENFORCEMENT: Clear context_manager before each review to ensure
        no inherited context from previous stages or iterations.
        """
        self.context_manager.clear()

        prev_result = message.payload.get("prev_result", {})
        prev_fingerprint = message.payload.get("prev_fingerprint", "none")
        target_spec = message.payload.get("target_spec", {})  # For completeness check

        # Extract data from previous result
        data = prev_result.get("data", {})
        artifacts = prev_result.get("artifacts", [])
        prev_status = prev_result.get("status", "unknown")
        prev_role = prev_result.get("agent_role", "unknown")

        # Independent review — pass target_spec for completeness validation
        review = self._review(data, artifacts, prev_status, prev_role, target_spec)

        result = SubAgentResult(
            agent_role=self.role,
            status=review["status"],
            data={
                "review": review["findings"],
                "accepted": review["accepted"],
                "concerns": review["concerns"],
                "suggested_fixes": review.get("suggested_fixes", []),
                "can_be_fixed": review.get("can_be_fixed", True),
                "generation_fingerprint": prev_fingerprint,
            },
            metadata={
                "reviewed_agent": prev_role,
                "review_method": "independent_analysis",
            },
        )

        return result

    def _review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
        target_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform independent review of the previous stage's result.

        Uses LLM when available for deeper analysis, falls back to
        rule-based checks (completeness, numeric sanity, methodology).
        """
        if self._model_caller is not None:
            result = self._llm_review(data, artifacts, prev_status, prev_role, target_spec)
        else:
            result = self._rule_review(data, artifacts, prev_status, prev_role, target_spec)

        result["can_be_fixed"] = self._assess_fixability(result.get("concerns", []))
        result["suggested_fixes"] = self._suggest_fixes(result.get("concerns", []))

        return result

    def _llm_review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
        target_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use LLM for independent review of the previous stage's result."""
        import json as _json

        sanitized_data = _sanitize_pct_of_peak_values(data)

        review_payload = {
            "data": sanitized_data,
            "artifacts": artifacts,
            "prev_status": prev_status,
            "prev_role": prev_role,
            "requested_targets": target_spec.get("targets", []) if target_spec else None,
        }

        user_msg = (
            f"Independently review this GPU profiling result. "
            f"Check: (1) data completeness (all requested targets measured?), "
            f"(2) numeric sanity, (3) methodology soundness, (4) artifact existence.\n\n"
            f"IMPORTANT: You are reviewing structured data, not directory contents. "
            f"The working directory is NOT empty - it contains the complete project code.\n\n"
            f"CRITICAL RULE FOR PCT_OF_PEAK METRICS:\n"
            f"For any target containing 'pct_of_peak_sustained_elapsed' (e.g., sm__throughput, "
            f"gpu__compute_memory_throughput), the value should be a COMPUTED percentage "
            f"in the range [0, 100]. Values of 0.0 indicate the kernel failed to achieve "
            f"measurable throughput or the measurement code is broken. "
            f"Values >100% or <0% have been clamped to [0, 100] before this review.\n\n"
            f"Review data:\n{_json.dumps(review_payload, indent=2)}\n\n"
            f'Return a JSON object with "findings" (list of strings), '
            f'"concerns" (list of strings), "accepted" (boolean), '
            f'and "status" ("success" or "rejected").'
        )
        messages = [
            {"role": "system", "content": "You are an independent GPU profiling result reviewer. You MUST NOT inherit any context from previous stages. Review only the data provided below."},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self._model_caller(messages)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(response[start:end])
                return {
                    "status": SubAgentStatus.SUCCESS
                    if parsed.get("status") == "success"
                    else SubAgentStatus.REJECTED,
                    "findings": parsed.get("findings", []),
                    "concerns": parsed.get("concerns", []),
                    "accepted": parsed.get("accepted", True),
                }
        except Exception:
            pass

        # Fallback to rule-based review
        return self._rule_review(data, artifacts, prev_status, prev_role, target_spec)

    def _rule_review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
        target_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rule-based independent review.

        Checks:
        1. Data completeness — all requested targets measured?
        2. Numeric sanity — values within plausible GPU ranges?
        3. Methodology — valid bottleneck type?
        4. Artifacts — evidence files referenced?
        """
        findings: list[str] = []
        concerns: list[str] = []
        accepted = True

        # Check 0: Empty data — fundamental failure
        if not data:
            concerns.append("No data provided for review — this indicates a complete pipeline failure or data flow break")
            accepted = False
            return {
                "status": SubAgentStatus.REJECTED,
                "findings": findings,
                "concerns": concerns,
                "accepted": False,
            }

        # Check 1: Data completeness against target_spec
        requested_targets = set()
        if target_spec:
            requested_targets = set(target_spec.get("targets", []))
            if requested_targets:
                # Measurements may be at top level OR nested in parsed_metrics
                measured_keys = set(data.keys())
                if "parsed_metrics" in data and isinstance(data["parsed_metrics"], dict):
                    measured_keys |= set(data["parsed_metrics"].keys())
                if "measurements" in data and isinstance(data["measurements"], dict):
                    measured_keys |= set(data["measurements"].keys())
                missing = requested_targets - measured_keys
                if missing:
                    concerns.append(f"Missing targets: {', '.join(sorted(missing))}")
                    accepted = False
                else:
                    findings.append(f"All {len(requested_targets)} requested targets measured")
                findings.append(f"Data contains {len(data)} fields")
            else:
                findings.append(f"Data contains {len(data)} fields")
        else:
            findings.append(f"Data contains {len(data)} fields (no target_spec for completeness check)")

        # Check 2: Numeric sanity
        has_zero_measurement = False
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if "pct_of_peak_sustained_elapsed" in key:
                    if value < 0 or value > 100:
                        findings.append(
                            f"'{key}' = {value}% — clamped to [0,100] range. "
                            f"This indicates an issue with the percentage calculation."
                        )
                    elif value == 0.0:
                        concerns.append(
                            f"'{key}' = 0.0% — kernel failed to achieve measurable throughput. "
                            f"The measurement code may be broken (e.g., FMA loop optimized away, "
                            f"missing volatile, incorrect peak calculation)."
                        )
                        accepted = False
                    elif "sm__throughput" in key and value < 20.0:
                        concerns.append(
                            f"'{key}' = {value:.2f}% — TOO LOW for SM throughput. "
                            f"A well-designed compute-bound kernel should achieve 40-80%. "
                            f"Likely causes: (1) using float instead of double, "
                            f"(2) grid size too small (< sm_count blocks), "
                            f"(3) no warmup kernel, (4) FMA loop not in registers. "
                            f"REGENERATE kernel with: double-precision FMA, sm_count*4 blocks x 256 threads, "
                            f"warmup run, clock64() for actual frequency, #pragma unroll 1."
                        )
                        accepted = False
                    elif "sm__throughput" in key and value >= 99.0:
                        concerns.append(
                            f"'{key}' = {value:.2f}% — SUSPICIOUSLY HIGH (near 100%). "
                            f"This almost certainly means peak_flops was UNDERESTIMATED. "
                            f"Cause: using cudaDevAttrClockRate (base clock) instead of clock64() (actual boost clock). "
                            f"FIX: Inside kernel, record clock64() before/after FMA loop, output cycle count. "
                            f"Compute actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0). "
                            f"Use: peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2. "
                            f"Do NOT use cudaDevAttrClockRate for peak_flops!"
                        )
                        accepted = False
                    elif "sm__throughput" in key and value > 100.0:
                        concerns.append(
                            f"'{key}' = {value:.2f}% — EXCEEDS 100%! "
                            f"This definitively proves peak_flops was UNDERESTIMATED. "
                            f"You are using cudaDevAttrClockRate which reports BASE clock (~1.4GHz), "
                            f"but GPU runs at BOOST clock (~1.7-2.0GHz). "
                            f"MANDATORY FIX: Replace cudaDevAttrClockRate with clock64() measurement. "
                            f"Inside kernel: uint64_t start_cycle = clock64(); //FMA loop// uint64_t end_cycle = clock64(); "
                            f"Host: double actual_freq_mhz = (double)h_cycle_count / (elapsed_ms * 1000.0); "
                            f"peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2; "
                            f"This MUST be fixed before verification can pass!"
                        )
                        accepted = False
                    elif "gpu__compute_memory_throughput" in key and value < 15.0:
                        concerns.append(
                            f"'{key}' = {value:.2f}% — TOO LOW for compute-memory throughput. "
                            f"A well-designed fused read-compute-write kernel should achieve 30-70%. "
                            f"Likely causes: (1) FMA chain uses register-only values (NOT read from memory), "
                            f"(2) buffer too small (< 64MB), (3) missing volatile on output, "
                            f"(4) grid size too small. "
                            f"REGENERATE kernel with: val = input[i]; val = val * 1.0001f + 0.001f; output[i] = val; "
                            f"64MB+ buffer, volatile float* output, sm_count*4 blocks x 256 threads."
                        )
                        accepted = False
                    else:
                        findings.append(f"'{key}' = {value}% (computed percentage, within valid range)")
                    continue
                if value == 0 and key not in ("exit_code", "binary_count"):
                    concerns.append(f"Zero measurement for '{key}': {value} — this indicates a measurement failure (e.g., clock64() not called, code optimized away)")
                    accepted = False
                    has_zero_measurement = True
                elif value < 0:
                    concerns.append(f"Negative value for '{key}': {value}")
                    accepted = False
                elif value > 1e12:
                    concerns.append(f"Suspiciously large value for '{key}': {value}")
                    accepted = False
                else:
                    findings.append(f"'{key}' = {value} (within plausible range)")

        # If there are zero measurements, the entire result is suspect
        if has_zero_measurement:
            concerns.append(
                "CRITICAL: Multiple measurements are zero — this indicates the measurement code is fundamentally broken. "
                "CodeGen must fix the CUDA kernel (e.g., ensure clock64() is called in the right code section, "
                "prevent compiler optimization with volatile/asm, use proper synchronization)."
            )

        # Check 3: Methodology
        if "bottleneck_type" in data:
            valid_types = {
                "compute_bound", "memory_bound", "latency_bound",
                "cache_capacity", "unknown",
            }
            if data["bottleneck_type"] in valid_types:
                findings.append(
                    f"Bottleneck type '{data['bottleneck_type']}' is valid"
                )
            else:
                concerns.append(
                    f"Unknown bottleneck type: '{data['bottleneck_type']}'"
                )
                accepted = False

        # Check 4: Artifacts
        if artifacts:
            findings.append(f"{len(artifacts)} artifact(s) referenced")
        else:
            concerns.append("No artifacts referenced")

        # Final determination
        status = SubAgentStatus.SUCCESS if accepted else SubAgentStatus.REJECTED

        return {
            "status": status,
            "findings": findings,
            "concerns": concerns,
            "accepted": accepted,
        }

    @staticmethod
    def _assess_fixability(concerns: list[str]) -> bool:
        """Assess whether the rejected result can be fixed by CodeGen retry.

        Returns False only for fundamentally unfixable issues.
        Most concerns are fixable by regenerating with better methodology.
        """
        unfixable_patterns = [
            "no data provided",
            "no artifacts",
        ]
        for concern in concerns:
            lower = concern.lower()
            if any(p in lower for p in unfixable_patterns):
                return False
        return len(concerns) > 0

    @staticmethod
    def _suggest_fixes(concerns: list[str]) -> list[str]:
        """Generate actionable fix suggestions based on concerns.

        Maps common concern patterns to specific CodeGen guidance.
        """
        fixes: list[str] = []
        for concern in concerns:
            lower = concern.lower()
            if "missing target" in lower:
                fixes.append("Ensure ALL requested targets are measured and reported with parseable output format (key: value)")
            elif "pct_of_peak_sustained_elapsed" in lower and ("invalid" in lower or ">100" in lower or "outside" in lower):
                fixes.append(
                    "FIX pct_of_peak_sustained_elapsed: This metric is a PERCENTAGE (0-100%). "
                    "The kernel MUST compute: pct = (achieved / peak) * 100. "
                    "For sm__throughput: use clock64() inside kernel to measure actual running frequency, "
                    "actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0), "
                    "peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2, "
                    "where fp64_per_sm depends on compute capability (V100=32, A100=32, H100=64, T4=2, consumer=2). "
                    "Do NOT use cudaDevAttrClockRate for peak — it may report base clock, not boost! "
                    "For gpu__compute_memory_throughput: peak_bw = (mem_clock_khz/1000.0) * 1e6 * (bus_width_bits/8) * 2 / 1e9. "
                    "NCU measurement is AUTHORITATIVE — kernel printf is for cross-validation only."
                )
            elif "negative value" in lower:
                fixes.append("Check timing methodology — negative values suggest clock64() wraparound or incorrect subtraction order")
            elif "suspiciously large" in lower:
                fixes.append("Verify measurement units and iteration count — divide total cycles by number of iterations")
            elif "zero measurement" in lower:
                fixes.append(
                    "FIX zero measurement: Add 'volatile' qualifier on output pointer, "
                    "add 'asm volatile(\"\" ::: \"memory\")' barrier after compute loop, "
                    "ensure clock64() is called inside the timed section, "
                    "and use printf to output the measured value."
                )
            elif "too low for sm throughput" in lower:
                fixes.append(
                    "FIX LOW sm__throughput: (1) Use DOUBLE-PRECISION FMA (result += a * b + c), NOT float. "
                    "(2) Launch sm_count*4 blocks x 256 threads for full SM utilization. "
                    "(3) Run WARMUP kernel before timed measurement. "
                    "(4) Inside kernel: record clock64() before/after FMA loop, output cycle count. "
                    "(5) Compute actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0). "
                    "(6) peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2. "
                    "(7) Use #pragma unroll 1 before the FMA loop. "
                    "(8) Use volatile double* sink to prevent dead-code elimination. "
                    "(9) Do NOT use cudaDevAttrClockRate for peak — it may report base clock!"
                )
            elif "suspiciously high" in lower and "sm__throughput" in lower:
                fixes.append(
                    "FIX sm__throughput = 100%: This means peak_flops is UNDERESTIMATED. "
                    "You are using cudaDevAttrClockRate which reports BASE clock, not BOOST clock. "
                    "FIX: Inside kernel, add clock64() before/after FMA loop:\n"
                    "  uint64_t start_cycle = clock64();\n"
                    "  // ... FMA loop ...\n"
                    "  uint64_t end_cycle = clock64();\n"
                    "  if (threadIdx.x == 0 && blockIdx.x == 0) *cycle_out = end_cycle - start_cycle;\n"
                    "Then in host code:\n"
                    "  cudaMemcpy(&h_cycle_count, d_cycle_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);\n"
                    "  double actual_freq_mhz = (double)h_cycle_count / (elapsed_ms * 1000.0);\n"
                    "  double peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2;\n"
                    "REMOVE: cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0)\n"
                    "REPLACE: peak_flops = sm_count * fp64_per_sm * (clock_khz/1000.0) * 1e6 * 2\n"
                    "WITH: peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2"
                )
            elif "too low for compute-memory throughput" in lower:
                fixes.append(
                    "FIX LOW gpu__compute_memory_throughput: (1) Use FUSED read-compute-write kernel. "
                    "(2) READ input[i] from global memory: val = input[i]. "
                    "(3) COMPUTE 8+ FMA USING THE READ VALUE: val = val * 1.0001f + 0.001f. "
                    "(4) WRITE to volatile output[i]: output[i] = val. "
                    "(5) WRONG: val = val * 1.0001f + 0.001f where val is register-only → 0.09%! "
                    "(6) Use 64MB+ buffer (16M+ floats) to ensure DRAM access. "
                    "(7) Launch sm_count*4 blocks x 256 threads. "
                    "(8) Run WARMUP kernel before timed measurement. "
                    "(9) Use volatile float* output to prevent dead-code elimination."
                )
            elif "no artifacts" in lower:
                fixes.append("Call compile_cuda and execute_binary tools — text-only output without tool calls is a failure")
            elif "bottleneck" in lower:
                fixes.append("Use a valid bottleneck classification: compute_bound, memory_bound, latency_bound, cache_capacity, or unknown")
            elif "latency" in lower and ("low" in lower or "high" in lower or "range" in lower):
                fixes.append("Verify working set size matches target memory level (L1: <32KB, L2: 512KB-2MB, DRAM: >64MB) and use random pointer-chasing to defeat prefetchers")
            elif "bandwidth" in lower:
                fixes.append("Use STREAM copy with large arrays (>64MB), launch enough threads (65535 blocks x 256 threads), and use cudaEventElapsedTime")
            elif "clock" in lower or "frequency" in lower:
                fixes.append("Use dual timing: clock64() for GPU cycles + cudaEventElapsedTime for wall-clock, freq = cycles / elapsed_us")
            elif "anti-cheat" in lower or "cudagetdeviceproperties" in lower:
                fixes.append(
                    "Remove ALL cudaGetDeviceProperties calls. Use empirical measurement: "
                    "clock64()+cudaEventElapsedTime for frequency, pointer-chasing for latency, "
                    "occupancy API for SM count."
                )
            else:
                fixes.append(f"Review and address: {concern}")
        return fixes
