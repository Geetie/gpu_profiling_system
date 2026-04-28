"""Pipeline shared context — mutable state passed between stages.

Encapsulates the data that flows through the pipeline:
- previous result from the last completed stage
- CodeGen data preserved for final result assembly
- the previous stage identifier
- conversation history for cross-Stage context inheritance
- iteration tracking for REJECT feedback loops
- versioned measurement snapshots with rollback capability

This is a pure data holder with no business logic.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from src.domain.subagent import PipelineStage, SubAgentResult, SubAgentStatus

logger = logging.getLogger(__name__)


_THROUGHPUT_KEYWORDS = ("throughput", "bytes", "bandwidth", "per_second", "gbps")
_LATENCY_KEYWORDS = ("latency", "penalty", "cycles")
_ATTRIBUTE_KEYWORDS = ("sm_count", "bus_width", "frequency_khz", "max_gpu", "max_mem", "fb_bus")
_IMPROVEMENT_SCALE = 5.0
_IMPROVEMENT_THRESHOLD = 0.03


@dataclass
class MeasurementVersion:
    """A versioned snapshot of all measurements produced by one CodeGen pass.

    Each CodeGen pass (initial or optimization) produces a complete set of
    measurements. Instead of merging individual values, we store the entire
    set as an atomic unit, enabling:
    - No cross-version mixing of measurements
    - Rollback to a previous stable version
    - Quality comparison between versions (data quality + optimization progress)
    """
    version: int
    measurements: dict[str, Any]
    code_hash: str
    timestamp: float
    is_optimization: bool
    quality_ok: bool = True
    quality_score: float = 0.0
    improvement_score: float = 0.0

    def compute_quality_score(self) -> float:
        """Score this version's measurements for intrinsic data quality.

        Higher score = better quality. Used as ONE factor in version comparison.

        Scoring has three components:
        1. COVERAGE: Each valid measurement contributes a base score.
           More targets measured = higher coverage score.
        2. SANITY: Physically impossible values receive heavy penalties.
           This catches inf/NaN, absurdly high percentages, etc.
        3. REASONABLENESS: Values within plausible range but near the
           boundary get a small penalty. Values in the "sweet spot"
           get a small bonus. This distinguishes "technically valid
           but suspicious" from "clearly correct".

        IMPORTANT: quality_score measures DATA QUALITY only — it does NOT
        measure whether measurements improved vs. a previous version.
        Use compare_with_previous() for that.
        """
        COVERAGE_WEIGHT = 10.0
        SANITY_PENALTY = 50.0
        REASONABLENESS_RANGE = 2.0

        coverage_score = 0.0
        sanity_penalty = 0.0
        reasonableness_score = 0.0
        n_valid = 0
        n_total = len(self.measurements)

        for key, val in self.measurements.items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                sanity_penalty += SANITY_PENALTY
                continue

            if not math.isfinite(fval):
                sanity_penalty += SANITY_PENALTY
                continue

            if fval == 0.0:
                if any(k in key for k in _ATTRIBUTE_KEYWORDS):
                    sanity_penalty += SANITY_PENALTY
                    continue
                if any(kw in key for kw in ("penalty", "conflict")):
                    n_valid += 1
                    coverage_score += COVERAGE_WEIGHT
                    continue
                if "pct_of_peak_sustained_elapsed" in key:
                    # Zero value for throughput metrics means measurement FAILED
                    sanity_penalty += SANITY_PENALTY * 0.5
                    continue
                sanity_penalty += SANITY_PENALTY * 0.3
                continue

            if "pct_of_peak" in key or "throughput.avg" in key:
                if fval > 100.0:
                    sanity_penalty += SANITY_PENALTY
                    continue
                if fval < 0.0:
                    sanity_penalty += SANITY_PENALTY
                    continue
                if fval > 100.0:
                    reasonableness_score -= REASONABLENESS_RANGE * 0.5
                elif fval > 0.0:
                    reasonableness_score += REASONABLENESS_RANGE * 0.1

            if "bytes" in key and "per_second" in key:
                if fval > 2_000_000_000_000.0:
                    sanity_penalty += SANITY_PENALTY
                    continue
                if fval < 1_000_000_000.0 and fval > 0.0:
                    reasonableness_score -= REASONABLENESS_RANGE * 0.3

            if "sm_count" in key:
                if fval < 1 or fval > 200:
                    sanity_penalty += SANITY_PENALTY
                    continue

            if "bus_width" in key:
                if fval < 32 or fval > 8192:
                    sanity_penalty += SANITY_PENALTY
                    continue

            if "frequency_khz" in key:
                if fval < 100_000 or fval > 20_000_000:
                    sanity_penalty += SANITY_PENALTY
                    continue

            n_valid += 1
            coverage_score += COVERAGE_WEIGHT

        if n_total > 0 and n_valid > 0:
            coverage_score *= (n_valid / n_total)

        score = coverage_score - sanity_penalty + reasonableness_score

        if n_valid == 0:
            score = -100.0

        self.quality_score = score
        self.quality_ok = score >= 0.0 and sanity_penalty == 0.0
        return score

    def compare_with_previous(self, prev: MeasurementVersion) -> tuple[bool, str]:
        """Compare this version with a previous version for overall superiority.

        Returns (is_better, reason).

        Decision hierarchy:
        1. If coverage dropped by >1 target → worse (lost too many measurements)
        2. If new sanity violations appeared → worse (data integrity compromised)
        3. Compare actual measurement VALUES for shared performance targets:
           - throughput/bandwidth targets: higher is better
           - latency targets: lower is better
           - device attributes: should be consistent (skip)
        4. If quality_score is similar, prefer the version with more improvements

        This fixes the core flaw where quality_score alone was used for version
        comparison: two versions with identical quality_score but very different
        actual measurement values were considered equal.
        """
        if prev is None:
            return True, "no previous version"

        curr_valid = sum(
            1 for v in self.measurements.values()
            if isinstance(v, (int, float)) and math.isfinite(v) and v != 0
        )
        prev_valid = sum(
            1 for v in prev.measurements.values()
            if isinstance(v, (int, float)) and math.isfinite(v) and v != 0
        )
        coverage_delta = curr_valid - prev_valid

        if coverage_delta < -1:
            return False, f"coverage dropped by {abs(coverage_delta)} targets ({prev_valid}->{curr_valid})"

        curr_sanity = self._count_sanity_violations()
        prev_sanity = prev._count_sanity_violations()
        if curr_sanity > prev_sanity:
            return False, f"new sanity violations ({curr_sanity} vs {prev_sanity})"

        shared_keys = set(self.measurements.keys()) & set(prev.measurements.keys())
        improvements = 0
        regressions = 0
        improvement_details = []
        regression_details = []

        for key in shared_keys:
            try:
                curr_val = float(self.measurements[key])
                prev_val = float(prev.measurements[key])
            except (TypeError, ValueError):
                continue

            if not (math.isfinite(curr_val) and math.isfinite(prev_val)):
                continue

            if any(kw in key for kw in _ATTRIBUTE_KEYWORDS):
                continue

            if curr_val == 0 or prev_val == 0:
                continue

            rel_change = (curr_val - prev_val) / abs(prev_val) if prev_val != 0 else 0

            if any(kw in key for kw in _THROUGHPUT_KEYWORDS):
                if rel_change > _IMPROVEMENT_THRESHOLD:
                    improvements += 1
                    improvement_details.append(
                        f"{key}: {prev_val:.1f}->{curr_val:.1f} (+{rel_change*100:.1f}%)"
                    )
                elif rel_change < -_IMPROVEMENT_THRESHOLD:
                    regressions += 1
                    regression_details.append(
                        f"{key}: {prev_val:.1f}->{curr_val:.1f} ({rel_change*100:.1f}%)"
                    )
            elif any(kw in key for kw in _LATENCY_KEYWORDS):
                if rel_change < -_IMPROVEMENT_THRESHOLD:
                    improvements += 1
                    improvement_details.append(
                        f"{key}: {prev_val:.1f}->{curr_val:.1f} ({rel_change*100:.1f}%)"
                    )
                elif rel_change > _IMPROVEMENT_THRESHOLD:
                    regressions += 1
                    regression_details.append(
                        f"{key}: {prev_val:.1f}->{curr_val:.1f} (+{rel_change*100:.1f}%)"
                    )

        net = improvements - regressions

        if net > 0:
            detail = improvement_details[:2]
            return True, f"net improvement +{improvements}/-{regressions} [{', '.join(detail)}]"

        if net < 0 and coverage_delta <= 0:
            detail = regression_details[:2]
            return False, f"net regression +{improvements}/-{regressions} [{', '.join(detail)}]"

        if coverage_delta > 0:
            return True, f"more coverage (+{coverage_delta}) with equal perf"

        quality_delta = self.quality_score - prev.quality_score
        if quality_delta >= -5.0:
            return True, f"similar quality (delta={quality_delta:.1f}), no regression"
        return False, f"quality dropped ({quality_delta:.1f}) with no improvement"

    def compute_combined_score(self, baseline: MeasurementVersion | None) -> float:
        """Compute a combined score for version ranking across all versions.

        Uses the FIRST version as baseline for improvement comparison,
        ensuring transitive ordering (all versions compared to same baseline).

        combined_score = quality_score + improvement_vs_baseline
        """
        base = self.quality_score

        if baseline is None or baseline is self:
            self.improvement_score = 0.0
            return base

        shared_keys = set(self.measurements.keys()) & set(baseline.measurements.keys())
        improvement_bonus = 0.0

        for key in shared_keys:
            try:
                curr_val = float(self.measurements[key])
                base_val = float(baseline.measurements[key])
            except (TypeError, ValueError):
                continue

            if not (math.isfinite(curr_val) and math.isfinite(base_val)):
                continue

            if any(kw in key for kw in _ATTRIBUTE_KEYWORDS):
                continue

            if curr_val == 0 or base_val == 0:
                continue

            rel_change = (curr_val - base_val) / abs(base_val) if base_val != 0 else 0

            if any(kw in key for kw in _THROUGHPUT_KEYWORDS):
                improvement_bonus += rel_change * _IMPROVEMENT_SCALE
            elif any(kw in key for kw in _LATENCY_KEYWORDS):
                improvement_bonus -= rel_change * _IMPROVEMENT_SCALE

        self.improvement_score = improvement_bonus
        return base + improvement_bonus

    def _count_sanity_violations(self) -> int:
        """Count the number of sanity violations in this version's measurements."""
        count = 0
        for key, val in self.measurements.items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                count += 1
                continue
            if not math.isfinite(fval):
                count += 1
                continue
            if fval == 0.0 and not any(k in key for k in _ATTRIBUTE_KEYWORDS):
                if not any(kw in key for kw in ("penalty", "conflict")):
                    # Zero value for throughput metrics means measurement FAILED
                    if "pct_of_peak" in key or "throughput.avg" in key:
                        count += 1
                    continue
            if ("pct_of_peak" in key or "throughput.avg" in key) and (fval > 100.0 or fval < 0.0):
                count += 1
                continue
            if "bytes" in key and "per_second" in key and fval > 2_000_000_000_000.0:
                count += 1
                continue
            if "sm_count" in key and (fval < 1 or fval > 200):
                count += 1
                continue
            if "bus_width" in key and (fval < 32 or fval > 8192):
                count += 1
                continue
            if "frequency_khz" in key and (fval < 100_000 or fval > 20_000_000):
                count += 1
        return count


@dataclass
class PipelineContext:
    """Mutable accumulator for pipeline execution state.

    Each stage reads from and writes to this context.
    The context is the single source of truth for inter-stage data flow.

    Memory Architecture (4 layers):
    - L0 (Permanent): Architecture info, target spec — never compressed
    - L1 (High): CodeGen measurements, binary paths — preserved across stages
    - L2 (Medium): MetricAnalysis results, error patterns — compressed on budget pressure
    - L3 (Low): Conversation history, Control Plane snapshots — aggressively compressed

    Version Control:
    - measurement_versions: list of MeasurementVersion snapshots
    - current_version_idx: index of the active version
    - Supports rollback to any previous version
    """

    prev_result: SubAgentResult | None = None
    prev_stage: PipelineStage | None = None
    code_gen_data: dict[str, Any] | None = None
    target_spec: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3
    rejection_history: list[dict[str, Any]] = field(default_factory=list)
    metric_feedback: list[dict[str, Any]] = field(default_factory=list)
    _stage_results: dict[str, SubAgentResult] = field(default_factory=dict)
    _stage_durations: dict[str, float] = field(default_factory=dict)

    # L0: Permanent memory — never compressed
    architecture_info: dict[str, Any] = field(default_factory=dict)
    # L1: High-priority memory — preserved across stages
    key_measurements: dict[str, Any] = field(default_factory=dict)
    binary_paths: list[str] = field(default_factory=list)
    # L2: Medium-priority memory — compressed on budget pressure
    stage_summaries: dict[str, str] = field(default_factory=dict)
    error_patterns: list[str] = field(default_factory=list)

    # Optimization targeting: specific targets CodeGen must re-optimize
    optimization_targets: list[dict[str, Any]] = field(default_factory=list)
    is_optimization_round: bool = False

    # Version control: measurement snapshots with rollback
    measurement_versions: list[MeasurementVersion] = field(default_factory=list)
    current_version_idx: int = -1

    def _compute_code_hash(self, code_gen_data: dict[str, Any] | None) -> str:
        """Compute a hash of the CUDA source code for version tracking."""
        if not code_gen_data:
            return ""
        source = ""
        if "tool_results" in code_gen_data:
            for tr in code_gen_data.get("tool_results", []):
                if isinstance(tr, dict) and "source" in tr:
                    source += str(tr["source"])
        if not source:
            for key in ("final_output", "code_gen_output"):
                if key in code_gen_data:
                    source += str(code_gen_data[key])[:2000]
        if not source:
            return f"v{len(self.measurement_versions)}"
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def _commit_measurement_version(self, measurements: dict[str, Any],
                                     code_hash: str, is_optimization: bool) -> MeasurementVersion:
        """Commit a new measurement version as an atomic snapshot.

        Instead of merging individual values, the entire measurement set
        is stored as one version. This prevents cross-version mixing.
        """
        version_num = len(self.measurement_versions)
        snapshot = MeasurementVersion(
            version=version_num,
            measurements=copy.deepcopy(measurements),
            code_hash=code_hash,
            timestamp=time.time(),
            is_optimization=is_optimization,
        )
        snapshot.compute_quality_score()
        self.measurement_versions.append(snapshot)
        self.current_version_idx = len(self.measurement_versions) - 1

        logger.info(
            "[PipelineContext] Committed measurement version #%d: %d measurements, "
            "quality_score=%.1f, quality_ok=%s, is_optimization=%s",
            version_num, len(measurements), snapshot.quality_score,
            snapshot.quality_ok, is_optimization,
        )
        return snapshot

    def rollback_to_best_version(self) -> dict[str, Any]:
        """Rollback to the best measurement version using combined scoring.

        Called when the current version has degraded quality (e.g., over-optimization).
        Returns the measurements from the best version.

        Uses combined_score = quality_score + improvement_vs_baseline to rank
        versions. This ensures that a version with slightly lower quality_score
        but significantly better measurements is preferred over a version with
        higher quality_score but mediocre measurements.
        """
        if not self.measurement_versions:
            return dict(self.key_measurements)

        baseline = self.measurement_versions[0]
        best_idx = 0
        best_combined = baseline.compute_combined_score(baseline)
        for i, ver in enumerate(self.measurement_versions[1:], 1):
            combined = ver.compute_combined_score(baseline)
            if combined > best_combined:
                best_combined = combined
                best_idx = i

        old_idx = self.current_version_idx
        self.current_version_idx = best_idx
        best_measurements = copy.deepcopy(self.measurement_versions[best_idx].measurements)

        best_ver = self.measurement_versions[best_idx]
        if old_idx != best_idx:
            old_ver = self.measurement_versions[old_idx] if 0 <= old_idx < len(self.measurement_versions) else None
            logger.info(
                "[PipelineContext] ROLLBACK: version %d -> %d "
                "(quality %.1f->%.1f, improvement %.1f->%.1f, combined %.1f->%.1f)",
                old_idx, best_idx,
                old_ver.quality_score if old_ver else 0, best_ver.quality_score,
                old_ver.improvement_score if old_ver else 0, best_ver.improvement_score,
                old_ver.compute_combined_score(baseline) if old_ver else 0, best_combined,
            )
            ncu_preserved = {}
            for k, v in self.key_measurements.items():
                if "pct_of_peak_sustained_elapsed" in k and isinstance(v, (int, float)) and v > 0.0:
                    ncu_preserved[k] = v
            self.key_measurements = best_measurements
            for k, v in ncu_preserved.items():
                if k not in self.key_measurements or (isinstance(self.key_measurements.get(k), (int, float)) and self.key_measurements[k] == 0.0):
                    self.key_measurements[k] = v
        else:
            logger.info(
                "[PipelineContext] Current version #%d is already the best "
                "(quality=%.1f, improvement=%.1f, combined=%.1f)",
                best_idx, best_ver.quality_score, best_ver.improvement_score, best_combined,
            )

        return best_measurements

    def get_current_measurements(self) -> dict[str, Any]:
        """Get measurements from the current active version."""
        if 0 <= self.current_version_idx < len(self.measurement_versions):
            return copy.deepcopy(self.measurement_versions[self.current_version_idx].measurements)
        return dict(self.key_measurements)

    def update(self, stage: PipelineStage, result: SubAgentResult) -> None:
        """Advance the context after a stage completes.
        
        Populates the layered memory architecture:
        - L0: Architecture info from any stage that detects it
        - L1: Key measurements and binary paths from CodeGen (version-controlled)
        - L2: Stage summaries and error patterns from all stages

        VERSION CONTROL: CodeGen measurements are committed as atomic snapshots.
        Each CodeGen pass produces a complete MeasurementVersion. If the new
        version has worse quality than the previous, we rollback automatically.
        """
        if stage == PipelineStage.CODE_GEN:
            self.code_gen_data = dict(result.data)
            code_hash = self._compute_code_hash(result.data)

            if "measurements" in result.data and isinstance(result.data["measurements"], dict):
                new_measurements = result.data["measurements"]

                prev_version = None
                if 0 <= self.current_version_idx < len(self.measurement_versions):
                    prev_version = self.measurement_versions[self.current_version_idx]

                snapshot = self._commit_measurement_version(
                    new_measurements, code_hash, self.is_optimization_round,
                )

                if prev_version is not None:
                    is_better, reason = snapshot.compare_with_previous(prev_version)
                    if not is_better:
                        logger.info(
                            "[PipelineContext] Version #%d is WORSE than #%d: %s. "
                            "quality_score %.1f->%.1f. Rolling back to best version.",
                            snapshot.version, prev_version.version, reason,
                            prev_version.quality_score, snapshot.quality_score,
                        )
                        best = self.rollback_to_best_version()
                        self.key_measurements = best
                    else:
                        merged = dict(self.key_measurements)
                        for k, v in new_measurements.items():
                            if "pct_of_peak_sustained_elapsed" in k and k in self.key_measurements:
                                existing_val = self.key_measurements[k]
                                if isinstance(existing_val, (int, float)) and existing_val > 0:
                                    continue
                            merged[k] = v
                        self.key_measurements = merged
                        logger.info(
                            "[PipelineContext] Version #%d is BETTER than #%d: %s. "
                            "quality_score %.1f->%.1f, improvement_score=%.1f",
                            snapshot.version, prev_version.version, reason,
                            prev_version.quality_score, snapshot.quality_score,
                            snapshot.improvement_score,
                        )
                else:
                    merged = dict(self.key_measurements)
                    for k, v in new_measurements.items():
                        if "pct_of_peak_sustained_elapsed" in k and k in self.key_measurements:
                            existing_val = self.key_measurements[k]
                            if isinstance(existing_val, (int, float)) and existing_val > 0:
                                continue
                        merged[k] = v
                    self.key_measurements = merged
                    logger.info(
                        "[PipelineContext] Updated key_measurements from version #%d (%d entries, first version)",
                        snapshot.version, len(new_measurements),
                    )

            if "binary_path" in result.data:
                bp = result.data["binary_path"]
                if isinstance(bp, str) and bp not in self.binary_paths:
                    self.binary_paths.append(bp)
            if "tool_results" in result.data and isinstance(result.data["tool_results"], list):
                for tr in result.data["tool_results"]:
                    if isinstance(tr, dict) and "binary_path" in tr:
                        bp = tr["binary_path"]
                        if isinstance(bp, str) and bp not in self.binary_paths:
                            self.binary_paths.append(bp)

        # L2: Record stage summary
        if result.is_success():
            summary = result.data.get("final_output", "")[:500] if result.data.get("final_output") else ""
            self.stage_summaries[stage.value] = summary
        else:
            error_msg = result.data.get("errors", result.data.get("error", "unknown"))
            self.error_patterns.append(f"{stage.value}: {str(error_msg)[:200]}")

        # L0: Extract architecture info if present
        if "arch" in result.data:
            self.architecture_info["gpu_arch"] = result.data["arch"]
        if "architecture" in result.data:
            self.architecture_info["gpu_architecture"] = result.data["architecture"]

        self.prev_result = result
        self.prev_stage = stage
        self._stage_results[stage.value] = result

    def append_history(self, role: str, content: str) -> None:
        """Append a conversation entry for cross-Stage context inheritance.

        Automatically trims to MAX_HISTORY_ENTRIES to prevent unbounded growth.
        """
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def get_stage_result(self, stage: PipelineStage) -> SubAgentResult | None:
        """Retrieve the result from a specific pipeline stage.

        Returns None if the stage has not yet completed.
        """
        return self._stage_results.get(stage.value)

    def get_history(self, limit: int = 20) -> list[dict[str, str]]:
        """Return the most recent conversation history entries."""
        return self.conversation_history[-limit:]

    def add_rejection(self, stage: str, concerns: list[str], suggested_fixes: list[str] | None = None) -> None:
        """Record a rejection event for iteration tracking."""
        self.rejection_history.append({
            "stage": stage,
            "concerns": concerns,
            "suggested_fixes": suggested_fixes or [],
            "iteration": self.iteration_count,
        })

    def add_metric_feedback(self, suggested_fixes: list[str], bottleneck_type: str = "",
                            bottleneck_sub_type: str = "", recommendations: list[str] | None = None) -> None:
        """Record MetricAnalysis feedback for CodeGen optimization.

        This enables the MetricAnalysis → CodeGen feedback loop:
        MetricAnalysis identifies bottlenecks and generates recommendations,
        which are then injected into CodeGen's task prompt for optimization.

        Automatically trims to last 10 entries to prevent unbounded growth.
        """
        self.metric_feedback.append({
            "stage": "metric_analysis",
            "suggested_fixes": suggested_fixes,
            "bottleneck_type": bottleneck_type,
            "bottleneck_sub_type": bottleneck_sub_type,
            "recommendations": recommendations or [],
            "iteration": self.iteration_count,
        })
        if len(self.metric_feedback) > 10:
            self.metric_feedback = self.metric_feedback[-10:]

    def can_retry(self) -> bool:
        """Check if another iteration is allowed."""
        return self.iteration_count < self.max_iterations

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.iteration_count += 1

    def set_optimization_targets(self, targets: list[dict[str, Any]]) -> None:
        """Set specific targets that CodeGen must re-optimize.

        Each entry contains:
        - target: the measurement target name
        - bottleneck_type: the identified bottleneck
        - optimization_strategy: specific optimization approach to apply
        - current_value: the current measured value
        """
        self.optimization_targets = targets
        self.is_optimization_round = bool(targets)
        logger.info(
            "[PipelineContext] Set %d optimization targets: %s",
            len(targets),
            [t.get("target", "?") for t in targets],
        )

    def get_optimization_targets(self) -> list[dict[str, Any]]:
        """Return the list of targets that need re-optimization."""
        return self.optimization_targets

    def clear_optimization_targets(self) -> None:
        """Clear optimization targets after CodeGen has processed them."""
        self.optimization_targets = []
        self.is_optimization_round = False

    def is_optimization_converged(self) -> bool:
        """Detect whether optimization iterations have converged.

        Convergence is detected via THREE independent signals:

        Signal 1 — Semantic convergence with no actionable feedback:
          The last two MetricAnalysis rounds report the SAME bottleneck_type
          AND neither produced concrete suggested_fixes or recommendations.

        Signal 2 — Diminishing returns:
          The last two CodeGen optimization rounds show a >50% decrease
          in execution time, indicating the LLM made no substantial changes.

        Signal 3 — Repeated bottleneck WITH actionable feedback but CodeGen
          already attempted optimization (3+ iterations with same bottleneck).
          This prevents premature convergence when MetricAnalysis gives
          recommendations but CodeGen hasn't fully applied them yet.

        IMPORTANT: If MetricAnalysis provides actionable recommendations,
        we do NOT converge just because the bottleneck type is the same.
        The bottleneck may persist because CodeGen's optimization was
        insufficient, not because optimization is impossible.
        """
        if len(self.metric_feedback) < 2:
            return False

        last = self.metric_feedback[-1]
        prev = self.metric_feedback[-2]

        same_bottleneck = (
            last.get("bottleneck_type", "") == prev.get("bottleneck_type", "")
            and last.get("bottleneck_type", "") != ""
        )

        last_has_fixes = bool(
            last.get("suggested_fixes", []) or last.get("recommendations", [])
        )
        prev_has_fixes = bool(
            prev.get("suggested_fixes", []) or prev.get("recommendations", [])
        )
        no_fixes = not last_has_fixes and not prev_has_fixes

        semantic_converged = same_bottleneck and no_fixes

        if same_bottleneck and last_has_fixes:
            if self.iteration_count >= self.max_iterations:
                logger.info(
                    "[PipelineContext] Convergence: same bottleneck '%s' with fixes but max iterations reached",
                    last.get("bottleneck_type", ""),
                )
                return True
            logger.info(
                "[PipelineContext] NOT converged: same bottleneck '%s' but MetricAnalysis still provides actionable fixes (iteration %d/%d)",
                last.get("bottleneck_type", ""),
                self.iteration_count,
                self.max_iterations,
            )
            return False

        diminishing_returns = False
        cg_durations = self._stage_durations
        cg_keys = sorted(k for k in cg_durations if k.startswith("code_gen_opt_"))
        if len(cg_keys) >= 2:
            last_dur = cg_durations[cg_keys[-1]]
            prev_dur = cg_durations[cg_keys[-2]]
            if prev_dur > 0 and last_dur < prev_dur * 0.5 and last_dur < 30.0:
                diminishing_returns = True

        return semantic_converged or diminishing_returns

    def record_stage_duration(self, stage: str, duration: float, iteration: int = 0) -> None:
        """Record the execution duration of a pipeline stage.

        For iterative stages, includes iteration number in the key
        to enable diminishing-returns detection.
        """
        if iteration > 0 and stage == "code_gen":
            key = f"code_gen_opt_{iteration}"
        else:
            key = f"{stage}_{iteration}" if iteration > 0 else stage
        self._stage_durations[key] = duration

    def get_feedback_for_codegen(self) -> dict[str, Any] | None:
        """Extract combined feedback from Verification and MetricAnalysis for CodeGen retry.

        Merges:
        - Verification rejection concerns and suggested fixes
        - MetricAnalysis bottleneck identification and optimization recommendations

        Returns None if no feedback is available.
        """
        feedback: dict[str, Any] = {}

        if self.rejection_history:
            last = self.rejection_history[-1]
            feedback["concerns"] = last.get("concerns", [])
            feedback["suggested_fixes"] = last.get("suggested_fixes", [])
            feedback["iteration"] = last.get("iteration", 0)

        if self.metric_feedback:
            last_metric = self.metric_feedback[-1]
            feedback["metric_recommendations"] = last_metric.get("recommendations", [])
            feedback["metric_suggested_fixes"] = last_metric.get("suggested_fixes", [])
            feedback["bottleneck_type"] = last_metric.get("bottleneck_type", "")
            feedback["bottleneck_sub_type"] = last_metric.get("bottleneck_sub_type", "")

        return feedback if feedback else None

    def bubble_codegen_data(self, result: SubAgentResult) -> SubAgentResult:
        """Propagate CodeGen measurements into a downstream result.

        Ensures MetricAnalysis and Verification can see the full chain
        of CodeGen measurements even if they don't produce their own.
        
        CRITICAL FIX: Always propagate measurements, even if downstream stage failed.
        This ensures measurements are available for final results even when
        MetricAnalysis or Verification fails.
        """
        if not self.code_gen_data:
            return result

        # Always propagate measurements, regardless of result status
        # Always merge measurements - CodeGen's are the primary source
        # CRITICAL: Merge even if result is failed, measurements must be preserved
        if "measurements" in self.code_gen_data:
            existing = result.data.get("measurements", {})
            if isinstance(existing, dict):
                for k, v in self.code_gen_data["measurements"].items():
                    if k not in existing:
                        existing[k] = v
                result.data["measurements"] = existing
            else:
                result.data["measurements"] = self.code_gen_data["measurements"]

        # Only propagate other data if result is successful
        if result.is_success():
            carry_keys = [
                "code_gen_output", "tool_results",
                "code_gen_final_output",
            ]
            for key in carry_keys:
                src_key = "final_output" if key == "code_gen_final_output" else key
                if key not in result.data and src_key in self.code_gen_data:
                    result.data[key] = self.code_gen_data[src_key]

        return result

    def _get_best_available_measurements(self) -> dict[str, Any]:
        """Get the best available measurements from all sources.

        Fallback chain:
        1. self.key_measurements (may have been rolled back to best version)
        2. Best measurement version from self.measurement_versions
        3. self.code_gen_data["measurements"] (last CodeGen pass)
        4. Empty dict
        """
        if self.key_measurements:
            return dict(self.key_measurements)

        if self.measurement_versions:
            baseline = self.measurement_versions[0]
            best_idx = 0
            best_combined = baseline.compute_combined_score(baseline)
            for i, ver in enumerate(self.measurement_versions[1:], 1):
                combined = ver.compute_combined_score(baseline)
                if combined > best_combined:
                    best_combined = combined
                    best_idx = i
            best = self.measurement_versions[best_idx]
            if best.measurements:
                logger.info(
                    "[PipelineContext] key_measurements empty, using best version #%d (%d measurements, quality=%.1f)",
                    best_idx, len(best.measurements), best.quality_score,
                )
                return copy.deepcopy(best.measurements)

        if self.code_gen_data and "measurements" in self.code_gen_data:
            cg_meas = self.code_gen_data["measurements"]
            if isinstance(cg_meas, dict) and cg_meas:
                logger.info("[PipelineContext] key_measurements empty, using code_gen_data measurements (%d entries)", len(cg_meas))
                return dict(cg_meas)

        return {}

    def assemble_final_result(self, result: SubAgentResult) -> SubAgentResult:
        """Merge CodeGen data and layered memory into the final pipeline result.
        
        Uses the 4-layer memory architecture:
        - L0: Architecture info always included
        - L1: Key measurements and binary paths always included
        - L2: Stage summaries and error patterns included for debugging
        - L3: Conversation history excluded from final result (too verbose)
        
        CRITICAL: Measurements MUST always be included, even if pipeline failed.
        Uses _get_best_available_measurements() for robust fallback when
        key_measurements is empty (e.g., after rejected verification).
        """
        best_measurements = self._get_best_available_measurements()

        if not self.code_gen_data:
            if self.architecture_info:
                result.data["architecture_info"] = dict(self.architecture_info)
            if best_measurements:
                existing = result.data.get("measurements", {})
                if isinstance(existing, dict):
                    for k, v in best_measurements.items():
                        if k not in existing:
                            existing[k] = v
                    result.data["measurements"] = existing
                else:
                    result.data["measurements"] = best_measurements
                result.data["key_measurements"] = dict(best_measurements)
            if self.binary_paths:
                result.data["binary_paths"] = list(self.binary_paths)
            return result

        merge_keys = [
            "analysis_method", "code_gen_output",
            "tool_results", "binary_path",
            "code_gen_final_output",
        ]
        for key in merge_keys:
            if key in self.code_gen_data and key not in result.data:
                result.data[key] = self.code_gen_data[key]

        if best_measurements:
            existing = result.data.get("measurements", {})
            if isinstance(existing, dict):
                for k, v in best_measurements.items():
                    if k not in existing:
                        existing[k] = v
                    elif "pct_of_peak_sustained_elapsed" in k:
                        old_val = existing[k]
                        existing[k] = v
                        if old_val != v:
                            logger.info(
                                "[PipelineContext] assemble_final_result: NCU override %s = %.2f (was %.2f from LLM)",
                                k, v, old_val,
                            )
                result.data["measurements"] = existing
            else:
                result.data["measurements"] = best_measurements
            result.data["key_measurements"] = dict(best_measurements)

        if self.architecture_info:
            result.data["architecture_info"] = dict(self.architecture_info)

        if self.binary_paths:
            result.data["binary_paths"] = list(self.binary_paths)

        if self.stage_summaries:
            result.data["stage_summaries"] = dict(self.stage_summaries)

        if self.error_patterns:
            result.data["error_patterns"] = list(self.error_patterns)

        if self.measurement_versions:
            result.data["num_measurement_versions"] = len(self.measurement_versions)

        return result
