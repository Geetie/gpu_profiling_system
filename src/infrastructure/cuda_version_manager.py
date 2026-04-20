"""CUDA Code Version Manager — Systematic code iteration tracking.

Provides version control for generated CUDA C++ source code with:
- Automatic versioning on each compile/execute cycle
- Performance baseline tracking and comparison
- Iteration history with measurable improvements
- Code quality scoring based on compilation success rate
- Feedback-driven optimization loop support

Integration points:
- CodeGenAgent: Records each generated kernel version
- MetricAnalysisAgent: Provides performance feedback for next iteration
- PipelineContext: Maintains cross-iteration state
- StageExecutor: Uses version history for quality assessment

Usage:
    from src.infrastructure.cuda_version_manager import CUDAVersionManager
    
    manager = CUDAVersionManager(state_dir=".state")
    
    # Record a new code generation attempt
    version_id = manager.record_generation(
        target="dram_latency_cycles",
        source_code=cuda_source,
        iteration=1,
        metadata={"method": "pointer_chasing", "working_set_mb": 128}
    )
    
    # Record compilation result
    manager.record_compilation(version_id, success=True, warnings=2)
    
    # Record execution result with measurements
    manager.record_execution(
        version_id,
        success=True,
        measurements={"dram_latency_cycles": 485.0},
        execution_time_ms=12.5
    )
    
    # Get performance trend
    trend = manager.get_performance_trend("dram_latency_cycles")
    print(f"Improvement: {trend['improvement_pct']:.1f}%")
"""
from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class CodeVersion:
    """Represents a single version of generated CUDA code."""
    version_id: str
    target: str
    iteration: int
    timestamp: str
    source_code_hash: str
    source_code_length: int
    compilation_status: Optional[str] = None  # "success", "error", "success_with_warning"
    compilation_warnings: int = 0
    compilation_errors: str = ""
    execution_status: Optional[str] = None
    measurements: dict[str, float] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    quality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    feedback_applied: list[str] = field(default_factory=list)


@dataclass
class PerformanceTrend:
    """Performance trend analysis for a target."""
    target: str
    total_versions: int
    successful_versions: int
    best_measurement: Optional[float]
    worst_measurement: Optional[float]
    latest_measurement: Optional[float]
    improvement_pct: float
    stability_score: float  # 0.0 (unstable) → 1.0 (very stable)
    avg_compilation_time_s: float
    recommendations: list[str]


class CUDAVersionManager:
    """Manages CUDA code versions with systematic tracking.

    Features:
    1. Version Control: Each code generation gets a unique version ID
    2. Baseline Tracking: First successful measurement becomes baseline
    3. Trend Analysis: Track improvements across iterations
    4. Quality Scoring: Composite score based on multiple factors
    5. Feedback Integration: Record which MetricAnalysis suggestions were applied

    Storage: JSON file in state_dir for persistence across sessions.
    """

    def __init__(self, state_dir: str = ".state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.state_dir / "cuda_versions.json"
        self.versions: dict[str, CodeVersion] = {}
        self._load_versions()

    def _load_versions(self) -> None:
        """Load versions from persistent storage."""
        if not self.version_file.exists():
            return
        
        try:
            with open(self.version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for version_data in data.get("versions", []):
                version = CodeVersion(**version_data)
                self.versions[version.version_id] = version
                
        except Exception as e:
            print(f"[CUDAVersionManager] Warning: Failed to load versions: {e}")

    def _save_versions(self) -> None:
        """Save versions to persistent storage."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "total_versions": len(self.versions),
                "versions": [asdict(v) for v in self.versions.values()]
            }
            
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[CUDAVersionManager] Error: Failed to save versions: {e}")

    def generate_version_id(self, target: str, iteration: int) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(f"{target}{iteration}{time.time()}".encode()).hexdigest()[:8]
        return f"v{iteration:03d}_{target}_{timestamp}_{hash_suffix}"

    @staticmethod
    def compute_source_hash(source_code: str) -> str:
        """Compute SHA-256 hash of source code for deduplication."""
        return hashlib.sha256(source_code.encode('utf-8')).hexdigest()[:16]

    def record_generation(
        self,
        target: str,
        source_code: str,
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a new code generation attempt.

        Args:
            target: Measurement target name (e.g., "dram_latency_cycles")
            source_code: Generated CUDA C++ source code
            iteration: Iteration number (1-based)
            metadata: Additional info about the generation (methodology, etc.)

        Returns:
            version_id: Unique identifier for this version
        """
        version_id = self.generate_version_id(target, iteration)
        
        version = CodeVersion(
            version_id=version_id,
            target=target,
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            source_code_hash=self.compute_source_hash(source_code),
            source_code_length=len(source_code),
            metadata=metadata or {},
        )
        
        self.versions[version_id] = version
        self._save_versions()
        
        print(f"[CUDAVersionManager] 📝 Recorded v{iteration:03d} for {target} "
              f"(hash={version.source_code_hash[:8]}..., size={len(source_code)} bytes)")
        
        return version_id

    def record_compilation(
        self,
        version_id: str,
        success: bool,
        warnings: int = 0,
        errors: str = "",
    ) -> None:
        """Record compilation result for a version.

        Args:
            version_id: Version identifier from record_generation()
            success: Whether compilation succeeded
            warnings: Number of compiler warnings
            errors: Compiler error message (if failed)
        """
        if version_id not in self.versions:
            print(f"[CUDAVersionManager] ⚠️ Unknown version_id: {version_id}")
            return
        
        version = self.versions[version_id]
        version.compilation_status = "success" if success else "error"
        if success and warnings > 0:
            version.compilation_status = "success_with_warning"
        version.compilation_warnings = warnings
        version.compilation_errors = errors
        
        self._save_versions()
        
        status_icon = "✅" if success else "❌"
        print(f"[CUDAVersionManager] {status_icon} Compilation result for {version.target}: "
              f"{version.compilation_status} ({warnings} warnings)")

    def record_execution(
        self,
        version_id: str,
        success: bool,
        measurements: dict[str, float] | None = None,
        execution_time_ms: float = 0.0,
    ) -> None:
        """Record execution result with measurements.

        Args:
            version_id: Version identifier
            success: Whether execution completed successfully
            measurements: Dictionary of measured values (e.g., {"latency": 485.0})
            execution_time_ms: Kernel execution time in milliseconds
        """
        if version_id not in self.versions:
            print(f"[CUDAVersionManager] ⚠️ Unknown version_id: {version_id}")
            return
        
        version = self.versions[version_id]
        version.execution_status = "success" if success else "failed"
        version.measurements = measurements or {}
        version.execution_time_ms = execution_time_ms
        version.quality_score = self._compute_quality_score(version)
        
        self._save_versions()
        
        status_icon = "✅" if success else "❌"
        meas_str = ", ".join(f"{k}={v}" for k, v in (measurements or {}).items())
        print(f"[CUDAVersionManager] {status_icon} Execution result for {version.target}: "
              f"{meas_str} ({execution_time_ms:.2f}ms)")

    def apply_feedback(
        self,
        version_id: str,
        feedback_items: list[str],
    ) -> None:
        """Record which MetricAnalysis feedback items were applied.

        This creates an audit trail showing the feedback loop in action:
        MetricAnalysis identifies bottleneck → suggests fix → CodeGen applies it → re-measure

        Args:
            version_id: Version that incorporated the feedback
            feedback_items: List of suggestion IDs or descriptions that were applied
        """
        if version_id not in self.versions:
            return
        
        version = self.versions[version_id]
        version.feedback_applied.extend(feedback_items)
        self._save_versions()
        
        print(f"[CUDAVersionManager] 🔄 Applied {len(feedback_items)} feedback items to "
              f"{version.target} v{version.iteration:03d}")

    def get_performance_trend(self, target: str) -> PerformanceTrend:
        """Analyze performance trend for a specific target.

        Returns comprehensive trend analysis including:
        - Improvement percentage from first to last measurement
        - Stability score (consistency across iterations)
        - Recommendations for further optimization

        Args:
            target: Target name to analyze

        Returns:
            PerformanceTrend object with analysis results
        """
        target_versions = [
            v for v in self.versions.values()
            if v.target == target and v.execution_status == "success" and v.measurements
        ]
        
        if not target_versions:
            return PerformanceTrend(
                target=target,
                total_versions=0,
                successful_versions=0,
                best_measurement=None,
                worst_measurement=None,
                latest_measurement=None,
                improvement_pct=0.0,
                stability_score=0.0,
                avg_compilation_time_s=0.0,
                recommendations=["No successful measurements recorded yet"],
            )
        
        # Extract primary measurement (first key in measurements dict)
        all_measurements = []
        for v in target_versions:
            if v.measurements:
                first_key = list(v.measurements.keys())[0]
                all_measurements.append((v.iteration, v.measurements.get(first_key)))
        
        if not all_measurements:
            return PerformanceTrend(
                target=target,
                total_versions=len(target_versions),
                successful_versions=0,
                best_measurement=None,
                worst_measurement=None,
                latest_measurement=None,
                improvement_pct=0.0,
                stability_score=0.0,
                avg_compilation_time_s=0.0,
                recommendations=["Versions exist but no valid measurements"],
            )
        
        measurements_only = [m for _, m in all_measurements]
        iterations = [i for i, _ in all_measurements]
        
        best = min(measurements_only)
        worst = max(measurements_only)
        latest = measurements_only[-1]
        first = measurements_only[0]
        
        # Calculate improvement (negative is good for latency metrics)
        if first != 0:
            improvement_pct = ((first - latest) / first) * 100
        else:
            improvement_pct = 0.0
        
        # Calculate stability (coefficient of variation)
        if len(measurements_only) > 1 and sum(measurements_only) > 0:
            mean = sum(measurements_only) / len(measurements_only)
            variance = sum((m - mean) ** 2 for m in measurements_only) / len(measurements_only)
            std_dev = variance ** 0.5
            cv = std_dev / mean if mean > 0 else 1.0
            stability_score = max(0.0, 1.0 - cv)  # Lower CV = higher stability
        else:
            stability_score = 1.0 if len(measurements_only) == 1 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            target, best, worst, latest, improvement_pct, stability_score, len(target_versions)
        )
        
        return PerformanceTrend(
            target=target,
            total_versions=len([v for v in self.versions.values() if v.target == target]),
            successful_versions=len(target_versions),
            best_measurement=best,
            worst_measurement=worst,
            latest_measurement=latest,
            improvement_pct=improvement_pct,
            stability_score=stability_score,
            avg_compilation_time_s=sum(v.execution_time_ms for v in target_versions) / max(1, len(target_versions)) / 1000.0,
            recommendations=recommendations,
        )

    def _compute_quality_score(self, version: CodeVersion) -> float:
        """Compute composite quality score for a version.

        Scoring criteria (total 10 points):
        - Compilation success: +3 points
        - No warnings: +1 point (warnings reduce by 0.2 each, max -1)
        - Execution success: +3 points
        - Has measurements: +2 points
        - Reasonable code length (>100 chars): +1 point

        Returns:
            Score from 0.0 to 1.0 (normalized)
        """
        score = 0.0
        
        # Compilation (4 points max)
        if version.compilation_status == "success":
            score += 3.0
        elif version.compilation_status == "success_with_warning":
            score += 2.5
            score -= min(1.0, version.compilation_warnings * 0.2)
        
        # Code length (1 point)
        if version.source_code_length > 100:
            score += 1.0
        elif version.source_code_length > 50:
            score += 0.5
        
        # Execution (3 points)
        if version.execution_status == "success":
            score += 3.0
        
        # Measurements (2 points)
        if version.measurements:
            score += 2.0
        
        # Normalize to 0.0-1.0
        return min(1.0, score / 9.0)

    def _generate_recommendations(
        self,
        target: str,
        best: float,
        worst: float,
        latest: float,
        improvement_pct: float,
        stability_score: float,
        num_versions: int,
    ) -> list[str]:
        """Generate actionable recommendations based on trend analysis."""
        recommendations = []
        
        # Stability-based recommendations
        if stability_score < 0.7:
            recommendations.append(
                f"⚠️ Low stability (score={stability_score:.2f}): "
                f"Consider running more iterations to establish reliable baseline"
            )
        elif stability_score >= 0.9:
            recommendations.append(
                f"✅ High stability (score={stability_score:.2f}): "
                f"Measurements are consistent and reliable"
            )
        
        # Improvement-based recommendations
        if improvement_pct > 10:
            recommendations.append(
                f"🎉 Excellent improvement (+{improvement_pct:.1f}%): "
                f"Optimization strategy is working well"
            )
        elif improvement_pct < -10:
            recommendations.append(
                f"⚠️ Performance degraded ({improvement_pct:+.1f}%): "
                f"Review recent changes for regressions"
            )
        elif abs(improvement_pct) <= 5 and num_versions >= 3:
            recommendations.append(
                f"📊 Plateau reached (±5% over {num_versions} versions): "
                f"Consider alternative approaches or accept current optimum"
            )
        
        # Range-based recommendations
        if worst > 0:
            range_pct = ((worst - best) / best) * 100
            if range_pct > 20:
                recommendations.append(
                    f"📈 Wide variation ({range_pct:.1f}% between best/worst): "
                    f"Investigate outliers and measurement methodology"
                )
        
        # Iteration count recommendation
        if num_versions < 3:
            recommendations.append(
                f"💡 Limited data ({num_versions} versions): "
                f"More iterations needed for statistically significant conclusions"
            )
        
        if not recommendations:
            recommendations.append("✅ Performance looks good — continue monitoring")
        
        return recommendations

    def get_latest_version(self, target: str) -> CodeVersion | None:
        """Get the most recent version for a target."""
        target_versions = [
            v for v in self.versions.values()
            if v.target == target
        ]
        
        if not target_versions:
            return None
        
        return max(target_versions, key=lambda v: v.timestamp)

    def get_best_version(self, target: str) -> CodeVersion | None:
        """Get the version with the best quality score for a target."""
        target_versions = [
            v for v in self.versions.values()
            if v.target == target and v.quality_score > 0
        ]
        
        if not target_versions:
            return None
        
        return max(target_versions, key=lambda v: v.quality_score)

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all targets.

        Returns:
            Dictionary with aggregate statistics for dashboard/reporting
        """
        targets = set(v.target for v in self.versions.values())
        
        stats = {
            "total_versions": len(self.versions),
            "unique_targets": len(targets),
            "targets": {},
            "last_updated": datetime.now().isoformat(),
        }
        
        for target in sorted(targets):
            trend = self.get_performance_trend(target)
            latest = self.get_latest_version(target)
            
            stats["targets"][target] = {
                "total_versions": trend.total_versions,
                "successful_versions": trend.successful_versions,
                "latest_measurement": trend.latest_measurement,
                "improvement_pct": round(trend.improvement_pct, 2),
                "stability_score": round(trend.stability_score, 3),
                "quality_score": round(latest.quality_score, 3) if latest else 0.0,
                "recommendations": trend.recommendations[:3],  # Top 3 only
            }
        
        return stats

    def export_history(self, filepath: str | Path) -> None:
        """Export full version history to external file for audit/analysis.

        Args:
            filepath: Output file path (JSON format)
        """
        filepath = Path(filepath)
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_summary_stats(),
            "versions": [asdict(v) for v in self.versions.values()],
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"[CUDAVersionManager] 📤 Exported {len(self.versions)} versions to {filepath}")
