"""Feedback Loop Enhancer — Bridges MetricAnalysis → CodeGen optimization cycle.

Implements a systematic feedback mechanism where:
1. MetricAnalysis identifies performance bottlenecks and generates recommendations
2. FeedbackEnhancer prioritizes and formats recommendations for CodeGen
3. CodeGen receives targeted guidance in its next iteration
4. VersionManager tracks which suggestions were applied and their impact

This closes the loop: Measure → Analyze → Suggest → Improve → Re-measure

Integration:
- Called by StageExecutor after MetricAnalysis completes
- Modifies PipelineContext.metric_feedback for next CodeGen iteration
- Works with CUDAVersionManager to track improvement history

Design Principles:
- Actionable: Only include suggestions CodeGen can actually implement
- Prioritized: Most impactful suggestions first (token budget awareness)
- Traceable: Each suggestion has ID for tracking in version history
- Adaptive: Adjusts suggestion detail based on available context
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OptimizationSuggestion:
    """A single actionable optimization suggestion.
    
    Attributes:
        id: Unique identifier for tracking (e.g., "mem_001")
        category: Bottleneck type (memory_bound, compute_bound, latency_bound, etc.)
        sub_category: Specific bottleneck subtype (dram, l2, l1, fp32, tensor_core)
        priority: Importance level 1-5 (5 = critical)
        title: Short human-readable description
        description: Detailed explanation of the issue
        code_hint: Specific code pattern or API to use (for LLM consumption)
        expected_impact: Estimated performance improvement percentage
        complexity: Implementation difficulty (low/medium/high)
        applicable_architectures: List of compute capabilities this applies to
        source: Where this came from ("metric_analysis", "verification", "best_practice")
    """
    id: str
    category: str
    sub_category: str
    priority: int  # 1-5
    title: str
    description: str
    code_hint: str = ""
    expected_impact: float = 0.0  # Percentage improvement
    complexity: str = "medium"  # low, medium, high
    applicable_architectures: list[str] = field(default_factory=list)
    source: str = "metric_analysis"
    
    def format_for_llm(self, max_tokens: int = 200) -> str:
        """Format suggestion for injection into LLM prompt.
        
        Optimized for token efficiency while preserving actionability.
        
        Args:
            max_tokens: Approximate token budget for this suggestion
            
        Returns:
            Formatted string ready for system/user message injection
        """
        parts = []
        
        # Priority indicator
        priority_icon = {5: "🔴", 4: "🟠", 3: "🟡", 2: "🟢", 1: "⚪"}.get(self.priority, "⚪")
        parts.append(f"{priority_icon} [{self.id}] {self.title}")
        
        # Description (always included)
        if self.description:
            parts.append(f"   Issue: {self.description}")
        
        # Code hint (high value for LLM)
        if self.code_hint:
            parts.append(f"   Fix: {self.code_hint}")
        
        # Expected impact (motivational)
        if self.expected_impact > 0:
            parts.append(f"   Impact: ~{self.expected_impact:.0f}% improvement")
        
        result = "\n".join(parts)
        
        # Truncate if over budget (keep priority and code_hint)
        if len(result) > max_tokens * 4:  # Rough token estimate
            return f"{priority_icon} [{self.id}] {self.title}\n   Fix: {self.code_hint}"
        
        return result


@dataclass
class FeedbackReport:
    """Complete feedback report from MetricAnalysis to CodeGen.
    
    Contains all analyzed data and prioritized suggestions.
    """
    target: str
    stage: str  # "metric_analysis" or "verification"
    measurements: dict[str, float]
    bottlenecks_identified: list[str]
    suggestions: list[OptimizationSuggestion]
    overall_verdict: str  # "ACCEPT", "REJECT", or "CONDITIONAL_ACCEPT"
    confidence_score: float  # 0.0 - 1.0
    raw_analysis_text: str = ""
    additional_context: dict[str, Any] = field(default_factory=dict)


class FeedbackEnhancer:
    """Enhances and manages the MetricAnalysis → CodeGen feedback loop.
    
    Responsibilities:
    1. Parse MetricAnalysis output into structured suggestions
    2. Prioritize suggestions based on impact and feasibility
    3. Filter suggestions for current GPU architecture
    4. Format suggestions for optimal LLM comprehension
    5. Track suggestion application and results
    
    Usage:
        enhancer = FeedbackEnhancer()
        
        # After MetricAnalysis completes
        report = enhancer.create_feedback_report(
            target="dram_latency_cycles",
            metric_analysis_output=analysis_text,
            measurements={"dram_latency_cycles": 485.0},
            compute_capability="sm_60",
        )
        
        # Get formatted suggestions for CodeGen prompt
        prompt_injection = enhancer.format_for_codegen(report, max_suggestions=3)
    """

    def __init__(self):
        # Common bottleneck patterns to detect in analysis text
        self._bottleneck_patterns = {
            r"(?i)(memory.*bound|bandwidth.*limited|dram.*bottleneck)": (
                "memory_bound", "dram", 4
            ),
            r"(?i)(l2.*cache.*miss|l2.*thrashing|l2.*capacity)": (
                "cache_capacity", "l2", 3
            ),
            r"(?i)(shared.*memory.*bank.*conflict|bank.*conflict)": (
                "memory_bound", "shared_memory", 3
            ),
            r"(?i)(register.*spill|local.*memory.*usage)": (
                "compute_bound", "fp32", 2
            ),
            r"(?i)(low.*occupancy|low.*utilization|warp.*stall)": (
                "latency_bound", "occupancy", 3
            ),
            r"(?i)(instruction.*cache|fetch.*bottleneck)": (
                "compute_bound", "instruction", 2
            ),
            r"(?i)(coalescing.*issue|unaligned.*access|strided.*access)": (
                "memory_bound", "global_memory", 4
            ),
        }

    def create_feedback_report(
        self,
        target: str,
        metric_analysis_output: str,
        measurements: dict[str, float],
        compute_capability: str | None = None,
        stage: str = "metric_analysis",
        raw_measurements: dict[str, float] | None = None,
        binary_paths: list[str] | None = None,
    ) -> FeedbackReport:
        """Create a comprehensive feedback report from analysis output.

        Parses unstructured LLM analysis text into actionable suggestions.

        Args:
            target: Measurement target name
            metric_analysis_output: Text output from MetricAnalysis agent
            measurements: Dictionary of measured values
            compute_capability: Current GPU architecture (optional)
            stage: Which stage produced this analysis
            raw_measurements: Additional raw metrics if available
            binary_paths: List of profiled binaries

        Returns:
            FeedbackReport with structured suggestions
        """
        suggestions = self._extract_suggestions(
            metric_analysis_output, target, compute_capability
        )
        
        bottlenecks = self._identify_bottlenecks(
            metric_analysis_output, measurements
        )
        
        verdict, confidence = self._determine_verdict(
            measurements, suggestions, metric_analysis_output
        )
        
        report = FeedbackReport(
            target=target,
            stage=stage,
            measurements=measurements,
            bottlenecks_identified=bottlenecks,
            suggestions=suggestions,
            overall_verdict=verdict,
            confidence_score=confidence,
            raw_analysis_text=metric_analysis_output[:2000],  # Truncate very long outputs
            additional_context={
                "compute_capability": compute_capability,
                "binary_count": len(binary_paths) if binary_paths else 0,
                "raw_metrics_available": raw_measurements is not None,
            },
        )
        
        print(f"[FeedbackEnhancer] 📊 Created report for {target}: "
              f"{len(suggestions)} suggestions, verdict={verdict}")
        
        return report

    def _extract_suggestions(
        self,
        analysis_text: str,
        target: str,
        compute_capability: str | None = None,
    ) -> list[OptimizationSuggestion]:
        """Extract optimization suggestions from analysis text.

        Uses pattern matching to find structured recommendations in LLM output.
        Also applies domain knowledge to generate suggestions based on detected patterns.

        Args:
            analysis_text: MetricAnalysis agent's output
            target: Target being measured
            compute_capability: Current GPU architecture

        Returns:
            List of OptimizationSuggestion objects sorted by priority
        """
        suggestions = []
        suggestion_id = 0
        
        # Pattern 1: Explicit recommendation lists
        # Look for bullet points or numbered lists with action verbs
        recommendation_patterns = [
            r'(?:^|\n)\s*[-•*]\s+(.+)',  # Bullet points
            r'(?:^|\n)\s*\d+\.\s+(.+)',   # Numbered lists
            r'(?i)(?:recommend|suggest|should|consider|use|implement)\s+(.+)',
        ]
        
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, analysis_text, re.MULTILINE)
            for match in matches:
                suggestion_id += 1
                
                # Classify the suggestion
                category, sub_cat, priority = self._classify_suggestion(match)
                
                # Generate code hint based on classification
                code_hint = self._generate_code_hint(category, sub_cat, match)
                
                # Estimate impact
                impact = self._estimate_impact(priority, category)
                
                suggestion = OptimizationSuggestion(
                    id=f"sug_{suggestion_id:03d}",
                    category=category,
                    sub_category=sub_cat,
                    priority=priority,
                    title=self._extract_title(match),
                    description=match.strip(),
                    code_hint=code_hint,
                    expected_impact=impact,
                    complexity="medium",
                    applicable_architectures=[compute_capability] if compute_capability else [],
                    source="metric_analysis",
                )
                
                suggestions.append(suggestion)
        
        # Pattern 2: Detect implicit issues from bottleneck keywords
        for pattern, (cat, sub_cat, base_priority) in self._bottleneck_patterns.items():
            if re.search(pattern, analysis_text) and not any(
                s.category == cat and s.sub_category == sub_cat 
                for s in suggestions
            ):
                suggestion_id += 1
                
                # Extract context around match
                context_match = re.search(pattern + r'.{0,200}', analysis_text)
                context = context_match.group(0) if context_match else ""
                
                suggestion = OptimizationSuggestion(
                    id=f"auto_{suggestion_id:03d}",
                    category=cat,
                    sub_category=sub_cat,
                    priority=base_priority,
                    title=f"Address {sub_cat.replace('_', ' ').title()} Issue",
                    description=context[:150],
                    code_hint=self._generate_code_hint(cat, sub_cat, context),
                    expected_impact=self._estimate_impact(base_priority, cat),
                    complexity="medium",
                    applicable_architectures=[compute_capability] if compute_capability else [],
                    source="auto_detected",
                )
                
                suggestions.append(suggestion)
        
        # Pattern 3: Target-specific best practices
        target_suggestions = self._get_target_best_practices(target, compute_capability)
        suggestions.extend(target_suggestions)
        
        # Sort by priority (descending)
        suggestions.sort(key=lambda s: (-s.priority, -s.expected_impact))
        
        # Assign final IDs after sorting
        for i, sug in enumerate(suggestions):
            if not sug.id.startswith("sug_"):
                sug.id = f"opt_{i+1:03d}"
        
        return suggestions[:10]  # Limit to top 10 suggestions

    def _classify_suggestion(self, text: str) -> tuple[str, str, int]:
        """Classify a suggestion into category/sub_category/priority.

        Uses keyword matching to determine the nature of the suggestion.

        Args:
            text: Suggestion text to classify

        Returns:
            Tuple of (category, sub_category, priority 1-5)
        """
        text_lower = text.lower()
        
        # Memory-related
        if any(w in text_lower for w in ["shared memory", "__shared__", "shmem"]):
            return "memory_bound", "shared_memory", 3
        if any(w in text_lower for w in ["coalesc", "align", "stride"]):
            return "memory_bound", "global_memory", 4
        if any(w in text_lower for w in ["ldg", "read-only", "texture", "const"]):
            return "memory_bound", "l1", 2
        
        # Cache-related
        if any(w in text_lower for w in ["l2 cache", "l2_", "capacity"]):
            return "cache_capacity", "l2", 3
        if any(w in text_lower for w in ["tiling", "block", "tile"]):
            return "cache_capacity", "tiling", 3
        
        # Compute-related
        if any(w in text_lower for w in ["tensor core", "wmma", "cutlass"]):
            return "compute_bound", "tensor_core", 4
        if any(w in text_lower for w in ["fp16", "half", "bf16", "mixed precision"]):
            return "compute_bound", "fp32", 3
        if any(w in text_lower for w in ["register", "spill", "local mem"]):
            return "compute_bound", "register_pressure", 2
        
        # Occupancy/Latency
        if any(w in text_lower for w in ["occupancy", "block", "thread", "warp"]):
            return "latency_bound", "occupancy", 3
        if any(w in text_lower for w in ["prefetch", "async", "overlap"]):
            return "latency_bound", "hiding", 2
        
        # Default classification
        return "balanced", "general", 2

    def _generate_code_hint(
        self, 
        category: str, 
        sub_category: str, 
        context: str
    ) -> str:
        """Generate a concrete code hint for a suggestion.

        Provides actual CUDA APIs or patterns that implement the fix.

        Args:
            category: Suggestion category
            sub_category: Sub-category
            context: Original context for relevance

        Returns:
            Code snippet or API reference as hint
        """
        hints = {
            ("memory_bound", "shared_memory"): (
                "Use __shared__ memory with appropriate tiling; "
                "ensure __syncthreads() after writes; "
                "consider __launch_bounds__ for register allocation"
            ),
            ("memory_bound", "global_memory"): (
                "Ensure coalesced access: consecutive threads access consecutive "
                "32-bit/64-bit words; use struct-of-arrays layout; "
                "align arrays to 128-byte boundaries"
            ),
            ("memory_bound", "l1"): (
                "Use __ldg() intrinsic for read-only data; "
                "leverage texture cache via cudaBindTexture; "
                "enable read-only cache with -Xptxas=-dlcm=cg"
            ),
            ("cache_capacity", "l2"): (
                "Reduce working set size per thread block; "
                "process data in tiles that fit L2; "
                "use software-managed caching for hot data"
            ),
            ("cache_capacity", "tiling"): (
                "Implement 2D tiling: TILE_X x TILE_Y blocks processing "
                "TILE_X x TILE_Y elements each; typical sizes: 16x16 or 32x32"
            ),
            ("compute_bound", "tensor_core"): (
                "#include <mma.h> and use wmma::fragment, "
                "wmma::load_matrix_sync, wmma::mma_sync, "
                "wmma::store_matrix_sync for WMMA operations"
            ),
            ("compute_bound", "fp32"): (
                "Consider __half arithmetic with FP32 accumulation; "
                "use FMA instructions (a*b+c); "
                "check --ptxas-options=-v for instruction mix"
            ),
            ("latency_bound", "occupancy"): (
                "Increase occupancy: reduce registers per thread, "
                "reduce shared memory per block, increase grid size; "
                "target >50% occupancy via cudaOccupancyMaxActiveBlocksPerMultiprocessor"
            ),
            ("latency_bound", "hiding"): (
                "Use cp.async for async copies (Ampere+); "
                "double-buffer between computation and memory transfers; "
                "increase concurrent kernel launches"
            ),
        }
        
        key = (category, sub_category)
        base_hint = hints.get(key, "Review NVIDIA CUDA Best Practices Guide for optimization techniques")
        
        # Add architecture-specific notes
        if "async" in base_hint.lower() and "cp.async" in base_hint.lower():
            base_hint += " [Requires sm_80+ / Ampere architecture]"
        
        return base_hint

    def _estimate_impact(self, priority: int, category: str) -> float:
        """Estimate potential performance improvement.

        Based on historical data and typical optimization impacts.

        Args:
            priority: Suggestion priority (1-5)
            category: Category of optimization

        Returns:
            Estimated percentage improvement
        """
        base_impacts = {
            "memory_bound": 15.0,
            "compute_bound": 20.0,
            "latency_bound": 12.0,
            "cache_capacity": 8.0,
            "balanced": 3.0,
        }
        
        base = base_impacts.get(category, 5.0)
        priority_multiplier = 0.6 + (priority * 0.15)  # 0.75x to 1.35x
        
        return round(base * priority_multiplier, 1)

    def _extract_title(self, text: str) -> str:
        """Extract a short title from suggestion text.

        Takes first meaningful phrase (up to 80 chars).

        Args:
            text: Full suggestion text

        Returns:
            Short title string
        """
        # Remove common prefixes
        text = re.sub(r'^(?:recommendation|suggestion|note|tip):\s*', '', text, flags=re.IGNORECASE)
        
        # Take first sentence or clause
        match = re.match(r'^([^.!?]+[.!?])?', text.strip())
        if match:
            title = match.group(1).strip() if match.group(1) else text.strip()
        else:
            title = text.strip()
        
        # Truncate to reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
        
        return title

    def _identify_bottlenecks(
        self,
        analysis_text: str,
        measurements: dict[str, float],
    ) -> list[str]:
        """Identify specific bottlenecks mentioned in analysis.

        Args:
            analysis_text: Analysis output
            measurements: Current measurements

        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []
        
        # Check for explicit bottleneck mentions
        bottleneck_keywords = [
            "bottleneck", "limited by", "bound by", 
            "throttling", "saturation", "underutilized",
        ]
        
        sentences = re.split(r'[.!?]+', analysis_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if any(kw in sentence.lower() for kw in bottleneck_keywords):
                bottlenecks.append(sentence[:150])
        
        # Check measurement-specific concerns
        for key, value in measurements.items():
            if "latency" in key.lower() and value > 1000:
                bottlenecks.append(f"High latency measured for {key}: {value}")
            elif "bandwidth" in key.lower():
                bottlenecks.append(f"Bandwidth analysis for {key}: {value} GB/s")
        
        return bottlenecks[:5]  # Top 5 bottlenecks

    def _determine_verdict(
        self,
        measurements: dict[str, float],
        suggestions: list[OptimizationSuggestion],
        analysis_text: str,
    ) -> tuple[str, float]:
        """Determine overall verdict on measurement quality.

        Args:
            measurements: Measured values
            suggestions: Generated suggestions
            analysis_text: Full analysis text

        Returns:
            Tuple of (verdict, confidence)
        """
        high_priority_count = sum(1 for s in suggestions if s.priority >= 4)
        medium_priority_count = sum(1 for s in suggestions if s.priority >= 3)
        
        # Verdict logic
        if high_priority_count >= 2:
            verdict = "REJECT"
            confidence = 0.7
        elif high_priority_count == 1 or medium_priority_count >= 3:
            verdict = "CONDITIONAL_ACCEPT"
            confidence = 0.8
        else:
            verdict = "ACCEPT"
            confidence = 0.9
        
        # Check for explicit acceptance/rejection indicators
        if re.search(r'(?i)(accept|valid|good|reasonable|within.*range)', analysis_text):
            if verdict != "ACCEPT":
                verdict = "CONDITIONAL_ACCEPT"
                confidence = min(confidence + 0.1, 1.0)
        
        if re.search(r'(?i)(reject|invalid|unreliable|failed|error)', analysis_text):
            verdict = "REJECT"
            confidence = max(confidence, 0.85)
        
        return verdict, confidence

    def _get_target_best_practices(
        self,
        target: str,
        compute_capability: str | None = None,
    ) -> list[OptimizationSuggestion]:
        """Generate target-specific best practice suggestions.

        These are proactive suggestions based on known challenges for each target type.

        Args:
            target: Measurement target name
            compute_capability: Current GPU arch

        Returns:
            List of best-practice suggestions
        """
        practices = {
            "dram_latency_cycles": [
                OptimizationSuggestion(
                    id="bp_dram_001",
                    category="methodology",
                    sub_category="accuracy",
                    priority=3,
                    title="Validate DRAM latency with multiple working set sizes",
                    description=(
                        "Run measurements with 64MB, 128MB, and 256MB working sets "
                        "to confirm DRAM latency is stable across different cache miss rates"
                    ),
                    code_hint="Test with working_set_mb=[64, 128, 256]; expect <5% variation",
                    expected_impact=5.0,
                    complexity="low",
                    source="best_practice",
                ),
            ],
            "l2_cache_size_mb": [
                OptimizationSuggestion(
                    id="bp_l2_001",
                    category="methodology",
                    sub_category="accuracy",
                    priority=4,
                    title="Verify L2 measurement with multiple detection methods",
                    description=(
                        "Cliff detection may underestimate partitioned L2; "
                        "consider using hardware query as cross-check"
                    ),
                    code_hint="Cross-validate with cudaDeviceGetAttribute(L2_CACHE_SIZE)",
                    expected_impact=10.0,
                    complexity="low",
                    source="best_practice",
                ),
            ],
            "actual_boost_clock_mhz": [
                OptimizationSuggestion(
                    id="bp_clock_001",
                    category="methodology",
                    sub_category="stability",
                    priority=3,
                    title="Confirm clock stability with sustained workload",
                    description=(
                        "Boost clock varies with thermal/load conditions; "
                        "run multiple iterations to establish stable reading"
                    ),
                    code_hint="Execute kernel for >100ms, repeat 5 times, take median",
                    expected_impact=3.0,
                    complexity="low",
                    source="best_practice",
                ),
            ],
        }
        
        return practices.get(target, [])

    def format_for_codegen(
        self,
        report: FeedbackReport,
        max_suggestions: int = 3,
        max_tokens: int = 800,
    ) -> str:
        """Format feedback report for injection into CodeGen's prompt.

        Creates an optimized prompt section that:
        - Fits within token budget constraints
        - Presents most important suggestions first
        - Includes actionable code hints
        - Maintains professional tone

        Args:
            report: FeedbackReport from create_feedback_report()
            max_suggestions: Maximum number of suggestions to include
            max_tokens: Approximate token budget for entire section

        Returns:
            Formatted string ready for system/user message
        """
        if not report.suggestions:
            return ""
        
        # Select top N suggestions
        selected = report.suggestions[:max_suggestions]
        
        lines = []
        lines.append("📈 **Performance Analysis Feedback**")
        lines.append(f"Target: {report.target} | Verdict: {report.overall_verdict}")
        lines.append("")
        
        # Current measurements summary
        if report.measurements:
            meas_str = ", ".join(f"{k}={v}" for k, v in report.measurements.items())
            lines.append(f"Current Measurements: {meas_str}")
            lines.append("")
        
        # Identified bottlenecks
        if report.bottlenecks_identified:
            lines.append("Identified Issues:")
            for bottleneck in report.bottlenecks_identified[:3]:
                lines.append(f"  ⚠️ {bottleneck[:120]}")
            lines.append("")
        
        # Prioritized suggestions
        lines.append("Optimization Recommendations (priority order):")
        lines.append("")
        
        for i, suggestion in enumerate(selected, 1):
            formatted = suggestion.format_for_llm(max_tokens=max_tokens // max_suggestions)
            lines.append(f"{i}. {formatted}")
            lines.append("")
        
        # Verdict-based guidance
        if report.overall_verdict == "REJECT":
            lines.append("⚡ ACTION REQUIRED: Implement at least one high-priority suggestion before proceeding.")
        elif report.overall_verdict == "CONDITIONAL_ACCEPT":
            lines.append("💡 RECOMMENDED: Consider implementing suggestions to improve accuracy.")
        else:
            lines.append("✅ Measurements are acceptable. Suggestions are optional improvements.")
        
        result = "\n".join(lines)
        
        # Verify approximate token count (rough estimate: 1 token ≈ 4 chars)
        estimated_tokens = len(result) // 4
        if estimated_tokens > max_tokens:
            # Truncate suggestions to fit budget
            lines = lines[:int(len(lines) * max_tokens / estimated_tokens * 0.9)]
            result = "\n".join(lines) + "\n\n[Truncated for token budget]"
        
        return result

    def get_action_items_for_pipeline_context(
        self,
        report: FeedbackReport,
    ) -> dict[str, Any]:
        """Extract action items for PipelineContext.metric_feedback.

        This is what gets stored in the pipeline context for the next iteration.

        Args:
            report: FeedbackReport object

        Returns:
            Dictionary compatible with PipelineContext.add_metric_feedback()
        """
        suggested_fixes = [s.title for s in report.suggestions[:5]]
        bottleneck_type = report.suggestions[0].category if report.suggestions else ""
        bottleneck_sub_type = report.suggestions[0].sub_category if report.suggestions else ""
        recommendations = [s.format_for_llm(150) for s in report.suggestions[:3]]
        
        return {
            "suggested_fixes": suggested_fixes,
            "bottleneck_type": bottleneck_type,
            "bottleneck_sub_type": bottleneck_sub_type,
            "recommendations": recommendations,
            "verdict": report.overall_verdict,
            "confidence": report.confidence_score,
            "suggestion_ids": [s.id for s in report.suggestions],
        }


# Global singleton instance
feedback_enhancer = FeedbackEnhancer()
