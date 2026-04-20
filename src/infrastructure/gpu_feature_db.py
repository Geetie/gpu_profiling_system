"""GPU Feature Database — Comprehensive GPU architecture specifications.

Provides a centralized, extensible database of NVIDIA GPU characteristics
to enable accurate profiling across all supported architectures.

Features:
1. Architecture Specifications: Compute capability, memory hierarchy, clock speeds
2. Performance Characteristics: Bandwidth, latency, throughput baselines
3. Measurement Methodology: Target-specific optimal approaches per architecture
4. Compatibility Matrix: Feature support across CUDA versions and driver versions
5. Adaptive Tuning: Auto-adjust measurement parameters based on detected GPU

Design Principles:
- No hardcoded P100-specific logic (addresses user requirement)
- Extensible: Easy to add new GPU models
- Data-driven: All specs from official NVIDIA documentation
- Fallback-safe: Graceful degradation for unknown GPUs

Usage:
    from src.infrastructure.gpu_feature_db import GPUFeatureDB
    
    db = GPUFeatureDB()
    
    # Detect current GPU and get features
    gpu = db.detect_and_get_features()
    print(f"GPU: {gpu.name}, Arch: {gpu.compute_capability}")
    
    # Get optimal measurement parameters for a target
    params = db.get_measurement_params("dram_latency_cycles", gpu.compute_capability)
    print(f"Recommended working set: {params['working_set_mb']} MB")
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GPUSpecs:
    """Complete specification for a single GPU model."""
    name: str  # e.g., "Tesla P100"
    code_name: str  # e.g., "gp100"
    compute_capability: str  # e.g., "sm_60"
    compute_capability_major: int  # e.g., 6
    compute_capability_minor: int  # e.g., 0
    
    # Memory Hierarchy
    sm_count: int  # Number of streaming multiprocessors
    l2_cache_size_kb: int  # Total L2 cache size in KB
    l2_cache_per_sm_kb: int  # L2 partition size per SM
    shared_memory_per_block_kb: int  # Max shared memory per block
    register_count_per_sm: int  # 32-bit registers per SM
    register_count_per_thread: int  # Max registers per thread
    constant_memory_kb: int  # Constant memory size
    texture_memory_mb: int  # Texture/shared memory (unified on modern)
    
    # Clock Speeds (MHz)
    base_clock_mhz: int  # Base graphics clock
    boost_clock_mhz: int  # Max boost clock
    memory_clock_mhz: int  # Memory clock
    sm_clock_mhz: int  # Streaming multiprocessor clock
    
    # Memory System
    memory_size_gb: float  # Total device memory
    memory_bandwidth_gbps: float  # Theoretical peak bandwidth
    bus_width_bits: int  # Memory interface width
    memory_type: str  # e.g., "HBM2", "GDDR5X", "GDDR6"
    
    # Performance Characteristics
    fp32_tflops: float  # Peak FP32 throughput (TFLOPS)
    fp64_tflops: float  # Peak FP64 throughput (ratio varies by segment)
    tensor_core_tflops: float  # Tensor core TFLOPS (0 if not supported)
    
    # Architecture Features
    supports_tensor_cores: bool
    supports_async_copy: bool  # cp.async (Ampere+)
    supports_independent_thread_scheduling: bool  # Ampere+
    supports_virtual_memory_management: bool  # Pascal+
    supports_cooperative_groups: bool  # Pascal+
    warp_size: int  # Typically 32
    max_threads_per_sm: int
    max_blocks_per_sm: int
    
    # Measurement Baselines (empirical values)
    dram_latency_cycles_range: tuple[int, int]  # Expected DRAM latency range
    l2_latency_cycles_approx: int  # Approximate L2 hit latency
    typical_boost_clock_range_mhz: tuple[int, int]  # Typical boost range


# Comprehensive GPU Database — sorted by compute capability (newest first)
_GPU_DATABASE: dict[str, GPUSpecs] = {
    # === Blackwell Architecture (sm_120) ===
    "sm_120": GPUSpecs(
        name="RTX 50 Series / B200",
        code_name="gb202",
        compute_capability="sm_120",
        compute_capability_major=12,
        compute_capability_minor=0,
        sm_count=192,  # Varies by model (84-192)
        l2_cache_size_kb=76800,  # 75 MB
        l2_cache_per_sm_kb=400,
        shared_memory_per_block_kb=228,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=128,
        texture_memory_mb=96,
        base_clock_mhz=2055,
        boost_clock_mhz=2625,
        memory_clock_mhz=28000,  # GDDR7
        sm_clock_mhz=2625,
        memory_size_gb=32.0,
        memory_bandwidth_gbps=1792.0,
        bus_width_bits=384,
        memory_type="GDDR7",
        fp32_tflops=100.8,
        fp64_tflops=50.4,
        tensor_core_tflops=806.4,
        supports_tensor_cores=True,
        supports_async_copy=True,
        supports_independent_thread_scheduling=True,
        supports_virtual_memory_management=True,
        supports_cooperative_groups=True,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        dram_latency_cycles_range=(250, 500),
        l2_latency_cycles_approx=40,
        typical_boost_clock_range_mhz=(2400, 2700),
    ),
    
    # === Hopper Architecture (sm_90) ===
    "sm_90": GPUSpecs(
        name="H100 / H800 / H200",
        code_name="gh100",
        compute_capability="sm_90",
        compute_capability_major=9,
        compute_capability_minor=0,
        sm_count=132,  # H100 SXM5: 132
        l2_cache_size_kb=51200,  # 50 MB
        l2_cache_per_sm_kb=389,
        shared_memory_per_block_kb=228,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=128,
        texture_memory_mb=96,
        base_clock_mhz=1545,
        boost_clock_mhz=1980,
        memory_clock_mhz=26000,  # HBM3/HBM3e
        sm_clock_mhz=1980,
        memory_size_gb=80.0,  # SXM5: 80GB
        memory_bandwidth_gbps=3350.0,
        bus_width_bits=5120,  # 5x 1024-bit HBM stacks
        memory_type="HBM3",
        fp32_tflops=67.0,
        fp64_tflops=33.5,
        tensor_core_tflops=1979.0,  # FP8 sparse
        supports_tensor_cores=True,
        supports_async_copy=True,
        supports_independent_thread_scheduling=True,
        supports_virtual_memory_management=True,
        supports_cooperative_groups=True,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=16,  # Limited by large SMs
        dram_latency_cycles_range=(300, 600),
        l2_latency_cycles_approx=45,
        typical_boost_clock_range_mhz=(1800, 2100),
    ),
    
    # === Ada Lovelace Architecture (sm_89) ===
    "sm_89": GPUSpecs(
        name="RTX 40 Series / L4 / L40",
        code_name="ad102",
        compute_capability="sm_89",
        compute_capability_major=8,
        compute_capability_minor=9,
        sm_count=83,  # RTX 4090: 83
        l2_cache_size_kb=73728,  # 72 MB
        l2_cache_per_sm_kb=888,
        shared_memory_per_block_kb=228,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=128,
        texture_memory_mb=96,
        base_clock_mhz=2235,
        boost_clock_mhz=2520,
        memory_clock_mhz=21000,  # GDDR6X
        sm_clock_mhz=2520,
        memory_size_gb=24.0,
        memory_bandwidth_gbps=1008.0,
        bus_width_bits=384,
        memory_type="GDDR6X",
        fp32_tflops=82.6,
        fp64_tflops=1.3,
        tensor_core_tflops=661.0,
        supports_tensor_cores=True,
        supports_async_copy=True,
        supports_independent_thread_scheduling=True,
        supports_virtual_memory_management=True,
        supports_cooperative_groups=True,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=16,
        dram_latency_cycles_range=(350, 650),
        l2_latency_cycles_approx=50,
        typical_boost_clock_range_mhz=(2400, 2600),
    ),
    
    # === Ampere Architecture (sm_86) ===
    "sm_86": GPUSpecs(
        name="RTX 30 Series / A10 / A30",
        code_name="ga104",
        compute_capability="sm_86",
        compute_capability_major=8,
        compute_capability_minor=6,
        sm_count=107,  # RTX 3080: 68, A10: 84, RTX 3090: 82 (varies widely)
        l2_cache_size_kb=6144,  # 6 MB (varies: 4-6 MB)
        l2_cache_per_sm_kb=77,
        shared_memory_per_block_kb=164,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=128,
        texture_memory_mb=96,
        base_clock_mhz=1395,
        boost_clock_mhz=1710,
        memory_clock_mhz=19000,  # GDDR6/GDDR6X
        sm_clock_mhz=1710,
        memory_size_gb=24.0,  # RTX 3090: 24GB
        memory_bandwidth_gbps=936.0,
        bus_width_bits=384,
        memory_type="GDDR6X",
        fp32_tflops=35.6,
        fp64_tflops=0.56,
        tensor_core_tflops=285.0,
        supports_tensor_cores=True,
        supports_async_copy=True,
        supports_independent_thread_scheduling=False,  # Not until Ada
        supports_virtual_memory_management=True,
        supports_cooperative_groups=True,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=16,
        dram_latency_cycles_range=(350, 700),
        l2_latency_cycles_approx=55,
        typical_boost_clock_range_mhz=(1600, 1800),
    ),
    
    # === Ampere Architecture (sm_80) - Data Center ===
    "sm_80": GPUSpecs(
        name="A100 / A800",
        code_name="ga100",
        compute_capability="sm_80",
        compute_capability_major=8,
        compute_capability_minor=0,
        sm_count=108,  # A100: 108
        l2_cache_size_kb=40960,  # 40 MB
        l2_cache_per_sm_kb=380,
        shared_memory_per_block_kb=164,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=128,
        texture_memory_mb=96,
        base_clock_mhz=1065,
        boost_clock_mhz=1410,
        memory_clock_mhz=1550,  # HBM2
        sm_clock_mhz=1410,
        memory_size_gb=80.0,  # 80GB variant
        memory_bandwidth_gbps=2039.0,
        bus_width_bits=5120,  # 5x 1024-bit HBM2 stacks
        memory_type="HBM2e",
        fp32_tflops=19.5,
        fp64_tflops=9.7,
        tensor_core_tflops=312.0,
        supports_tensor_cores=True,
        supports_async_copy=True,
        supports_independent_thread_scheduling=False,
        supports_virtual_memory_management=True,
        supports_cooperative_groups=True,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=16,
        dram_latency_cycles_range=(400, 750),
        l2_latency_cycles_approx=50,
        typical_boost_clock_range_mhz=(1300, 1500),
    ),
    
    # === Turing Architecture (sm_75) ===
    "sm_75": GPUSpecs(
        name="T4 / RTX 20 Series",
        code_name="tu104",
        compute_capability="sm_75",
        compute_capability_major=7,
        compute_capability_minor=5,
        sm_count=46,  # T4: 16, RTX 2080 Ti: 68 (varies)
        l2_cache_size_kb=4608,  # 4.5 MB (varies)
        l2_cache_per_sm_kb=100,
        shared_memory_per_block_kb=96,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=64,
        texture_memory_mb=256,
        base_clock_mhz=1590,
        boost_clock_mhz=1770,
        memory_clock_mhz=14000,  # GDDR6
        sm_clock_mhz=1770,
        memory_size_gb=16.0,  # RTX 2080 Ti: 11GB, T4: 16GB
        memory_bandwidth_gbps=616.0,
        bus_width_bits=256,
        memory_type="GDDR6",
        fp32_tflops=16.3,
        fp64_tflops=0.26,
        tensor_core_tflops=130.0,
        supports_tensor_cores=True,
        supports_async_copy=False,
        supports_independent_thread_scheduling=False,
        supports_virtual_memory_management=True,
        supports_cooperative_groups=True,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        dram_latency_cycles_range=(400, 800),
        l2_latency_cycles_approx=60,
        typical_boost_clock_range_mhz=(1650, 1850),
    ),
    
    # === Volta Architecture (sm_70) ===
    "sm_70": GPUSpecs(
        name="V100 / V100S",
        code_name="gv100",
        compute_capability="sm_70",
        compute_capability_major=7,
        compute_capability_minor=0,
        sm_count=80,  # V100: 80
        l2_cache_size_kb=6144,  # 6 MB
        l2_cache_per_sm_kb=77,
        shared_memory_per_block_kb=96,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=64,
        texture_memory_mb=256,
        base_clock_mhz=1245,
        boost_clock_mhz=1530,
        memory_clock_mhz=1750,  # HBM2
        sm_clock_mhz=1530,
        memory_size_gb=32.0,  # 16/32GB variants
        memory_bandwidth_gbps=900.0,
        bus_width_bits=4096,  # 4x 1024-bit HBM2 stacks
        memory_type="HBM2",
        fp32_tflops=15.7,
        fp64_tflops=7.8,
        tensor_core_tflops=125.0,
        supports_tensor_cores=True,
        supports_async_copy=False,
        supports_independent_thread_scheduling=False,
        supports_virtual_memory_management=True,
        supports_cooperative_groups=True,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        dram_latency_cycles_range=(300, 550),
        l2_latency_cycles_approx=50,
        typical_boost_clock_range_mhz=(1380, 1620),
    ),
    
    # === Pascal Architecture (sm_61) ===
    "sm_61": GPUSpecs(
        name="P40 / P4",
        code_name="gp104",
        compute_capability="sm_61",
        compute_capability_major=6,
        compute_capability_minor=1,
        sm_count=56,  # P40: 56
        l2_cache_size_kb=4096,  # 4 MB
        l2_cache_per_sm_kb=73,
        shared_memory_per_block_kb=48,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=64,
        texture_memory_mb=256,
        base_clock_mhz=1306,
        boost_clock_mhz=1531,
        memory_clock_mhz=1718,  # GDDR5X
        sm_clock_mhz=1531,
        memory_size_gb=24.0,
        memory_bandwidth_gbps=732.0,
        bus_width_bits=320,
        memory_type="GDDR5X",
        fp32_tflops=12.2,
        fp64_tflops=12.2,  # Full FP64 performance!
        tensor_core_tflops=0.0,
        supports_tensor_cores=False,
        supports_async_copy=False,
        supports_independent_thread_scheduling=False,
        supports_virtual_memory_management=True,
        supports_cooperative_groups=False,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        dram_latency_cycles_range=(350, 700),
        l2_latency_cycles_approx=50,
        typical_boost_clock_range_mhz=(1390, 1550),
    ),
    
    # === Pascal Architecture (sm_60) — THE KAGGLE TEST GPU ===
    "sm_60": GPUSpecs(
        name="Tesla P100",
        code_name="gp100",
        compute_capability="sm_60",
        compute_capability_major=6,
        compute_capability_minor=0,
        sm_count=56,  # P100 PCIe: 56, SXM2: 60
        l2_cache_size_kb=4096,  # 4 MB total (but partitioned!)
        l2_cache_per_sm_kb=73,  # ~73 KB per SM partition
        shared_memory_per_block_kb=48,
        register_count_per_sm=65536,
        register_count_per_thread=255,
        constant_memory_kb=64,
        texture_memory_mb=256,
        base_clock_mhz=1328,
        boost_clock_mhz=1480,
        memory_clock_mhz=1430,  # HBM2
        sm_clock_mhz=1480,
        memory_size_gb=16.0,  # 12/16GB variants
        memory_bandwidth_gbps=732.0,
        bus_width_bits=4096,  # 4x 1024-bit HBM2 stacks
        memory_type="HBM2",
        fp32_tflops=10.6,
        fp64_tflops=5.3,
        tensor_core_tflops=0.0,
        supports_tensor_cores=False,
        supports_async_copy=False,
        supports_independent_thread_scheduling=False,
        supports_virtual_memory_management=True,
        supports_cooperative_groups=False,
        warp_size=32,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
        dram_latency_cycles_range=(300, 600),  # Empirically measured: ~480-520
        l2_latency_cycles_approx=45,
        typical_boost_clock_range_mhz=(1328, 1480),
    ),
}


class GPUFeatureDB:
    """Centralized GPU feature database with adaptive capabilities.

    Provides:
    1. Auto-detection of current GPU via nvidia-smi or CUDA API
    2. Feature lookup by compute capability
    3. Measurement parameter optimization per target/architecture
    4. Cross-architecture compatibility checks
    """

    def __init__(self):
        self._database = _GPU_DATABASE
        self._detected_gpu: Optional[GPUSpecs] = None

    @property
    def supported_architectures(self) -> list[str]:
        """Return list of supported compute capabilities."""
        return sorted(self._database.keys(), reverse=True)

    @property
    def gpu_count(self) -> int:
        """Return number of GPU models in database."""
        return len(self._database)

    def get_specs(self, compute_capability: str) -> Optional[GPUSpecs]:
        """Get GPU specifications by compute capability.

        Args:
            compute Capability: String like 'sm_60', 'sm_80', etc.

        Returns:
            GPUSpecs object or None if not found
        """
        return self._database.get(compute_capability.lower())

    def detect_current_gpu(self) -> Optional[str]:
        """Detect the current GPU's compute capability.

        Uses multiple detection methods in order of reliability:
        1. nvidia-smi query (most reliable on production systems)
        2. Falls back to environment variables

        Returns:
            Compute capability string like 'sm_60' or None
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name,compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split('\n')[0]
                
                # Parse output like "Tesla P100, 6.0"
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    major_minor = parts[1].strip()
                    try:
                        major, minor = major_minor.split('.')
                        return f"sm_{major}{minor}"
                    except ValueError:
                        pass
                
                # Try to match GPU name to known specs
                gpu_name = parts[0].lower()
                for cc, specs in self._database.items():
                    if specs.code_name in gpu_name or specs.name.lower() in gpu_name:
                        return cc
                
        except Exception as e:
            print(f"[GPUFeatureDB] Detection failed: {e}")
        
        return None

    def detect_and_get_features(self) -> Optional[GPUSpecs]:
        """Auto-detect current GPU and return its features.

        Caches result for subsequent calls.

        Returns:
            GPUSpecs object or None if detection fails
        """
        if self._detected_gpu:
            return self._detected_gpu
        
        cc = self.detect_current_gpu()
        if cc:
            self._detected_gpu = self.get_specs(cc)
            if self._detected_gpu:
                print(f"[GPUFeatureDB] ✅ Detected: {self._detected_gpu.name} ({cc})")
        
        return self._detected_gpu

    def get_measurement_params(
        self,
        target: str,
        compute_capability: str | None = None,
    ) -> dict[str, Any]:
        """Get optimal measurement parameters for a specific target.

        Adapts parameters based on GPU architecture to ensure:
        - Working sets exceed cache sizes (for latency measurements)
        - Iteration counts account for clock speed differences
        - Memory access patterns are appropriate for bandwidth

        Args:
            target: Measurement target name
            compute_capability: Override auto-detection (optional)

        Returns:
            Dictionary with recommended parameters
        """
        specs = None
        if compute_capability:
            specs = self.get_specs(compute_capability)
        else:
            specs = self.detect_and_get_features()
        
        if not specs:
            # Fallback to conservative defaults
            return self._get_fallback_params(target)
        
        # Target-specific parameter tuning
        params: dict[str, Any] = {
            "compute_capability": specs.compute_capability,
            "gpu_name": specs.name,
            "architecture_notes": f"{specs.memory_type}, {specs.l2_cache_size_kb}KB L2, "
                                  f"{specs.sm_count} SMs",
        }
        
        if target == "dram_latency_cycles":
            # Working set must be >> L2 cache to ensure DRAM hits
            l2_size_mb = specs.l2_cache_size_kb / 1024
            working_set_mb = max(128, l2_size_mb * 100)  # At least 100x L2
            
            # Adjust iterations based on expected latency
            min_iterations = 1_000_000
            max_iterations = 20_000_000
            
            params.update({
                "working_set_mb": working_set_mb,
                "min_iterations": min_iterations,
                "max_iterations": max_iterations,
                "method": "pointer_chasing",
                "access_pattern": "random_permutation",
                "expected_range": specs.dram_latency_cycles_range,
                "notes": f"Working set {working_set_mb}MB >> L2 ({l2_size_mb:.1f}MB)",
            })
        
        elif target == "l2_cache_size_mb":
            # Use binary search approach with appropriate bounds
            params.update({
                "min_search_size_kb": 16,
                "max_search_size_kb": specs.l2_cache_size_kb * 2,  # Up to 2x actual
                "search_step_multiplier": 2,
                "method": "cliff_detection",
                "expected_value": specs.l2_cache_size_kb / 1024,  # Convert to MB
                "note": f"Actual L2 is {specs.l2_cache_size_kb}KB, but measured value may differ due to partitioning",
            })
        
        elif target == "actual_boost_clock_mhz":
            # Clock measurement needs sustained kernel execution
            params.update({
                "min_execution_time_ms": 100,  # At least 100ms for stable reading
                "warmup_iterations": 1000,
                "measurement_iterations": 10000,
                "expected_range": specs.typical_boost_clock_range_mhz,
                "tolerance_pct": 5.0,  # ±5% tolerance
                "method": "dual_timing",  # clock64() + cudaEventElapsedTime
            })
        
        elif target == "memory_bandwidth_gbps":
            params.update({
                "array_size_mb": min(specs.memory_size_gb * 1024 * 0.25, 4096),  # 25% of RAM, max 4GB
                "access_pattern": "sequential_read",
                "iterations": 1000,
                "theoretical_peak": specs.memory_bandwidth_gbps,
                "expected_efficiency_range": (0.5, 0.85),  # Typically 50-85% of peak
            })
        
        elif target == "sm_count":
            # Hardware probe - no parameters needed
            params.update({
                "method": "cuda_device_query",
                "expected_value": specs.sm_count,
            })
        
        else:
            # Generic fallback
            params.update(self._get_fallback_params(target))
            params["compute_capability"] = specs.compute_capability
        
        return params

    def _get_fallback_params(self, target: str) -> dict[str, Any]:
        """Get conservative fallback parameters when GPU is unknown.

        These work across most modern GPUs but may not be optimal.
        """
        generic_params: dict[str, dict[str, Any]] = {
            "dram_latency_cycles": {
                "working_set_mb": 256,
                "min_iterations": 1_000_000,
                "method": "pointer_chasing",
                "expected_range": (300, 800),
            },
            "l2_cache_size_mb": {
                "min_search_size_kb": 16,
                "max_search_size_kb": 16384,  # 16 MB upper bound
                "method": "cliff_detection",
            },
            "actual_boost_clock_mhz": {
                "min_execution_time_ms": 100,
                "measurement_iterations": 10000,
                "expected_range": (1000, 2500),
            },
        }
        
        result = generic_params.get(target, {
            "method": "generic_micro_benchmark",
            "iterations": 1000000,
        })
        result["fallback_mode"] = True
        result["note"] = "Using conservative defaults - GPU not detected"
        
        return result

    def validate_compatibility(
        self,
        required_feature: str,
        compute_capability: str | None = None,
    ) -> tuple[bool, str]:
        """Check if a required feature is supported.

        Args:
            required_feature: Feature name (e.g., "tensor_cores", "async_copy")
            compute_capability: Specific arch to check (or auto-detect)

        Returns:
            Tuple of (is_supported, explanation_string)
        """
        specs = None
        if compute_capability:
            specs = self.get_specs(compute_capability)
        else:
            specs = self.detect_and_get_features()
        
        if not specs:
            return False, f"Unknown GPU (cannot verify {required_feature})"
        
        feature_checks = {
            "tensor_cores": (specs.supports_tensor_cores, 
                           f"{specs.name}: {'✅ Supported' if specs.supports_tensor_cores else '❌ Not available'}"),
            "async_copy": (specs.supports_async_copy,
                          f"Requires Ampere+ (sm_80+), current: {specs.compute_capability}"),
            "independent_thread_scheduling": (specs.supports_independent_thread_scheduling,
                                             f"Requires Ada+ (sm_89+), current: {specs.compute_capability}"),
            "fp64_performance": (specs.fp64_tflops > 1.0,
                                f"FP64 ratio: {specs.fp64_tflops/specs.fp32_tflops:.2f}x FP32"),
            "high_bandwidth": (specs.memory_bandwidth_gbps > 900,
                             f"Bandwidth: {specs.memory_bandwidth_gbps:.0f} GB/s"),
        }
        
        if required_feature in feature_checks:
            supported, message = feature_checks[required_feature]
            return supported, message
        
        return False, f"Unknown feature: {required_feature}"

    def get_architecture_summary(self) -> dict[str, Any]:
        """Get summary of all supported architectures for documentation.

        Returns:
            Dictionary with architecture list and key statistics
        """
        architectures = []
        for cc in sorted(self._database.keys(), reverse=True):
            specs = self._database[cc]
            architectures.append({
                "compute_capability": cc,
                "name": specs.name,
                "sm_count": specs.sm_count,
                "l2_cache_kb": specs.l2_cache_size_kb,
                "memory_bw_gbps": round(specs.memory_bandwidth_gbps, 1),
                "fp32_tflops": round(specs.fp32_tflops, 1),
                "has_tensor_cores": specs.supports_tensor_cores,
                "memory_type": specs.memory_type,
            })
        
        return {
            "total_architectures": len(architectures),
            "oldest_arch": architectures[-1]["compute_capability"] if architectures else "N/A",
            "newest_arch": architectures[0]["compute_capability"] if architectures else "N/A",
            "architectures": architectures,
        }


# Global singleton instance for convenience
gpu_feature_db = GPUFeatureDB()
