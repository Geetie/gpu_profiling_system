#!/usr/bin/env python3
"""GPU Profiling System - Kaggle Test Kernel.

Clones the project from GitHub, then runs the GPU profiling system.
The multi-agent system autonomously generates CUDA code based on targets.
No hardcoded CUDA source in this kernel - Agent generates everything.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

# Redirect stdout/stderr to a log file for debugging
WORKING_DIR = "/kaggle/working"
os.makedirs(WORKING_DIR, exist_ok=True)
LOG_FILE = os.path.join(WORKING_DIR, "execution.log")
log_fh = open(LOG_FILE, "w", buffering=1)  # line-buffered

class TeeWriter:
    def __init__(self, stream, file):
        self._stream = stream
        self._file = file
    def write(self, s):
        self._stream.write(s)
        self._file.write(s)
        self._file.flush()
    def flush(self):
        self._stream.flush()
        self._file.flush()

sys.stdout = TeeWriter(sys.stdout, log_fh)
sys.stderr = TeeWriter(sys.stderr, log_fh)

os.chdir(WORKING_DIR)
print(f"Working dir: {WORKING_DIR}")
print(f"Python: {sys.version}")
print(f"Log file: {LOG_FILE}")


def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_cmd(cmd, timeout=300, description="", wd=None, extra_env=None):
    if description:
        print(f"  {description}")
    cwd = wd if wd else WORKING_DIR
    print(f"[cwd={cwd}] $ {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, cwd=cwd, env=env)
        if r.stdout:
            out = r.stdout
            if len(out) > 5000:
                print(f"... ({len(out)} chars total, showing first 3000 + last 2000)")
                print(out[:3000])
                print("\n... [truncated] ...\n")
                print(out[-2000:])
            else:
                print(out)
        if r.stderr:
            print(f"STDERR: {r.stderr[-1000:]}")
        print(f"Exit code: {r.returncode}")
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        print(f"Timeout after {timeout}s")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error: {e}")
        return False, "", str(e)


def get_kaggle_secret(secret_name: str) -> str | None:
    """Read a secret from Kaggle's Secrets storage.

    Kaggle Secrets are NOT automatically environment variables.
    They must be read explicitly via the Kaggle API.
    """
    # Method 1: Try kaggle_secrets module (available in Kaggle kernels)
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        val = client.get_secret(secret_name)
        if val:
            print(f"[secret] {secret_name}: found via kaggle_secrets")
            return val
    except Exception as e:
        print(f"[secret] kaggle_secrets failed for {secret_name}: {e}")

    # Method 2: Try reading from /kaggle/secrets/ (mounted secret directory)
    secret_path = f"/kaggle/secrets/{secret_name}"
    if os.path.isfile(secret_path):
        with open(secret_path) as f:
            val = f.read().strip()
        if val:
            print(f"[secret] {secret_name}: found via file mount")
            return val

    # Method 3: Check environment variable (direct env var)
    val = os.environ.get(secret_name, "")
    if val:
        print(f"[secret] {secret_name}: found via env var")
        return val

    # Method 4: Check KAGGLE_DATA_PROXY_TOKEN based path
    # Kaggle also mounts secrets under /kaggle/input/ for some configurations
    secret_input_path = f"/kaggle/input/{secret_name}"
    if os.path.isfile(secret_input_path):
        with open(secret_input_path) as f:
            val = f.read().strip()
        if val:
            print(f"[secret] {secret_name}: found via /kaggle/input")
            return val

    print(f"[secret] {secret_name}: not found")
    return None


def check_environment():
    banner("1. Environment Check")
    ok, out, err = run_cmd(["nvidia-smi", "-L"], description="GPU check")
    if not ok:
        print("No GPU detected -- aborting")
        return False
    ok, out, err = run_cmd(["nvcc", "--version"], description="nvcc check")
    if not ok:
        print("nvcc not found, trying install...")
        run_cmd(["apt-get", "update"], timeout=120)
        run_cmd(["apt-get", "install", "-y", "cuda-toolkit-12-0"], timeout=300)
        ok, out, err = run_cmd(["nvcc", "--version"])
        if not ok:
            print("nvcc still not available -- aborting")
            return False
    ok, _, _ = run_cmd(["which", "ncu"], description="ncu check")
    if ok:
        print("ncu available -- will use Nsight Compute profiling")
    else:
        print("ncu not found -- will use cudaEventElapsedTime fallback")
    return True


def configure_api():
    banner("2. API Configuration")
    config_path = os.path.join(PROJECT_ROOT, "config", "api_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        token = cfg.get("env", {}).get("ANTHROPIC_AUTH_TOKEN", "")
        if token and len(token) > 30 and not token.startswith("mock") and not token.startswith("YOUR"):
            print("API config already exists with valid token")
            return True

    # Read secrets from Kaggle Secrets storage
    print("Reading secrets from Kaggle...")
    longcat_key = get_kaggle_secret("LONGCAT_API_KEY") or ""
    dashscope_key = get_kaggle_secret("DASHSCOPE_API_KEY") or ""
    anthropic_key = get_kaggle_secret("ANTHROPIC_API_KEY") or ""

    if longcat_key and len(longcat_key) > 10:
        env = {
            "ANTHROPIC_BASE_URL": "https://api.longcat.com/openaicompatible/api/v1/chat/completions",
            "ANTHROPIC_AUTH_TOKEN": longcat_key,
            "ANTHROPIC_MODEL": "longcat-flash-chat",
            "ANTHROPIC_REASONING_MODEL": "longcat-flash-chat",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "longcat-flash-chat",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "longcat-flash-chat",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "longcat-flash-chat"
        }
        cfg = {"env": env, "includeCoAuthoredBy": False, "effortLevel": "high"}
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("API configured from LONGCAT_API_KEY")
        return True
    elif dashscope_key and len(dashscope_key) > 10:
        env = {
            "ANTHROPIC_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "ANTHROPIC_AUTH_TOKEN": dashscope_key,
            "ANTHROPIC_MODEL": "qwen-max",
            "ANTHROPIC_REASONING_MODEL": "qwen-max",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen-turbo",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen-plus",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen-max"
        }
        cfg = {"env": env, "includeCoAuthoredBy": False, "effortLevel": "high"}
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("API configured from DASHSCOPE_API_KEY")
        return True
    elif anthropic_key and len(anthropic_key) > 30:
        env = {
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            "ANTHROPIC_AUTH_TOKEN": anthropic_key,
            "ANTHROPIC_MODEL": "claude-sonnet-4-5-20250514",
            "ANTHROPIC_REASONING_MODEL": "claude-sonnet-4-5-20250514",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5-20251001",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-5-20250514",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-6-20250514"
        }
        cfg = {"env": env, "includeCoAuthoredBy": False, "effortLevel": "high"}
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("API configured from ANTHROPIC_API_KEY")
        return True
    else:
        print("No LLM API keys found -- probes-only mode")
        return False


def create_target_spec():
    ts_path = os.path.join(PROJECT_ROOT, "config", "target_spec.json")
    if not os.path.isfile(ts_path):
        spec = {"targets": ["dram_latency_cycles", "l2_cache_size_mb",
            "actual_boost_clock_mhz", "max_shmem_per_block_kb",
            "dram_bandwidth_gbps", "shmem_bandwidth_gbps",
            "bank_conflict_penalty_ratio", "sm_count"]}
        os.makedirs(os.path.dirname(ts_path), exist_ok=True)
        with open(ts_path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"Created target_spec.json: {ts_path}")
    else:
        with open(ts_path) as f:
            print(f"Loaded target_spec: {json.load(f)}")
    return ts_path


def run_probes():
    banner("3. Hardware Probes (probes-only)")
    print("Running: python -m src.main --probes-only --no-docker")
    ok, out, err = run_cmd([
        sys.executable, "-m", "src.main",
        "--probes-only",
        "--output-dir", WORKING_DIR,
        "--state-dir", os.path.join(WORKING_DIR, ".state"),
        "--no-docker",
    ], timeout=600, wd=PROJECT_ROOT)
    print(f"Probes completed: success={ok}")
    return ok


def run_pipeline(target_spec_path):
    banner("4. Multi-Agent Pipeline")
    print("Planner -> CodeGen -> MetricAnalysis -> Verification")
    print("Agent will autonomously generate CUDA kernels based on targets")
    ok, out, err = run_cmd([
        sys.executable, "-m", "src.main",
        "Profile GPU hardware parameters for the given targets",
        "--pipeline",
        "--target-spec", target_spec_path,
        "--output-dir", WORKING_DIR,
        "--state-dir", os.path.join(WORKING_DIR, ".state"),
        "--no-docker",
        "--mode", "high_autonomy",
        "--max-turns", "50",
        "--max-tokens", "16000",
    ], timeout=3600, wd=PROJECT_ROOT)
    print(f"Pipeline completed: success={ok}")
    return ok


def analyze_results():
    banner("5. Results Analysis")
    results_path = os.path.join(WORKING_DIR, "results.json")
    if not os.path.isfile(results_path):
        print(f"ERROR: results.json not found at {results_path}")
        print("Directory listing:")
        for f in sorted(os.listdir(WORKING_DIR)):
            print(f"  {f}")
        return False
    with open(results_path) as f:
        results = json.load(f)
    m = results.get("measurements", {})
    print()
    print("Measurements:")
    key_metrics = ["actual_boost_clock_mhz", "dram_latency_cycles",
        "l2_latency_cycles", "l1_latency_cycles", "l2_cache_size_mb",
        "dram_bandwidth_gbps", "shmem_bandwidth_gbps",
        "max_shmem_per_block_kb", "bank_conflict_penalty_ratio",
        "sm_count", "likely_gpu_family"]
    for metric in key_metrics:
        if metric in m:
            print(f"  {metric}: {m[metric]}")
    cv = results.get("cross_validation", {})
    if cv:
        passed = sum(1 for v in cv.values() if v is True)
        print(f"Cross-validation: {passed}/{len(cv)} passed")
    methodology = results.get("methodology", "")
    print(f"Methodology length: {len(methodology)} chars")
    return True


# ============================================================
# MAIN
# ============================================================

PROJECT_ROOT = None
all_errors = []
# Defaults — safe for finally block if error occurs before assignment
probe_ok = False
pipeline_ok = False
api_configured = False

try:
    banner("GPU Profiling System - Kaggle Test")

    # Step 0: Clone repo
    banner("0. Clone Project Repository")
    PROJECT_ROOT = os.path.join(WORKING_DIR, "gpu_profiling_system")
    if os.path.isfile(os.path.join(PROJECT_ROOT, "src", "main.py")):
        print("Project already cloned -- skipping")
    else:
        ok, out, err = run_cmd([
            "git", "clone",
            "https://github.com/Geetie/gpu_profiling_system.git",
            PROJECT_ROOT,
        ], timeout=120, description="Clone repository")
        if not ok:
            all_errors.append("Failed to clone repository")
            print("ERROR: Failed to clone repository")
            sys.exit(1)

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Current working dir: {os.getcwd()}")
    print(f"src/main.py exists: {os.path.isfile(os.path.join(PROJECT_ROOT, 'src', 'main.py'))}")


    if not check_environment():
        all_errors.append("Environment check failed")
        print("Environment check failed")
        sys.exit(1)

    api_configured = configure_api()

    target_spec_path = create_target_spec()

    # Run probes first
    probe_ok = run_probes()
    if not probe_ok:
        all_errors.append("Hardware probes failed")

    # Run pipeline if API available
    pipeline_ok = False
    if api_configured and probe_ok:
        pipeline_ok = run_pipeline(target_spec_path)
        if not pipeline_ok:
            all_errors.append("Pipeline failed")
    else:
        banner("Pipeline - SKIPPED (no API or probes failed)")
        all_errors.append(f"Pipeline skipped: api_configured={api_configured}, probe_ok={probe_ok}")

    results_ok = analyze_results()
    if not results_ok:
        all_errors.append("Results analysis failed")

except Exception as e:
    all_errors.append(f"Fatal error: {str(e)}")
    print(f"\nFATAL ERROR: {e}")
    traceback.print_exc()

finally:
    # Write execution summary
    banner("Execution Summary")
    rp = os.path.join(WORKING_DIR, "results.json")
    ps = "PASS" if probe_ok else "FAIL"
    pl = "PASS" if pipeline_ok else "SKIP"
    rs = "FOUND" if os.path.isfile(rp) else "MISSING"
    print("  Hardware probes:", ps)
    print("  Pipeline mode:  ", pl)
    print("  results.json:   ", rs)
    if os.path.isfile(rp):
        print("  Size:", os.path.getsize(rp), "bytes")

    # Write summary file for debugging
    summary = {
        "probe_ok": probe_ok,
        "pipeline_ok": pipeline_ok,
        "results_found": os.path.isfile(rp) if "rp" in dir() else False,
        "api_configured": api_configured,
        "errors": all_errors,
        "project_root": PROJECT_ROOT,
    }
    summary_path = os.path.join(WORKING_DIR, "execution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nExecution summary written to: {summary_path}")
    if all_errors:
        print(f"\nERRORS ({len(all_errors)}):")
        for i, err in enumerate(all_errors, 1):
            print(f"  {i}. {err}")

    print("\nTest complete.")
    print(f"Log file: {LOG_FILE}")

    # Close log file
    log_fh.close()
