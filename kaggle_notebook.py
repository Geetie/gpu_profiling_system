# ============================================================
# Kaggle Notebook — GPU Profiling System
# ============================================================
# 使用方法:
#   1. Kaggle 新建 Notebook → 设置 GPU (T4)
#   2. 将整个项目目录 zip 打包
#   3. 上传 zip 到 Kaggle Dataset
#   4. 或直接 git clone（如果项目在 GitHub 上）
#   5. 编辑 config/api_config.json 填入 API Key
#   6. 按顺序运行以下 Cell
# ============================================================

# ────────────────────────────────────────────────────────────
# Cell 1: 环境验证
# ────────────────────────────────────────────────────────────
import subprocess, sys, os, json

print("=== GPU Environment Check ===")
print("Python:", sys.version)
print()

# Check CUDA
print("--- CUDA ---")
r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(r.stdout[:500] if r.stdout else r.stderr[:200])

# Check nvcc
print("--- NVCC ---")
r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print(r.stdout[:200] if r.returncode == 0 else "nvcc not found")

# Check Docker
print("--- Docker ---")
r = subprocess.run(["docker", "--version"], capture_output=True, text=True)
print(r.stdout.strip() if r.returncode == 0 else "Docker not available")

print("\nEnvironment check complete.")


# ────────────────────────────────────────────────────────────
# Cell 2: 安装依赖
# ────────────────────────────────────────────────────────────
!pip install pytest requests 2>&1 | tail -3


# ────────────────────────────────────────────────────────────
# Cell 3: 克隆/加载项目
# ────────────────────────────────────────────────────────────
# 选项 A: 从 GitHub 克隆 (推荐)
# !git clone https://github.com/YOUR_USERNAME/GPU_Profiling_System.git
# %cd /kaggle/working/GPU_Profiling_System

# 选项 B: 从 Dataset 解压 (如果没有 GitHub)
# 将 zip 上传到 Kaggle Input，然后:
# import shutil
# shutil.unpack_archive('/kaggle/input/gpu-profiling-system/gpu_profiling_system.zip', '/kaggle/working')
# %cd /kaggle/working/GPU_Profiling_System

# 选项 C: 直接在 Notebook 中写代码 (当前)
# 如果以上都不可行，需要手动创建文件

print(f"Working directory: {os.getcwd()}")
print(f"Contents: {os.listdir('.')}")


# ────────────────────────────────────────────────────────────
# Cell 4: 配置 API Key
# ────────────────────────────────────────────────────────────
# ⚠️ 重要：填入你的 API Key
# ⚠️ 建议使用 Kaggle Secrets 而不是硬编码

# 方式 1: 编辑配置文件 (测试用)
# config_path = "config/api_config.json"
# with open(config_path, "r") as f:
#     config = json.load(f)
# config["env"]["ANTHROPIC_AUTH_TOKEN"] = "你的API Key"
# with open(config_path, "w") as f:
#     json.dump(config, f, indent=2)

# 方式 2: 使用 Kaggle Secrets (生产推荐)
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# api_key = user_secrets.get_secret("ANTHROPIC_AUTH_TOKEN")

# 方式 3: 环境变量 (最简单)
# os.environ["ANTHROPIC_AUTH_TOKEN"] = "你的API Key"

# 当前状态检查
config_path = "config/api_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    key = cfg["env"]["ANTHROPIC_AUTH_TOKEN"]
    if key and not key.startswith("在此填入"):
        print(f"✅ API Key 已配置: {key[:8]}...{key[-4:]}")
    else:
        print("❌ API Key 未配置，请编辑 config/api_config.json")
        print("   将 ANTHROPIC_AUTH_TOKEN 的值改为你自己的 Key")
else:
    print("❌ config/api_config.json 不存在")


# ────────────────────────────────────────────────────────────
# Cell 5: 运行单元测试
# ────────────────────────────────────────────────────────────
print("=== Running Unit Tests ===")
r = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True, text=True, cwd=os.getcwd()
)
print(r.stdout[-2000:])  # 最后 2000 字符
if r.returncode != 0:
    print("STDERR:", r.stderr[-500:])


# ────────────────────────────────────────────────────────────
# Cell 6: 运行集成测试
# ────────────────────────────────────────────────────────────
print("=== Running Integration Tests ===")
r = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_integration/", "-v", "--tb=short"],
    capture_output=True, text=True, cwd=os.getcwd()
)
print(r.stdout[-2000:])


# ────────────────────────────────────────────────────────────
# Cell 7: 端到端 GPU 探测 (需要 API Key)
# ────────────────────────────────────────────────────────────
print("=== End-to-End GPU Profiling ===")

# 验证 target_spec
with open("config/target_spec.json") as f:
    target_spec = json.load(f)
print(f"Targets: {target_spec['targets']}")

# 运行系统
r = subprocess.run(
    [sys.executable, "-m", "src.main",
     "Profile dram_latency_cycles on T4 GPU",
     "--no-docker",  # Kaggle 容器内不需要额外 Docker
     "--max-turns", "10",
     "--target-spec", "config/target_spec.json"],
    capture_output=True, text=True, cwd=os.getcwd(),
    timeout=300,
)
print(r.stdout[-3000:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-1000:])


# ────────────────────────────────────────────────────────────
# Cell 8: 对抗性测试 — 频率锁定
# ────────────────────────────────────────────────────────────
print("=== Adversarial: Frequency Lock ===")

# Kaggle T4 默认频率 ~1590MHz
# 我们可以用 nvidia-smi 限制频率
r = subprocess.run(
    ["nvidia-smi", "-q", "-d", "CLOCK"],
    capture_output=True, text=True
)
print(r.stdout[:1000])


# ────────────────────────────────────────────────────────────
# Cell 9: 对抗性测试 — SM 资源限制
# ────────────────────────────────────────────────────────────
print("=== Adversarial: SM Visibility ===")

# 查看当前 GPU 信息
r = subprocess.run(
    ["nvidia-smi", "query", "--gpu=0",
     "--format=csv,nounits,noheader",
     "--query-gpu=compute_mode,clocks.sm,clocks.mem,clocks.gr"],
    capture_output=True, text=True
)
print(r.stdout)


# ────────────────────────────────────────────────────────────
# Cell 10: Pipeline 多智能体模式 (需要 API Key + nvcc)
# ────────────────────────────────────────────────────────────
print("=== Pipeline Multi-Agent Mode ===")

r = subprocess.run(
    [sys.executable, "-m", "src.main",
     "Profile T4 GPU dram latency and L2 cache",
     "--pipeline",
     "--no-docker",
     "--max-turns", "15",
     "--target-spec", "config/target_spec.json"],
    capture_output=True, text=True, cwd=os.getcwd(),
    timeout=600,
)
print(r.stdout[-3000:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-1000:])


# ────────────────────────────────────────────────────────────
# Cell 11: 查看输出结果
# ────────────────────────────────────────────────────────────
print("=== Results ===")

# 检查生成的文件
for f in ["results.json", "target_spec.json"]:
    if os.path.exists(f):
        with open(f) as fh:
            print(f"\n--- {f} ---")
            print(fh.read()[:2000])

# 查看 session log
import glob
for f in glob.glob(".state/*.jsonl"):
    print(f"\n--- {f} ({os.path.getsize(f)} bytes) ---")
    with open(f) as fh:
        lines = fh.readlines()
        print(f"Total entries: {len(lines)}")
        if lines:
            print("Last entry:", lines[-1][:200])
