# GPU Profiling System

**基于多智能体协作的 GPU 硬件性能分析系统**

## 📋 项目简介

本项目是一个基于多智能体协作的 GPU 性能分析系统，能够自动生成 CUDA 微基准测试代码，精确测量 GPU 硬件参数，包括：

- **时钟频率** - GPU 核心频率测量
- **内存层次延迟** - L1/L2/DRAM 延迟测量
- **缓存容量** - L1/L2 缓存容量检测
- **内存带宽** - 共享内存/全局内存带宽测量
- **架构特性** - SM 数量、寄存器文件容量等

## 🎯 核心特性

### 1. 多智能体协作
- **Planner** - 任务分解与规划
- **CodeGen** - CUDA 代码生成
- **MetricAnalysis** - 测量结果分析
- **Verification** - 结果验证与交叉验证

### 2. 安全性保障
- **Sandbox 隔离** - 所有 CUDA 编译和执 行在隔离环境中进行
- **不变量追踪** - M1/M2/M3/M4 不变量自动验证
- **Circuit Breaker** - 异常检测和熔断机制
- **Handoff Validation** - 阶段间数据验证

### 3. 自动化流程
- **自动架构检测** - 智能识别 GPU 架构 (sm_75+)
- **自动编译优化** - 根据架构选择最佳编译选项
- **错误自修复** - CodeGen 支持 3 次编译重试

## 📁 项目结构

```
e:\GPU_Profiling_System/
├── src/                          # 源代码
│   ├── application/              # 应用层 (Agents, Tools, Pipeline)
│   ├── domain/                   # 领域层 (核心业务逻辑)
│   ├── infrastructure/           # 基础设施层 (Sandbox, Probes)
│   └── presentation/             # 展示层 (UI, CLI)
├── config/                       # 配置文件
├── docs/                         # 项目文档
│   ├── spec.md                   # 系统规格说明
│   ├── PJ 需求.md                 # 项目需求
│   └── ...                       # 其他技术文档
├── reports/                      # 审查报告
│   ├── CodeGen 能力审查报告.md
│   ├── Bug 修复审查报告.md
│   └── ...                       # 其他审查报告
├── kaggle_docs/                  # Kaggle 相关文档
│   ├── KAGGLE 文件下载指南.md
│   ├── README_CLEANUP.md
│   └── cleanup.py                # 清理脚本
├── kaggle_results/               # Kaggle 测试结果 (已清理)
│   ├── execution.log             # 执行日志
│   ├── pipeline_log.jsonl        # Pipeline 状态
│   └── audit_report.md           # 审计报告
└── test_output/                  # 测试输出
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.10+
- **CUDA**: 12.0+
- **GPU**: NVIDIA GPU (Compute Capability 6.0+)
- **可选**: Docker (用于生产环境隔离)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API

编辑 `config/api_config.json` 配置 LLM API：

```json
{
  "provider": "longcat",
  "model": "LongCat-Flash-Chat",
  "api_key": "your-api-key"
}
```

### 运行 Pipeline

```bash
# 基本用法
python -m src.main Profile GPU hardware --pipeline --target-spec config/target_spec.json

# 高级选项
python -m src.main Profile GPU hardware \
  --pipeline \
  --target-spec config/target_spec.json \
  --output-dir ./results \
  --state-dir ./.state \
  --no-docker \
  --mode high_autonomy \
  --max-turns 50 \
  --max-tokens 16000
```

## 📊 测量目标示例

### 1. DRAM 延迟测量

```json
{
  "target": "dram_latency_cycles",
  "category": "memory_hierarchy",
  "method": "pointer_chasing"
}
```

### 2. L2 缓存容量测量

```json
{
  "target": "l2_cache_size_mb",
  "category": "cache_capacity",
  "method": "capacity_sweep"
}
```

### 3. 实际 boost 频率测量

```json
{
  "target": "actual_boost_clock_mhz",
  "category": "clock",
  "method": "clock_measurement"
}
```

## 🔍 Kaggle 测试

### 下载测试结果

**核心文件** (必须下载，总计 ~70 KB)：

| 文件 | 路径 | 大小 | 重要性 |
|------|------|------|--------|
| `execution.log` | `/kaggle/working/gpu_profiling_system/` | ~56 KB | ⭐⭐⭐⭐⭐ |
| `pipeline_log.jsonl` | `/kaggle/working/gpu_profiling_system/.state/` | ~1 KB | ⭐⭐⭐⭐⭐ |
| `session_log.jsonl` | `/kaggle/working/gpu_profiling_system/.state/` | ~3 KB | ⭐⭐⭐⭐ |
| `audit_report.md` | `/kaggle/working/gpu_profiling_system/audit/` | ~0.25 KB | ⭐⭐⭐⭐ |
| `results.json` | `/kaggle/working/gpu_profiling_system/` | ~1 KB | ⭐⭐⭐⭐⭐ |

**不要下载** (临时文件，~3 MB)：
- `source.cu` - 临时 CUDA 源文件
- `freq_*` - 编译产物
- 其他二进制文件

### 清理临时文件

```bash
# 运行清理脚本
python kaggle_docs/cleanup.py
```

## 📚 文档索引

### 项目规范
- [`docs/spec.md`](docs/spec.md) - 系统规格说明
- [`docs/PJ 需求.md`](docs/PJ 需求.md) - 项目需求

### 审查报告
- [`reports/CodeGen 能力审查报告.md`](reports/CodeGen 能力审查报告.md) - CodeGen 能力审查
- [`reports/Bug 修复审查报告.md`](reports/Bug 修复审查报告.md) - Bug 修复审查
- [`reports/修复效果审查报告.md`](reports/修复效果审查报告.md) - 修复效果验证

### Kaggle 文档
- [`kaggle_docs/KAGGLE 文件下载指南.md`](kaggle_docs/KAGGLE 文件下载指南.md) - Kaggle 下载指南
- [`kaggle_docs/README_CLEANUP.md`](kaggle_docs/README_CLEANUP.md) - 仓库整理说明
- [`kaggle_docs/cleanup.py`](kaggle_docs/cleanup.py) - 清理脚本

### 技术文档
- [`docs/lecture6pdf-解读.md`](docs/lecture6pdf-解读.md) - 技术讲座解读
- [`docs/第六章讲义简版--Claude Code 内部解读笔记.md`](docs/第六章讲义简版--Claude Code 内部解读笔记.md) - 多智能体基础设施

## 🛠️ 开发工具

### 清理脚本

```bash
# 清理临时文件和编译产物
python kaggle_docs/cleanup.py
```

### Git 工作流

```bash
# 提交前清理
python kaggle_docs/cleanup.py
git status
git add .
git commit -m "描述"
git push origin master
```

## 📈 性能指标

### 测量精度

- **时钟频率**: ±1% 误差
- **缓存容量**: 精确测量
- **内存延迟**: ±5% 误差
- **内存带宽**: ±3% 误差

### 执行时间

- **单次测量**: 1-10 秒
- **完整 Pipeline**: 10-30 分钟
- **Kaggle 测试**: 15-45 分钟

## ⚠️ 注意事项

### 1. Sandbox 隔离

- **生产环境**: 必须使用 DockerSandbox
- **开发环境**: 可使用 LocalSandbox
- **Kaggle 环境**: 自动使用 `.kaggle_sandbox` 子目录

### 2. 架构兼容性

- **最低架构**: sm_75 (Turing+)
- **推荐架构**: sm_80 (Ampere+)
- **CUDA 版本**: 12.0+

### 3. 错误处理

- **编译失败**: CodeGen 自动重试 3 次
- **执行失败**: 自动切换测量方法
- **API 失败**: 自动切换 API 提供商

## 📝 许可证

本项目仅供学习和研究使用。

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请提交 Issue 或联系维护者。

---

**最后更新**: 2026-04-17  
**版本**: 1.0.0
