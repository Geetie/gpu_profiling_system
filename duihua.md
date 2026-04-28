## 找到基准测试规模过小的根本原因！

### 🔴 问题分析

从提示词中可以看到明确的硬编码要求：

- IGNORE --size/--repeats. HARDCODE iterations=10000000, grid=sm\_count\*4 blocks x 256 threads
- IGNORE --size/--repeats. HARDCODE n=16777216, grid=sm\_count\*4 blocks x 256 threads
  但是 ，从日志看到 Grid Size = 2，这说明 LLM 没有遵循这些指令 ！

### 🔍 检查是否有验证机制

让我搜索是否有对生成代码的参数验证：

在工作区搜索 'grid|block|threads|iteration'

Thought

让我查看 \_detect\_kernel\_quality\_issues 函数，看看是否有对参数的验证：

src\infrastructure\tools\compile\_cuda.py

Thought

让我继续查看 \_detect\_kernel\_quality\_issues 函数，看看是否有对 grid size 的验证：

src\infrastructure\tools\compile\_cuda.py

Thought

## 🔍 发现问题：缺少对 Grid Size 的验证

从代码分析可以看到， \_detect\_kernel\_quality\_issues 函数检查了：

- clock64() 的使用
- cudaDevAttrClockRate 的禁止
- float/double 类型
- volatile 关键字
- 全局内存读写位置
- warmup 执行
- \#pragma unroll
  但是 ，它 没有检查 grid size ！这就是为什么 LLM 生成的代码使用了 Grid Size = 2，远小于要求的 sm\_count\*4 。

让我添加对 grid size 的检测：

src\infrastructure\tools\compile\_cuda.py

Thought

现在让我添加对 grid size 的检测。我需要检查 kernel 启动配置是否足够大：

compile\_cuda.py
src\infrastructure\tools\compile\_cuda.py

- 2 - 0

查看变更

Thought

现在让我添加对 grid size 的检测逻辑：

compile\_cuda.py
src\infrastructure\tools\compile\_cuda.py

- 4 - 0

查看变更

Thought

现在让我在函数末尾添加对 grid size 的验证：

compile\_cuda.py
src\infrastructure\tools\compile\_cuda.py

- 15 - 0

查看变更

Thought

现在让我上传修复后的代码：

GPU\_Profiling\_System
白名单运行
在终端查看

$ scp -o StrictHostKeyChecking=no -P 39333 e:\GPU\_Profiling\_System\src\infrastructure\tools\compile\_cuda.py root\@10.176.37.31:/workspace/mlsys-project-main/agent/src/infrastructure/tools/

compile\_cuda.py 100% 29KB 2.6MB/s 00:00

Thought

## 🎯 基准测试规模问题分析与修复

### 🔍 根本原因分析

从日志看到 Grid Size = 2（对于82个SM的GPU来说太小），原因是：

1. 提示词指令未被遵循 ：提示词明确告诉LLM要：
   - IGNORE --size/--repeats
   - HARDCODE grid=sm\_count\*4 blocks x 256 threads
   - 但LLM生成的代码使用了很小的固定值
2. 缺少验证机制 ： \_detect\_kernel\_quality\_issues 函数没有检查 grid size 是否足够大

### ✅ 修复方案

在 compile\_cuda.py 中添加了对 grid size 的检测：

```
# 检测 kernel 启动配置
if '<<<' in line and '>>>' in line:
    if 'sm_count' in line.lower() or '*' in line:
        grid_size_ok = True

# 验证阶段
if not grid_size_ok:
    warnings.append("KERNEL QUALITY WARNING: Grid 
    size may be too small! "
                   "You MUST use sm_count*4 blocks 
                   to saturate the GPU.")
if not uses_sm_count:
    warnings.append("KERNEL QUALITY WARNING: 
    sm_count not detected in code.")
```

### 📋 修复内容汇总

问题 文件 修复内容 检测顺序错误 compile\_cuda.py 检测移至枚举替换之前 迭代循环限制过严 pipeline.py 从1轮增加到3轮 迭代限制错误 agent.py 从 < 1 改为 < 3 基准测试规模过小 compile\_cuda.py 新增 grid size 检测\
\
📊 日志深度分析报告

### ✅ 好消息：迭代优化循环开始工作了！

```
[Pipeline] Stage verification completed in 22.93s 
with status: rejected
[Pipeline] DEBUG: Verification REJECTED — 
iteration_count=0, max_iterations=3, can_retry=True
[Pipeline] Verification REJECTED (iteration 1/3). 
Retrying from CodeGen with feedback.
```

这证明修复的迭代循环限制（从1轮增加到3轮）起作用了！

### 🔴 问题分析

指标 值 状态 sm\_\_throughput 100.0% ❌ 仍然不正确 gpu\_\_compute\_memory\_throughput 40.29% ✅ 合理

根本原因 ：从日志第443行可以看到：

```
[compile_cuda] HARNESS: 1 kernel quality issue(s) 
detected for target 'sm__throughput.avg.
pct_of_peak_sustained_elapsed'
```

检测到了问题，但只是警告，代码仍然被编译执行了！

LLM生成的代码问题 ：

- 编译错误： calling a **device** function("clock64") from a **host** function
- LLM把 clock64() 放在了主机代码中，而不是kernel中！

### 🎯 已实施的修复

修复项 文件 修复内容 检测顺序错误 compile\_cuda.py 检测移至枚举替换之前 迭代循环限制过严 pipeline.py 从1轮增加到3轮 迭代限制错误 agent.py 从 < 1 改为 < 3 基准测试规模检测 compile\_cuda.py 新增 grid size 检测 添加调试信息 compile\_cuda.py 新增 DEBUG 日志输出

