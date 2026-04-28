# GPU Profiling System - Windows PowerShell 稳定性测试脚本
# 使用方法: .\run_stability_test.ps1

param(
    [string]$ServerIP = "10.176.37.31",
    [string]$StudentID = "",
    [int]$SSHPort = 0
)

$LOCAL_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "GPU Profiling System - 稳定性测试" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Step 1: 检查或启动服务器环境
if ($SSHPort -eq 0) {
    Write-Host ""
    Write-Host "[Step 1] 检查服务器环境..." -ForegroundColor Yellow
    
    # 如果没有提供 SSH 端口，尝试通过 API 启动环境
    if ($StudentID -eq "") {
        Write-Host "错误: 需要指定 StudentID (--StudentID <your_id>) 来自动启动环境" -ForegroundColor Red
        Write-Host ""
        Write-Host "或者手动启动环境后提供 SSH 端口:" -ForegroundColor Yellow
        Write-Host "  .\run_stability_test.ps1 -SSHPort <port_number>" -ForegroundColor White
        exit 1
    }
    
    Write-Host "尝试启动 GPU 环境..." -ForegroundColor Yellow
    try {
        $response = Invoke-RestMethod -Method POST `
            -Uri "http://${ServerIP}:8080/start" `
            -ContentType "application/json" `
            -Body "{`"id`":`"${StudentID}`",`"gpu`":1}"
        
        if ($response.ok) {
            $SSHPort = $response.ssh_port
            Write-Host "✓ 环境已启动! SSH 端口: $SSHPort" -ForegroundColor Green
            Write-Host "  GPU ID: $($response.gpu_id)" -ForegroundColor Green
        } else {
            Write-Host "✗ 环境启动失败: $($response | ConvertTo-Json)" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "✗ API 请求失败: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[Step 1] 使用提供的 SSH 端口: $SSHPort" -ForegroundColor Yellow
}

# Step 2: 测试 SSH 连接
Write-Host ""
Write-Host "[Step 2] 测试 SSH 连接..." -ForegroundColor Yellow
try {
    $testResult = ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSHPort "root@${ServerIP}" "echo 'Connection OK'" 2>&1
    if ($testResult -match "Connection OK") {
        Write-Host "✓ SSH 连接成功!" -ForegroundColor Green
    } else {
        throw "连接测试失败"
    }
} catch {
    Write-Host "✗ SSH 连接失败: $_" -ForegroundColor Red
    Write-Host "请检查:" -ForegroundColor Yellow
    Write-Host "  1. 服务器是否正在运行" -ForegroundColor Yellow
    Write-Host "  2. SSH 端口是否正确" -ForegroundColor Yellow
    Write-Host "  3. 防火墙设置" -ForegroundColor Yellow
    exit 1
}

# Step 3: 清理服务器缓存
Write-Host ""
Write-Host "[Step 3] 清理服务器缓存和旧代码..." -ForegroundColor Yellow
$cleanupCommands = @"
echo '清理构建产物...' &&
rm -rf /workspace/mlsys-project-main/build/* 2>/dev/null &&
rm -rf /workspace/mlsys-project-main/.state/* 2>/dev/null &&
rm -rf /workspace/mlsys-project-main/benchmarks/* 2>/dev/null &&
rm -f /workspace/mlsys-project-main/output.json 2>/dev/null &&
rm -f /workspace/mlsys-project-main/results.log 2>/dev/null &&
echo '清理 Python 缓存...' &&
find /workspace/mlsys-project-main -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true &&
find /workspace/mlsys-project-main -type f -name '*.pyc' -delete 2>/dev/null || true &&
echo '✓ 清理完成'
"@

ssh -o StrictHostKeyChecking=no -p $SSHPort "root@${ServerIP}" $cleanupCommands

# Step 4: 上传新代码
Write-Host ""
Write-Host "[Step 4] 上传优化后的代码..." -ForegroundColor Yellow

$uploadTargets = @(
    @{"local" = "agent"; "remote" = "/workspace/mlsys-project-main/agent"},
    @{"local" = "runner"; "remote" = "/workspace/mlsys-project-main/runner"},
    @{"local" = "llm"; "remote" = "/workspace/mlsys-project-main/llm"},
    @{"local" = "benchmarks"; "remote" = "/workspace/mlsys-project-main/benchmarks"},
    @{"local" = "run.sh"; "remote" = "/workspace/mlsys-project-main/run.sh"},
    @{"local" = "target_spec_sample.json"; "remote" = "/workspace/mlsys-project-main/target_spec_sample.json"}
)

foreach ($target in $uploadTargets) {
    $localPath = Join-Path $LOCAL_DIR $target.local
    $remotePath = $target.remote
    
    if (Test-Path $localPath) {
        Write-Host "  上传 $($target.local)..." -ForegroundColor Gray
        if ($target.local -like "*.sh" -or $target.local -like "*.json") {
            scp -o StrictHostKeyChecking=no -P $SSHPort $localPath "root@${ServerIP}:${remotePath}" 2>&1 | Out-Null
        } else {
            scp -o StrictHostKeyChecking=no -P $SSHPort -r $localPath "root@${ServerIP}:${remotePath}" 2>&1 | Out-Null
        }
    } else {
        Write-Host "  ✗ 跳过 $($target.local) - 文件不存在" -ForegroundColor Red
    }
}

# Step 5: 创建必要目录并验证
Write-Host ""
Write-Host "[Step 5] 创建远程目录..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=no -p $SSHPort "root@${ServerIP}" @"
mkdir -p /workspace/mlsys-project-main/build
mkdir -p /workspace/mlsys-project-main/build/profiles
mkdir -p /workspace/mlsys-project-main/benchmarks
mkdir -p /workspace/mlsys-project-main/.state
chmod +x /workspace/mlsys-project-main/run.sh
echo '✓ 目录创建完成'
"@

Write-Host ""
Write-Host "[Step 6] 验证部署..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=no -p $SSHPort "root@${ServerIP}" @"
echo '检查文件结构...'
test -f /workspace/mlsys-project-main/run.sh && echo '  ✓ run.sh' || echo '  ✗ run.sh 缺失!'
test -f /workspace/mlsys-project-main/agent/agent_framework.py && echo '  ✓ agent_framework.py' || echo '  ✗ agent_framework.py 缺失!'
test -f /workspace/mlsys-project-main/agent/agent.py && echo '  ✓ agent.py' || echo '  ✗ agent.py 缺失!'
test -f /workspace/mlsys-project-main/agent/prompts/generate_benchmark.txt && echo '  ✓ generate_benchmark.txt' || echo '  ✗ generate_benchmark.txt 缺失!'
test -f /workspace/mlsys-project-main/agent/prompts/analyze_metrics.txt && echo '  ✓ analyze_metrics.txt' || echo '  ✗ analyze_metrics.txt 缺失!'
test -f /workspace/mlsys-project-main/runner/run.py && echo '  ✓ run.py' || echo '  ✗ run.py 缺失!'
test -f /workspace/mlsys-project-main/llm/openai_client.py && echo '  ✓ openai_client.py' || echo '  ✗ openai_client.py 缺失!'
echo '✓ 验证完成'
"@

# Step 7: 执行稳定性测试
Write-Host ""
Write-Host "[Step 7] 执行稳定性测试..." -ForegroundColor Yellow
Write-Host "这将需要一些时间，请耐心等待..." -ForegroundColor Gray
Write-Host ""

$testCommands = @"
cd /workspace/mlsys-project-main &&
echo '开始测试...' &&
echo '========================================' &&
bash run.sh 2>&1 | tee /tmp/test_output.log &&
echo '========================================' &&
echo '测试完成'
"@

ssh -o StrictHostKeyChecking=no -p $SSHPort "root@${ServerIP}" $testCommands

# Step 8: 获取结果
Write-Host ""
Write-Host "[Step 8] 获取测试结果..." -ForegroundColor Yellow

Write-Host ""
Write-Host "--- Output JSON ---" -ForegroundColor Cyan
ssh -o StrictHostKeyChecking=no -p $SSHPort "root@${ServerIP}" "cat /workspace/output.json 2>/dev/null || echo '文件不存在'"

Write-Host ""
Write-Host "--- Results Log (最后 100 行) ---" -ForegroundColor Cyan
ssh -o StrictHostKeyChecking=no -p $SSHPort "root@${ServerIP}" "tail -100 /workspace/results.log 2>/dev/null || echo '文件不存在'"

# Step 9: 分析结果
Write-Host ""
Write-Host "[Step 9] 分析测试结果..." -ForegroundColor Yellow

$resultsJson = ssh -o StrictHostKeyChecking=no -p $SSHPort "root@${ServerIP}" "cat /workspace/output.json 2>/dev/null"

if ($resultsJson) {
    $results = $resultsJson | ConvertFrom-Json -ErrorAction SilentlyContinue
    
    if ($results) {
        Write-Host ""
        Write-Host "=== 关键指标 ===" -ForegroundColor Green
        
        # 检查关键指标
        $smThroughput = $results."sm__throughput.avg.pct_of_peak_sustained_elapsed"
        $memThroughput = $results."gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
        
        if ($smThroughput) {
            $color = if ($smThroughput -gt 50) { "Green" } elseif ($smThroughput -gt 30) { "Yellow" } else { "Red" }
            Write-Host "sm__throughput: $smThroughput%" -ForegroundColor $color
        }
        
        if ($memThroughput) {
            $color = if ($memThroughput -gt 30) { "Green" } elseif ($memThroughput -gt 20) { "Yellow" } else { "Red" }
            Write-Host "gpu__compute_memory_throughput: $memThroughput%" -ForegroundColor $color
        }
    }
} else {
    Write-Host "✗ 无法获取测试结果" -ForegroundColor Red
}

# Step 10: 完成
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "稳定性测试完成!" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "要手动检查结果，请运行:" -ForegroundColor Yellow
Write-Host "  ssh -p $SSHPort root@${ServerIP}" -ForegroundColor White
Write-Host "  cat /workspace/output.json" -ForegroundColor White
Write-Host "  cat /workspace/results.log" -ForegroundColor White
