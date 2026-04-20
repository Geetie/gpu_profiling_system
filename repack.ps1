# Windows PowerShell 专用 - 正确的打包命令
# 复制以下所有内容到PowerShell中一次性执行

Write-Host "============================================"
Write-Host "  GPU Profiling System - 重新打包"
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================"
Write-Host ""

# 切换到项目目录
Set-Location "E:\GPU_Profiling_System"

# 删除旧的打包文件（如果存在）
if (Test-Path "$HOME\Desktop\gpu_profiling_submit.tar.gz") {
    Remove-Item "$HOME\Desktop\gpu_profiling_submit.tar.gz" -Force
    Write-Host "[✓] 已删除旧文件"
}

Write-Host "[1/4] 正在打包项目文件..."
Write-Host ""

# 使用单行命令（避免续行符问题）
tar -czvf "$HOME\Desktop\gpu_profiling_submit.tar.gz" --exclude='.git' --exclude='kaggle_results' --exclude='__pycache__' --exclude='*.pyc' --exclude='.vscode' --exclude='*.log' src config run.sh report.md output_id.txt docs

Write-Host ""
Write-Host "[2/4] 验证打包结果..."

if (Test-Path "$HOME\Desktop\gpu_profiling_submit.tar.gz") {
    $fileInfo = Get-Item "$HOME\Desktop\gpu_profiling_submit.tar.gz"
    $sizeKB = [math]::Round($fileInfo.Length / 1KB, 2)
    
    Write-Host "[✓] 打包成功!"
    Write-Host ""
    Write-Host "📁 文件信息:"
    Write-Host "   名称: $($fileInfo.Name)"
    Write-Host "   大小: $sizeKB KB ($($fileInfo.Length) bytes)"
    Write-Host "   位置: $($fileInfo.FullName)"
    Write-Host "   时间: $($fileInfo.LastWriteTime)"
    Write-Host ""
    
    Write-Host "[3/4] 检查文件列表..."
    Write-Host ""
    Write-Host "📦 包含的文件/目录:"
    tar -tzf "$HOME\Desktop\gpu_profiling_submit.tar.gz" | ForEach-Object { Write-Host "  ✓ $_" }
    Write-Host ""
    
    # 统计文件数量
    $fileCount = (tar -tf "$HOME\Desktop\gpu_profiling_submit.tar.gz").Count
    Write-Host "[INFO] 总计: $fileCount 个文件/目录"
    Write-Host ""
    
    if ($fileCount -lt 10) {
        Write-Host "[⚠️] 警告: 文件数量过少，可能缺少核心代码!"
        Write-Host "       预期应该有 50+ 个文件"
    } else {
        Write-Host "[✅] 文件数量正常"
    }
} else {
    Write-Host "[❌] 打包失败! 未生成文件"
    exit 1
}

Write-Host ""
Write-Host "[4/4] 下一步操作指南"
Write-Host "============================================"
Write-Host ""
Write-Host "1. 使用 WinSCP 上传此文件到服务器:"
Write-Host "   文件: $HOME\Desktop\gpu_profiling_submit.tar.gz"
Write-Host "   目标: /workspace/"
Write-Host ""
Write-Host "2. 在服务器上解压:"
Write-Host "   cd /workspace && tar -xzvf gpu_profiling_submit.tar.gz"
Write-Host ""
Write-Host "3. 提交评估 (替换YOUR_ID为学号):"
Write-Host '   curl.exe -X POST http://10.176.37.31:8080/submit -H "Content-Type: application/json" -d "{\"id\": \"YOUR_ID\", \"gpu\": 1}"'
Write-Host ""
Write-Host "⭐ 重要: 立即保存返回的 output_file 字段!"
Write-Host "============================================"
