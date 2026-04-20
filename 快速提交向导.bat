@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════════════╗
echo ║     GPU Profiling System - 快速提交向导          ║
echo ╚══════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

REM 检查必要文件
if not exist run.sh (
    echo [ERROR] 未找到 run.sh 文件
    pause & exit /b 1
)

if not exist src\main.py (
    echo [ERROR] 未找到 src\main.py 文件
    pause & exit /b 1
)

echo [✓] 项目文件检查通过
echo.

REM 打包
echo [1/3] 正在打包项目文件...

set OUTPUT_FILE=%USERPROFILE%\Desktop\gpu_profiling_%date:~0,4%%date:~5,2%%date:~8,2%.tar.gz

tar -czvf "%OUTPUT_FILE%" ^
    --exclude='.git' ^
    --exclude='kaggle_results' ^
    --exclude='__pycache__' ^
    --exclude='*.pyc' ^
    --exclude='*.log' ^
    src config run.sh report.md output_id.txt docs >nul 2>&1

if %errorlevel% neq 0 (
    echo [ERROR] 打包失败
    pause & exit /b 1
)

echo [✓] 打包成功!
echo.
echo [2/3] 文件信息:
for %%A in ("%OUTPUT_FILE%") do (
    echo     文件名: %%~nxA
    echo     大小: %%~zA bytes
    echo     位置: %%~dpA
)
echo.
echo [3/3] 下一步操作说明:
echo.
echo ┌────────────────────────────────────────────────────┐
echo │  方法1: WinSCP (推荐)                              │
echo │  ① 打开 WinSCP                                    │
echo │  ② 连接 10.176.37.21:22                           │
echo │  ③ 拖拽桌面上的 %OUTPUT_FILE% 到 /workspace/       │
echo │  ④ 右键解压                                        │
echo ├────────────────────────────────────────────────────┤
echo │  方法2: PowerShell                                 │
echo │  scp "%OUTPUT_FILE%" user@10.176.37.31:/workspace/ │
echo ├────────────────────────────────────────────────────┤
echo │  解压命令 (在服务器上):                            │
echo │  cd /workspace ^&^& tar -xzvf gpu_profiling_*.tgz   │
echo ├────────────────────────────────────────────────────┤
echo │  提交命令:                                         │
echo │  curl -X POST http://10.176.37.31:8080/submit      │
echo │    -H "Content-Type: application/json"             │
echo │    -d "{\"id\": \"YOUR_ID\", \"gpu\": 1}"            │
echo └────────────────────────────────────────────────────┘
echo.
echo ⚠️  重要: 请立即保存返回的 output_file 字段!
echo.
pause
