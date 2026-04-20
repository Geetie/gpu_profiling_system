@echo off
REM ============================================================
REM  GPU Profiling System - Windows Submit Script
REM  用于在Windows环境下打包并准备提交MLSYS项目
REM ============================================================

echo.
echo ============================================
echo   GPU Profiling System - 打包工具
echo   %date% %time%
echo ============================================
echo.

cd /d "%~dp0"

REM 检查tar是否可用
where tar >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到tar命令
    echo 请使用Windows 10 (1803+) 或安装Git for Windows
    pause
    exit /b 1
)

echo [INFO] 检测到tar版本:
tar --version | findstr "tar"
echo.

REM 创建临时目录用于打包
set TEMP_DIR=%TEMP%\gpu_submit_%random%
mkdir "%TEMP_DIR%" 2>nul

echo [STEP 1] 复制必要文件...
REM 复制核心文件（排除.git, kaggle_results, __pycache__, *.pyc等）
xcopy /E /I /Q src "%TEMP_DIR%\src\" >nul 2>&1
xcopy /E /I /Q config "%TEMP_DIR%\config\" >nul 2>&1
copy run.sh "%TEMP_DIR%\" >nul 2>&1
copy report.md "%TEMP_DIR%\" >nul 2>&1
copy output_id.txt "%TEMP_DIR%\" >nul 2>&1

REM 删除__pycache__和.pyc文件
for /r "%TEMP_DIR%" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q "%TEMP_DIR%\*.pyc" 2>nul

echo [STEP 2] 打包项目文件...

REM 切换到临时目录并打包
pushd "%TEMP_DIR%"
tar -czvf ..\gpu_profiling_submit.tar.gz . --exclude='.git' --exclude='kaggle_results' --exclude='__pycache__' --exclude='*.pyc'
popd

REM 移动到用户桌面方便查找
move "%TEMP_DIR%\..\gpu_profiling_submit.tar.gz" "%USERPROFILE%\Desktop\"

echo.
echo [SUCCESS] 打包完成!
echo.
echo 📁 文件位置: %USERPROFILE%\Desktop\gpu_profiling_submit.tar.gz
echo 📊 文件大小: 
for %%A in ("%USERPROFILE%\Desktop\gpu_profiling_submit.tar.gz") do echo    ~%%~zA bytes
echo.

REM 清理临时目录
rd /s /q "%TEMP_DIR%" 2>nul

echo ============================================
echo   下一步操作:
echo ============================================
echo.
echo 1. 使用SCP工具上传到服务器:
echo    scp %USERPROFILE%\Desktop\gpu_profiling_submit.tar.gz your_user@10.176.37.31:/workspace/
echo.
echo 2. 或使用WinSCP/FileZilla图形界面上传
echo.
echo 3. 上传后在服务器执行:
echo    cd /workspace ^&^& tar -xzvf gpu_profiling_submit.tar.gz
echo.
echo 4. 提交评估:
echo    curl -X POST http://10.176.37.31:8080/submit -H "Content-Type: application/json" -d "{\"id\": \"YOUR_ID\", \"gpu\": 1}"
echo.
echo ============================================
pause
