@echo off
chcp 65001 >nul 2>&1
echo.
echo ============================================
echo   MLSYS Project - API Submit Test
echo   %date% %time%
echo ============================================
echo.

REM Create JSON file
echo {"id": "23301010041", "gpu": 1} > "%TEMP%\submit.json"

echo [INFO] Testing submit-test endpoint...
echo.

REM Use curl to submit (test mode)
curl.exe -s -X POST http://10.176.37.31:8080/submit-test ^
  -H "Content-Type: application/json" ^
  -d @"%TEMP%\submit.json"

echo.
echo ============================================
echo If you see a JSON response above with "ok": true,
echo then the API is working!
echo.
echo Next steps:
echo 1. Check if SSH is really needed, or
echo 2. Contact TA for correct credentials
echo ============================================
pause
