@echo off
chcp 65001 >nul
title 停止 VoiceMind 声智助手
cd /d "%~dp0"

if not exist "%~dp0funasr_env\Scripts\python.exe" (
    echo [错误] 找不到虚拟环境解释器：%~dp0funasr_env\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

"%~dp0funasr_env\Scripts\python.exe" -u "%~dp0code\stop.py"

echo.
pause
