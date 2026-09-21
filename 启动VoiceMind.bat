@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 已有 Web 服务在运行：只重新打开页面，不重复启动
netstat -ano | findstr "LISTENING" | findstr ":8000" >nul
if %errorlevel%==0 (
    start "" http://127.0.0.1:8000/index.html
    exit
)

REM 用 pythonw.exe 无窗口常驻启动（日志见 logs\launcher.log）
if exist "%~dp0funasr_env\Scripts\pythonw.exe" (
    start "" "%~dp0funasr_env\Scripts\pythonw.exe" "%~dp0code\launcher.py"
)
exit
