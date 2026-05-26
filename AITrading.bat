@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
call "%~dp0start_ai_trading.bat"
exit /b %ERRORLEVEL%
