@echo off
chcp 65001 > nul
title Futures Dashboard Server

echo =========================================
echo   Signal Dashboard
echo   http://localhost:5000/signal_dashboard.html
echo =========================================
echo.

cd /d "%~dp0"

py -3.11 -m pip install flask pandas pyarrow tensorflow --quiet

echo Starting server...
echo.
start "Open Dashboard" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:5000/signal_dashboard.html'"
py -3.11 server.py

pause
