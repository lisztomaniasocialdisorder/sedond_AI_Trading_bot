@echo off
setlocal
cd /d C:\Users\brian\trading

if not exist logs\strategy_recalibrate mkdir logs\strategy_recalibrate
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_FILE=C:\Users\brian\trading\logs\strategy_recalibrate\weekly_strategy_%RUN_TS%.log

echo [%date% %time%] start weekly strategy recalibrate > "%LOG_FILE%"
py -3.11 harvesters\weekly_strategy_recalibrate.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] py -3.11 failed, fallback to python >> "%LOG_FILE%"
  python harvesters\weekly_strategy_recalibrate.py >> "%LOG_FILE%" 2>&1
)
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] finished exit=%EXIT_CODE% >> "%LOG_FILE%"
exit /b %EXIT_CODE%
endlocal
