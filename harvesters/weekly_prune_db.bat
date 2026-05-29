@echo off
setlocal
cd /d C:\Users\brian\trading

if not exist logs\parquet_prune mkdir logs\parquet_prune
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_FILE=C:\Users\brian\trading\logs\parquet_prune\weekly_prune_%RUN_TS%.log

echo [%date% %time%] start weekly prune > "%LOG_FILE%"
echo [%date% %time%] stopping harvesters >> "%LOG_FILE%"
call harvesters\stop_harvesters.bat >> "%LOG_FILE%" 2>&1

echo [%date% %time%] run archive_prune_db --prune --vacuum >> "%LOG_FILE%"
python harvesters\archive_prune_db.py --symbols BTC,ADA --table all --prune --vacuum >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] restarting harvesters >> "%LOG_FILE%"
call harvesters\start_harvesters.bat >> "%LOG_FILE%" 2>&1

echo [%date% %time%] finished exit=%EXIT_CODE% >> "%LOG_FILE%"
exit /b %EXIT_CODE%
endlocal

