@echo off
setlocal
cd /d C:\Users\brian\trading

if not exist logs\parquet_rollup mkdir logs\parquet_rollup

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_FILE=C:\Users\brian\trading\logs\parquet_rollup\rollup_%RUN_TS%.log

echo [%date% %time%] start parquet rollup > "%LOG_FILE%"
python harvesters\archive_prune_db.py --symbols BTC,ADA --table all --prune --vacuum %* >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] finished with exit code %EXIT_CODE% >> "%LOG_FILE%"

exit /b %EXIT_CODE%
endlocal
