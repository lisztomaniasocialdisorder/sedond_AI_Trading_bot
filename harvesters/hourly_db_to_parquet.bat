@echo off
setlocal
cd /d C:\Users\brian\trading

if not exist logs\parquet_signal mkdir logs\parquet_signal
if not exist logs\locks mkdir logs\locks
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_FILE=C:\Users\brian\trading\logs\parquet_signal\hourly_%RUN_TS%.log
set LOCK_DIR=C:\Users\brian\trading\logs\locks\parquet_pipeline.lock

echo [%date% %time%] start hourly pipeline: spool -^> parquet_rollup > "%LOG_FILE%"
mkdir "%LOCK_DIR%" 2>nul
if errorlevel 1 (
  echo [%date% %time%] skipped: lock exists (%LOCK_DIR%) >> "%LOG_FILE%"
  exit /b 0
)
echo [%date% %time%] lock acquired: %LOCK_DIR% >> "%LOG_FILE%"
echo [%date% %time%] step1/1 compact open-day spool >> "%LOG_FILE%"
python harvesters\compact_parquet_spool.py --symbols BTC,ADA --table all --include-open-day >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :fail

echo [%date% %time%] finished exit=0 >> "%LOG_FILE%"
rmdir "%LOCK_DIR%" 2>nul
exit /b 0

:fail
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] finished exit=%EXIT_CODE% >> "%LOG_FILE%"
rmdir "%LOCK_DIR%" 2>nul
exit /b %EXIT_CODE%
endlocal
