@echo off
setlocal
cd /d C:\Users\brian\trading

if not exist logs\parquet_rollup mkdir logs\parquet_rollup
if not exist logs\locks mkdir logs\locks
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_FILE=C:\Users\brian\trading\logs\parquet_rollup\midnight_aggregate_%RUN_TS%.log
set LOCK_DIR=C:\Users\brian\trading\logs\locks\parquet_pipeline.lock

echo [%date% %time%] start nightly pipeline: aggregate -^> features -^> obi_edge -^> train > "%LOG_FILE%"
mkdir "%LOCK_DIR%" 2>nul
if errorlevel 1 (
  echo [%date% %time%] skipped: lock exists (%LOCK_DIR%) >> "%LOG_FILE%"
  exit /b 0
)
echo [%date% %time%] lock acquired: %LOCK_DIR% >> "%LOG_FILE%"

echo [%date% %time%] step1/4 compact spool -^> rollup (include open day, keep spool) >> "%LOG_FILE%"
python harvesters\compact_parquet_spool.py --symbols BTC,ADA --table all --include-open-day >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :fail

echo [%date% %time%] step2/4 parquet -^> features parquet >> "%LOG_FILE%"
python harvesters\parquet_to_features.py --symbols BTC,ADA >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :fail

echo [%date% %time%] step3/4 build OBI edge report >> "%LOG_FILE%"
py -3.11 research\obi_edge_report.py --symbols BTC,ADA --days 7 --horizon-sec 5 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] py -3.11 failed, fallback to python >> "%LOG_FILE%"
  python research\obi_edge_report.py --symbols BTC,ADA --days 7 --horizon-sec 5 >> "%LOG_FILE%" 2>&1
  if errorlevel 1 goto :fail
)

echo [%date% %time%] step4/4 train model >> "%LOG_FILE%"
py -3.11 run.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] py -3.11 failed, fallback to python >> "%LOG_FILE%"
  python run.py >> "%LOG_FILE%" 2>&1
  if errorlevel 1 goto :fail
)

echo [%date% %time%] finished exit=0 >> "%LOG_FILE%"
rmdir "%LOCK_DIR%" 2>nul
exit /b 0

:fail
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] finished exit=%EXIT_CODE% >> "%LOG_FILE%"
rmdir "%LOCK_DIR%" 2>nul
exit /b %EXIT_CODE%
endlocal
