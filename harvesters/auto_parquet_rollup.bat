@echo off
setlocal
cd /d C:\Users\brian\trading

if not exist logs\parquet_rollup mkdir logs\parquet_rollup

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_FILE=C:\Users\brian\trading\logs\parquet_rollup\rollup_%RUN_TS%.log

echo [%date% %time%] start parquet rollup > "%LOG_FILE%"
echo [%date% %time%] stopping harvesters >> "%LOG_FILE%"
call harvesters\stop_harvesters.bat >> "%LOG_FILE%" 2>&1

python harvesters\compact_parquet_spool.py --symbols BTC,ADA --table all --delete-spool %* >> "%LOG_FILE%" 2>&1
set SPOOL_EXIT=%ERRORLEVEL%
echo [%date% %time%] spool compact finished with exit code %SPOOL_EXIT% >> "%LOG_FILE%"
if not "%SPOOL_EXIT%"=="0" (
  echo [%date% %time%] rollup failed, restarting harvesters >> "%LOG_FILE%"
  call harvesters\start_harvesters.bat >> "%LOG_FILE%" 2>&1
  exit /b %SPOOL_EXIT%
)

echo [%date% %time%] start archive_prune_db with timeout 2h >> "%LOG_FILE%"
start "" /B cmd /c "python harvesters\archive_prune_db.py --symbols BTC,ADA --table all --prune --vacuum %* >> \"%LOG_FILE%\" 2>&1"
set WAIT_SEC=7200
:WAIT_PRUNE
timeout /t 5 /nobreak >nul
set /a WAIT_SEC-=5
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match 'archive_prune_db\.py' }; if ($p) { exit 1 } else { exit 0 }"
if "%ERRORLEVEL%"=="0" goto PRUNE_DONE
if %WAIT_SEC% LEQ 0 goto PRUNE_TIMEOUT
goto WAIT_PRUNE

:PRUNE_DONE
echo [%date% %time%] archive_prune_db finished within timeout >> "%LOG_FILE%"
set EXIT_CODE=0
goto AFTER_PRUNE

:PRUNE_TIMEOUT
echo [%date% %time%] archive_prune_db timeout hit, force killing process >> "%LOG_FILE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match 'archive_prune_db\.py' }; if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }"
set EXIT_CODE=124

:AFTER_PRUNE
echo [%date% %time%] finished with exit code %EXIT_CODE% >> "%LOG_FILE%"

echo [%date% %time%] restarting harvesters >> "%LOG_FILE%"
call harvesters\start_harvesters.bat >> "%LOG_FILE%" 2>&1

exit /b %EXIT_CODE%
endlocal
