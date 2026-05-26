@echo off
setlocal

set TASK_NAME=Trading Parquet Rollup
set SCRIPT=C:\Users\brian\trading\harvesters\auto_parquet_rollup.bat

schtasks /Create /F /TN "%TASK_NAME%" /SC DAILY /ST 03:00 /TR "%SCRIPT%"

echo.
echo Installed scheduled task: %TASK_NAME%
echo Time: daily 03:00
echo Script: %SCRIPT%
echo Logs: C:\Users\brian\trading\logs\parquet_rollup

endlocal
