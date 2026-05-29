@echo off
setlocal

set TASK_NAME=Trading Weekly DB Prune
set SCRIPT=C:\Users\brian\trading\harvesters\weekly_prune_db.bat

schtasks /Create /F /TN "%TASK_NAME%" /SC WEEKLY /D SUN /ST 01:30 /TR "%SCRIPT%"

echo.
echo Installed scheduled task: %TASK_NAME%
echo Schedule: Every Sunday 01:30
echo Script: %SCRIPT%
echo Logs: C:\Users\brian\trading\logs\parquet_prune

endlocal

