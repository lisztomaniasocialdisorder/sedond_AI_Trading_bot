@echo off
setlocal

set TASK_NAME=Trading DB To Parquet Hourly
set SCRIPT=C:\Users\brian\trading\harvesters\hourly_db_to_parquet.bat

schtasks /Create /F /TN "%TASK_NAME%" /SC HOURLY /MO 1 /TR "%SCRIPT%"

echo.
echo Installed scheduled task: %TASK_NAME%
echo Frequency: every 1 hour
echo Script: %SCRIPT%
echo Logs: C:\Users\brian\trading\logs\parquet_signal

endlocal

