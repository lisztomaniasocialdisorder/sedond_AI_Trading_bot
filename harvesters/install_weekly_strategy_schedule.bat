@echo off
setlocal

set TASK_NAME=Trading Weekly Strategy Recalibrate
set SCRIPT=C:\Users\brian\trading\harvesters\weekly_strategy_recalibrate.bat

schtasks /Create /F /TN "%TASK_NAME%" /SC WEEKLY /D SUN /ST 02:30 /TR "%SCRIPT%"

echo.
echo Installed scheduled task: %TASK_NAME%
echo Schedule: Every Sunday 02:30
echo Script: %SCRIPT%
echo Logs: C:\Users\brian\trading\logs\strategy_recalibrate

endlocal
