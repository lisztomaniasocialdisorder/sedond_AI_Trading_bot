@echo off
setlocal
cd /d "%~dp0"

echo [INFO] Stopping BTC/ADA harvesters...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$targets = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match 'btc_harvester\.py' -or $_.CommandLine -match 'ada_harvester\.py' };" ^
  "if (-not $targets) { Write-Host 'No harvester process found.'; exit 0 };" ^
  "$targets | ForEach-Object { Write-Host ('Stopping PID ' + $_.ProcessId + ' :: ' + $_.CommandLine); Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue };" ^
  "Start-Sleep -Seconds 2;" ^
  "$left = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match 'btc_harvester\.py' -or $_.CommandLine -match 'ada_harvester\.py' };" ^
  "if ($left) { $left | ForEach-Object { Write-Host ('Force kill PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }"

endlocal
