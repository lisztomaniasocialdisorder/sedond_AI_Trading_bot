@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$btc = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match 'btc_harvester\.py' };" ^
  "if ($btc) { Write-Host 'BTC harvester already running.' } else { Start-Process cmd -ArgumentList '/k python btc_harvester.py' -WorkingDirectory '%~dp0' -WindowStyle Normal };" ^
  "$ada = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match 'ada_harvester\.py' };" ^
  "if ($ada) { Write-Host 'ADA harvester already running.' } else { Start-Process cmd -ArgumentList '/k python ada_harvester.py' -WorkingDirectory '%~dp0' -WindowStyle Normal };"

endlocal
