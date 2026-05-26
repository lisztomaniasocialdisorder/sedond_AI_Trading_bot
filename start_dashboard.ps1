param()

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$teacherOutput = "C:\Users\brian\OneDrive\桌面\btc_teacher_outputs_ult\outputs"
if ((-not $env:TEACHER_OUTPUT_DIR) -and (Test-Path -LiteralPath $teacherOutput)) {
  $env:TEACHER_OUTPUT_DIR = $teacherOutput
}

$port = 8501
$url = "http://127.0.0.1:$port"

$streamlitPath = $null
if (Test-Path -LiteralPath ".venv311\Scripts\streamlit.exe") {
  $streamlitPath = ".venv311\Scripts\streamlit.exe"
} elseif (Test-Path -LiteralPath ".venv\Scripts\streamlit.exe") {
  $streamlitPath = ".venv\Scripts\streamlit.exe"
} else {
  $cmd = Get-Command streamlit -ErrorAction SilentlyContinue
  if ($cmd) {
    $streamlitPath = $cmd.Source
  }
}

if (-not $streamlitPath) {
  Write-Host "[ERROR] Cannot find streamlit executable (.venv311/.venv/PATH)."
  exit 1
}

if (-not (Test-Path -LiteralPath "dashboard.py")) {
  Write-Host "[ERROR] Cannot find dashboard.py"
  exit 1
}

# Prefer existing healthy service bound to the target port.
try {
  $health = Invoke-WebRequest -UseBasicParsing -Uri "$url/_stcore/health" -TimeoutSec 2
  if (($health.Content -as [string]) -match "ok") {
    Write-Host "[INFO] Dashboard already healthy on $url. Opening browser only."
    Start-Process $url
    exit 0
  }
} catch {}

$existing = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -in @("streamlit.exe", "python.exe", "pythonw.exe")) -and
  $_.CommandLine -like "*streamlit*" -and
  $_.CommandLine -like "*run*dashboard.py*"
}

if ($existing) {
  if ($existing.Count -gt 1) {
    Write-Host "[WARN] Multiple dashboard processes detected ($($existing.Count)); restarting cleanly."
    $existing | ForEach-Object {
      try { Stop-Process -Id $_.ProcessId -Force } catch {}
    }
    Start-Sleep -Seconds 1
  } else {
    Write-Host "[INFO] Dashboard process already running. Opening browser only."
    Start-Process $url
    exit 0
  }
}

Write-Host "[INFO] Starting dashboard service on port $port..."
Start-Process -FilePath $streamlitPath -ArgumentList @("run", "dashboard.py", "--server.port", $port, "--server.headless", "true") -WindowStyle Hidden
Start-Sleep -Seconds 2
Start-Process $url
Write-Host "[INFO] Dashboard opening: $url"
