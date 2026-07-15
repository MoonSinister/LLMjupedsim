$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = $env:JUPEDSIM_PYTHON
if (-not $PythonExe) {
  $PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

Push-Location $ProjectRoot

& $PythonExe "src\run_experiment_suite.py" @args

Pop-Location
