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

& $PythonExe "src\atc_reference.py" `
  --atc-raw-path "data\atc-20121114\atc-20121114.csv" `
  --atc-regions "data\map\localization_grid_regions.json" `
  --max-persons 500 `
  --min-points 300 `
  --output "outputs\summaries\atc_20121114_reference_metrics.json" `
  @args

Pop-Location
