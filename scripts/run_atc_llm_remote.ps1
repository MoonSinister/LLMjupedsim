$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

$PythonExe = "D:\anaconda\envs\python3.12\python.exe"
if (-not (Test-Path $PythonExe)) {
  $PythonExe = "C:\Users\MoonSinister\miniconda3\python.exe"
}

$AtcProject = Get-ChildItem "D:\AAAWorkSpace\file" -Directory |
  ForEach-Object { Join-Path $_.FullName "ATC-map" } |
  Where-Object { Test-Path $_ } |
  Select-Object -First 1

if (-not $AtcProject) {
  throw "Could not find ATC-map under D:\AAAWorkSpace\file"
}

$AtcRawPath = Join-Path $AtcProject "data\raw\atc-tracking-part1"
$AtcRegions = Join-Path $AtcProject "data\map\localization_grid_regions.json"

Push-Location $ProjectRoot

& $PythonExe "src\demo_map_simulation.py" `
  --profile-source "atc" `
  --atc-raw-path $AtcRawPath `
  --atc-regions $AtcRegions `
  --atc-max-persons 120 `
  --atc-min-points 300 `
  --atc-profile-cache "outputs\profiles\demo_map_atc_profiles.json" `
  -n 120 `
  --spawn-interval 300 `
  --spawn-jitter 180 `
  --max-iters 150000 `
  --movement-model "avm" `
  --agent-radius 0.11 `
  --agent-time-gap 0.9 `
  --neighbor-repulsion-strength 10.0 `
  --neighbor-repulsion-range 0.25 `
  --avm-wall-buffer-distance 0.05 `
  --avm-anticipation-time 1.8 `
  --avm-reaction-time 0.2 `
  --max-active-agents 70 `
  --agent-max-lifetime-seconds 240 `
  --congestion-check-interval 300 `
  --congestion-grace-iterations 900 `
  --stuck-distance-threshold 0.06 `
  --stuck-checks-before-reroute 4 `
  --congestion-release-radius 2.0 `
  --output "outputs\trajectories\demo_map_atc_llm_remote.sqlite" `
  --llm-routing `
  --llm-batch-size 25 `
  --llm-max-tokens 6000 `
  --llm-plan-output "outputs\plans\demo_map_atc_llm_plan.json" `
  --region-target-margin 0.25 `
  --waypoint-distance 1.8 `
  --routing-waypoint-distance 1.4 `
  --routing-waypoint-max-per-leg 8 `
  --llm-base-url "http://127.0.0.1:8600/v1" `
  --llm-model "qwen3.6-27b:q8"

Pop-Location
