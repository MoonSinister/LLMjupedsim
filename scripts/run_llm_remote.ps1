$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

& "D:\anaconda\envs\python3.12\python.exe" "src\demo_map_simulation.py" `
  -n 200 `
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
  --output "outputs\trajectories\demo_map_llm_remote.sqlite" `
  --llm-routing `
  --llm-batch-size 25 `
  --llm-max-tokens 6000 `
  --llm-plan-output "outputs\plans\demo_map_llm_plan.json" `
  --region-target-margin 0.25 `
  --waypoint-distance 1.8 `
  --routing-waypoint-distance 1.4 `
  --routing-waypoint-max-per-leg 8 `
  --llm-base-url "http://127.0.0.1:8600/v1" `
  --llm-model "qwen3.6-27b:q8"

Pop-Location
