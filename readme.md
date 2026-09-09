# JuPedSim Mall Crowd Simulation

This project runs an indoor crowd simulation on a drawn mall/building map. It combines:

- JuPedSim physical simulation
- semantic map regions
- local/remote LLM route planning
- optional ATC or LLMob-derived pedestrian profiles

Run commands from the project root:

```powershell
cd D:\AAAWorkSpace\code\Jupedsim
```

## Directory Layout

```text
.
|-- src/                         # Python source code
|   |-- jupedsim_mall/            # Main Python package
|   |   |-- simulation/           # JuPedSim execution
|   |   |-- geometry/             # Map editing and conversion
|   |   |-- profiles/             # Mall/ATC/LLMob profiles
|   |   |-- planning/             # LLM prompts and route planning
|   |   |-- experiments/          # Scenario validation and batch runs
|   |   `-- analysis/             # Metrics and realism evaluation
|   `-- *.py                      # Backward-compatible script entry points
|-- configs/
|   |-- scenarios/                # Reproducible experiment scenarios
|   `-- schemas/                  # Versioned JSON Schemas
|-- scripts/                      # PowerShell launch scripts
|   |-- run_llm_remote.ps1        # LLM-routed mall simulation
|   `-- run_atc_llm_remote.ps1    # ATC-profile + LLM-routed simulation
|-- data/
|   `-- map/                      # Map inputs and manually drawn geometry
|       |-- geometry.wkt
|       |-- stages.json
|       |-- localization_grid_regions.json
|       |-- drawn_geometry.json
|       `-- localization_grid.pgm
|-- outputs/
|   |-- runs/<run_id>/            # Isolated run artifacts and manifest
|   |-- trajectories/             # Main SQLite trajectory outputs
|   |-- plans/                    # LLM per-agent intent/route plans
|   |-- profiles/                 # ATC/LLMob extracted profile caches
|   `-- smoke/                    # Small test/smoke output files
|-- docs/assets/                  # Figures and documentation assets
|-- jupedsim/                     # Upstream JuPedSim source checkout
|-- tests/                        # Unit, integration, fixture, and golden tests
`-- pyproject.toml                # Package metadata and unified CLI
```

## Environment

Install the Python dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

## One-click Start

On Windows, double-click `start.bat` from the project root to run the default
120-agent `baseline_random` scenario. The launcher selects Python in this order: `JUPEDSIM_PYTHON`,
`.venv\Scripts\python.exe`, `D:\anaconda\envs\python3.12\python.exe`, then
`python` on `PATH`.

Command-line examples:

```powershell
.\start.bat
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1 -Agents 200
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1 -List
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1 -Scenario baseline_random
```

After a run finishes, double-click `visualize.bat` to open the newest
`trajectory.sqlite` in the JuPedSim visualizer. You can also choose a specific
trajectory:

```powershell
.\visualize.bat
.\visualize.bat -Path outputs\runs\<run_id>\trajectory.sqlite
```

For presentation, double-click `visualize_3d.bat` to export the newest
trajectory as an interactive Three.js browser replay:

```powershell
.\visualize_3d.bat
.\visualize_3d.bat -MaxAgents 300 -FrameStride 10
.\visualize_3d.bat -Path outputs\runs\<run_id>\trajectory.sqlite
```

The exported page is written to `outputs\visualizations\latest_3d_replay.html`.
Use `visualize_3d.bat` to open it because the launcher serves the page from a
local `http://127.0.0.1` address; opening the HTML directly as `file://` may
trigger browser module-security errors.

The exact direct dependency versions used for the batch-A baseline are recorded
in `requirements-lock.txt`.

By default, the PowerShell scripts first use `JUPEDSIM_PYTHON` when it is set,
then fall back to `.venv\Scripts\python.exe`, and finally
to `python` on `PATH`.

Example:

```powershell
$env:JUPEDSIM_PYTHON='D:\anaconda\envs\python3.12\python.exe'
```

## Remote LLM Setup

The LLM server is remote, so first create an SSH tunnel:

```powershell
ssh -L 8600:127.0.0.1:8600 file_share
```

The scripts call the OpenAI-compatible endpoint:

```text
http://127.0.0.1:8600/v1
```

Current model name:

```text
qwen3.6-27b:q8
```

The scripts can be configured without editing files:

```powershell
$env:LOCAL_LLM_BASE_URL='http://127.0.0.1:8600/v1'
$env:LOCAL_LLM_MODEL='qwen3.6-27b:q8'
```

For ATC-profile runs, the repository sample is used by default. Set
`ATC_RAW_PATH` and optionally `ATC_REGIONS` to use a larger external dataset.

## Run Simulations

The package provides a unified command line interface. When the package has not
been installed in editable mode, set `PYTHONPATH=src` or continue to use the
compatible scripts shown below.

```powershell
$env:PYTHONPATH='src'
python -m jupedsim_mall doctor
python -m jupedsim_mall scenarios list
python -m jupedsim_mall scenarios validate
python -m jupedsim_mall map validate --map data/map `
  --output outputs/datasets/map_quality_report.json `
  --preview outputs/datasets/map_quality_preview.png
python -m jupedsim_mall run --dry-run smoke_baseline
python -m jupedsim_mall run smoke_baseline
python -m jupedsim_mall runs verify outputs/runs
```

Prepare and audit a paired-seed experiment matrix before launching it:

```powershell
python -m jupedsim_mall matrix prepare configs/matrices/thesis_core_v1.json
python -m jupedsim_mall matrix run outputs/matrices/thesis_core_v1/<fingerprint>/plan.json --dry-run
python -m jupedsim_mall matrix run outputs/matrices/thesis_core_v1/<fingerprint>/plan.json
python -m jupedsim_mall matrix status outputs/matrices/thesis_core_v1/<fingerprint>/plan.json
python -m jupedsim_mall matrix resume outputs/matrices/thesis_core_v1/<fingerprint>/plan.json --retry-failed
python -m jupedsim_mall matrix cancel-local outputs/matrices/thesis_core_v1/<fingerprint>/plan.json
```

Run the complete analysis chain for an already completed plan, or add `--run`
to execute the matrix first:

```powershell
python -m jupedsim_mall pipeline outputs/matrices/<name>/<fingerprint>/plan.json `
  --reference outputs/summaries/atc_reference_20121114_metrics.json `
  --output-dir outputs/formal-analysis
```

Before a formal batch, run the full gate and freeze an immutable snapshot:

```powershell
python -m jupedsim_mall quality-gate
python -m jupedsim_mall experiment preflight configs/matrices/thesis_core_v1.json --check-llm
python -m jupedsim_mall experiment freeze configs/matrices/thesis_core_v1.json --check-llm
```

Matrix plans use stable fingerprints and frozen seed lists. Completed runs with
matching fingerprints are skipped during resume. Failed attempts remain in the
plan history, while retries receive new run IDs. CPU concurrency is capped by
the configured worker and memory budgets; LLM runs have a separate concurrency
limit. Cancellation is cooperative so active SQLite writers can close cleanly.

Prepare an external profile source without starting a simulation:

```powershell
python -m jupedsim_mall profiles prepare `
  --source atc `
  --data-path data/atc-20121114/atc-20121114.csv `
  --regions data/map/localization_grid_regions.json `
  --max-persons 50 `
  --cache-output outputs/profiles/atc_profiles.json `
  --report-output outputs/datasets/atc_profile_report.json
```

List reproducible scenario configs:

```powershell
.\scripts\run_scenario.ps1 --list
```

Dry-run two scenarios without starting simulation:

```powershell
.\scripts\run_scenario.ps1 --dry-run baseline_random llm_remote
```

Validate scenario JSON files before running a large experiment batch:

```powershell
D:\anaconda\envs\python3.12\python.exe src\validate_scenarios.py
```

Run repeated experiments with controlled seeds:

```powershell
.\scripts\run_scenario.ps1 --repeat 5 --seed-start 2026 baseline_random baseline_nearest llm_exit_only llm_remote
```

Run a configured smoke test:

```powershell
.\scripts\run_scenario.ps1 smoke_baseline
```

Run the standard LLM-routed simulation:

```powershell
.\scripts\run_llm_remote.ps1
```

Run the ATC-profile-based LLM simulation:

```powershell
.\scripts\run_atc_llm_remote.ps1
```

Run the disabled-by-default LLMob-profile scenario after setting the LLMob data
root if the default path is not available:

```powershell
$env:LLMOB_DATA_ROOT='D:\AAAWorkSpace\code\LLMob\LLMob\data'
.\scripts\run_scenario.ps1 --include-disabled llmob_llm_remote
```

Run a small no-LLM smoke test:

```powershell
$env:PYTHONUTF8='1'
D:\anaconda\envs\python3.12\python.exe src\demo_map_simulation.py `
  -n 10 `
  --max-iters 5000 `
  --movement-model avm `
  --output outputs\smoke\demo_map_smoke.sqlite
```

Replay a saved plan without contacting the LLM service:

```powershell
python src/demo_map_simulation.py `
  -n 10 --seed 2026 --max-iters 5000 `
  --replay-plan outputs/plans/smoke_baseline_plan_seed2026_run01.json `
  --output outputs/smoke/replay.sqlite `
  --llm-plan-output outputs/smoke/replay_plan.json
```

LLM planning responses are cached under `outputs/llm_cache` by default. Call
metadata, prompt hashes, token usage, latency, retries, and failures are written
to `outputs/runs/llm_calls.jsonl`.

## Outputs

Configured runs now write all related artifacts into one
`outputs/runs/<scenario>_seed<seed>_<timestamp>_<hash>/` directory:

- `trajectory.sqlite`: JuPedSim trajectory database.
- `plans.json`: agent profiles, routes, activities, reroutes and terminal states.
- `events.jsonl`: spawn, wait, reroute, completion and removal events.
- `llm_calls.jsonl`: LLM requests, cache hits, latency and failures when applicable.
- `resolved_scenario.json`: complete resolved arguments, environment and seed.
- `run.log`: combined simulation stdout/stderr.
- `manifest.json`: status, timings, hashes, versions and artifact index.

The parent `outputs/runs/experiment_manifest_*.json` files index batches.
Use `run --legacy-output-layout` only when an older script still depends on
scenario-configured output paths.

Visualize a trajectory:

```powershell
D:\anaconda\envs\python3.12\python.exe src\demo_visualize.py outputs\trajectories\demo_map_llm_remote.sqlite
```

Summarize experiment artifacts for tables:

```powershell
D:\anaconda\envs\python3.12\python.exe src\experiment_summary.py
```

Equivalent package command:

```powershell
python -m jupedsim_mall summarize
```

Build ATC reference metrics for realism evaluation:

```powershell
python -m jupedsim_mall atc-reference --config configs/data/atc_reference_20121114.json

D:\anaconda\envs\python3.12\python.exe src\atc_reference.py `
  --atc-raw-path "$env:ATC_RAW_PATH" `
  --atc-regions "$env:ATC_REGIONS" `
  --output outputs\summaries\atc_reference_metrics.json
```

Evaluate simulation realism against the ATC reference:

```powershell
D:\anaconda\envs\python3.12\python.exe src\realism_evaluation.py `
  --reference outputs\summaries\atc_reference_metrics.json `
  --sim outputs\trajectories\demo_map_llm_remote.sqlite
```

The equivalent package commands are `python -m jupedsim_mall atc-reference`
and `python -m jupedsim_mall realism`, followed by the same options.

Build and verify a portable reproduction package after reporting:

```powershell
python -m jupedsim_mall release build --output release `
  --plan outputs/matrices/<name>/<fingerprint>/plan.json `
  --report-dir outputs/formal-analysis/report
python -m jupedsim_mall release verify release
```

The summary command writes both per-artifact metrics and cross-run aggregate
metrics, including plan validation status, TTL removal rate, reroute rate, exit
distribution, observed trajectory duration, and mean/std values for repeated
runs. It also writes baseline-comparison JSD metrics:

```text
outputs/summaries/experiment_summary_comparison.json
outputs/summaries/experiment_summary_comparison.csv
```

The graduation-design workflow is documented in:

```text
docs/research_and_design_workflow.md
```

The engineering refactor and technical route are documented in:

```text
docs/engineering_technical_route.md
```

The academic experiment pipeline is documented in:

```text
docs/academic_experiment_pipeline.md
```

The staged implementation checklist is documented in:

```text
docs/full_experiment_pipeline_implementation_plan.md
```

## Map Editing

Edit or redraw geometry/stages:

```powershell
D:\anaconda\envs\python3.12\python.exe src\manual_draw_geometry.py
```

The editor saves:

- `data/map/geometry.wkt`
- `data/map/geometry.py`
- `data/map/drawn_geometry.json`
- `data/map/stages.json`

## Agent Intent and Profiles

Per-run agent intent is saved in:

```text
outputs/plans/demo_map_llm_plan.json
outputs/plans/demo_map_atc_llm_plan.json
```

Profile caches are saved in:

```text
outputs/profiles/demo_map_atc_profiles.json
outputs/profiles/demo_map_llmob_profiles.json
```

For ATC profiles, `preferences` are not purely random. They are built from:

```text
role default preferences + regions visited by the ATC trajectory
```

For example, a commuter may start with `direct exit`, `train corridor`, and `wide corridor`, then append `region_2` if the ATC trajectory crossed that map region.

## Key Simulation Parameters

Common parameters:

- `-n`: total number of agents.
- `--spawn-interval`: base interval between spawned agents.
- `--spawn-jitter`: random spawn delay.
- `--max-active-agents`: caps active population.
- `--agent-max-lifetime-seconds`: removes agents that remain too long, treating them as stuck.
- `--movement-model avm`: currently preferred for opposing pedestrian flows.
- `--llm-routing`: enables LLM route planning.
- `--llm-batch-size`: groups agents for LLM planning.
- `--waypoint-distance`: stage arrival tolerance.
- `--routing-waypoint-distance`: RoutingEngine waypoint arrival tolerance.
- `--routing-waypoint-max-per-leg`: maximum inserted navigation waypoints per semantic route segment.

## Notes

- The project root should be the working directory for manual commands.
- PowerShell scripts automatically switch to the project root before running.
- The upstream JuPedSim checkout under `jupedsim/` is left untouched by this project layout.
