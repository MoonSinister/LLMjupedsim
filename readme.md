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
|   |-- demo_map_simulation.py    # Main simulation entry
|   |-- demo_visualize.py         # Opens a SQLite trajectory in jupedsim_visualizer
|   |-- agent_model.py            # Agent IDs, profile queue, fallback routing
|   |-- llm_prompts.py            # LLM routing prompt/schema
|   |-- llmob_adapter.py          # Built-in mall personas
|   |-- llmob_training_adapter.py # LLMob/ATC profile extraction
|   |-- manual_draw_geometry.py   # Interactive map/stage editor
|   |-- geometry_editor.py        # Geometry editing helper
|   |-- pgm_to_geometry.py        # Convert PGM map to geometry
|   `-- plot_free_points.py       # Inspect/edit localization free points
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
|   |-- trajectories/             # Main SQLite trajectory outputs
|   |-- plans/                    # LLM per-agent intent/route plans
|   |-- profiles/                 # ATC/LLMob extracted profile caches
|   `-- smoke/                    # Small test/smoke output files
|-- docs/assets/                  # Figures and documentation assets
`-- jupedsim/                     # Upstream JuPedSim source checkout
```

## Environment

The scripts assume Python is available at:

```powershell
D:\anaconda\envs\python3.12\python.exe
```

The project currently uses packages such as `jupedsim`, `shapely`, and plotting/GUI dependencies used by the map tools.

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

## Run Simulations

Run the standard LLM-routed simulation:

```powershell
.\scripts\run_llm_remote.ps1
```

Run the ATC-profile-based LLM simulation:

```powershell
.\scripts\run_atc_llm_remote.ps1
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

## Outputs

Important output files:

- `outputs/trajectories/*.sqlite`: JuPedSim trajectories for visualization.
- `outputs/plans/*.json`: per-agent LLM plans, including role, subtype, intent, activities, target regions, final exit, TTL removal status, and reroute records.
- `outputs/profiles/*.json`: profile caches generated from ATC or LLMob data.
- `outputs/smoke/*.sqlite`: short test runs.

Visualize a trajectory:

```powershell
D:\anaconda\envs\python3.12\python.exe src\demo_visualize.py outputs\trajectories\demo_map_llm_remote.sqlite
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
