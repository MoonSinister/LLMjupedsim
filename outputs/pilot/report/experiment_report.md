# JuPedSim Mall Experiment Report

Generated from standardized analysis tables; figures are not manually edited.

## Run Status

- Matrix counts: {'pending': 0, 'running': 0, 'completed': 2, 'failed': 0, 'interrupted': 0, 'cancelled': 0, 'skipped': 0}
- Planned sample size satisfied: True
- Metric runs included: 2

## Figures

### primary_completion_rate

![primary_completion_rate](figures/primary_completion_rate.png)

Source rows: 2.

### primary_mean_travel_time_seconds

![primary_mean_travel_time_seconds](figures/primary_mean_travel_time_seconds.png)

Source rows: 2.

### primary_mean_path_length_m

![primary_mean_path_length_m](figures/primary_mean_path_length_m.png)

Source rows: 2.

### primary_mean_speed_mps

![primary_mean_speed_mps](figures/primary_mean_speed_mps.png)

Source rows: 2.

### exit_distribution

![exit_distribution](figures/exit_distribution.png)

Source rows: 22.

### region_visits

![region_visits](figures/region_visits.png)

Source rows: 22.

### realism_submetrics

![realism_submetrics](figures/realism_submetrics.png)

Source rows: 10.

### trajectory_density

![trajectory_density](figures/trajectory_density.png)

Source rows: 8351.

### representative_trajectories

![representative_trajectories](figures/representative_trajectories.png)

Source rows: 5026.

### state_transitions

![state_transitions](figures/state_transitions.png)

Source rows: 4.

### state_dwell

![state_dwell](figures/state_dwell.png)

Source rows: 2.

## Tables

- tables\aggregates.csv
- tables\aggregates.md
- tables\aggregates.tex
- tables\paired_comparisons.csv
- tables\paired_comparisons.md
- tables\paired_comparisons.tex

## Exceptions And Limitations

- ATC and simulated pedestrians are compared as distributions, not one-to-one trajectories.
- LLM outputs may vary across model versions; prompt and response hashes are retained.
- Results generalize only to the validated map, population assumptions, and parameter ranges.
- Failed and missing runs remain visible and are not silently discarded.
