from jupedsim_mall.analysis.behavior_states import fit_states


def test_behavior_state_clustering_is_reproducible_and_has_holdout():
    rows = []
    for run_index in range(3):
        for index in range(10):
            high = index >= 5
            rows.append({
                "run_id": f"run-{run_index}",
                "agent_id": f"a-{index}",
                "window_start_seconds": index,
                "role": "visitor" if high else "staff",
                "profile_source": "mall",
                "terminal_state": "completed",
                "rerouted": False,
                "mean_speed_mps": 1.5 if high else 0.2,
                "speed_std_mps": 0.1,
                "slow_ratio": 0.0 if high else 0.9,
                "turning_rad": 0.1 if high else 0.8,
                "local_density_per_m2": 0.3 if high else 2.0,
                "stop_ratio": 0.0 if high else 0.8,
                "region_code": 1 if high else 0,
            })
    config = {
        "features": [
            "mean_speed_mps", "speed_std_mps", "slow_ratio", "turning_rad",
            "local_density_per_m2", "stop_ratio", "region_code",
        ],
        "candidate_clusters": [2, 3],
        "random_seed": 7,
        "n_init": 10,
        "holdout_run_fraction": 0.34,
        "stride_seconds": 2.5,
    }
    first, labeled = fit_states([dict(row) for row in rows], config)
    second, _ = fit_states([dict(row) for row in rows], config)
    assert first["selected_clusters"] == 2
    assert first["state_centers"] == second["state_centers"]
    assert first["holdout"]["windows"] == 10
    assert {row["split"] for row in labeled} == {"train", "holdout"}
