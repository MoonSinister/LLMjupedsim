#!/usr/bin/env python3
"""Build ATC reference metrics for realism evaluation.

The reference is intentionally distribution-based. It converts real ATC tracks
into comparable behavioral statistics instead of assuming point-wise alignment
with the mall simulation map.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
import pathlib
from collections.abc import Sequence

from jupedsim_mall.analysis.metrics import histogram_counts
from jupedsim_mall.project import PROJECT_ROOT
from jupedsim_mall.profiles.providers import source_fingerprint
from jupedsim_mall.profiles.llmob_training_adapter import load_atc_tracks


def track_metrics(track) -> dict:
    duration = max(0.0, (track.last_time or 0.0) - (track.first_time or 0.0))
    mean_speed = track.speed_sum / max(track.count, 1)
    slow_ratio = track.slow_count / max(track.count, 1)
    regions = track.region_counter or Counter()
    return {
        "person_id": track.person_id,
        "source": track.source,
        "points": track.count,
        "duration_seconds": round(duration, 3),
        "path_length_m": round(track.path_length_m, 3),
        "mean_speed_mps": round(mean_speed, 4),
        "slow_ratio": round(slow_ratio, 4),
        "region_visits": dict(regions),
    }


def build_reference(tracks, provenance: dict | None = None) -> dict:
    rows = [track_metrics(track) for track in tracks]
    region_counts: Counter[str] = Counter()
    for row in rows:
        for region, count in row["region_visits"].items():
            region_counts[str(region)] += int(count)

    durations = [row["duration_seconds"] for row in rows]
    lengths = [row["path_length_m"] for row in rows]
    speeds = [row["mean_speed_mps"] for row in rows]
    slow_ratios = [row["slow_ratio"] for row in rows]

    binning = (provenance or {}).get("binning", {})

    def bins(name: str, default_size: float, default_max: float) -> tuple[float, float]:
        specification = binning.get(name, {})
        if isinstance(specification, dict):
            return float(specification.get("bin_size", default_size)), float(specification.get("max_value", default_max))
        return default_size, default_max

    duration_bin, duration_max = bins("duration_seconds", 30, 300)
    length_bin, length_max = bins("path_length_m", 5, 80)
    speed_bin, speed_max = bins("mean_speed_mps", 0.25, 2.5)
    slow_bin, slow_max = bins("slow_ratio", 0.1, 1.0)
    return {
        "metadata": {
            "source": "atc",
            "track_count": len(rows),
            "description": "Distributional ATC reference metrics for JuPedSim realism evaluation.",
            "provenance": provenance or {},
        },
        "distributions": {
            "duration_seconds": histogram_counts(durations, bin_size=duration_bin, max_value=duration_max),
            "path_length_m": histogram_counts(lengths, bin_size=length_bin, max_value=length_max),
            "mean_speed_mps": histogram_counts(speeds, bin_size=speed_bin, max_value=speed_max),
            "slow_ratio": histogram_counts(slow_ratios, bin_size=slow_bin, max_value=slow_max),
            "region_visits": dict(region_counts),
        },
        "tracks": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--atc-raw-path")
    parser.add_argument("--atc-regions", default="")
    parser.add_argument("--max-persons", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-points", type=int, default=300)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_ROOT / "outputs" / "summaries" / "atc_reference_metrics.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    raw_path = args.atc_raw_path or config.get("atc_raw_path")
    if not raw_path:
        raise SystemExit("--atc-raw-path or --config with atc_raw_path is required")
    regions_path = args.atc_regions or config.get("atc_regions", "")
    max_persons = config.get("max_persons", args.max_persons) if args.max_persons == 200 else args.max_persons
    max_rows = config.get("max_rows", args.max_rows) if args.max_rows == 0 else args.max_rows
    min_points = config.get("min_points", args.min_points) if args.min_points == 300 else args.min_points
    if args.config and args.output == PROJECT_ROOT / "outputs" / "summaries" / "atc_reference_metrics.json":
        args.output = PROJECT_ROOT / config.get("output", str(args.output.relative_to(PROJECT_ROOT)))
    raw_hash = source_fingerprint(raw_path)
    regions_hash = source_fingerprint(regions_path) if regions_path else None
    if config.get("expected_raw_sha256") and raw_hash != config["expected_raw_sha256"]:
        raise SystemExit("ATC raw file hash does not match frozen reference config")
    if config.get("expected_regions_sha256") and regions_hash != config["expected_regions_sha256"]:
        raise SystemExit("ATC region mapping hash does not match frozen reference config")
    tracks = load_atc_tracks(
        raw_path,
        regions_path=regions_path,
        max_persons=max_persons,
        max_rows=max_rows,
        min_points=min_points,
        partition=config.get("partition", "evaluation"),
    )
    if not tracks:
        raise SystemExit("No ATC tracks matched the given filters.")
    reference = build_reference(tracks, {
        "reference_date": config.get("reference_date", "2012-11-14"),
        "raw_path": str(raw_path),
        "raw_sha256": raw_hash,
        "regions_path": str(regions_path),
        "regions_sha256": regions_hash,
        "filters": {"max_persons": max_persons, "max_rows": max_rows, "min_points": min_points},
        "partition": config.get("partition", "evaluation"),
        "sampling_seed": config.get("sampling_seed", 2026),
        "binning": config.get("binning", {}),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ATC reference tracks: {len(tracks)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
