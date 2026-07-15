from jupedsim_mall.geometry.map_io import (
    keep_largest_connected_area,
    load_grid_transform,
    load_geometry,
    load_manual_stages,
    load_regions,
    validate_map,
)
from jupedsim_mall.project import PROJECT_ROOT


def test_repository_map_is_valid_and_fingerprinted():
    report = validate_map(PROJECT_ROOT / "data" / "map")

    assert report.valid
    assert report.metrics["usable_exits"] == 11
    assert report.metrics["usable_regions"] == 11
    assert report.metrics["connected_components"] == 8
    assert report.metrics["entrance_exit_pairs"] == 110
    assert report.metrics["unreachable_entrance_exit_pairs"] == 0
    assert len(report.file_hashes["geometry.wkt"]) == 64


def test_map_loaders_return_simulation_ready_objects():
    geometry = keep_largest_connected_area(
        load_geometry(PROJECT_ROOT / "data" / "map" / "geometry.wkt"),
        announce=False,
    )
    exits, entrances = load_manual_stages(geometry)
    regions = load_regions(geometry)

    assert len(exits) == 11
    assert len(entrances) == 10
    assert len(regions) == 11
    assert all(geometry.intersects(polygon) for polygon, _ in exits + entrances)
    assert all(region["area"] > 0 for region in regions)


def test_small_map_fixture_and_coordinate_transform():
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "map_small"
    report = validate_map(fixture)
    transform = load_grid_transform(fixture)

    assert report.valid
    assert report.metrics["entrance_exit_pairs"] == 1
    assert transform.pixel_to_world(25, 75) == (2.5, 2.5)
    assert transform.world_to_pixel(2.5, 2.5) == (25.0, 75.0)
