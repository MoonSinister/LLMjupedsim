"""Versioned map loading and spatial quality validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import shapely
import yaml
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import triangulate

from jupedsim_mall.project import PROJECT_ROOT


DEFAULT_MAP_DIR = PROJECT_ROOT / "data" / "map"


@dataclass(frozen=True)
class GridTransform:
    resolution_m_per_pixel: float
    origin_x_m: float
    origin_y_m: float
    image_height_px: int

    def pixel_to_world(self, x_px: float, y_px: float) -> tuple[float, float]:
        return (
            self.origin_x_m + x_px * self.resolution_m_per_pixel,
            self.origin_y_m + (self.image_height_px - y_px) * self.resolution_m_per_pixel,
        )

    def world_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        return (
            (x_m - self.origin_x_m) / self.resolution_m_per_pixel,
            self.image_height_px - (y_m - self.origin_y_m) / self.resolution_m_per_pixel,
        )


def load_grid_transform(map_dir: str | Path = DEFAULT_MAP_DIR) -> GridTransform:
    root = _resolve(map_dir)
    yaml_path = root / "localization_grid.yaml"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    region_payload = load_region_payload(root / "localization_grid_regions.json")
    map_metadata = region_payload.get("map", {})
    origin = payload.get("origin") or map_metadata.get("origin")
    resolution = payload.get("resolution") or map_metadata.get("resolution")
    height = map_metadata.get("height")
    if not origin or resolution is None or height is None:
        raise ValueError("grid transform requires origin, resolution, and image height")
    return GridTransform(float(resolution), float(origin[0]), float(origin[1]), int(height))


def geometry_parts(geometry) -> list:
    return list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]


def largest_polygon(geometry) -> Polygon | None:
    parts = [part for part in geometry_parts(geometry) if isinstance(part, Polygon) and not part.is_empty]
    return max(parts, key=lambda part: part.area) if parts else None


def keep_largest_connected_area(geometry, *, announce: bool = True):
    parts = [part for part in geometry_parts(geometry) if isinstance(part, Polygon) and not part.is_empty]
    if len(parts) <= 1:
        return geometry
    parts.sort(key=lambda part: part.area, reverse=True)
    largest = parts[0]
    dropped_area = sum(part.area for part in parts[1:])
    if announce:
        print(
            "Note: JuPedSim requires one connected walkable area; "
            f"kept largest area {largest.area:.1f} m2 and ignored "
            f"{len(parts) - 1} smaller areas ({dropped_area:.1f} m2)."
        )
    return largest


def convex_stage_polygon(stage_area) -> Polygon | None:
    stage_area = largest_polygon(stage_area)
    if stage_area is None or stage_area.area <= 0:
        return None
    hull = stage_area.convex_hull
    tolerance = max(1e-6, stage_area.area * 0.01)
    if stage_area.covers(hull) or abs(hull.area - stage_area.area) <= tolerance:
        return hull
    triangles = [
        triangle
        for triangle in triangulate(stage_area)
        if isinstance(triangle, Polygon) and triangle.area > 1e-6 and stage_area.covers(triangle)
    ]
    if triangles:
        return max(triangles, key=lambda triangle: triangle.area)
    return None


def load_geometry(path: str | Path, *, fallback_python: bool = True):
    wkt_path = Path(path)
    if not wkt_path.is_absolute():
        wkt_path = PROJECT_ROOT / wkt_path
    if wkt_path.exists():
        geometry = shapely.from_wkt(wkt_path.read_text(encoding="utf-8"))
        if geometry.is_empty:
            raise ValueError(f"geometry is empty: {wkt_path}")
        return geometry
    if fallback_python:
        py_path = DEFAULT_MAP_DIR / "geometry.py"
        if py_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("jupedsim_mall_generated_geometry", py_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load geometry module: {py_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.get_geometry()
    raise FileNotFoundError(f"No geometry file: {wkt_path}")


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def load_stage_payload(path: str | Path) -> dict[str, Any]:
    stage_path = _resolve(path)
    if not stage_path.exists():
        return {"exits": [], "entrances": []}
    data = json.loads(stage_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"stage root must be an object: {stage_path}")
    return data


def load_manual_stages(geometry, stages_path: str | Path = DEFAULT_MAP_DIR / "stages.json"):
    path = _resolve(stages_path)
    data = load_stage_payload(path)
    exits = []
    spawns = []
    for index, item in enumerate(data.get("exits", [])):
        polygon = Polygon(item["polygon"])
        clipped = convex_stage_polygon(polygon.intersection(geometry))
        if clipped is None or clipped.area < 0.01:
            print(f"  skip exit {item.get('name', index)}: outside walkable area")
            continue
        exits.append((clipped, item.get("name", f"exit_{index}")))
    entrances = data.get("entrances", data.get("spawns", []))
    for index, item in enumerate(entrances):
        polygon = Polygon(item["polygon"])
        clipped = largest_polygon(polygon.intersection(geometry))
        if clipped is None or clipped.area < 0.02:
            print(f"  skip entrance {item.get('name', index)}: outside walkable area or too small")
            continue
        spawns.append((clipped, item.get("name", f"entrance_{index}")))
    if exits or spawns:
        print(f"\nLoaded manual stages: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
        print(f"  exits: {len(exits)}")
        print(f"  entrances/spawns: {len(spawns)}")
    return exits, spawns


def load_region_payload(path: str | Path) -> dict[str, Any]:
    region_path = _resolve(path)
    if not region_path.exists():
        return {"regions": []}
    data = json.loads(region_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"region root must be an object: {region_path}")
    return data


def load_regions(geometry, regions_path: str | Path = DEFAULT_MAP_DIR / "localization_grid_regions.json"):
    path = _resolve(regions_path)
    data = load_region_payload(path)
    regions = []
    for item in data.get("regions", []):
        points = item.get("points_world") or []
        if len(points) < 3:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        clipped = largest_polygon(polygon.intersection(geometry))
        region_geometry = clipped or polygon
        center = region_geometry.representative_point()
        if not geometry.covers(center):
            center = geometry.representative_point()
        regions.append({
            "name": item.get("name", f"region_{len(regions) + 1}"),
            "description": item.get("description", ""),
            "color": item.get("color", ""),
            "category": item.get("category", "unknown"),
            "polygon": region_geometry,
            "position": (center.x, center.y),
            "area": region_geometry.area,
        })
    if regions:
        print(f"\nLoaded semantic regions: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
        print(f"  regions: {len(regions)}")
    return regions


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class MapQualityReport:
    map_id: str
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    file_hashes: dict[str, str | None] = field(default_factory=dict)
    schema_version: str = "1.0"

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.valid = False

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "map_id": self.map_id,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "file_hashes": self.file_hashes,
        }


def _duplicate_names(items: list[dict[str, Any]]) -> list[str]:
    names = [str(item.get("name", "")) for item in items]
    return sorted({name for name in names if name and names.count(name) > 1})


def validate_map(map_dir: str | Path = DEFAULT_MAP_DIR) -> MapQualityReport:
    root = _resolve(map_dir)
    geometry_path = root / "geometry.wkt"
    stages_path = root / "stages.json"
    regions_path = root / "localization_grid_regions.json"
    report = MapQualityReport(map_id=root.name)
    report.file_hashes = {
        "geometry.wkt": file_sha256(geometry_path),
        "stages.json": file_sha256(stages_path),
        "localization_grid_regions.json": file_sha256(regions_path),
    }
    try:
        geometry = load_geometry(geometry_path, fallback_python=False)
    except (OSError, ValueError) as exc:
        report.add_error(str(exc))
        return report
    if not geometry.is_valid:
        report.add_error(f"invalid geometry: {shapely.is_valid_reason(geometry)}")
    parts = [part for part in geometry_parts(geometry) if isinstance(part, Polygon) and not part.is_empty]
    main_geometry = keep_largest_connected_area(geometry, announce=False)
    report.metrics.update({
        "geometry_area_m2": round(float(geometry.area), 3),
        "connected_components": len(parts),
        "main_component_area_m2": round(float(main_geometry.area), 3),
        "dropped_component_area_m2": round(float(geometry.area - main_geometry.area), 3),
        "minimum_clearance_m": round(float(main_geometry.minimum_clearance), 6),
        "interior_holes": sum(len(part.interiors) for part in parts),
    })
    if len(parts) > 1:
        report.warnings.append(f"geometry has {len(parts)} components; simulation keeps the largest")

    stages = load_stage_payload(stages_path)
    exits = stages.get("exits", [])
    entrances = stages.get("entrances", stages.get("spawns", []))
    report.metrics["declared_exits"] = len(exits)
    report.metrics["declared_entrances"] = len(entrances)
    for duplicate in _duplicate_names(exits + entrances):
        report.add_error(f"duplicate stage name: {duplicate}")
    usable_exits = []
    exit_points: list[tuple[str, tuple[float, float]]] = []
    entrance_points: list[tuple[str, tuple[float, float]]] = []
    for kind, items, minimum_area in (("exit", exits, 0.01), ("entrance", entrances, 0.02)):
        for index, item in enumerate(items):
            name = item.get("name", f"{kind}_{index}")
            try:
                polygon = Polygon(item.get("polygon", []))
            except (TypeError, ValueError):
                report.add_error(f"{kind} {name} has invalid polygon coordinates")
                continue
            if not polygon.is_valid or polygon.is_empty:
                report.add_error(f"{kind} {name} polygon is invalid or empty")
                continue
            overlap = polygon.intersection(main_geometry).area
            ratio = overlap / max(polygon.area, 1e-12)
            if overlap < minimum_area:
                report.add_error(f"{kind} {name} does not intersect the simulated walkable area")
            elif ratio < 0.95:
                report.warnings.append(f"{kind} {name} is only {ratio:.1%} inside the simulated walkable area")
            if kind == "exit" and overlap >= minimum_area:
                usable_exits.append(name)
                point = polygon.intersection(main_geometry).representative_point()
                exit_points.append((name, (point.x, point.y)))
            elif kind == "entrance" and overlap >= minimum_area:
                point = polygon.intersection(main_geometry).representative_point()
                entrance_points.append((name, (point.x, point.y)))
    if not usable_exits:
        report.add_error("no usable exits")

    region_payload = load_region_payload(regions_path)
    region_items = region_payload.get("regions", [])
    report.metrics["declared_regions"] = len(region_items)
    for duplicate in _duplicate_names(region_items):
        report.add_error(f"duplicate region name: {duplicate}")
    usable_regions = 0
    for index, item in enumerate(region_items):
        name = item.get("name", f"region_{index + 1}")
        points = item.get("points_world") or []
        if len(points) < 3:
            report.add_error(f"region {name} has fewer than 3 world points")
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            report.add_error(f"region {name} is invalid: {shapely.is_valid_reason(polygon)}")
            continue
        overlap = polygon.intersection(main_geometry).area
        if overlap <= 0.01:
            report.warnings.append(f"region {name} does not overlap the simulated walkable area")
        else:
            usable_regions += 1
    report.metrics["usable_regions"] = usable_regions
    report.metrics["usable_exits"] = len(usable_exits)
    unreachable_pairs = []
    try:
        import jupedsim as jps

        routing = jps.RoutingEngine(main_geometry)
        for entrance_name, entrance_point in entrance_points:
            for exit_name, exit_point in exit_points:
                try:
                    waypoints = routing.compute_waypoints(entrance_point, exit_point)
                    if not waypoints:
                        unreachable_pairs.append(f"{entrance_name}->{exit_name}")
                except RuntimeError:
                    unreachable_pairs.append(f"{entrance_name}->{exit_name}")
    except (ImportError, RuntimeError) as exc:
        report.warnings.append(f"routing reachability check unavailable: {exc}")
    report.metrics["entrance_exit_pairs"] = len(entrance_points) * len(exit_points)
    report.metrics["unreachable_entrance_exit_pairs"] = len(unreachable_pairs)
    if unreachable_pairs:
        report.add_error(f"unreachable entrance/exit pairs: {', '.join(unreachable_pairs[:10])}")
    return report


def write_map_preview(map_dir: str | Path, output: str | Path) -> Path:
    import matplotlib.pyplot as plt

    root = _resolve(map_dir)
    geometry = keep_largest_connected_area(load_geometry(root / "geometry.wkt"), announce=False)
    exits, entrances = load_manual_stages(geometry, root / "stages.json")
    regions = load_regions(geometry, root / "localization_grid_regions.json")
    figure, axis = plt.subplots(figsize=(12, 7))
    for part in geometry_parts(geometry):
        x, y = part.exterior.xy
        axis.fill(x, y, color="#e9ecef", edgecolor="#343a40", linewidth=0.8)
    for region in regions:
        x, y = region["polygon"].exterior.xy
        axis.fill(x, y, color=region.get("color") or "#4c78a8", alpha=0.25)
        axis.text(*region["position"], region["name"], fontsize=7)
    for polygon, name in entrances:
        x, y = polygon.exterior.xy
        axis.plot(x, y, color="#2a9d8f", linewidth=1.5)
        axis.text(
            polygon.centroid.x,
            polygon.centroid.y + 0.3,
            name,
            fontsize=5,
            color="#147d6f",
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.4},
        )
    for polygon, name in exits:
        x, y = polygon.exterior.xy
        axis.plot(x, y, color="#e63946", linewidth=1.5)
        axis.text(
            polygon.centroid.x,
            polygon.centroid.y - 0.3,
            name,
            fontsize=5,
            color="#b4232f",
            ha="center",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.4},
        )
    axis.set_aspect("equal")
    axis.set_title("Map quality preview")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    figure.tight_layout()
    output_path = _resolve(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path
