#!/usr/bin/env python3
"""
Draw the final JuPedSim geometry directly.

What you draw is what JuPedSim receives:
  - blue polygons are walkable areas
  - red lines are obstacle walls
  - open red wall lines become thick obstacles
  - closed red wall loops also become filled blocked areas
  - data/map/geometry.wkt is the final walkable geometry used by JuPedSim

Typical use:
  python src/manual_draw_geometry.py
  python src/manual_draw_geometry.py --wall-thickness 0.2
  python src/manual_draw_geometry.py --blank
  python src/manual_draw_geometry.py --load-geometry data/map/geometry.wkt

Controls:
  left click        add vertex
  right click/Enter finish current shape
  a                 draw walkable polygon
  h                 draw obstacle wall line
  e                 draw exit area
  n                 draw entrance/spawn area
  d                 delete nearest obstacle wall line
  w                 cycle wall thickness
  u                 undo last vertex
  z                 undo last finished shape
  q/Esc             cancel current shape
  c                 clear all drawn shapes
  s                 save data/map/geometry.wkt + data/map/geometry.py
  t                 switch background mode
  p                 toggle final geometry preview
  r                 reset view
  +/-               change vertex marker size
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shapely
import yaml
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSON = PROJECT_ROOT / "data/map/drawn_geometry.json"
STAGES_JSON = PROJECT_ROOT / "data/map/stages.json"
OUTPUT_WKT = PROJECT_ROOT / "data/map/geometry.wkt"
OUTPUT_PY = PROJECT_ROOT / "data/map/geometry.py"


def resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists() or candidate.parts[:2] == ("data", "map"):
        return project_candidate
    return candidate


@dataclass
class MapBackground:
    rgba: np.ndarray
    free_points: tuple[np.ndarray, np.ndarray]
    extent: list[float]
    resolution: float
    origin: list[float]
    shape: tuple[int, int]


def load_pgm_background(pgm_path: Path, yaml_path: Path) -> MapBackground:
    with yaml_path.open(encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    resolution = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"]]
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read PGM image: {pgm_path}")

    height, width = img.shape
    flipped = np.flipud(img)
    free_val = np.unique(img)[0]

    rgba = np.ones((height, width, 4), dtype=float)
    rgba[flipped == free_val] = [0.92, 0.97, 1.0, 1.0]
    rgba[flipped > free_val] = [0.18, 0.19, 0.21, 1.0]

    free_mask = img == free_val
    py, px = np.where(free_mask)
    wx = origin[0] + px * resolution
    wy = origin[1] + (height - 1 - py) * resolution
    if len(wx) > 90000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(wx), 90000, replace=False)
        wx, wy = wx[idx], wy[idx]

    extent = [
        origin[0],
        origin[0] + width * resolution,
        origin[1],
        origin[1] + height * resolution,
    ]
    return MapBackground(rgba, (wx, wy), extent, resolution, origin, img.shape)


def polygons_from_geometry(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if hasattr(geometry, "geoms"):
        return [g for g in geometry.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def normalize_geometry(geometry, min_area: float = 0.05):
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    polygons = [p for p in polygons_from_geometry(geometry) if p.area >= min_area]
    if not polygons:
        return None
    result = unary_union(polygons)
    if not result.is_valid:
        result = result.buffer(0)
    return result if not result.is_empty else None


def make_polygon(points: list[tuple[float, float]]) -> Polygon | None:
    if len(points) < 3:
        return None
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    parts = polygons_from_geometry(poly)
    if not parts:
        return None
    return max(parts, key=lambda p: p.area)


def make_line(points: list[tuple[float, float]]) -> LineString | None:
    if len(points) < 2:
        return None
    line = LineString(points)
    if line.is_empty or line.length <= 0:
        return None
    return line


def draw_geometry(
    ax,
    geometry,
    *,
    facecolor,
    edgecolor,
    alpha,
    zorder,
    label_prefix=None,
    label_start=0,
):
    for i, poly in enumerate(polygons_from_geometry(geometry)):
        x, y = poly.exterior.xy
        ax.fill(x, y, fc=facecolor, ec=edgecolor, alpha=alpha, linewidth=1.2, zorder=zorder)
        if label_prefix:
            rp = poly.representative_point()
            ax.annotate(
                f"{label_prefix}{label_start + i}",
                (rp.x, rp.y),
                fontsize=8,
                color=edgecolor,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=zorder + 1,
            )
        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, fc="white", ec="#d62828", alpha=0.95, linewidth=0.8, zorder=zorder + 1)


class FinalGeometryDrawer:
    def __init__(
        self,
        pgm_path: Path,
        yaml_path: Path,
        *,
        load_geometry: Path | None,
        source_json: Path,
        blank: bool,
        wall_thickness: float,
    ):
        self.background = load_pgm_background(pgm_path, yaml_path)
        self.source_json = source_json

        self.walkables: list[Polygon] = []
        self.blocking_lines: list[LineString] = []
        self.exit_areas: list[Polygon] = []
        self.spawn_areas: list[Polygon] = []
        self.history: list[tuple[str, Polygon | LineString]] = []

        self.current_vertices: list[tuple[float, float]] = []
        self.draw_mode = "walkable"
        self.display_mode = "hybrid"
        self.show_preview = True
        self.marker_size = 7
        self.snap_tolerance = max(self.background.resolution * 4, 0.15)
        self.delete_tolerance = max(self.background.resolution * 10, 0.45)
        self.wall_thickness = wall_thickness

        if not blank:
            self._load_initial_geometry(load_geometry)

        self.fig, self.ax = plt.subplots(figsize=(18, 8), num="JuPedSim Final Geometry Drawer")
        self._connect_events()
        self._draw()
        self.ax.set_xlim(self.background.extent[0], self.background.extent[1])
        self.ax.set_ylim(self.background.extent[2], self.background.extent[3])

    def _load_initial_geometry(self, load_geometry: Path | None):
        if self.source_json.exists():
            self._load_source_json(self.source_json)
            print(f"Loaded editable source: {self.source_json}")
            return

        if load_geometry is None or not load_geometry.exists():
            return

        geometry = shapely.from_wkt(load_geometry.read_text(encoding="utf-8"))
        for poly in polygons_from_geometry(geometry):
            self.walkables.append(poly)
            self.history.append(("walkable", poly))
        print(f"Loaded final geometry as editable walkable polygons: {load_geometry}")

    def _load_source_json(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.wall_thickness = float(data.get("wall_thickness", self.wall_thickness))
        self.walkables = [Polygon(item["exterior"], holes=item.get("holes", [])) for item in data.get("walkable", [])]

        self.blocking_lines = [
            LineString(item["points"])
            for item in data.get("blocked_lines", [])
            if len(item.get("points", [])) >= 2
        ]

        # Compatibility with the previous polygon-based source file.
        for item in data.get("blocked", []):
            exterior = item.get("exterior", [])
            if len(exterior) >= 2:
                self.blocking_lines.append(LineString(exterior))

        self.exit_areas = [
            Polygon(item["exterior"])
            for item in data.get("exits", [])
            if len(item.get("exterior", [])) >= 3
        ]
        self.spawn_areas = [
            Polygon(item["exterior"])
            for item in data.get("entrances", data.get("spawns", []))
            if len(item.get("exterior", [])) >= 3
        ]

        self.history = []
        for poly in self.walkables:
            self.history.append(("walkable", poly))
        for line in self.blocking_lines:
            self.history.append(("blocked_line", line))
        for poly in self.exit_areas:
            self.history.append(("exit", poly))
        for poly in self.spawn_areas:
            self.history.append(("spawn", poly))

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _compute_blocked_geometry(self):
        if not self.blocking_lines:
            return None
        obstacles = []

        wall_bodies = [
            line.buffer(
                self.wall_thickness / 2,
                cap_style="flat",
                join_style="mitre",
            )
            for line in self.blocking_lines
            if not line.is_empty and line.length > 0
        ]
        obstacles.extend(wall_bodies)

        closed_polygons = list(polygonize(unary_union(self.blocking_lines)))
        obstacles.extend(closed_polygons)

        if not obstacles:
            return None
        return normalize_geometry(unary_union(obstacles))

    def _compute_final_geometry(self):
        walkable = normalize_geometry(unary_union(self.walkables)) if self.walkables else None
        if walkable is None:
            return None
        obstacles = self._compute_blocked_geometry()
        if obstacles is not None:
            walkable = walkable.difference(obstacles)
        return normalize_geometry(walkable)

    def _draw_background(self):
        bg = self.background
        if self.display_mode in ("image", "hybrid"):
            self.ax.imshow(
                bg.rgba,
                extent=bg.extent,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
                zorder=0,
            )
        if self.display_mode in ("points", "hybrid"):
            wx, wy = bg.free_points
            self.ax.scatter(
                wx,
                wy,
                c="#2f6f9f",
                s=0.35,
                alpha=0.28,
                marker="s",
                edgecolors="none",
                zorder=1,
            )

    def _draw(self):
        self.ax.clear()
        self.ax.set_aspect("equal")
        self._draw_background()

        for i, poly in enumerate(self.walkables):
            draw_geometry(
                self.ax,
                poly,
                facecolor="#78c6e7",
                edgecolor="#145c78",
                alpha=0.42,
                zorder=3,
                label_prefix="A",
                label_start=i,
            )

        blocked_geometry = self._compute_blocked_geometry()
        if blocked_geometry is not None:
            draw_geometry(
                self.ax,
                blocked_geometry,
                facecolor="#e63946",
                edgecolor="#7f111b",
                alpha=0.38,
                zorder=5,
            )

        for i, line in enumerate(self.blocking_lines):
            x, y = line.xy
            self.ax.plot(
                x,
                y,
                "-",
                color="#c1121f",
                linewidth=2.2,
                solid_capstyle="round",
                zorder=6,
            )
            mid = line.interpolate(0.5, normalized=True)
            self.ax.annotate(
                f"W{i}",
                (mid.x, mid.y),
                fontsize=8,
                color="#7f111b",
                fontweight="bold",
                ha="center",
                va="center",
                zorder=7,
            )

        for i, poly in enumerate(self.exit_areas):
            draw_geometry(
                self.ax,
                poly,
                facecolor="#ffd166",
                edgecolor="#8a5a00",
                alpha=0.62,
                zorder=8,
                label_prefix="E",
                label_start=i,
            )

        for i, poly in enumerate(self.spawn_areas):
            draw_geometry(
                self.ax,
                poly,
                facecolor="#b185db",
                edgecolor="#5a189a",
                alpha=0.58,
                zorder=8,
                label_prefix="N",
                label_start=i,
            )

        final_geometry = self._compute_final_geometry()
        if self.show_preview and final_geometry is not None:
            draw_geometry(
                self.ax,
                final_geometry,
                facecolor="#4caf50",
                edgecolor="#1b5e20",
                alpha=0.28,
                zorder=2,
            )

        self._draw_current_shape()
        self._draw_legend()
        self._draw_status(final_geometry, blocked_geometry)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.18, linestyle="--")
        self.fig.canvas.draw_idle()

    def _draw_current_shape(self):
        if not self.current_vertices:
            return
        colors = {
            "walkable": ("#0b6fa4", "#78c6e7"),
            "blocked": ("#c1121f", "#e63946"),
            "exit": ("#8a5a00", "#ffd166"),
            "spawn": ("#5a189a", "#b185db"),
        }
        color, fill = colors[self.draw_mode]
        xs, ys = zip(*self.current_vertices)
        self.ax.plot(
            xs,
            ys,
            "o-",
            color=color,
            linewidth=2.2,
            markersize=self.marker_size,
            zorder=9,
        )
        if self.draw_mode != "blocked" and len(self.current_vertices) >= 3:
            preview = Polygon(self.current_vertices)
            if not preview.is_empty:
                x, y = preview.exterior.xy
                self.ax.fill(x, y, fc=fill, ec=color, alpha=0.22, linewidth=1.4, zorder=8)

    def _draw_legend(self):
        patches = [
            mpatches.Patch(color="#78c6e7", label="Blue: walkable area"),
            mpatches.Patch(color="#c1121f", label="Red line: wall centerline"),
            mpatches.Patch(color="#e63946", label="Red fill: wall/closed obstacle"),
            mpatches.Patch(color="#ffd166", label="Yellow: exit area"),
            mpatches.Patch(color="#b185db", label="Purple: entrance/spawn area"),
            mpatches.Patch(color="#4caf50", label="Green: final JuPedSim map"),
        ]
        self.ax.legend(handles=patches, loc="upper right", framealpha=0.88)

    def _draw_status(self, final_geometry, blocked_geometry):
        mode_labels = {
            "walkable": "walkable polygon",
            "blocked": "obstacle wall line",
        "exit": "exit area",
        "spawn": "entrance/spawn area",
        "delete_wall": "delete obstacle wall",
    }
        mode_label = mode_labels[self.draw_mode]
        area = final_geometry.area if final_geometry is not None else 0.0
        regions = len(polygons_from_geometry(final_geometry)) if final_geometry is not None else 0
        holes = (
            sum(len(poly.interiors) for poly in polygons_from_geometry(final_geometry))
            if final_geometry is not None
            else 0
        )
        obstacle_count = len(polygons_from_geometry(blocked_geometry)) if blocked_geometry is not None else 0
        self.ax.set_title(
            f"Mode: {mode_label} | vertices: {len(self.current_vertices)} | "
            f"walkable polygons: {len(self.walkables)} | wall lines: {len(self.blocking_lines)} | "
            f"wall thickness: {self.wall_thickness:.2f}m | "
            f"exits: {len(self.exit_areas)} | entrances: {len(self.spawn_areas)} | "
            f"obstacles: {obstacle_count} | final: {regions} regions, {holes} holes, "
            f"{area:.1f} m² | background: {self.display_mode}"
        )

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            if self.draw_mode == "delete_wall":
                self._delete_nearest_wall(float(event.xdata), float(event.ydata))
                return
            point = self._maybe_snap_point(float(event.xdata), float(event.ydata))
            self.current_vertices.append(point)
            print(f"+ vertex {len(self.current_vertices)}: ({point[0]:.2f}, {point[1]:.2f})")
            self._draw()
        elif event.button == 3:
            self._finish_current_shape()

    def _on_key(self, event):
        key = event.key.lower() if event.key else ""
        if key in ("enter", "return"):
            self._finish_current_shape()
        elif key == "a":
            self.draw_mode = "walkable"
            print("Draw mode: walkable polygon")
            self._draw()
        elif key == "h":
            self.draw_mode = "blocked"
            print("Draw mode: obstacle wall line")
            self._draw()
        elif key == "e":
            self.draw_mode = "exit"
            print("Draw mode: exit area")
            self._draw()
        elif key == "n":
            self.draw_mode = "spawn"
            print("Draw mode: entrance/spawn area")
            self._draw()
        elif key == "d":
            self.current_vertices = []
            self.draw_mode = "delete_wall"
            print("Delete mode: click the red obstacle wall line to remove")
            self._draw()
        elif key == "w":
            thicknesses = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
            nearest = min(range(len(thicknesses)), key=lambda i: abs(thicknesses[i] - self.wall_thickness))
            self.wall_thickness = thicknesses[(nearest + 1) % len(thicknesses)]
            print(f"Wall thickness: {self.wall_thickness:.2f} m")
            self._draw()
        elif key == "u":
            if self.current_vertices:
                removed = self.current_vertices.pop()
                print(f"Undo vertex: ({removed[0]:.2f}, {removed[1]:.2f})")
                self._draw()
        elif key == "z":
            self._undo_finished_shape()
        elif key in ("q", "escape"):
            if self.current_vertices:
                print(f"Canceled current shape ({len(self.current_vertices)} vertices)")
                self.current_vertices = []
                self._draw()
        elif key == "c":
            self.walkables = []
            self.blocking_lines = []
            self.exit_areas = []
            self.spawn_areas = []
            self.history = []
            self.current_vertices = []
            print("Cleared all shapes")
            self._draw()
        elif key == "s":
            self._save()
        elif key == "t":
            modes = ["hybrid", "image", "points", "none"]
            self.display_mode = modes[(modes.index(self.display_mode) + 1) % len(modes)]
            print(f"Background mode: {self.display_mode}")
            self._draw()
        elif key == "p":
            self.show_preview = not self.show_preview
            print(f"Final preview: {'on' if self.show_preview else 'off'}")
            self._draw()
        elif key == "r":
            self.ax.set_xlim(self.background.extent[0], self.background.extent[1])
            self.ax.set_ylim(self.background.extent[2], self.background.extent[3])
            self._draw()
        elif key in ("+", "="):
            self.marker_size = min(20, self.marker_size + 1)
            self._draw()
        elif key == "-":
            self.marker_size = max(3, self.marker_size - 1)
            self._draw()

    def _finish_current_shape(self):
        if self.draw_mode == "blocked":
            line = make_line(self.current_vertices)
            if line is None:
                print("Need at least 2 valid points to finish an obstacle wall line")
                return
            self.blocking_lines.append(line)
            self.history.append(("blocked_line", line))
            print(
                f"Added obstacle wall W{len(self.blocking_lines) - 1}: "
                f"{len(self.current_vertices)} vertices, {line.length:.2f} m"
            )
            self.current_vertices = []
            self._draw()
            return

        poly = make_polygon(self.current_vertices)
        if poly is None or poly.area < 0.05:
            print("Need at least 3 valid points to finish this polygon")
            return

        if self.draw_mode == "walkable":
            self.walkables.append(poly)
            self.history.append(("walkable", poly))
            print(f"Added walkable polygon A{len(self.walkables) - 1}: {poly.area:.2f} m2")
        elif self.draw_mode == "exit":
            self.exit_areas.append(poly)
            self.history.append(("exit", poly))
            print(f"Added exit area E{len(self.exit_areas) - 1}: {poly.area:.2f} m2")
        elif self.draw_mode == "spawn":
            self.spawn_areas.append(poly)
            self.history.append(("spawn", poly))
            print(f"Added entrance/spawn area N{len(self.spawn_areas) - 1}: {poly.area:.2f} m2")

        self.current_vertices = []
        self._draw()

    def _undo_finished_shape(self):
        if not self.history:
            return
        kind, shape = self.history.pop()
        targets = {
            "walkable": self.walkables,
            "blocked_line": self.blocking_lines,
            "exit": self.exit_areas,
            "spawn": self.spawn_areas,
        }
        target = targets[kind]
        for idx in range(len(target) - 1, -1, -1):
            if target[idx].equals_exact(shape, 1e-9) or target[idx].equals(shape):
                target.pop(idx)
                break
        print(f"Removed last {kind}")
        self._draw()

    def _delete_nearest_wall(self, x: float, y: float):
        if not self.blocking_lines:
            print("No obstacle wall lines to delete")
            return

        point = Point(x, y)
        scored = [
            (point.distance(line), index, line)
            for index, line in enumerate(self.blocking_lines)
            if not line.is_empty
        ]
        if not scored:
            print("No obstacle wall lines to delete")
            return

        distance, index, line = min(scored, key=lambda item: item[0])
        if distance > self.delete_tolerance:
            print(
                f"No wall close enough to delete "
                f"(nearest W{index}, distance {distance:.2f} m)"
            )
            return

        self.blocking_lines.pop(index)
        for history_index in range(len(self.history) - 1, -1, -1):
            kind, shape = self.history[history_index]
            if kind == "blocked_line" and (shape.equals_exact(line, 1e-9) or shape.equals(line)):
                self.history.pop(history_index)
                break
        print(f"Deleted obstacle wall W{index} at distance {distance:.2f} m")
        self._draw()

    def _maybe_snap_point(self, x: float, y: float) -> tuple[float, float]:
        if self.draw_mode != "blocked":
            return (x, y)

        candidates: list[tuple[float, float]] = []
        if self.current_vertices:
            candidates.append(self.current_vertices[0])
            candidates.append(self.current_vertices[-1])
        for line in self.blocking_lines:
            coords = list(line.coords)
            if coords:
                candidates.append(tuple(coords[0]))
                candidates.append(tuple(coords[-1]))

        point = Point(x, y)
        best = None
        best_dist = self.snap_tolerance
        for candidate in candidates:
            dist = point.distance(Point(candidate))
            if dist <= best_dist:
                best = candidate
                best_dist = dist

        if best is not None:
            print(f"  snapped to ({best[0]:.2f}, {best[1]:.2f})")
            return (float(best[0]), float(best[1]))
        return (x, y)

    def _save(self):
        final_geometry = self._compute_final_geometry()
        if final_geometry is None or final_geometry.is_empty:
            print("No final walkable geometry. Draw at least one blue walkable polygon first.")
            return

        OUTPUT_WKT.parent.mkdir(parents=True, exist_ok=True)
        wkt = shapely.to_wkt(final_geometry, rounding_precision=3)
        OUTPUT_WKT.write_text(wkt + "\n", encoding="utf-8")
        self._save_python_module(final_geometry, OUTPUT_PY)
        self._save_source_json()
        self._save_stages_json()

        polygons = polygons_from_geometry(final_geometry)
        holes = sum(len(poly.interiors) for poly in polygons)
        vertices = sum(len(poly.exterior.coords) for poly in polygons)
        blocked_geometry = self._compute_blocked_geometry()
        obstacles = len(polygons_from_geometry(blocked_geometry)) if blocked_geometry is not None else 0
        print()
        print("Saved final JuPedSim geometry")
        print(f"  WKT       : {OUTPUT_WKT} ({len(wkt)} chars)")
        print(f"  Python    : {OUTPUT_PY}")
        print(f"  Source    : {self.source_json}")
        print(f"  Stages    : {STAGES_JSON}")
        print(f"  Obstacles : {obstacles} wall/closed obstacle region(s)")
        print(f"  Wall width: {self.wall_thickness:.2f} m")
        print(f"  Exits     : {len(self.exit_areas)}")
        print(f"  Entrances : {len(self.spawn_areas)}")
        print(f"  Final     : {len(polygons)} region(s), {holes} hole(s), {vertices} exterior vertices")
        print(f"  Area      : {final_geometry.area:.2f} m²")
        print()
        print("Run simulation with: python src/demo_map_simulation.py")

    def _save_source_json(self):
        def serialize_polygon(poly: Polygon):
            return {
                "exterior": [(round(x, 3), round(y, 3)) for x, y in poly.exterior.coords],
                "holes": [
                    [(round(x, 3), round(y, 3)) for x, y in interior.coords]
                    for interior in poly.interiors
                ],
            }

        data = {
            "format": "jupedsim-final-geometry-drawing-v2",
            "meaning": "walkable polygons minus thick wall lines and closed wall loops equals data/map/geometry.wkt",
            "wall_thickness": self.wall_thickness,
            "walkable": [serialize_polygon(poly) for poly in self.walkables],
            "blocked_lines": [
                {"points": [(round(x, 3), round(y, 3)) for x, y in line.coords]}
                for line in self.blocking_lines
            ],
            "exits": [serialize_polygon(poly) for poly in self.exit_areas],
            "entrances": [serialize_polygon(poly) for poly in self.spawn_areas],
        }
        self.source_json.parent.mkdir(parents=True, exist_ok=True)
        self.source_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_stages_json(self):
        def serialize_stage(prefix: str, idx: int, poly: Polygon):
            return {
                "name": f"{prefix}_{idx}",
                "polygon": [(round(x, 3), round(y, 3)) for x, y in poly.exterior.coords],
            }

        data = {
            "format": "jupedsim-manual-stages-v1",
            "exits": [
                serialize_stage("exit", idx, poly)
                for idx, poly in enumerate(self.exit_areas)
            ],
            "entrances": [
                serialize_stage("entrance", idx, poly)
                for idx, poly in enumerate(self.spawn_areas)
            ],
        }
        STAGES_JSON.parent.mkdir(parents=True, exist_ok=True)
        STAGES_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _save_python_module(geometry, output_path: Path):
        polygons = polygons_from_geometry(geometry)
        lines = [
            '"""Auto-generated final walkable geometry for JuPedSim."""',
            "import shapely",
            "",
            "def get_geometry():",
            '    """Return the final walkable area as a Shapely geometry."""',
        ]
        parts = []
        for poly in polygons:
            exterior = list(poly.exterior.coords)
            holes = [list(interior.coords) for interior in poly.interiors]
            if holes:
                parts.append(f"shapely.Polygon({exterior!r}, holes={holes!r})")
            else:
                parts.append(f"shapely.Polygon({exterior!r})")
        if len(parts) == 1:
            lines.append(f"    return {parts[0]}")
        else:
            lines.append(f"    return shapely.unary_union([{', '.join(parts)}])")
        lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def run(self):
        final_geometry = self._compute_final_geometry()
        area = final_geometry.area if final_geometry is not None else 0.0
        width = self.background.shape[1]
        height = self.background.shape[0]

        print()
        print("=" * 72)
        print("JuPedSim final-geometry drawer")
        print("=" * 72)
        print(f"Background: {width}x{height}px, resolution={self.background.resolution}m/px")
        print(f"Current final walkable area: {area:.2f} m²")
        print()
        print("Meaning:")
        print("  Blue polygons are walkable. This is the map JuPedSim can use.")
        print("  Red lines are wall centerlines and become thick obstacles.")
        print("  Closed red wall loops also become filled blocked areas.")
        print("  Yellow polygons are exits.")
        print("  Purple polygons are entrances/spawn areas.")
        print("  Green preview is the final geometry saved to data/map/geometry.wkt.")
        print()
        print("Keys:")
        print("  left click        add vertex")
        print("  right click/Enter finish current shape")
        print("  a                 draw walkable polygon")
        print("  h                 draw obstacle wall line")
        print("  e                 draw exit area")
        print("  n                 draw entrance/spawn area")
        print("  d                 delete nearest obstacle wall line")
        print("  w                 cycle wall thickness")
        print("  u                 undo last vertex")
        print("  z                 undo last finished shape")
        print("  q/Esc             cancel current shape")
        print("  c                 clear all shapes")
        print("  s                 save final JuPedSim geometry")
        print("  t                 switch background mode")
        print("  p                 toggle green final preview")
        print("  r                 reset view")
        print("=" * 72)
        print()

        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw the final JuPedSim walkable geometry directly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pgm", nargs="?", default="data/map/localization_grid.pgm")
    parser.add_argument("--yaml", "-y", default=None)
    parser.add_argument(
        "--load-geometry",
        default=str(OUTPUT_WKT),
        help="WKT geometry to load if no editable source JSON exists.",
    )
    parser.add_argument(
        "--source-json",
        default=str(SOURCE_JSON),
        help="Editable drawing source JSON.",
    )
    parser.add_argument(
        "--blank",
        action="store_true",
        help="Start with an empty drawing instead of loading existing geometry/source.",
    )
    parser.add_argument(
        "--wall-thickness",
        "-w",
        type=float,
        default=0.2,
        help="Obstacle wall thickness in meters. Default: 0.2",
    )
    parser.add_argument("--load-walls", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bbox", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    pgm_path = resolve_project_path(args.pgm)
    yaml_path = resolve_project_path(args.yaml) if args.yaml else pgm_path.with_suffix(".yaml")
    load_geometry = resolve_project_path(args.load_geometry)
    source_json = resolve_project_path(args.source_json)

    if args.load_walls or args.bbox:
        print(
            "Note: wall-based drawing options are deprecated. "
            "This tool now draws final walkable geometry and thick obstacle walls directly."
        )

    try:
        drawer = FinalGeometryDrawer(
            pgm_path,
            yaml_path,
            load_geometry=load_geometry,
            source_json=source_json,
            blank=args.blank,
            wall_thickness=args.wall_thickness,
        )
        drawer.run()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
