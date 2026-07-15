#!/usr/bin/env python3
"""Convert a ROS-style PGM occupancy grid map to JuPedSim-compatible geometry.

Handles:
  - Lidar speckle/noise removal: GaussianBlur + area filtering + morphological close
  - Coordinate flip: OpenCV (Y-down, origin top-left) → world coords (Y-up, origin bottom-left)
  - Output as Shapely Polygon/MultiPolygon (WKT format) ready for JuPedSim

Usage:
    python src/pgm_to_geometry.py data/map/localization_grid.pgm -o data/map/geometry.wkt
    python src/pgm_to_geometry.py data/map/localization_grid.pgm --save-debug outputs/debug_
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import shapely
import yaml
from shapely.geometry import Polygon
from shapely.ops import unary_union


def load_map(pgm_path: str):
    """Load PGM image and its YAML metadata."""
    pgm = Path(pgm_path)
    yaml_path = pgm.with_suffix(".yaml")
    if not yaml_path.exists():
        yaml_path = pgm.with_suffix(".yml")

    if not yaml_path.exists():
        raise FileNotFoundError(f"No .yaml metadata found for {pgm_path}")

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    resolution = float(meta["resolution"])
    origin = [float(o) for o in meta["origin"]]

    img = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {pgm_path} as an image")

    return img, resolution, origin


def binarize_free_space(img: np.ndarray, free_thresh: int | None = None) -> np.ndarray:
    """Convert grayscale PGM to binary: 255 = free/walkable, 0 = obstacle/unknown.

    If free_thresh is None, auto-detects the darkest pixel value as "free".
    """
    if free_thresh is None:
        free_val = np.unique(img)[0]  # typically 0
        return (img == free_val).astype(np.uint8) * 255
    else:
        return (img < free_thresh).astype(np.uint8) * 255


def clean_map(binary: np.ndarray, close_kernel: int, min_area: int,
              blur_sigma: float = 2.0) -> np.ndarray:
    """Clean lidar speckle and bridge gaps in the walkable area.

    1. GaussianBlur to soften jagged edges from lidar scans
    2. Morphological CLOSE (dilate → erode) to connect nearby free-space patches
    3. Remove small noise components (isolated speckle pixels)
    """
    blurred = cv2.GaussianBlur(binary, (5, 5), sigmaX=blur_sigma)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Remove tiny components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clean = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    return clean


def pixel_to_world(px: float, py: float, origin: list, height: int,
                   resolution: float) -> tuple[float, float]:
    """Convert OpenCV pixel coord (Y-down, origin top-left) to world coord (Y-up)."""
    wx = origin[0] + px * resolution
    wy = origin[1] + (height - 1 - py) * resolution
    return (wx, wy)


def contour_to_world(cnt, resolution, origin, height, epsilon_px):
    """Convert a single OpenCV contour to world-coordinate polygon."""
    approx = cv2.approxPolyDP(cnt, epsilon_px, True)
    pts = [pixel_to_world(float(pt[0][0]), float(pt[0][1]),
                          origin, height, resolution)
           for pt in approx]
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def extract_polygons(binary_walkable: np.ndarray, resolution: float,
                     origin: list, height: int,
                     simplify_epsilon: float = 0.05) -> shapely.Geometry:
    """Extract Shapely Polygon(s) with holes from binary walkable mask.

    1. Find external contours → walkable regions
    2. Find internal contours → holes (obstacles)
    3. Match holes to containing region by geometry containment
    """
    contours, hierarchy = cv2.findContours(
        binary_walkable, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found in binary map")

    epsilon_px = simplify_epsilon / resolution  # meters → pixels

    # Separate external and hole contours
    externals = []
    hole_candidates = []

    for i, (cnt, hier) in enumerate(zip(contours, hierarchy[0])):
        poly = contour_to_world(cnt, resolution, origin, height, epsilon_px)
        if poly is None:
            continue
        if hasattr(poly, "geoms"):
            # buffer(0) split it; treat each sub as separate
            sub_polys = list(poly.geoms)
        else:
            sub_polys = [poly]

        if hier[3] == -1:
            externals.extend(sub_polys)
        else:
            hole_candidates.extend(sub_polys)

    if not externals:
        raise ValueError("No external contours found")

    # Match holes to their containing external polygon
    result_polys = []
    for ext in externals:
        my_holes = []
        for h in hole_candidates:
            if h.area < simplify_epsilon ** 2:
                continue
            if ext.contains(h) or ext.covers(h):
                my_holes.append(h.exterior.coords)

        if my_holes:
            ext = Polygon(ext.exterior.coords, holes=my_holes)
            if not ext.is_valid:
                ext = ext.buffer(0)
        result_polys.append(ext)

    if len(result_polys) == 1:
        return result_polys[0]
    merged = unary_union(result_polys)
    # Drop tiny fragments (< 5 m2)
    if hasattr(merged, "geoms"):
        merged = unary_union([g for g in merged.geoms if g.area > 5.0])
    return merged


def save_wkt(geometry, output_path: str, rounding_precision: int = 2):
    """Save geometry as WKT file."""
    wkt = shapely.to_wkt(geometry, rounding_precision=rounding_precision)
    Path(output_path).write_text(wkt + "\n", encoding="utf-8")
    return wkt


def save_python_module(geometry, output_path: str):
    """Save geometry as a Python module that can be imported directly."""
    if hasattr(geometry, "geoms"):
        polys = list(geometry.geoms)
    else:
        polys = [geometry]

    lines = [
        '"""Auto-generated walkable geometry for JuPedSim."""',
        'import shapely',
        '',
        'def get_geometry():',
        '    """Return the walkable area as a Shapely geometry."""',
    ]

    if len(polys) == 1:
        p = polys[0]
        ext = list(p.exterior.coords)
        lines.append(f"    exterior = {ext}")
        holes = []
        for interior in p.interiors:
            holes.append(list(interior.coords))
        if holes:
            lines.append(f"    holes = {holes}")
            lines.append("    return shapely.Polygon(exterior, holes=holes)")
        else:
            lines.append("    return shapely.Polygon(exterior)")
    else:
        poly_exprs = []
        for p in polys:
            ext = list(p.exterior.coords)
            holes = [list(inter.coords) for inter in p.interiors]
            if holes:
                poly_exprs.append(f"shapely.Polygon({ext}, holes={holes})")
            else:
                poly_exprs.append(f"shapely.Polygon({ext})")
        lines.append(f"    return shapely.unary_union([{', '.join(poly_exprs)}])")

    lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PGM occupancy grid to JuPedSim geometry"
    )
    parser.add_argument("pgm", help="Path to .pgm file (needs .yaml alongside)")
    parser.add_argument("--close-kernel", type=int, default=70,
                        help="Morphological close kernel in pixels (default: 70 = 3.5m)")
    parser.add_argument("--min-area", type=int, default=500,
                        help="Minimum component area in pixels (default: 500 = 1.25 m2)")
    parser.add_argument("--blur-sigma", type=float, default=2.0,
                        help="GaussianBlur sigma (default: 2.0)")
    parser.add_argument("--simplify", type=float, default=0.1,
                        help="Contour simplify epsilon in meters (default: 0.1)")
    parser.add_argument("--free-thresh", type=int, default=None,
                        help="Free-space pixel threshold (default: auto-detect)")
    parser.add_argument("--output", "-o", default="geometry.wkt",
                        help="Output .wkt file path (default: geometry.wkt)")
    parser.add_argument("--output-py", default=None,
                        help="Also output as importable .py module")
    parser.add_argument("--save-debug", default=None,
                        help="Save intermediate debug images to this path prefix")
    args = parser.parse_args()

    # 1. Load
    print(f"Loading {args.pgm} ...")
    img, resolution, origin = load_map(args.pgm)
    h, w = img.shape
    print(f"  Image: {w}x{h}, resolution={resolution} m/px, origin=({origin[0]}, {origin[1]})")
    print(f"  World extent: x=[{origin[0]:.1f}, {origin[0] + w * resolution:.1f}], "
          f"y=[{origin[1]:.1f}, {origin[1] + h * resolution:.1f}]")

    # 2. Binarize
    print("Binarizing free space ...")
    binary = binarize_free_space(img, args.free_thresh)
    n_free = (binary > 0).sum()
    print(f"  Free pixels: {n_free} / {binary.size} ({n_free / binary.size * 100:.1f}%)")

    # 3. Clean
    print(f"Cleaning: blur_sigma={args.blur_sigma}, close={args.close_kernel}px "
          f"({args.close_kernel * resolution:.1f}m), min_area={args.min_area}px ...")
    clean = clean_map(binary, close_kernel=args.close_kernel, min_area=args.min_area,
                      blur_sigma=args.blur_sigma)

    n_clean = (clean > 0).sum()
    num_labels = cv2.connectedComponents(clean)[0] - 1
    print(f"  Clean pixels: {n_clean} / {binary.size} ({n_clean / binary.size * 100:.1f}%)")
    print(f"  Components kept: {num_labels}")

    if args.save_debug:
        scale = 0.25
        cv2.imwrite(f"{args.save_debug}binary.png",
                    cv2.resize(binary, None, fx=scale, fy=scale))
        cv2.imwrite(f"{args.save_debug}clean.png",
                    cv2.resize(clean, None, fx=scale, fy=scale))

    # 4. Extract geometry
    print(f"Extracting polygons (simplify={args.simplify}m) ...")
    geometry = extract_polygons(clean, resolution, origin, h, args.simplify)

    if hasattr(geometry, "geoms"):
        n_regions = len(geometry.geoms)
        total_area = sum(p.area for p in geometry.geoms)
    else:
        n_regions = 1
        total_area = geometry.area

    print(f"  Regions: {n_regions}, total area: {total_area:.1f} m2")

    # Print per-region stats
    polys = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    for i, p in enumerate(polys):
        n_holes = len(p.interiors)
        n_verts = len(p.exterior.coords)
        print(f"  Region {i}: area={p.area:.1f}m2, exterior_vertices={n_verts}, holes={n_holes}")

    # 5. Output
    wkt = save_wkt(geometry, args.output, rounding_precision=2)
    print(f"\nWKT saved to {args.output} ({len(wkt)} chars)")

    if args.output_py:
        save_python_module(geometry, args.output_py)
        print(f"Python module saved to {args.output_py}")

    return geometry


if __name__ == "__main__":
    main()
