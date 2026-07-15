#!/usr/bin/env python3
"""
查看和修改 JuPedSim 几何体 (geometry.wkt) 的交互工具。

功能:
  view      - 可视化查看几何体
  info      - 打印几何体统计信息
  simplify  - 简化顶点 (Douglas-Peucker)
  smooth    - 平滑锯齿边缘 (buffer + 反向 buffer)
  holes     - 列出 / 删除小孔洞
  cut       - 从可行走区域中挖掉一个矩形/多边形区域
  add       - 向可行走区域添加一个矩形/多边形区域
  crop      - 裁剪到指定范围
  hull      - 应用凸包 / 凹包
  export    - 同时导出 WKT 和 .py 模块

用法:
  python geometry_editor.py view
  python geometry_editor.py info
  python geometry_editor.py simplify --tolerance 0.15
  python geometry_editor.py smooth --distance 0.3
  python geometry_editor.py holes --list
  python geometry_editor.py holes --remove-smaller-than 2.0
  python geometry_editor.py cut --rect -10,10,0,15
  python geometry_editor.py add --rect 50,-30,55,-25
  python geometry_editor.py crop --rect -42,-30,56,16
  python geometry_editor.py hull --convex
  python geometry_editor.py export
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Polygon as MplPolygon
from shapely import Point, Polygon
from shapely.ops import unary_union

# ============================================================
# 配置
# ============================================================
DEFAULT_INPUT = "data/map/geometry_fine.wkt"
DEFAULT_OUTPUT_WKT = "data/map/geometry_fine.wkt"
DEFAULT_OUTPUT_PY = "data/map/geometry_fine.py"

# matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 工具函数
# ============================================================
def load_geometry(path: str = DEFAULT_INPUT) -> shapely.Geometry:
    path = Path(path)
    if path.suffix == ".wkt":
        return shapely.from_wkt(path.read_text(encoding="utf-8"))
    elif path.suffix == ".py":
        sys.path.insert(0, str(path.parent))
        mod = __import__(path.stem)
        return mod.get_geometry()
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")


def save_geometry(geo: shapely.Geometry, wkt_path: str = DEFAULT_OUTPUT_WKT,
                  py_path: str = None):
    wkt = shapely.to_wkt(geo, rounding_precision=2)
    Path(wkt_path).write_text(wkt + "\n", encoding="utf-8")
    print(f"✓ WKT 已保存到: {wkt_path}")

    if py_path:
        _save_py_module(geo, py_path)
        print(f"✓ Python 模块已保存到: {py_path}")


def _save_py_module(geo: shapely.Geometry, output_path: str):
    polys = list(geo.geoms) if hasattr(geo, "geoms") else [geo]
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
        holes = [list(interior.coords) for interior in p.interiors]
        if holes:
            lines.append(f"    holes = {holes}")
            lines.append("    return shapely.Polygon(exterior, holes=holes)")
        else:
            lines.append("    return shapely.Polygon(exterior)")
    else:
        parts = []
        for p in polys:
            ext = list(p.exterior.coords)
            holes = [list(interior.coords) for interior in p.interiors]
            if holes:
                parts.append(f"shapely.Polygon({ext}, holes={holes})")
            else:
                parts.append(f"shapely.Polygon({ext})")
        lines.append(f"    return shapely.unary_union([{', '.join(parts)}])")
    lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def get_polys(geo: shapely.Geometry) -> list:
    return list(geo.geoms) if hasattr(geo, "geoms") else [geo]


def parse_rect(s: str) -> tuple[float, float, float, float]:
    """解析 minx,miny,maxx,maxy"""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("rect 格式: minx,miny,maxx,maxy")
    return parts[0], parts[1], parts[2], parts[3]


def parse_polygon(s: str) -> Polygon:
    """解析逗号分隔的坐标序列: x1,y1,x2,y2,x3,y3,..."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) < 6 or len(parts) % 2 != 0:
        raise ValueError("polygon 格式: x1,y1,x2,y2,x3,y3,... (至少 3 对坐标)")
    coords = [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
    return Polygon(coords)


# ============================================================
# 命令实现
# ============================================================
def cmd_info(args):
    """打印几何体的统计信息"""
    geo = load_geometry(args.input)
    polys = get_polys(geo)

    total_area = geo.area
    bounds = geo.bounds
    total_vertices = sum(len(p.exterior.coords) for p in polys)
    total_holes = sum(len(p.interiors) for p in polys)
    hole_areas = []
    for p in polys:
        for h in p.interiors:
            hole_areas.append(Polygon(h).area)

    print("=" * 50)
    print("几何体信息")
    print("=" * 50)
    print(f"  文件            : {args.input}")
    print(f"  类型            : {geo.geom_type}")
    print(f"  区域数量        : {len(polys)}")
    print(f"  总面积          : {total_area:.1f} m²")
    print(f"  边界范围        : x=[{bounds[0]:.1f}, {bounds[2]:.1f}], "
          f"y=[{bounds[1]:.1f}, {bounds[3]:.1f}]")
    print(f"  世界尺寸        : {bounds[2] - bounds[0]:.1f}m × "
          f"{bounds[3] - bounds[1]:.1f}m")
    print(f"  总顶点数        : {total_vertices}")
    print(f"  孔洞总数        : {total_holes}")
    if hole_areas:
        hole_areas.sort()
        print(f"  孔洞面积范围    : {hole_areas[0]:.1f} ~ {hole_areas[-1]:.1f} m²")
        print(f"  小孔洞 (< 1m²)  : {sum(1 for a in hole_areas if a < 1.0)} 个")
        print(f"  中孔洞 (1-5m²)  : {sum(1 for a in hole_areas if 1.0 <= a < 5.0)} 个")
        print(f"  大孔洞 (>= 5m²) : {sum(1 for a in hole_areas if a >= 5.0)} 个")
    print()

    for i, p in enumerate(polys):
        n_verts = len(p.exterior.coords)
        n_holes = len(p.interiors)
        hole_info = ""
        if n_holes > 0:
            h_areas = [f"{Polygon(h).area:.1f}m²" for h in p.interiors]
            hole_info = f", holes=[{', '.join(h_areas)}]"
        print(f"  区域 {i}: area={p.area:.1f}m², vertices={n_verts}, "
              f"holes_count={n_holes}{hole_info}")

    print("=" * 50)


def cmd_view(args):
    """可视化几何体"""
    geo = load_geometry(args.input)
    polys = get_polys(geo)
    bounds = geo.bounds

    fig, ax = plt.subplots(figsize=getattr(args, 'figsize', (16, 8)), num="Geometry Viewer")

    color_walkable = "#a8d8ea"
    color_hole = "#ffffff"
    color_edge = "#333333"
    color_hole_edge = "#e63946"

    for poly in polys:
        # 外轮廓（可行走区域）
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.85, fc=color_walkable, ec=color_edge, linewidth=0.3)

        # 孔洞（障碍物）
        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, fc=color_hole, ec=color_hole_edge, linewidth=0.4)

    ax.set_aspect("equal")
    ax.set_title(
        f"Walkable Geometry — {len(polys)} region(s), "
        f"{geo.area:.0f} m², "
        f"bounds: [{bounds[0]:.0f}, {bounds[1]:.0f}] → [{bounds[2]:.0f}, {bounds[3]:.0f}]"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3, linestyle="--")

    # 交互状态
    state = {"selected": None, "annot": None}

    def on_mouse_move(event):
        if event.inaxes != ax:
            return
        # 检测鼠标所在的多边形区域
        pt = Point(event.xdata, event.ydata)
        for i, poly in enumerate(polys):
            if poly.contains(pt):
                if state["selected"] != i:
                    state["selected"] = i
                    p = polys[i]
                    n_holes = len(p.interiors)
                    n_verts = len(p.exterior.coords)
                    text = f"Region {i}: {p.area:.1f}m², {n_verts}v, {n_holes}h"
                    if state["annot"]:
                        state["annot"].remove()
                    state["annot"] = ax.annotate(
                        text, xy=(event.xdata, event.ydata),
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                        fontsize=9,
                    )
                    fig.canvas.draw_idle()
                return
        if state["annot"]:
            state["annot"].remove()
            state["annot"] = None
            state["selected"] = None
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == MouseButton.LEFT:
            print(f"  📍 点击坐标: ({event.xdata:.2f}, {event.ydata:.2f})")

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"✓ 图片保存到: {args.output}")

    print("交互功能：")
    print("  - 鼠标悬停 → 显示区域信息")
    print("  - 点击      → 打印坐标")
    print("  - 滚轮      → 缩放")
    print("  - 中键拖动  → 平移")
    print("  - 关闭窗口  → 退出")
    plt.tight_layout()
    plt.show()


def cmd_simplify(args):
    """简化顶点"""
    geo = load_geometry(args.input)
    before_v = sum(len(p.exterior.coords) for p in get_polys(geo))
    before_area = geo.area

    geo = geo.simplify(args.tolerance, preserve_topology=True)

    after_v = sum(len(p.exterior.coords) for p in get_polys(geo))
    after_area = geo.area

    print(f"简化: tolerance={args.tolerance}m")
    print(f"  顶点数: {before_v} → {after_v} (-{before_v - after_v}, "
          f"{(before_v - after_v) / before_v * 100:.1f}%)")
    print(f"  面积变化: {before_area:.1f} → {after_area:.1f} m² "
          f"({(after_area - before_area) / before_area * 100:+.2f}%)")

    save_geometry(geo, args.output, args.output_py)


def cmd_smooth(args):
    """平滑边缘"""
    geo = load_geometry(args.input)
    before_area = geo.area

    smoothed = geo.buffer(args.distance, resolution=8).buffer(-args.distance, resolution=8)

    after_area = smoothed.area
    if after_area <= 0:
        print("⚠ 平滑后面积为零！请减小 distance。")
        sys.exit(1)

    print(f"平滑: buffer ±{args.distance}m")
    print(f"  面积变化: {before_area:.1f} → {after_area:.1f} m² "
          f"({(after_area - before_area) / before_area * 100:+.2f}%)")

    save_geometry(smoothed, args.output, args.output_py)


def cmd_holes(args):
    """孔洞操作"""
    geo = load_geometry(args.input)
    polys = get_polys(geo)

    if args.list:
        all_holes = []
        for i, p in enumerate(polys):
            for j, hole in enumerate(p.interiors):
                h_poly = Polygon(hole)
                all_holes.append((i, j, h_poly))
        if not all_holes:
            print("没有孔洞。")
        else:
            all_holes.sort(key=lambda x: x[2].area)
            print(f"{'区域':<6} {'孔洞':<6} {'面积(m²)':<12} {'质心'}")
            print("-" * 50)
            for region_i, hole_j, hole_poly in all_holes:
                cx, cy = hole_poly.centroid.x, hole_poly.centroid.y
                print(f"{region_i:<6} {hole_j:<6} {hole_poly.area:<12.2f} "
                      f"({cx:.2f}, {cy:.2f})")
        return

    if args.remove_smaller_than is not None:
        min_area = args.remove_smaller_than
        new_polys = []
        removed = 0
        for p in polys:
            big_holes = [h for h in p.interiors if Polygon(h).area >= min_area]
            removed += len(p.interiors) - len(big_holes)
            new_polys.append(Polygon(p.exterior, holes=big_holes))
        geo = unary_union(new_polys)
        print(f"删除了 {removed} 个小于 {min_area}m² 的孔洞")
        save_geometry(geo, args.output, args.output_py)


def cmd_cut(args):
    """从可行走区域中挖掉指定区域"""
    geo = load_geometry(args.input)

    if args.rect:
        minx, miny, maxx, maxy = parse_rect(args.rect)
        cut_region = Polygon([
            (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)
        ])
    elif args.polygon:
        cut_region = parse_polygon(args.polygon)
    else:
        print("请指定 --rect 或 --polygon")
        return

    before_area = geo.area
    geo = geo.difference(cut_region)
    after_area = geo.area

    print(f"挖掉区域: area={cut_region.area:.1f}m²")
    print(f"  面积变化: {before_area:.1f} → {after_area:.1f} m²")

    save_geometry(geo, args.output, args.output_py)


def cmd_add(args):
    """向可行走区域添加指定区域"""
    geo = load_geometry(args.input)

    if args.rect:
        minx, miny, maxx, maxy = parse_rect(args.rect)
        add_region = Polygon([
            (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)
        ])
    elif args.polygon:
        add_region = parse_polygon(args.polygon)
    else:
        print("请指定 --rect 或 --polygon")
        return

    before_area = geo.area
    geo = geo.union(add_region)
    after_area = geo.area

    print(f"添加区域: area={add_region.area:.1f}m²")
    print(f"  面积变化: {before_area:.1f} → {after_area:.1f} m²")

    save_geometry(geo, args.output, args.output_py)


def cmd_crop(args):
    """裁剪到指定范围"""
    geo = load_geometry(args.input)
    before_area = geo.area

    if args.rect:
        minx, miny, maxx, maxy = parse_rect(args.rect)
        clip = Polygon([
            (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)
        ])
    else:
        print("请用 --rect 指定裁剪范围")
        return

    geo = geo.intersection(clip)
    after_area = geo.area

    print(f"裁剪到: x=[{minx}, {maxx}], y=[{miny}, {maxy}]")
    print(f"  面积变化: {before_area:.1f} → {after_area:.1f} m²")

    save_geometry(geo, args.output, args.output_py)


def cmd_hull(args):
    """凸包 / 凹包"""
    geo = load_geometry(args.input)
    before_area = geo.area

    if args.convex:
        geo = geo.convex_hull
        print("已应用凸包 (convex hull)")
    elif args.concave is not None:
        # 使用 alpha shape 近似凹包
        # 先采样点，然后做 concave hull
        from shapely.ops import triangulate
        ratio = args.concave
        geo_simplified = geo.simplify(0.1)
        polys = get_polys(geo_simplified)
        all_pts = []
        for p in polys:
            all_pts.extend(p.exterior.coords)
        from shapely import MultiPoint
        mp = MultiPoint(all_pts)
        geo = mp.convex_hull  # shapely 原生不支持 concave hull
        # 通过 buffer erosion 模拟
        geo = geo.buffer(-ratio).buffer(ratio * 1.5)
        print(f"已应用近似凹包 (ratio={ratio})")
    else:
        print("请指定 --convex 或 --concave RATIO")

    after_area = geo.area
    print(f"  面积变化: {before_area:.1f} → {after_area:.1f} m²")
    save_geometry(geo, args.output, args.output_py)


def cmd_export(args):
    """同时导出 WKT 和 Python 模块"""
    geo = load_geometry(args.input)
    save_geometry(geo, args.output, args.output_py or "data/map/geometry.py")


# ============================================================
# 交互式编辑模式
# ============================================================
def cmd_interactive(args):
    """交互式几何体编辑器"""
    geo = load_geometry(args.input)
    print("=" * 50)
    print("交互式几何体编辑器")
    print("=" * 50)
    print(f"已加载: {args.input}")
    print(f"区域数: {len(get_polys(geo))}, 总面积: {geo.area:.1f} m²")
    print()
    print("命令列表:")
    print("  view     - 可视化")
    print("  info     - 显示统计信息")
    print("  simplify <tolerance>   - 简化顶点")
    print("  smooth <distance>      - 平滑边缘")
    print("  holes list             - 列出孔洞")
    print("  holes remove <min_area> - 删除小孔洞")
    print("  cut rect <minx,miny,maxx,maxy>    - 挖掉矩形区域")
    print("  cut poly <x1,y1,x2,y2,...>        - 挖掉多边形")
    print("  add rect <minx,miny,maxx,maxy>    - 添加矩形区域")
    print("  save [path]            - 保存")
    print("  quit / q               - 退出")
    print()

    while True:
        try:
            cmd = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        if action in ("quit", "q", "exit"):
            break
        elif action == "view":
            cmd_view(argparse.Namespace(input=args.input, output=None, figsize=(16, 8)))
        elif action == "info":
            cmd_info(argparse.Namespace(input=args.input))
        elif action == "simplify":
            tol = float(parts[1]) if len(parts) > 1 else 0.15
            geo = geo.simplify(tol, preserve_topology=True)
            print(f"  简化完成 (tolerance={tol})")
        elif action == "smooth":
            dist = float(parts[1]) if len(parts) > 1 else 0.3
            geo = geo.buffer(dist, resolution=8).buffer(-dist, resolution=8)
            print(f"  平滑完成 (distance=±{dist})")
        elif action == "holes":
            if len(parts) < 2:
                print("  用法: holes list | holes remove <min_area>")
                continue
            sub = parts[1]
            polys = get_polys(geo)
            if sub == "list":
                for i, p in enumerate(polys):
                    for j, h in enumerate(p.interiors):
                        print(f"  region {i}, hole {j}: area={Polygon(h).area:.2f}m²")
            elif sub == "remove" and len(parts) >= 3:
                min_a = float(parts[2])
                new_polys = []
                removed = 0
                for p in polys:
                    big = [h for h in p.interiors if Polygon(h).area >= min_a]
                    removed += len(p.interiors) - len(big)
                    new_polys.append(Polygon(p.exterior, holes=big))
                geo = unary_union(new_polys)
                print(f"  删除了 {removed} 个孔洞")
        elif action == "cut":
            if len(parts) < 3:
                print("  用法: cut rect minx,miny,maxx,maxy | cut poly x1,y1,x2,y2,...")
                continue
            if parts[1] == "rect" and len(parts) >= 3:
                minx, miny, maxx, maxy = parse_rect(parts[2])
                region = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            elif parts[1] == "poly" and len(parts) >= 3:
                region = parse_polygon(parts[2])
            else:
                print("  格式错误")
                continue
            before = geo.area
            geo = geo.difference(region)
            print(f"  挖掉后面积: {before:.1f} → {geo.area:.1f} m²")
        elif action == "add":
            if len(parts) < 3:
                print("  用法: add rect minx,miny,maxx,maxy | add poly x1,y1,x2,y2,...")
                continue
            if parts[1] == "rect" and len(parts) >= 3:
                minx, miny, maxx, maxy = parse_rect(parts[2])
                region = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            elif parts[1] == "poly" and len(parts) >= 3:
                region = parse_polygon(parts[2])
            else:
                print("  格式错误")
                continue
            before = geo.area
            geo = geo.union(region)
            print(f"  添加后面积: {before:.1f} → {geo.area:.1f} m²")
        elif action == "save":
            path = parts[1] if len(parts) > 1 else DEFAULT_OUTPUT_WKT
            save_geometry(geo, path)
        else:
            print(f"  未知命令: {action}")

    # 退出时提示保存
    print(f"退出。最终几何体: {geo.area:.1f} m²")
    ans = input("是否保存到 geometry.wkt? [y/N]: ").strip().lower()
    if ans == "y":
        save_geometry(geo, DEFAULT_OUTPUT_WKT, DEFAULT_OUTPUT_PY)


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="查看和修改 JuPedSim 几何体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s view                          # 可视化查看
  %(prog)s info                          # 查看统计信息
  %(prog)s simplify --tolerance 0.15     # 简化顶点
  %(prog)s smooth --distance 0.3         # 平滑边缘
  %(prog)s holes --list                  # 列出所有孔洞
  %(prog)s holes --remove-smaller-than 2.0  # 删除 < 2m² 的孔洞
  %(prog)s cut --rect -10,10,0,15       # 挖掉矩形区域
  %(prog)s add --rect 50,-30,55,-25     # 添加矩形区域
  %(prog)s crop --rect -42,-30,56,16    # 裁剪到指定范围
  %(prog)s hull --convex                 # 凸包
  %(prog)s export                        # 导出 WKT + .py
  %(prog)s interactive                   # 交互模式
        """,
    )

    parser.add_argument("--input", "-i", default=DEFAULT_INPUT,
                        help=f"输入文件 (默认: {DEFAULT_INPUT})")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_WKT,
                        help=f"输出 WKT 路径 (默认: {DEFAULT_OUTPUT_WKT})")
    parser.add_argument("--output-py", default=DEFAULT_OUTPUT_PY,
                        help=f"输出 Python 模块路径 (默认: {DEFAULT_OUTPUT_PY})")

    sub = parser.add_subparsers(dest="command", help="命令")

    # view
    p_view = sub.add_parser("view", help="可视化查看几何体")
    p_view.add_argument("--output", "-o", default=None,
                        help="保存图片 (如 map_preview.png)")

    # info
    sub.add_parser("info", help="打印统计信息")

    # simplify
    p_simplify = sub.add_parser("simplify", help="简化顶点")
    p_simplify.add_argument("--tolerance", "-t", type=float, default=0.15,
                            help="简化容差 (米), 默认 0.15")

    # smooth
    p_smooth = sub.add_parser("smooth", help="平滑几何体边缘")
    p_smooth.add_argument("--distance", "-d", type=float, default=0.3,
                          help="缓冲距离 (米), 默认 0.3")

    # holes
    p_holes = sub.add_parser("holes", help="孔洞操作")
    p_holes.add_argument("--list", action="store_true", help="列出所有孔洞")
    p_holes.add_argument("--remove-smaller-than", type=float, default=None,
                         help="删除面积小于此值的孔洞 (m²)")

    # cut
    p_cut = sub.add_parser("cut", help="挖掉区域")
    p_cut.add_argument("--rect", type=str, default=None,
                       help="矩形: minx,miny,maxx,maxy")
    p_cut.add_argument("--polygon", type=str, default=None,
                       help="多边形: x1,y1,x2,y2,x3,y3,...")

    # add
    p_add = sub.add_parser("add", help="添加区域")
    p_add.add_argument("--rect", type=str, default=None,
                       help="矩形: minx,miny,maxx,maxy")
    p_add.add_argument("--polygon", type=str, default=None,
                       help="多边形: x1,y1,x2,y2,x3,y3,...")

    # crop
    p_crop = sub.add_parser("crop", help="裁剪到指定范围")
    p_crop.add_argument("--rect", type=str, required=True,
                        help="矩形: minx,miny,maxx,maxy")

    # hull
    p_hull = sub.add_parser("hull", help="凸包 / 凹包")
    p_hull.add_argument("--convex", action="store_true", help="凸包")
    p_hull.add_argument("--concave", type=float, default=None,
                        help="凹包 ratio (越大越宽松)")

    # export
    sub.add_parser("export", help="同时导出 WKT 和 .py 模块")

    # interactive
    sub.add_parser("interactive", aliases=["i"], help="交互式编辑模式")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 分发命令
    cmd_map = {
        "view":        lambda: cmd_view(args),
        "info":        lambda: cmd_info(args),
        "simplify":    lambda: cmd_simplify(args),
        "smooth":      lambda: cmd_smooth(args),
        "holes":       lambda: cmd_holes(args),
        "cut":         lambda: cmd_cut(args),
        "add":         lambda: cmd_add(args),
        "crop":        lambda: cmd_crop(args),
        "hull":        lambda: cmd_hull(args),
        "export":      lambda: cmd_export(args),
        "interactive": lambda: cmd_interactive(args),
        "i":           lambda: cmd_interactive(args),
    }
    cmd_map[args.command]()


if __name__ == "__main__":
    main()
