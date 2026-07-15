#!/usr/bin/env python3
"""
将 PGM 地图中所有自由空间点绘制出来，支持交互式矩形框选区域，
框选的区域自动保存为 region 定义。

用法:
  python src/plot_free_points.py                    # 查看所有点
  python src/plot_free_points.py --save outputs/map_points.png
  python src/plot_free_points.py --max-points 10000

交互操作:
  - 鼠标悬停 → 显示像素/世界坐标
  - 左键拖拽 → 框选矩形区域，松开后自动保存
  - 右键点击 → 撤销上一个框选
  - 关闭窗口 → 打印所有已选区域并保存到 JSON
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_pgm_free_points(pgm_path, yaml_path, max_points=None):
    """加载 PGM 地图，返回所有自由空间点的世界坐标 (N, 2) 和元数据"""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = float(meta["resolution"])
    origin = [float(o) for o in meta["origin"]]

    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    free_val = np.unique(img)[0]
    free_mask = img == free_val
    py, px = np.where(free_mask)

    wx = origin[0] + px * resolution
    wy = origin[1] + (h - 1 - py) * resolution

    points = np.column_stack([wx, wy])

    if max_points and len(points) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]

    return points, resolution, origin, (h, w)


def world_to_pixel(wx, wy, origin, height, resolution):
    px = (wx - origin[0]) / resolution
    py = height - 1 - (wy - origin[1]) / resolution
    return (px, py)


def main():
    parser = argparse.ArgumentParser(description="绘制 PGM 地图的所有自由空间点")
    parser.add_argument("pgm", nargs="?", default="data/map/localization_grid.pgm")
    parser.add_argument("--yaml", "-y", default=None)
    parser.add_argument("--max-points", "-n", type=int, default=None,
                        help="随机采样点数（默认全部，数据量大时建议 20000）")
    parser.add_argument("--save", "-s", default=None, help="保存图片路径")
    parser.add_argument("--output-regions", "-o", default="data/map/my_regions.json",
                        help="框选区域的输出文件")
    parser.add_argument("--load-regions", "-r", default=None,
                        help="加载已有的 regions JSON，叠加显示")
    args = parser.parse_args()

    pgm_path = Path(args.pgm)
    yaml_path = Path(args.yaml) if args.yaml else pgm_path.with_suffix(".yaml")
    if not yaml_path.exists():
        yaml_path = pgm_path.with_suffix(".yml")

    print(f"加载 {pgm_path} ...")
    points, resolution, origin, (height, width) = load_pgm_free_points(
        str(pgm_path), str(yaml_path), args.max_points
    )
    print(f"  自由空间点数: {len(points)}")
    print(f"  分辨率: {resolution} m/px")
    print(f"  图像尺寸: {width}x{height} px")
    print(f"  世界范围: x=[{origin[0]:.1f}, {origin[0]+width*resolution:.1f}], "
          f"y=[{origin[1]:.1f}, {origin[1]+height*resolution:.1f}]")

    # ---- 加载已有 regions ----
    existing_regions = []
    if args.load_regions:
        er_path = Path(args.load_regions)
        if er_path.exists():
            data = json.loads(er_path.read_text(encoding="utf-8"))
            existing_regions = data.get("regions", [])
            print(f"  已加载 {len(existing_regions)} 个已有区域")

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(20, 8), num="PGM Free Space Points")

    # 背景：半透明显示原始图像
    bg = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    bg_flipped = np.flipud(bg)
    extent = [
        origin[0], origin[0] + width * resolution,
        origin[1], origin[1] + height * resolution,
    ]
    free_val = np.unique(bg)[0]
    bg_rgba = np.zeros((height, width, 4))
    bg_rgba[bg_flipped == free_val, :] = [0.12, 0.47, 0.71, 0.12]
    bg_rgba[bg_flipped != free_val, :] = [0.05, 0.05, 0.05, 0.03]
    ax.imshow(bg_rgba, extent=extent, origin="lower", aspect="equal", interpolation="nearest")

    # 散点：自由空间
    alpha = 0.25 if len(points) > 50000 else 0.45
    s = 0.3 if len(points) > 50000 else 1.5
    ax.scatter(points[:, 0], points[:, 1], c="#1f77b4", s=s, alpha=alpha,
               marker="s", edgecolors="none", label=f"Free space ({len(points):,} pts)")

    # 显示已有区域
    colors = plt.cm.tab10.colors
    for i, r in enumerate(existing_regions):
        rect = r.get("world_rect", r.get("rect", None))
        if rect:
            minx, miny, maxx, maxy = rect
            ax.add_patch(mpatches.Rectangle(
                (minx, miny), maxx-minx, maxy-miny,
                linewidth=2, edgecolor=colors[i % len(colors)],
                facecolor=colors[i % len(colors)], alpha=0.1
            ))
            ax.annotate(r["name"], ((minx+maxx)/2, (miny+maxy)/2),
                        ha="center", va="center", fontsize=8,
                        color=colors[i % len(colors)], fontweight="bold")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Walkable Free-Space Points — {len(points):,} points, "
        f"World: {width*resolution:.0f}m × {height*resolution:.0f}m"
    )
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    # ---- 交互式框选 ----
    regions = []  # 新框选的
    rect_patch = None
    start_xy = None
    is_dragging = False
    MIN_DRAG_PX = 5  # 最小拖拽像素，小于这个算单击

    def on_press(event):
        nonlocal rect_patch, start_xy, is_dragging
        if event.inaxes != ax:
            return

        if event.button == 1:  # 左键：开始拖拽
            start_xy = (event.xdata, event.ydata)
            is_dragging = False
            rect_patch = mpatches.Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=2, edgecolor="red", facecolor="red", alpha=0.15,
                linestyle="--",
            )
            ax.add_patch(rect_patch)

        elif event.button == 3:  # 右键：撤销
            if regions:
                removed = regions.pop()
                print(f"\n  ❌ 撤销: {removed['name']} (剩余 {len(regions)} 个)")
                # 删除最后一个绿色矩形
                for p in list(ax.patches):
                    fc = p.get_facecolor()
                    if hasattr(fc, '__len__') and len(fc) >= 3:
                        if fc[0] < 0.1 and fc[1] > 0.4 and fc[2] < 0.1:
                            p.remove()
                            break
                fig.canvas.draw_idle()

    def on_motion(event):
        nonlocal rect_patch, is_dragging
        if rect_patch is None or event.inaxes != ax or start_xy is None:
            return
        dx = event.xdata - start_xy[0]
        dy = event.ydata - start_xy[1]
        # 判断是否真的在拖拽
        dx_px = abs(event.x - event.xdata) if hasattr(event, 'x') else abs(dx)
        dy_px = abs(event.y - event.ydata) if hasattr(event, 'y') else abs(dy)
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            is_dragging = True
        rect_patch.set_width(dx)
        rect_patch.set_height(dy)

    def on_release(event):
        nonlocal rect_patch, start_xy, is_dragging
        if rect_patch is None or event.inaxes != ax or start_xy is None:
            return
        x1, y1 = start_xy
        x2, y2 = event.xdata, event.ydata
        minx, maxx = sorted([x1, x2])
        miny, maxy = sorted([y1, y2])
        area = (maxx - minx) * (maxy - miny)

        # 太小算单击（不是框选）
        if area < 0.01:
            rect_patch.remove()
            rect_patch = None
            start_xy = None
            # 单击：打印坐标
            wx, wy = event.xdata, event.ydata
            px, py = world_to_pixel(wx, wy, origin, height, resolution)
            print(f"  📍 世界: ({wx:.2f}, {wy:.2f})  ←  像素: ({px:.0f}, {py:.0f})")
            fig.canvas.draw_idle()
            return

        region = {
            "name": f"region_{len(regions)+1}",
            "world_rect": [round(minx, 2), round(miny, 2), round(maxx, 2), round(maxy, 2)],
            "area_approx_m2": round(area, 2),
        }
        regions.append(region)

        # 固定矩形
        rect_patch.set_facecolor("green")
        rect_patch.set_alpha(0.1)
        rect_patch.set_edgecolor("green")
        rect_patch.set_linestyle("-")
        rect_patch = None
        start_xy = None

        print(f"\n  ✅ {region['name']}: rect={region['world_rect']}, area≈{region['area_approx_m2']}m²")
        print(f"     当前共 {len(regions)} 个区域")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    print()
    print("=" * 60)
    print("交互说明:")
    print("  左键拖拽  →  框选矩形区域（松开后自动记录）")
    print("  左键单击  →  打印该点坐标")
    print("  右键点击  →  撤销上一个框选")
    print("  滚轮      →  缩放")
    print("  中键拖拽  →  平移")
    print("  关闭窗口  →  保存所有框选区域并退出")
    print("=" * 60)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\n图片已保存到: {args.save}")

    plt.tight_layout()
    plt.show()

    # ---- 输出 ----
    print()
    print("=" * 60)
    print("框选区域汇总")
    print("=" * 60)

    all_regions = existing_regions + regions

    if not all_regions:
        print("(未框选任何区域)")
    else:
        for r in all_regions:
            rect = r.get("world_rect", r.get("rect", "?"))
            area = r.get("area_approx_m2", r.get("area_m2", "?"))
            print(f"  {r['name']}: rect={rect}, area≈{area}m²")

        output = {
            "map": {
                "pgm": str(pgm_path.resolve()),
                "yaml": str(yaml_path.resolve()),
                "resolution": resolution,
                "origin": origin,
                "width": width,
                "height": height,
            },
            "regions": all_regions,
        }
        out_path = Path(args.output_regions)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  ✅ 已保存 {len(all_regions)} 个区域到: {out_path}")

    print()
    print("提示: 用 --load-regions data/map/my_regions.json 可叠加上次结果继续编辑")


if __name__ == "__main__":
    main()
