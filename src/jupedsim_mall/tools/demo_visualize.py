#! /usr/bin/env python3
"""Launch jupedsim_visualizer GUI and auto-load a .sqlite trajectory file.

Usage:
    python demo_visualize.py demo_bottleneck.sqlite
"""

import sys
from pathlib import Path

import jupedsim as jps
from PySide6.QtWidgets import QApplication
from jupedsim.recording import Recording
from jupedsim_visualizer.geometry import Geometry
from jupedsim_visualizer.main_window import MainWindow
from jupedsim_visualizer.replay_widget import ReplayWidget
from jupedsim_visualizer.trajectory import Trajectory


def main(trajectory_file: str):
    file = Path(trajectory_file).resolve()
    if not file.exists():
        print(f"File not found: {file}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow()

    rec = Recording(file.as_posix())
    navi = jps.RoutingEngine(rec.geometry())
    geo = Geometry(navi)
    trajectory = Trajectory(rec)
    tab = ReplayWidget(navi, rec, geo, trajectory, parent=window)
    tab.render_widget.show_grid(False)
    window.tabs.insertTab(0, tab, file.name)
    window.tabs.setCurrentIndex(0)

    sys.exit(app.exec())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <trajectory.sqlite>")
        sys.exit(1)
    main(sys.argv[1])
