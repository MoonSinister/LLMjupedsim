#!/usr/bin/env python3
"""Compatibility entry point for trajectory visualization."""

import sys

from jupedsim_mall.tools.demo_visualize import main


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <trajectory.sqlite>")
        raise SystemExit(1)
    main(sys.argv[1])
