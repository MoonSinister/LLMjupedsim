"""Export a JuPedSim SQLite trajectory as an interactive Three.js replay."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import pathlib
import socketserver
import statistics
import webbrowser
from collections.abc import Sequence

from jupedsim_mall.analysis.run_metrics import load_trajectories
from jupedsim_mall.project import OUTPUT_DIR, PROJECT_ROOT


def _latest_trajectory() -> pathlib.Path:
    patterns = (
        "outputs/runs/*/trajectory.sqlite",
        "outputs/trajectories/*.sqlite",
        "outputs/smoke/*.sqlite",
    )
    candidates: list[pathlib.Path] = []
    for pattern in patterns:
        candidates.extend(PROJECT_ROOT.glob(pattern))
    if not candidates:
        raise SystemExit("No trajectory SQLite file found. Run start.bat first.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _sample_trajectories(
    trajectories: dict[int, list[tuple[int, float, float]]],
    *,
    frame_stride: int,
    max_agents: int,
) -> tuple[list[int], dict[int, dict[int, tuple[float, float]]], dict[int, list[tuple[float, float]]]]:
    selected_ids = [
        agent_id
        for agent_id in sorted(trajectories)
        if trajectories[agent_id]
    ][:max_agents]
    if not selected_ids:
        return [], {}, {}

    first_frame = min(trajectories[agent_id][0][0] for agent_id in selected_ids)
    last_frame = max(trajectories[agent_id][-1][0] for agent_id in selected_ids)
    frames = list(range(first_frame, last_frame + 1, frame_stride))
    if not frames or frames[-1] != last_frame:
        frames.append(last_frame)

    sampled: dict[int, dict[int, tuple[float, float]]] = {}
    trails: dict[int, list[tuple[float, float]]] = {}
    for agent_id in selected_ids:
        points = trajectories[agent_id]
        trails[agent_id] = [
            (round(x, 3), round(y, 3))
            for index, (_, x, y) in enumerate(points)
            if index % frame_stride == 0 or index == len(points) - 1
        ]
        cursor = 0
        for frame in frames:
            if frame < points[0][0] or frame > points[-1][0]:
                continue
            while cursor + 1 < len(points) and points[cursor + 1][0] < frame:
                cursor += 1
            sampled.setdefault(frame, {})[agent_id] = _interpolate_position(points, cursor, frame)
    return frames, sampled, trails


def _interpolate_position(points: list[tuple[int, float, float]], cursor: int, frame: int) -> tuple[float, float]:
    current_frame, current_x, current_y = points[cursor]
    if current_frame == frame or cursor + 1 >= len(points):
        return round(current_x, 3), round(current_y, 3)

    next_frame, next_x, next_y = points[cursor + 1]
    if next_frame <= current_frame:
        return round(current_x, 3), round(current_y, 3)
    factor = (frame - current_frame) / (next_frame - current_frame)
    return (
        round(current_x + (next_x - current_x) * factor, 3),
        round(current_y + (next_y - current_y) * factor, 3),
    )


def _bounds(trails: dict[int, list[tuple[float, float]]]) -> dict[str, float]:
    xs = [x for points in trails.values() for x, _ in points]
    ys = [y for points in trails.values() for _, y in points]
    if not xs or not ys:
        return {"min_x": -10, "max_x": 10, "min_y": -10, "max_y": 10}
    padding = max(2.0, 0.08 * max(max(xs) - min(xs), max(ys) - min(ys)))
    return {
        "min_x": round(min(xs) - padding, 3),
        "max_x": round(max(xs) + padding, 3),
        "min_y": round(min(ys) - padding, 3),
        "max_y": round(max(ys) + padding, 3),
    }


def build_replay_payload(
    trajectory_path: pathlib.Path,
    *,
    frame_stride: int = 10,
    max_agents: int = 300,
) -> dict:
    _, fps, trajectories = load_trajectories(trajectory_path)
    frames, sampled, trails = _sample_trajectories(
        trajectories,
        frame_stride=max(1, frame_stride),
        max_agents=max(1, max_agents),
    )
    replay_frames = [
        {
            "frame": frame,
            "time": round((frame - frames[0]) / fps, 3) if frames else 0,
            "agents": [
                [agent_id, x, y]
                for agent_id, (x, y) in sorted(sampled.get(frame, {}).items())
            ],
        }
        for frame in frames
    ]
    observed_counts = [len(frame["agents"]) for frame in replay_frames]
    return {
        "source": str(trajectory_path),
        "fps": fps,
        "frame_stride": frame_stride,
        "agent_count": len(trajectories),
        "exported_agent_count": len(trails),
        "frame_count": len(replay_frames),
        "median_visible_agents": statistics.median(observed_counts) if observed_counts else 0,
        "bounds": _bounds(trails),
        "trails": {str(agent_id): points for agent_id, points in trails.items()},
        "frames": replay_frames,
    }


def _html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JuPedSim 3D Replay</title>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.165.0/build/three.module.js"
      }}
    }}
  </script>
  <style>
    html, body {{ margin: 0; height: 100%; overflow: hidden; background: #111317; color: #f4f4f2; font-family: Segoe UI, Arial, sans-serif; }}
    #scene {{ position: fixed; inset: 0; }}
    .hud {{ position: fixed; left: 16px; right: 16px; bottom: 16px; display: flex; gap: 12px; align-items: center; padding: 10px 12px; background: rgba(17,19,23,.84); border: 1px solid rgba(255,255,255,.16); border-radius: 8px; backdrop-filter: blur(8px); }}
    button {{ width: 42px; height: 34px; border: 1px solid rgba(255,255,255,.24); border-radius: 6px; background: #f4f4f2; color: #111317; font-weight: 700; cursor: pointer; }}
    input[type=range] {{ flex: 1; min-width: 120px; accent-color: #18a999; }}
    .stat {{ min-width: 190px; font-size: 13px; color: #d9d7d0; }}
  </style>
</head>
<body>
  <div id="scene"></div>
  <div class="hud">
    <button id="play" title="Play or pause">Pause</button>
    <input id="scrub" type="range" min="0" max="0" value="0">
    <div class="stat" id="stat"></div>
  </div>
  <script id="replay-data" type="application/json">{data}</script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js';

    const replay = JSON.parse(document.getElementById('replay-data').textContent);
    const root = document.getElementById('scene');
    const bounds = replay.bounds;
    const width = bounds.max_x - bounds.min_x;
    const depth = bounds.max_y - bounds.min_y;
    const cx = (bounds.min_x + bounds.max_x) / 2;
    const cy = (bounds.min_y + bounds.max_y) / 2;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111317);
    const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
    camera.position.set(cx, Math.max(width, depth) * 0.75, cy + Math.max(width, depth) * 0.9);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    root.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(cx, 0, cy);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x46505a, 2.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(cx - 8, 24, cy + 10);
    scene.add(sun);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(width, depth),
      new THREE.MeshStandardMaterial({{ color: 0x2c3037, roughness: 0.92, metalness: 0.02 }})
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(cx, -0.02, cy);
    scene.add(floor);

    const grid = new THREE.GridHelper(Math.max(width, depth), 24, 0x6f7b86, 0x343b44);
    grid.position.set(cx, 0, cy);
    scene.add(grid);

    const agentMaterial = new THREE.MeshStandardMaterial({{ color: 0xffc857, roughness: 0.55 }});
    const geometry = new THREE.CylinderGeometry(0.13, 0.13, 1.15, 14);
    const agents = new Map();

    const trailMaterial = new THREE.LineBasicMaterial({{ color: 0x18a999, transparent: true, opacity: 0.28 }});
    for (const points of Object.values(replay.trails)) {{
      if (points.length < 2) continue;
      const vertices = points.map(([x, y]) => new THREE.Vector3(x, 0.025, y));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(vertices), trailMaterial));
    }}

    function meshFor(id) {{
      if (!agents.has(id)) {{
        const mesh = new THREE.Mesh(geometry, agentMaterial.clone());
        mesh.position.y = 0.58;
        scene.add(mesh);
        agents.set(id, mesh);
      }}
      return agents.get(id);
    }}

    const scrub = document.getElementById('scrub');
    const stat = document.getElementById('stat');
    const play = document.getElementById('play');
    scrub.max = Math.max(0, replay.frames.length - 1);
    let index = 0;
    let playing = true;

    play.onclick = () => {{
      playing = !playing;
      play.textContent = playing ? 'Pause' : 'Play';
    }};
    scrub.oninput = () => {{ index = Number(scrub.value); drawFrame(); }};

    function drawFrame() {{
      const frame = replay.frames[index];
      const visible = new Set();
      for (const [id, x, y] of frame.agents) {{
        const mesh = meshFor(id);
        mesh.position.x = x;
        mesh.position.z = y;
        mesh.material = agentMaterial;
        mesh.visible = true;
        visible.add(id);
      }}
      for (const [id, mesh] of agents.entries()) {{
        if (!visible.has(id)) mesh.visible = false;
      }}
      scrub.value = index;
      stat.textContent = `${{frame.time.toFixed(1)}}s | visible ${{frame.agents.length}} | exported ${{replay.exported_agent_count}}/${{replay.agent_count}} agents`;
    }}

    let last = 0;
    function animate(now) {{
      requestAnimationFrame(animate);
      if (playing && now - last > 90) {{
        index = (index + 1) % replay.frames.length;
        drawFrame();
        last = now;
      }}
      controls.update();
      renderer.render(scene, camera);
    }}

    addEventListener('resize', () => {{
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    }});

    drawFrame();
    animate(0);
  </script>
</body>
</html>
"""


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def serve_and_open(path: pathlib.Path, *, preferred_port: int) -> None:
    directory = path.parent.resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + 20):
        try:
            with _ReusableTCPServer(("127.0.0.1", port), handler) as server:
                url = f"http://127.0.0.1:{port}/{path.name}"
                print(f"Serving: {url}")
                print("Close this terminal window or press Ctrl+C to stop the 3D replay server.")
                webbrowser.open(url)
                server.serve_forever()
        except OSError as exc:
            last_error = exc
            continue
        except KeyboardInterrupt:
            print("\n3D replay server stopped.")
            return
    raise SystemExit(f"Could not start local replay server: {last_error}")


def export_html(trajectory_path: pathlib.Path, output: pathlib.Path, *, frame_stride: int, max_agents: int) -> dict:
    trajectory_path = trajectory_path.resolve()
    output = output.resolve()
    payload = build_replay_payload(trajectory_path, frame_stride=frame_stride, max_agents=max_agents)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_html(payload), encoding="utf-8")
    return {"output": str(output), **{key: payload[key] for key in ("agent_count", "exported_agent_count", "frame_count")}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", nargs="?", type=pathlib.Path, help="Trajectory SQLite file. Defaults to newest run.")
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT_DIR / "visualizations" / "latest_3d_replay.html")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-agents", type=int, default=300)
    parser.add_argument("--open", action="store_true", help="Open the exported HTML in the default browser.")
    parser.add_argument("--serve-port", type=int, default=8765, help="First local port to try when opening the replay.")
    args = parser.parse_args(argv)

    trajectory = args.trajectory or _latest_trajectory()
    result = export_html(trajectory, args.output, frame_stride=args.frame_stride, max_agents=args.max_agents)
    print(f"3D replay: {result['output']}")
    print(f"Agents: {result['exported_agent_count']}/{result['agent_count']} | frames: {result['frame_count']}")
    if args.open:
        serve_and_open(pathlib.Path(result["output"]), preferred_port=args.serve_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
