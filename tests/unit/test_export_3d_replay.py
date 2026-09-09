import pathlib
import sqlite3

from jupedsim_mall.tools.export_3d_replay import build_replay_payload, export_html


def _trajectory(path: pathlib.Path) -> pathlib.Path:
    with sqlite3.connect(path) as connection:
        connection.execute("create table metadata(key text, value text)")
        connection.execute("insert into metadata values ('fps', '25')")
        connection.execute("create table trajectory_data(id integer, frame integer, pos_x real, pos_y real)")
        connection.executemany(
            "insert into trajectory_data values (?, ?, ?, ?)",
            [
                (1, 0, 0.0, 0.0),
                (1, 10, 1.0, 0.0),
                (2, 0, 0.0, 1.0),
                (2, 10, 1.0, 1.0),
            ],
        )
    return path


def test_build_replay_payload_samples_trajectory(tmp_path):
    payload = build_replay_payload(_trajectory(tmp_path / "trajectory.sqlite"), frame_stride=10, max_agents=10)

    assert payload["agent_count"] == 2
    assert payload["exported_agent_count"] == 2
    assert payload["frame_count"] == 2
    assert payload["frames"][0]["agents"] == [[1, 0.0, 0.0], [2, 0.0, 1.0]]


def test_build_replay_payload_interpolates_on_global_frames(tmp_path):
    path = tmp_path / "staggered.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("create table metadata(key text, value text)")
        connection.execute("insert into metadata values ('fps', '25')")
        connection.execute("create table trajectory_data(id integer, frame integer, pos_x real, pos_y real)")
        connection.executemany(
            "insert into trajectory_data values (?, ?, ?, ?)",
            [
                (1, 0, 0.0, 0.0),
                (1, 10, 10.0, 0.0),
                (2, 5, 0.0, 5.0),
                (2, 15, 10.0, 5.0),
            ],
        )

    payload = build_replay_payload(path, frame_stride=5, max_agents=10)
    frames = {frame["frame"]: frame["agents"] for frame in payload["frames"]}

    assert frames[0] == [[1, 0.0, 0.0]]
    assert frames[5] == [[1, 5.0, 0.0], [2, 0.0, 5.0]]
    assert frames[10] == [[1, 10.0, 0.0], [2, 5.0, 5.0]]
    assert frames[15] == [[2, 10.0, 5.0]]


def test_export_html_writes_threejs_replay(tmp_path):
    result = export_html(
        _trajectory(tmp_path / "trajectory.sqlite"),
        tmp_path / "replay.html",
        frame_stride=10,
        max_agents=10,
    )

    html = pathlib.Path(result["output"]).read_text(encoding="utf-8")
    assert "three.module.js" in html
    assert "OrbitControls" in html
    assert "trajectory.sqlite" in html
