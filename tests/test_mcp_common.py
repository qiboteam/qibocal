import asyncio
from types import SimpleNamespace

from plotly.subplots import make_subplots

from calibration_mcp_servers import automatic_calibration_server as calibration
from calibration_mcp_servers._common import _export_png


def test_export_png_with_matplotlib(tmp_path):
    figure = make_subplots(rows=1, cols=2)
    figure.add_scatter(
        x=[1, 2],
        y=[2, 3],
        mode="markers",
        marker={"color": "blue", "size": 5},
        name="data",
        row=1,
        col=1,
    )
    figure.add_scatter(
        x=[1, 2],
        y=[3, 4],
        mode="lines",
        line={"color": "black", "dash": "dash"},
        name="fit",
        row=1,
        col=2,
    )
    output = tmp_path / "figure.png"

    _export_png(figure, output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_checkpoint_dumps_live_platform(tmp_path, monkeypatch):
    output_dumps = []

    class FakeOutput:
        def __init__(self, history, meta):
            pass

        def dump(self, path):
            output_dumps.append(path)

        @staticmethod
        def update_platform(platform, path):
            updated_platform = path / "new_platform"
            updated_platform.mkdir()
            platform.dump(updated_platform)

    platform = SimpleNamespace(
        dump=lambda path: (path / "state.txt").write_text("updated")
    )
    executor = SimpleNamespace(
        history=SimpleNamespace(_order=[]),
        meta=object(),
        platform=platform,
    )
    session = calibration.CalibrationSession(executor, "mock", tmp_path)
    monkeypatch.setattr(calibration, "Output", FakeOutput)

    calibration._checkpoint(session)

    assert output_dumps == [tmp_path]
    assert (tmp_path / "new_platform" / "state.txt").read_text() == "updated"


def test_run_protocol_reuses_executor_and_forwards_update(tmp_path, monkeypatch):
    calls = []

    class Results:
        def __contains__(self, target):
            return target == 0

    completed = SimpleNamespace(
        task=SimpleNamespace(targets=[0], id="rabi"),
        results=Results(),
    )

    class FakeExecutor:
        def __init__(self):
            self.protocols = {"rabi_amplitude": object()}
            self.targets = [0]

        def run_protocol(self, **kwargs):
            calls.append(kwargs)
            return completed

    executor = FakeExecutor()
    calibration._active_session = calibration.CalibrationSession(
        executor, "mock", tmp_path
    )
    monkeypatch.setattr(calibration, "_checkpoint", lambda session: None)

    try:
        response = asyncio.run(
            calibration.run_protocol("rabi_amplitude", {"min_amp": 0.0}, update=False)
        )
    finally:
        calibration._active_session = None

    assert calls[0]["parameters"].update is False
    assert calls[0]["output"] == tmp_path
    assert response["successful_targets"] == ["0"]
    assert response["platform_updated_targets"] == []


def test_finish_calibration_publishes_new_platform_once(tmp_path, monkeypatch):
    published = []
    executor = SimpleNamespace(
        platform=SimpleNamespace(disconnect=lambda: None),
        meta=SimpleNamespace(end=lambda: None),
    )
    calibration._active_session = calibration.CalibrationSession(
        executor, "mock", tmp_path
    )
    monkeypatch.setattr(calibration, "_checkpoint", lambda session: None)
    monkeypatch.setattr(
        calibration,
        "publish_platform",
        lambda path, skip_qubits: published.append((path, skip_qubits)),
    )

    response = asyncio.run(calibration.finish_calibration(publish=True))

    assert published == [(tmp_path, None)]
    assert response["published"] is True
    assert calibration._active_session is None
