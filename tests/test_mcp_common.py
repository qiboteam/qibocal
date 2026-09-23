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


def test_run_protocol_selects_targets_per_step(tmp_path, monkeypatch):
    runcard_targets = []
    run_paths = []

    def make_runcard(experiments, targets, platform, update):
        runcard_targets.append(targets)
        return object()

    def write_runcard(runcard, output_dir):
        output_dir.mkdir(parents=True)
        path = output_dir / "runcard.yml"
        path.touch()
        return path

    async def run_qq(runcard_path, output_path, update, partition):
        run_paths.append(output_path)
        return ""

    calibration._active_session = calibration.CalibrationSession(
        platform_name="mock",
        targets=[0, 1, 2],
        path=tmp_path,
        partition=None,
    )
    monkeypatch.setattr(calibration, "make_runcard", make_runcard)
    monkeypatch.setattr(calibration, "write_runcard", write_runcard)
    monkeypatch.setattr(calibration, "run_qq", run_qq)
    monkeypatch.setattr(
        calibration,
        "report_content",
        lambda path: {"output_folder": str(path.resolve())},
    )

    try:
        single = asyncio.run(
            calibration.run_protocol(
                "rabi_amplitude", {}, targets=[1], update=False, step_id="step-n"
            )
        )
        all_targets = asyncio.run(
            calibration.run_protocol(
                "rabi_amplitude",
                {},
                targets=[0, 1, 2],
                update=False,
                step_id="step-n-plus-1",
            )
        )
        individual = asyncio.run(
            calibration.run_protocol(
                "rabi_amplitude",
                {},
                targets=[0, 2],
                update=False,
                step_id="step-n-plus-2",
                execution_mode="individual",
            )
        )
    finally:
        calibration._active_session = None

    assert runcard_targets == [[1], [0, 1, 2], [0], [2]]
    assert len(run_paths) == 4
    assert single["targets"] == [1]
    assert single["output_folder"].endswith("step-n-rabi_amplitude-qubits-1")
    assert all_targets["targets"] == [0, 1, 2]
    assert individual["execution_mode"] == "individual"
    assert [run["targets"] for run in individual["runs"]] == [[0], [2]]
    assert individual["output_folders"][0].endswith(
        "step-n-plus-2-rabi_amplitude-qubit-0"
    )
    assert individual["output_folders"][1].endswith(
        "step-n-plus-2-rabi_amplitude-qubit-2"
    )
