import os
import pathlib
import shutil

import pytest
from click.testing import CliRunner

from qibocal import create_calibration_platform
from qibocal.cli._base import command

test_runcards_dir = pathlib.Path(__file__).parent / "runcards"
DUMMY_ACTION = test_runcards_dir / "dummy_action.yml"
DUMMY_COMPARE = test_runcards_dir / "dummy_compare.yml"


@pytest.mark.parametrize("update", ["--update", "--no-update"])
def test_qq_update(update, tmp_path, monkeypatch, platform):
    """Testing qq update using mock."""
    output_folder = tmp_path / "out"
    runner = CliRunner()
    runner.invoke(
        command,
        ["run", str(DUMMY_ACTION), "-o", str(output_folder), "-f", update],
        catch_exceptions=False,
    )

    old_platform_path = pathlib.Path(os.getenv("QIBOLAB_PLATFORMS")) / "mock"
    platforms = tmp_path / "platforms"
    (tmp_path / "platforms").mkdir(parents=True, exist_ok=True)
    shutil.copytree(old_platform_path, platforms / "mock")
    monkeypatch.setenv("QIBOLAB_PLATFORMS", str(platforms))
    runner = CliRunner()
    if not update:
        while pytest.raises(FileNotFoundError):
            runner.invoke(
                command, ["update", str(output_folder)], catch_exceptions=False
            )
    else:
        runner.invoke(command, ["update", str(output_folder)], catch_exceptions=False)
        new_parameters = (
            pathlib.Path(os.getenv("QIBOLAB_PLATFORMS")) / "mock" / "parameters.json"
        )
        assert new_parameters.exists()


@pytest.mark.parametrize("skip_qubits", [None, "0"])
def test_skip_qubits_option(skip_qubits, tmp_path, monkeypatch, platform):
    """Testing skip_qubits options using mock."""
    output_folder = tmp_path / "out"
    runner = CliRunner()
    runner.invoke(
        command,
        ["run", str(DUMMY_ACTION), "-o", str(output_folder), "-f"],
        catch_exceptions=False,
    )
    old_platform_path = pathlib.Path(os.getenv("QIBOLAB_PLATFORMS")) / "mock"
    old_platform = create_calibration_platform("mock")
    platforms = tmp_path / "platforms"
    (tmp_path / "platforms").mkdir(parents=True, exist_ok=True)
    shutil.copytree(old_platform_path, platforms / "mock")
    monkeypatch.setenv("QIBOLAB_PLATFORMS", str(platforms))
    runner = CliRunner()
    if skip_qubits is None:
        runner.invoke(command, ["update", str(output_folder)], catch_exceptions=False)
        new_platform = create_calibration_platform("mock")
        for i in platform.qubits:
            assert old_platform.config(
                old_platform.qubits[i].acquisition
            ) != new_platform.config(new_platform.qubits[i].acquisition)

    else:
        runner.invoke(
            command,
            ["update", str(output_folder), "--skip-qubits", skip_qubits],
            catch_exceptions=False,
        )
        new_platform = create_calibration_platform("mock")
        for i in new_platform.qubits:
            if i in list(skip_qubits):
                assert old_platform.config(
                    old_platform.qubits[i].acquisition
                ) == new_platform.config(new_platform.qubits[i].acquisition)


def test_fit_command(tmp_path, platform):
    """Test qq fit behavior."""

    tmp_dir_1 = tmp_path / "temp_dir_1"
    tmp_dir_2 = tmp_path / "temp_dir_2"
    runner = CliRunner()
    runner.invoke(command, ["acquire", str(DUMMY_ACTION), "-o", str(tmp_dir_1), "-f"])

    # fit after acquisition same folder
    runner.invoke(command, ["fit", str(tmp_dir_1)], catch_exceptions=False)
    # raise error if fit on same folder without force
    with pytest.raises(RuntimeError):
        runner.invoke(command, ["fit", str(tmp_dir_1)], catch_exceptions=False)
    # do not raise error with force option
    runner.invoke(command, ["fit", str(tmp_dir_1), "-f"], catch_exceptions=False)

    # fit on separate folder
    runner.invoke(
        command, ["fit", str(tmp_dir_1), "-o", str(tmp_dir_2)], catch_exceptions=False
    )

    # raise error if separate folder exists
    with pytest.raises(RuntimeError):
        runner.invoke(
            command,
            ["fit", str(tmp_dir_1), "-o", str(tmp_dir_2)],
            catch_exceptions=False,
        )

    # do not raise error with force option
    runner.invoke(
        command,
        ["fit", str(tmp_dir_1), "-o", str(tmp_dir_2), "-f"],
        catch_exceptions=False,
    )
    # fit after acquisition different folder


def test_compare_report_dates(tmp_path, platform):
    report_dir_1 = tmp_path / "report_dir_1"
    report_dir_2 = tmp_path / "report_dir_2"
    compare_dir = tmp_path / "compare_dir"

    runner = CliRunner()
    runner.invoke(command, ["run", str(DUMMY_COMPARE), "-o", str(report_dir_1), "-f"])
    runner.invoke(command, ["run", str(DUMMY_COMPARE), "-o", str(report_dir_2), "-f"])

    runner.invoke(
        command,
        ["compare", str(report_dir_1), str(report_dir_2), "-o", str(compare_dir), "-f"],
    )
