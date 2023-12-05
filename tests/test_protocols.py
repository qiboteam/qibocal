"""Test routines' acquisition method using dummy platform"""
import pathlib

import pytest
import yaml
from click.testing import CliRunner
from qibolab import create_platform

from qibocal.cli._base import command
from qibocal.protocols.characterization.rabi.amplitude import RabiAmplitudeData
from qibocal.protocols.characterization.rabi.ef import RabiAmplitudeEFData
from qibocal.protocols.characterization.rabi.length import RabiLengthData
from qibocal.protocols.characterization.rabi.utils import (
    extract_rabi,
    rabi_amplitude_function,
    rabi_length_function,
)

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/protocols.yml"
PLATFORM = create_platform("dummy")
SINGLE_ACTION_RUNCARD = "action.yml"


def generate_runcard_single_protocol():
    actions = yaml.safe_load(PATH_TO_RUNCARD.read_text(encoding="utf-8"))
    with open(PATH_TO_RUNCARD) as file:
        actions = yaml.safe_load(file)
    for action in actions["actions"]:
        card = {"actions": [action], "qubits": list(PLATFORM.qubits)}
        yield card


def idfn(val):
    """Helper function to indentify the protocols when testing."""
    return val["actions"][0]["id"]


@pytest.mark.parametrize("platform", ["dummy"])
@pytest.mark.parametrize("backend", ["qibolab"])
@pytest.mark.parametrize("update", ["--update", "--no-update"])
@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_auto_command(runcard, update, platform, backend, tmp_path):
    """Test auto command pipeline."""

    (tmp_path / SINGLE_ACTION_RUNCARD).write_text(yaml.safe_dump(runcard))
    runner = CliRunner()
    results = runner.invoke(
        command,
        [
            "auto",
            str(tmp_path / SINGLE_ACTION_RUNCARD),
            "-o",
            f"{str(tmp_path)}",
            "-f",
            "--backend",
            backend,
            "--platform",
            platform,
            update,
        ],
    )
    assert not results.exception
    assert results.exit_code == 0


@pytest.mark.parametrize("platform", ["dummy"])
@pytest.mark.parametrize("backend", ["qibolab"])
@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_acquire_command(runcard, backend, platform, tmp_path):
    """Test acquire command pipeline and report generated."""
    (tmp_path / SINGLE_ACTION_RUNCARD).write_text(yaml.safe_dump(runcard))
    runner = CliRunner()

    # test acquisition
    results = runner.invoke(
        command,
        [
            "acquire",
            str(tmp_path / SINGLE_ACTION_RUNCARD),
            "-o",
            f"{str(tmp_path)}",
            "-f",
            "--backend",
            backend,
            "--platform",
            platform,
        ],
    )
    assert not results.exception
    assert results.exit_code == 0

    # generate report from acquired data
    results_report = runner.invoke(command, ["report", str(tmp_path)])
    assert not results_report.exception
    assert results_report.exit_code == 0


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_fit_command(runcard, tmp_path):
    """Test fit builder and report generated."""
    (tmp_path / SINGLE_ACTION_RUNCARD).write_text(yaml.safe_dump(runcard))
    runner = CliRunner()

    # test acquisition
    results = runner.invoke(
        command,
        [
            "acquire",
            str(tmp_path / SINGLE_ACTION_RUNCARD),
            "-o",
            f"{str(tmp_path)}",
            "-f",
        ],
    )
    assert not results.exception
    assert results.exit_code == 0

    # perform fit
    results_fit = runner.invoke(command, ["fit", str(tmp_path), "--no-update"])

    assert not results_fit.exception
    assert results_fit.exit_code == 0

    # generate report with fit and plot
    results_plot = runner.invoke(command, ["report", str(tmp_path)])
    assert not results_plot.exception
    assert results_plot.exit_code == 0


def test_extract_rabi():
    assert extract_rabi(RabiAmplitudeData()) == (
        "amp",
        "Amplitude (dimensionless)",
        rabi_amplitude_function,
    )
    assert extract_rabi(RabiLengthData()) == (
        "length",
        "Time (ns)",
        rabi_length_function,
    )
    with pytest.raises(RuntimeError):
        extract_rabi(RabiAmplitudeEFData)


# TODO: compare report by calling qq report
