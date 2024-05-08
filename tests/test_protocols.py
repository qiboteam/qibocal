"""Test routines' acquisition method using dummy_couplers platform"""

import pathlib

import pytest
import yaml
from click.testing import CliRunner
from qibolab import create_platform

from qibocal.auto.task import PLATFORM_DIR
from qibocal.cli import utils
from qibocal.cli._base import command
from qibocal.protocols.characterization.flux_dependence.resonator_flux_dependence import (
    ResonatorFluxParameters,
)
from qibocal.protocols.characterization.rabi.amplitude import RabiAmplitudeData
from qibocal.protocols.characterization.rabi.ef import RabiAmplitudeEFData
from qibocal.protocols.characterization.rabi.length import RabiLengthData
from qibocal.protocols.characterization.rabi.utils import (
    extract_rabi,
    rabi_amplitude_function,
    rabi_length_function,
)

SINGLE_ACTION_RUNCARD = "action.yml"
PLATFORM = create_platform("dummy")
PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/"
RUNCARDS_NAMES = ["protocols.yml", "rb_noise_protocols.yml", "protocols_couplers.yml"]


def generate_runcard_single_protocol():
    for runcard_name in RUNCARDS_NAMES:
        complete_path = PATH_TO_RUNCARD / runcard_name
        actions = yaml.safe_load(complete_path.read_text(encoding="utf-8"))
        if isinstance(actions["platform"], str):
            actions["platform"] = [actions["platform"]]
        for platform in actions["platform"]:
            if "backend" not in actions:
                backend = "qibolab"
            else:
                backend = actions["backend"]
            for action in actions["actions"]:
                card = {
                    "actions": [action],
                    "targets": list(PLATFORM.qubits),
                    "backend": backend,
                }
                if "platform" in actions:
                    card["platform"] = platform
                yield (card, runcard_name)


def idfn(val):
    """Helper function to indentify the protocols when testing."""
    return val[0]["platform"] + "-" + val[1] + "-" + val[0]["actions"][0]["id"]


@pytest.mark.parametrize("update", ["--update", "--no-update"])
@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_auto_command(runcard, update, tmp_path):
    """Test auto command pipeline."""
    runcard = runcard[0]
    protocol = runcard["actions"][0]["id"]

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
            update,
        ],
    )
    assert not results.exception
    assert results.exit_code == 0
    if update == "--update" and runcard["backend"] == "qibolab":
        assert (tmp_path / utils.UPDATED_PLATFORM).is_dir()
        assert (tmp_path / "data" / f"{protocol}_0" / PLATFORM_DIR).is_dir()


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_acquire_command(runcard, tmp_path):
    """Test acquire command pipeline and report generated."""
    runcard = runcard[0]
    protocol = runcard["actions"][0]["id"]

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
    assert (tmp_path / "data" / f"{protocol}_0").is_dir()

    # generate report from acquired data
    results_report = runner.invoke(command, ["report", str(tmp_path)])
    assert not results_report.exception
    assert results_report.exit_code == 0
    assert (tmp_path / "index.html").is_file()


@pytest.mark.parametrize("update", ["--update", "--no-update"])
@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_fit_command(runcard, update, tmp_path):
    """Test fit builder and report generated."""

    runcard = runcard[0]
    protocol = runcard["actions"][0]["id"]

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
    results_fit = runner.invoke(command, ["fit", str(tmp_path), update])

    assert not results_fit.exception
    assert results_fit.exit_code == 0

    if update == "--update" and runcard["backend"] == "qibolab":
        assert (tmp_path / utils.UPDATED_PLATFORM).is_dir()
        assert (tmp_path / "data" / f"{protocol}_0" / PLATFORM_DIR).is_dir()

    # generate report with fit and plot
    results_plot = runner.invoke(command, ["report", str(tmp_path)])
    assert not results_plot.exception
    assert results_plot.exit_code == 0
    assert (tmp_path / "index.html").is_file()


def test_extract_rabi():
    assert extract_rabi(RabiAmplitudeData()) == (
        "amp",
        "Amplitude [dimensionless]",
        rabi_amplitude_function,
    )
    assert extract_rabi(RabiLengthData()) == (
        "length",
        "Time [ns]",
        rabi_length_function,
    )
    with pytest.raises(RuntimeError):
        extract_rabi(RabiAmplitudeEFData)


def test_resonator_flux_bias():
    freq_width = 100_000
    freq_step = 10_000
    bias_width = 1
    bias_step = 0.2
    flux_start = -0.5
    flux_end = 0.4
    flux_step = 0.1
    ResonatorFluxParameters(freq_width, freq_step, bias_width, bias_step)
    ResonatorFluxParameters(freq_width, freq_step, flux_start, flux_end, flux_step)
    with pytest.raises(ValueError):
        ResonatorFluxParameters(
            freq_width,
            freq_step,
            bias_width,
            bias_step,
            flux_start,
            flux_end,
            flux_step,
        )


# TODO: compare report by calling qq report
