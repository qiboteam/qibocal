"""Test routines' acquisition method using dummy_couplers platform."""

import pathlib

import pytest
import yaml
from click.testing import CliRunner
from qibolab import create_platform

from qibocal.auto.output import PLATFORM as PLATFORM_DIR
from qibocal.cli._base import command
from qibocal.protocols.rabi.amplitude import RabiAmplitudeData
from qibocal.protocols.rabi.ef import RabiAmplitudeEFData
from qibocal.protocols.rabi.length import RabiLengthData
from qibocal.protocols.rabi.utils import (
    extract_rabi,
    rabi_amplitude_function,
    rabi_length_function,
)

SINGLE_ACTION_RUNCARD = "action.yml"
PLATFORM = create_platform("dummy")
PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/"
RUNCARDS_NAMES = ["protocols.yml", "rb_noise_protocols.yml", "protocols_couplers.yml"]

INVOKER_OPTIONS = dict(catch_exceptions=False)
"""Generate errors when calling qq."""


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
    runner.invoke(
        command,
        [
            "auto",
            str(tmp_path / SINGLE_ACTION_RUNCARD),
            "-o",
            f"{str(tmp_path)}",
            "-f",
            update,
        ],
        **INVOKER_OPTIONS,
    )
    if runcard["backend"] == "qibolab" and runcard["platform"] is not None:
        assert (tmp_path / utils.UPDATED_PLATFORM).is_dir()
        assert (tmp_path / "data" / f"{protocol}" / PLATFORM_DIR).is_dir()


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_acquire_command(runcard, tmp_path):
    """Test acquire command pipeline and report generated."""
    runcard = runcard[0]
    protocol = runcard["actions"][0]["id"]

    (tmp_path / SINGLE_ACTION_RUNCARD).write_text(yaml.safe_dump(runcard))
    runner = CliRunner()

    # test acquisition
    runner.invoke(
        command,
        [
            "acquire",
            str(tmp_path / SINGLE_ACTION_RUNCARD),
            "-o",
            f"{str(tmp_path)}",
            "-f",
        ],
        **INVOKER_OPTIONS,
    )

    assert (tmp_path / "data" / f"{protocol}").is_dir()

    # generate report from acquired data
    runner.invoke(command, ["report", str(tmp_path)], **INVOKER_OPTIONS)
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
    runner.invoke(
        command,
        [
            "acquire",
            str(tmp_path / SINGLE_ACTION_RUNCARD),
            "-o",
            f"{str(tmp_path)}",
            "-f",
        ],
        **INVOKER_OPTIONS,
    )

    # perform fit
    runner.invoke(command, ["fit", str(tmp_path), update], **INVOKER_OPTIONS)

    if runcard["backend"] == "qibolab" and runcard["platform"] is not None:
        assert (tmp_path / utils.UPDATED_PLATFORM).is_dir()
        assert (tmp_path / "data" / f"{protocol}" / PLATFORM_DIR).is_dir()

    # generate report with fit and plot
    runner.invoke(command, ["report", str(tmp_path)], **INVOKER_OPTIONS)
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


# TODO: compare report by calling qq report
