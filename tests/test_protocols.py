"""Test routines' acquisition method using dummy platform"""
import pathlib
import tempfile

import pytest
import yaml
from qibolab import create_platform

from qibocal.cli.acquisition import acquire
from qibocal.cli.autocalibration import autocalibrate
from qibocal.cli.fit import fit
from qibocal.cli.report import report

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/protocols.yml"
PLATFORM = create_platform("dummy")


def generate_runcard_single_protocol():
    with open(PATH_TO_RUNCARD) as file:
        actions = yaml.safe_load(file)
    for action in actions["actions"]:
        yield {"actions": [action], "qubits": list(PLATFORM.qubits)}


def idfn(val):
    """Helper function to indentify the protocols when testing."""
    return val["actions"][0]["id"]


@pytest.mark.parametrize("update", [True, False])
@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_action_builder(runcard, update):
    """Test ActionBuilder for all protocols."""
    path = pathlib.Path(tempfile.mkdtemp())
    autocalibrate(runcard, path, force=True, update=update)
    report(path)


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_acquisition_builder(runcard):
    """Test AcquisitionBuilder for all protocols."""
    path = pathlib.Path(tempfile.mkdtemp())
    acquire(runcard, path, force=True)
    report(path)


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_fit_builder(runcard):
    """Test FitBuilder."""
    output_folder = pathlib.Path(tempfile.mkdtemp())
    acquire(runcard, output_folder, force=True)
    fit(output_folder, update=False)
    report(output_folder)


# TODO: compare report by calling qq report
