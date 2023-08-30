"""Test routines' acquisition method using dummy platform"""
import pathlib
import tempfile

import pytest
import yaml
from qibolab import create_platform

from qibocal.cli.acquisition import acquire
from qibocal.cli.autocalibration import autocalibrate
from qibocal.cli.fit import fit

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


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_action_builder(runcard):
    """Test ActionBuilder for all protocols."""
    autocalibrate(runcard, tempfile.mkdtemp(), force=True, update=False)


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_acquisition_builder(runcard):
    """Test AcquisitionBuilder for all protocols."""
    acquire(runcard, tempfile.mkdtemp(), force=True)


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_fit_builder(runcard):
    """Test FitBuilder."""
    output_folder = tempfile.mkdtemp()
    acquire(runcard, output_folder, force=True)
    fit(output_folder, update=False)


# TODO: compare report by calling qq report
