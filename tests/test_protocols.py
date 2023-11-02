"""Test routines' acquisition method using dummy platform"""
import pathlib

import pytest
import yaml
from qibolab import create_platform

from qibocal.auto.runcard import Runcard
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
        card = {"actions": [action], "qubits": list(PLATFORM.qubits)}
        yield Runcard.load(card)


def idfn(val):
    """Helper function to indentify the protocols when testing."""
    return val.actions[0].id


@pytest.mark.parametrize("update", [True, False])
@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_action_builder(runcard, update, tmp_path):
    """Test ActionBuilder for all protocols."""
    autocalibrate(
        runcard,
        tmp_path,
        force=True,
        update=update,
    )
    report(tmp_path)


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_acquisition_builder(runcard, tmp_path):
    """Test AcquisitionBuilder for all protocols."""
    acquire(runcard, tmp_path, force=True)
    report(tmp_path)


@pytest.mark.parametrize("runcard", generate_runcard_single_protocol(), ids=idfn)
def test_fit_builder(runcard, tmp_path):
    """Test FitBuilder."""
    acquire(runcard, tmp_path, force=True)
    fit(tmp_path, update=False)
    report(tmp_path)


# TODO: compare report by calling qq report
