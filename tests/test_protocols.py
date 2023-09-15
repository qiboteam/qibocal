"""Test routines' acquisition method using dummy platform"""
import pathlib
import tempfile

import pytest
import yaml
from qibolab import create_platform

from qibocal.cli.builders import ActionBuilder

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
@pytest.mark.parametrize("update", [True, False])
def test_builder(runcard, update):
    """Test possible update combinations between global and local."""
    builder = ActionBuilder(runcard, tempfile.mkdtemp(), force=True, update=update)
    builder.run()
    builder.dump_platform_runcard()
    builder.dump_report()
