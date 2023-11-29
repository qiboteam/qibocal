"""Testing validators."""

import pathlib

import pytest
import yaml
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard
from qibocal.cli.report import ExecutionMode

RUNCARD = pathlib.Path(__file__).parent / "runcards/chi_squared.yml"
PLATFORM = create_platform("dummy")


@pytest.mark.parametrize("chi2_max_value", [1000, 1e-5])
def test_chi2(chi2_max_value, tmp_path):
    """Dummy test only for t1"""
    card = yaml.safe_load(RUNCARD.read_text(encoding="utf-8"))
    card["actions"][0]["validator"]["parameters"]["chi2_max_value"] = chi2_max_value
    executor = Executor.load(
        Runcard.load(card),
        tmp_path,
        PLATFORM,
        list(PLATFORM.qubits),
    )

    list(executor.run(mode=ExecutionMode.autocalibration))

    # for large chi2 value executor will execute 2 protocols
    # only 1 with low chi2
    assert len(executor.history.keys()) == 2 if chi2_max_value == 100 else 1
