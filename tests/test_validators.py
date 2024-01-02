"""Testing validators."""

import pathlib

import pytest
import yaml
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard
from qibocal.cli.report import ExecutionMode

RUNCARD_CHI_SQUARED = pathlib.Path(__file__).parent / "runcards/chi_squared.yml"
RUNCARD_EXCEPTION = pathlib.Path(__file__).parent / "runcards/handlers.yml"
PLATFORM = create_platform("dummy")


@pytest.mark.parametrize("chi2_max_value", [1000, 1e-5])
def test_chi2(chi2_max_value, tmp_path):
    """Dummy test only for t1"""
    card = yaml.safe_load(RUNCARD_CHI_SQUARED.read_text(encoding="utf-8"))
    card["actions"][0]["validator"]["parameters"]["thresholds"] = [chi2_max_value]
    executor = Executor.load(
        Runcard.load(card),
        tmp_path,
        PLATFORM,
        list(PLATFORM.qubits),
    )

    if chi2_max_value == 1e-5:
        with pytest.raises(ValueError):
            list(executor.run(mode=ExecutionMode.autocalibration))
    else:
        list(executor.run(mode=ExecutionMode.autocalibration))

    # for large chi2 value executor will execute 2 protocols
    # only 1 with low chi2
    assert len(executor.history.keys()) == 2 if chi2_max_value == 100 else 1


def test_validator_with_exception_handled(tmp_path):
    """Dummy test raising a MAX ITERATIONS error."""
    card = yaml.safe_load(RUNCARD_EXCEPTION.read_text(encoding="utf-8"))
    executor = Executor.load(
        Runcard.load(card),
        tmp_path,
        PLATFORM,
        list(PLATFORM.qubits),
    )

    with pytest.raises(ValueError):
        list(executor.run(mode=ExecutionMode.autocalibration))

    first_id = card["actions"][0]["id"]
    second_id = card["actions"][1]["id"]
    expected_history = []
    for i in range(card["max_iterations"] + 1):
        expected_history.append((first_id, i))
        expected_history.append((second_id, i))

    assert expected_history == list(executor.history)
