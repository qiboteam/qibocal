"""Testing validators."""

import pathlib

import pytest
import yaml
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.operation import Results
from qibocal.auto.runcard import Runcard
from qibocal.auto.validation import Validator
from qibocal.auto.validators.chi2 import Chi2Validator
from qibocal.cli.report import ExecutionMode

RUNCARD_CHI_SQUARED = pathlib.Path(__file__).parent / "runcards/chi_squared.yml"
RUNCARD_EXCEPTION = pathlib.Path(__file__).parent / "runcards/handlers.yml"
PLATFORM = create_platform("dummy")


def test_error_validator():
    """Test error with unknown validator."""

    with pytest.raises(KeyError):
        _ = Validator("unknown")


def test_chi2_error():
    """Test error when Results don't contain chi2."""
    assert (
        Chi2Validator(
            thresholds=[
                0,
            ]
        )(results=Results(), target=0)
        == None
    )


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

    list(executor.run(mode=ExecutionMode.autocalibration))

    assert len(executor.history.keys()) == 2 if chi2_max_value == 1000 else 1


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
