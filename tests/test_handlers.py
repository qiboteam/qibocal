"""Testing handlers."""

import pathlib

import pytest
import yaml
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard
from qibocal.cli.report import ExecutionMode

RUNCARD = pathlib.Path(__file__).parent / "runcards/handlers.yml"
PLATFORM = create_platform("dummy")

LOW_CHI2_THRESHOLD = 1e-5
HIGH_CHI2_THRESHOLD = 1000


def test_handler_fail(tmp_path):
    """Test handler error with max iterations."""
    card = yaml.safe_load(RUNCARD.read_text(encoding="utf-8"))
    card["actions"][1]["validator"]["parameters"]["chi2_max_value"] = LOW_CHI2_THRESHOLD
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
