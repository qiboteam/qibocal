"""Testing validators."""

import pytest
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard
from qibocal.cli.report import ExecutionMode

PLATFORM = create_platform("dummy")
QUBITS = list(PLATFORM.qubits)
DUMMY_CARD = {
    "qubits": QUBITS,
    "actions": [
        {
            "id": "t1",
            "priority": 0,
            "main": "single shot",
            "operation": "t1",
            "validator": {
                "scheme": "chi2",
                "parameters": {
                    "chi2_max_value": 0,
                },
            },
            "parameters": {
                "delay_before_readout_start": 0,
                "delay_before_readout_end": 20_000,
                "delay_before_readout_step": 2000,
                "nshots": 10,
            },
        },
        {
            "id": "single shot",
            "priority": 0,
            "operation": "single_shot_classification",
            "parameters": {
                "nshots": 10,
            },
        },
    ],
}

# TODO: generalize to all protocols that support chi2.


@pytest.mark.parametrize("chi2_max_value", [1000, 1e-5])
def test_chi2(chi2_max_value, tmp_path):
    """Dummy test only for t1"""
    DUMMY_CARD["actions"][0]["validator"]["parameters"][
        "chi2_max_value"
    ] = chi2_max_value
    executor = Executor.load(
        Runcard.load(DUMMY_CARD),
        tmp_path,
        PLATFORM,
        PLATFORM.qubits,
    )

    list(executor.run(mode=ExecutionMode.autocalibration))

    # for large chi2 value executor will execute 2 protocols
    # only 1 with low chi2
    assert len(executor.history.keys()) == 2 if chi2_max_value == 100 else 1
