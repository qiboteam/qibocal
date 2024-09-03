from qibocal.auto.runcard import Runcard

EXAMPLE = {
    "targets": [0, 1],
    "actions": [
        {
            "id": "readout characterization",
            "operation": "readout_characterization",
            "parameters": {
                "nshots": 5000,
                "delay": 1000,
            },
        }
    ],
}


def test_load():
    ex = Runcard.load(EXAMPLE)

    assert ex.targets == [0, 1]
    assert len(ex.actions) == 1
