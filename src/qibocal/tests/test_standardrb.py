import numpy as np
from qibo import models

from qibocal.calibrations.protocols.standardrb import SingleCliffordsInvFactory


def test_factory():
    qubits = [0]
    sequences, runs = [1, 3, 4], 2

    myfactory = SingleCliffordsInvFactory(qubits, sequences, runs)
    circuits_list = list(myfactory)
    assert len(circuits_list) == len(sequences) * runs
    for count, circuit in enumerate(myfactory):
        assert isinstance(circuit, models.Circuit)
        assert np.array_equal(circuit.unitary(), np.eye(2 ** len(qubits)))
        # There will be an inverse gate.
        assert len(circuit.queue) == sequences[count % len(sequences)] + 1


# import pdb
# pdb.set_trace()
