import numpy as np
from collections.abc import Iterable
from qibo import models, gates
from qibocal.calibrations.niGSC.basics.circuitfactory import *
from qibocal.calibrations.niGSC.standardrb import moduleFactory as SingleCliffordsInvFactory
import pytest
from copy import deepcopy
from qibocal.calibrations.niGSC.basics.utils import ONEQ_GATES

@pytest.fixture
def factories_singlequbitgates():
    thelist = [
        SingleCliffordsFactory,
        Qibo1qGatesFactory,
        SingleCliffordsInvFactory
    ]
    return thelist

@pytest.fixture
def depths():
    return [0, 1, 5, 10, 30]


def abstract_factorytest(gfactory):
    # The factory is an iterable.
    assert isinstance(gfactory, Iterable)
    # The objects it produces are of the type ``models.Circuit``.
    for circuit in gfactory:
        assert isinstance(circuit, models.Circuit)
    
def general_circuittest(gfactory: Circuitfactory):
    """Check if the circuits produced by the given factory are
    kind of random.

    Args:
        gfactory (Circuitfactory): Produces circuits which are checked.
    """
    factory1 = deepcopy(gfactory)
    factory2 = deepcopy(gfactory)
    same_count = 0
    count = 0
    for circuit1, circuit2 in zip(factory1, factory2):
        same_circuit = True
        for gate1, gate2 in zip(circuit1.queue[:-1], circuit2.queue[:-1]):
            same_circuit *= np.array_equal(gate1.matrix, gate2.matrix)
        same_count += same_circuit
        count += 1
    # Half of the runs should not give the same 
    assert same_count <= count * 0.5
        


def test_Circuitfactory():
    pass

@pytest.mark.parametrize("nqubits", [1,2,5])
@pytest.mark.parametrize("runs", [1, 4])
@pytest.mark.parametrize('qubits', [[0], [0,2]])
def test_general_singlequbitgates_factories(
    factories_singlequbitgates: list, nqubits: int, qubits:list, depths: list, runs: int
    ) -> None:
    """Check for how random circuits are produced and if the lengths, shape
    and randomness works.

    Args:
        qubits (list): List of qubits
        depths (list): list of depths for circuits
        runs (int): How many randomly drawn cirucit for one depth value

    """
    if max(qubits) >= nqubits:
        pass
    else:
        for factory_init in factories_singlequbitgates:
            factory = factory_init(nqubits, list(depths) * runs, qubits = qubits)
            abstract_factorytest(factory)
            # if factory.name not in ('XId', 'SingleCliffordsInv'):
            general_circuittest(factory)
            if 'inv' in factory.name or 'Inv' in factory.name:
                # When checking the depth of circuits, the measurement gate and inverse gate 
                # has to be taken into account
                additional_gates = 2
            else:
                # When checking the depth of circuits, measurement gate has to be taken into account
                additional_gates = 1
            for count, circuit in enumerate(factory):
                if circuit.ngates == 1:
                    assert isinstance(circuit.queue[0], gates.measurements.M)
                else:
                    assert circuit.ngates == depths[count % len(depths)] * len(qubits) + additional_gates
                    assert circuit.depth == depths[count % len(depths)] + additional_gates
                # Check the factories individual trades.
                if factory.name in ('Qibo1qGates'):
                    for gate in circuit.queue[:-1]:
                        assert gate.__class__.__name__ in ONEQ_GATES
                elif factory.name in ('SingleCliffords'):
                    for gate in circuit.queue[:-1]:
                        assert isinstance(gate, gates.Unitary)
                elif factory.name in ('SingleCliffordsInv'):
                    for gate in circuit.queue[:-1]:
                        assert isinstance(gate, gates.Unitary)
                elif factory.name in ('XId'):
                    for gate in circuit.queue[:-1]:
                        assert isinstance(gate, gates.X) or isinstance(gate, gates.I)
                else:
                    raise_error(ValueError, 'Unknown circuitfactory :{}'.format(factory.name))
