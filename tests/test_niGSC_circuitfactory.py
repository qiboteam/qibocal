from collections.abc import Iterable

import pytest
from qibo import gates, models

from qibocal.calibrations.niGSC.basics.circuitfactory import *
from qibocal.calibrations.niGSC.basics.utils import ONEQ_GATES
from qibocal.calibrations.niGSC.standardrb import (
    ModuleFactory as SingleCliffordsInvFactory,
)
from qibocal.calibrations.niGSC.XIdrb import ModuleFactory as XIdFactory


@pytest.fixture
def factories_singlequbitgates():
    thelist = [
        SingleCliffordsFactory,
        Qibo1qGatesFactory,
        SingleCliffordsInvFactory,
        XIdFactory,
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


def test_abstract_factory():
    cfactory = CircuitFactory(1, [1, 2] * 3, qubits=[0])
    with pytest.raises(NotImplementedError):
        list(cfactory)
    cfactory = CircuitFactory(1, 3, qubits=[0])
    assert cfactory.depths == [3]


@pytest.mark.parametrize("nqubits", [1, 2, 5])
@pytest.mark.parametrize("runs", [1, 4])
@pytest.mark.parametrize("qubits", [[0], [0, 2]])
def test_general_singlequbitgates_factories(
    factories_singlequbitgates: list,
    nqubits: int,
    qubits: list,
    depths: list,
    runs: int,
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
            # XId factory is only defined for 1 qubit.
            if max(qubits) > 0 and factory_init == XIdFactory:
                with pytest.raises(ValueError):
                    factory = factory_init(nqubits, list(depths) * runs, qubits=qubits)
            else:
                factory = factory_init(nqubits, list(depths) * runs, qubits=qubits)
                abstract_factorytest(factory)
                # if factory.name not in ('XId', 'SingleCliffordsInv'):
                if "inv" in factory.name or "Inv" in factory.name:
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
                        assert (
                            circuit.ngates
                            == depths[count % len(depths)] * len(qubits)
                            + additional_gates
                        )
                        assert (
                            circuit.depth
                            == depths[count % len(depths)] + additional_gates
                        )
                    # Check the factories individual trades.
                    if factory.name in ("Qibo1qGates"):
                        for gate in circuit.queue[:-1]:
                            assert gate.__class__.__name__ in ONEQ_GATES
                    elif factory.name in ("SingleCliffords"):
                        for gate in circuit.queue[:-1]:
                            assert isinstance(gate, gates.Unitary)
                    elif factory.name in ("SingleCliffordsInv"):
                        for gate in circuit.queue[:-1]:
                            assert isinstance(gate, gates.Unitary)
                    elif factory.name in ("XId"):
                        for gate in circuit.queue[:-1]:
                            assert isinstance(gate, (gates.X, gates.I))
                    else:
                        raise_error(
                            ValueError,
                            "Unknown circuitfactory :{}".format(factory.name),
                        )
