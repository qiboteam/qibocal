import pathlib

from qibolab import ConfigKinds
from qibolab._core.components import IqChannel
from qibolab._core.instruments.emulator.emulator import EmulatorController
from qibolab._core.instruments.emulator.hamiltonians import (
    DriveEmulatorConfig,
    HamiltonianConfig,
)
from qibolab._core.platform import Platform
from qibolab._core.qubits import Qubit, QubitPair

FOLDER = pathlib.Path(__file__).parent

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig])


def create() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    qubits = {}
    pairs = {}
    channels = {}

    for q in range(2):
        qubits[q] = qubit = Qubit.default(q, drive_qudits={(1, 2): f"{q}/drive12"})
        channels |= {
            qubit.drive: IqChannel(mixer=None, lo=None),
            qubits[q].drive_qudits[1, 2]: IqChannel(mixer=None, lo=None),
        }
    pairs[(0, 1)] = pair = QubitPair(drive="01/drive")
    channels |= {pair.drive: IqChannel(mixer=None, lo=None)}
    # register the instruments
    instruments = {
        "dummy": EmulatorController(address="0.0.0.0", channels=channels),
    }

    return Platform.load(
        path=FOLDER,
        instruments=instruments,
        qubits=qubits,
        qubit_pairs=pairs,
    )
