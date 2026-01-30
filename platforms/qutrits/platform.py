import pathlib

from qibolab import ConfigKinds
from qibolab._core.components import IqChannel
from qibolab._core.instruments.emulator.emulator import EmulatorController
from qibolab._core.instruments.emulator.hamiltonians import (
    DriveEmulatorConfig,
    HamiltonianConfig,
)
from qibolab._core.platform import Platform
from qibolab._core.qubits import Qubit

FOLDER = pathlib.Path(__file__).parent

ConfigKinds.extend([HamiltonianConfig, DriveEmulatorConfig])


def create() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    qubits = {}
    channels = {}

    qubits[0] = qubit = Qubit.default(
        0, drive_extra={(1, 2): "0/drive12", 1: "0/drive1"}
    )
    channels |= {
        qubit.drive: IqChannel(mixer=None, lo=None),
        qubits[0].drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
        qubits[0].drive_extra[1]: IqChannel(mixer=None, lo=None),
    }
    qubits[1] = qubit = Qubit.default(1, drive_extra={(1, 2): "1/drive12"})
    channels |= {
        qubit.drive: IqChannel(mixer=None, lo=None),
        qubits[1].drive_extra[1, 2]: IqChannel(mixer=None, lo=None),
    }
    # register the instruments
    instruments = {
        "dummy": EmulatorController(address="0.0.0.0", channels=channels),
    }

    return Platform.load(
        path=FOLDER,
        instruments=instruments,
        qubits=qubits,
    )
