import pathlib

from qibolab import ConfigKinds
from qibolab._core.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab._core.instruments.dummy import DummyInstrument, DummyLocalOscillator
from qibolab._core.parameters import Hardware
from qibolab._core.platform import Platform
from qibolab._core.qubits import Qubit

from qibocal.protocols.utils import DcFilteredConfig

ConfigKinds.extend([DcFilteredConfig])

FOLDER = pathlib.Path(__file__).parent


def create_mock_hardware() -> Hardware:
    """Create dummy hardware configuration based on the dummy instrument."""

    qubits = {}
    channels = {}
    # attach the channels
    pump_name = "twpa_pump"
    qubits[0] = Qubit.default(0, drive_extra={(1, 2): "0/drive12", 1: "01/drive"})

    channels |= {
        qubits[0].probe: IqChannel(mixer=None, lo="01/probe_lo"),
        qubits[0].acquisition: AcquisitionChannel(
            twpa_pump=pump_name, probe=qubits[0].probe
        ),
        qubits[0].drive: IqChannel(mixer=None, lo="0/drive_lo"),
        qubits[0].drive_extra[1, 2]: IqChannel(mixer=None, lo="0/drive_lo"),
        qubits[0].drive_extra[1]: IqChannel(mixer=None, lo="0/drive_lo"),
        qubits[0].flux: DcChannel(),
    }
    qubits[1] = Qubit.default(1, drive_extra={(1, 2): "0/drive12"})
    channels |= {
        qubits[1].probe: IqChannel(mixer=None, lo="01/probe_lo"),
        qubits[1].acquisition: AcquisitionChannel(
            twpa_pump=pump_name, probe=qubits[1].probe
        ),
        qubits[1].drive: IqChannel(mixer=None, lo="1/drive_lo"),
        qubits[1].drive_extra[1, 2]: IqChannel(mixer=None, lo="1/drive_lo"),
        qubits[1].flux: DcChannel(),
    }
    couplers = {}
    couplers["01"] = coupler = Qubit(flux="coupler_01/flux")
    channels |= {coupler.flux: DcChannel()}
    # register the instruments
    instruments = {
        "dummy": DummyInstrument(address="0.0.0.0", channels=channels),
        pump_name: DummyLocalOscillator(address="0.0.0.0"),
    }
    return Hardware(instruments=instruments, qubits=qubits, couplers=couplers)


def create() -> Platform:
    """Create a dummy platform using the dummy instrument."""
    hardware = create_mock_hardware()
    return Platform.load(path=FOLDER, **vars(hardware))
