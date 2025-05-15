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
    for q in range(2):
        drive12 = f"{q}/drive12"
        qubits[q] = qubit = Qubit.default(q, drive_extra={(1, 2): drive12})
        channels |= {
            qubit.probe: IqChannel(mixer=None, lo="01/probe_lo"),
            qubit.acquisition: AcquisitionChannel(
                twpa_pump=pump_name, probe=qubit.probe
            ),
            qubit.drive: IqChannel(mixer=None, lo=f"{q}/drive_lo"),
            drive12: IqChannel(mixer=None, lo=f"{q}/drive_lo"),
            qubit.flux: DcChannel(),
        }

    couplers = {}
    couplers["coupler_0"] = coupler = Qubit(flux="coupler_0/flux")
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
