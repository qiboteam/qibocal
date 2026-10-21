from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants as const
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Pulse,
    PulseId,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import (
    Data,
    Parameters,
    Protocol,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import Range, RangeLike, readout_frequency, to_range
from qibocal.result import magnitude

__all__ = ["coupler_spectroscopy"]


CouplerSpectroscopyType = np.dtype(
    [
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
    ]
)


@dataclass
class CouplerSpectroscopyParameters(Parameters):
    """Coupler readout runcard inputs."""

    qubit_pulse_amplitude: float
    qubit_pulse_duration: float
    coupler_frequency: RangeLike
    coupler_pulse_duration: float = 40e3
    coupler_pulse_amplitude: float = 0.99
    coupler_power: float = -2.0

    def frequency_range(self, center: float | None = None) -> Range:
        return to_range(self.coupler_frequency, center)


@dataclass
class CouplerSpectroscopyResults(Results):
    """Coupler readout fit."""

    def __contains__(self, key):
        return True


@dataclass
class CouplerSpectroscopyData(Data):
    """Coupler acquisition outputs."""

    data: dict[QubitPairId, CouplerSpectroscopyType] = field(default_factory=dict)

    def coupler_frequency(self, pair: QubitPairId) -> npt.NDArray[float]:
        return np.unique(self.data[pair]["coupler"])

    def return_data_per_coupler_freq(
        self, pair: QubitPairId, coupler_freq: float
    ) -> npt.NDArray[CouplerSpectroscopyType]:
        return self.data[pair][np.array(self.data[pair])["coupler"] == coupler_freq]


def _acquisition(
    params: CouplerSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerSpectroscopyData:
    """Protocol to perform sweep to locate coupler."""

    assert platform.calibration is not None

    spectr_pulse = Pulse(
        duration=params.qubit_pulse_duration,
        amplitude=params.qubit_pulse_amplitude,
        envelope=Rectangular(),
    )
    coupler_pulse = Pulse(
        duration=params.coupler_pulse_duration,
        amplitude=params.coupler_pulse_amplitude,
        envelope=Rectangular(),
    )

    full_sequence = PulseSequence()
    frequency_parsweeper: dict[QubitPairId, Sweeper] = {}
    ro_pulses_id: dict[QubitPairId, PulseId] = {}
    updates: dict = {}
    for pair in targets:
        qubit1, qubit2 = pair
        qubit1_natives = platform.natives.single_qubit[qubit1]
        qd_channel, _ = qubit1_natives.RX()[0]
        ro_channel, ro_pulse = qubit1_natives.MZ()[0]

        qubit2_natives = platform.natives.single_qubit[qubit2]
        coupler_ch, _ = qubit2_natives.RX()[0]

        sequence = PulseSequence([(coupler_ch, coupler_pulse)])
        sequence |= PulseSequence([(qd_channel, spectr_pulse)])
        sequence |= PulseSequence([(ro_channel, ro_pulse)])

        full_sequence += sequence
        ro_pulses_id[pair] = ro_pulse.id

        coupler_id = platform.calibration.two_qubits[pair].coupler
        if coupler_id is None:
            raise ValueError("Coupler ID should not be None")

        coupler_freq = platform.calibration.couplers[coupler_id].qubit.frequency_01
        coupler_range = params.frequency_range(coupler_freq)
        frequency_parsweeper[pair] = Sweeper(
            parameter=Parameter.frequency,
            range=coupler_range,
            channels=[coupler_ch],
        )

        channel_obj = platform.channels[coupler_ch]
        updates |= {
            platform.qubits[qubit1].probe: {
                "frequency": readout_frequency(qubit1, platform)
            },
            coupler_ch: {"frequency": coupler_range[0], "power": params.coupler_power},
            channel_obj.lo: {"frequency": coupler_range[0] - 50e6},
        }

    data = CouplerSpectroscopyData()
    results = platform.execute(
        [full_sequence],
        [list(frequency_parsweeper.values())],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for pair in targets:
        freq_values = np.array(frequency_parsweeper[pair].values)
        quadratures = np.asarray(results[ro_pulses_id[pair]])

        # Store absolute bias here; fitting and updates must not add DC again.
        data.register_qubit(
            CouplerSpectroscopyType,
            pair,
            {
                "freq": freq_values,
                "i": quadratures[..., 0].ravel(),
                "q": quadratures[..., 1].ravel(),
            },
        )

    return data


def _fit(data: CouplerSpectroscopyData) -> CouplerSpectroscopyResults:
    """Fitting routine for Coupler readout experiment."""

    return CouplerSpectroscopyResults()


def _plot(
    data: CouplerSpectroscopyData,
    target: QubitPairId,
    fit: CouplerSpectroscopyResults | None = None,
):
    """Plotting function for CouplerSpectroscopy Experiment."""
    figures = []
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.data[target]["freq"] * const.nano,
            y=magnitude(
                np.stack([data.data[target]["i"], data.data[target]["q"]], axis=-1)
            ),
        ),
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Coupler Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
    )
    figures.append(fig)
    fitting_report = ""

    return figures, fitting_report


coupler_spectroscopy = Protocol(_acquisition, _fit, _plot)
"""CouplerSpectroscopy Routine object."""
