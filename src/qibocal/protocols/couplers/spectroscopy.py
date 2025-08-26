from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace

from ...result import magnitude
from ..utils import (
    HZ_TO_GHZ,
    readout_frequency,
)

__all__ = ["coupler_spectroscopy"]


@dataclass
class CouplerSpectroscopyParameters(Parameters):
    """Coupler readout runcard inputs."""

    freq_width: float
    freq_step: float
    bias_width: float
    bias_step: float
    drive_duration: float
    drive_amplitude: float

    @property
    def frequency_span(self):
        return np.arange(-self.freq_width / 2, self.freq_width / 2, self.freq_step)

    @property
    def bias_span(self):
        return np.arange(-self.bias_width / 2, self.bias_width / 2, self.bias_step)


@dataclass
class CouplerSpectroscopyResults(Results):
    """Coupler readout fit."""

    def __contains__(self, key):
        return True


@dataclass
class CouplerSpectroscopyData(Data):
    """Coupler acquisition outputs."""

    frequencies_swept: dict[QubitPairId, list]
    biases_swept: dict[QubitPairId, list]
    data: dict[QubitPairId, np.ndarray] = field(default_factory=dict)


def _acquisition(
    params: CouplerSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerSpectroscopyData:
    """Protocol to perform sweep to locate coupler."""

    assert len(targets) == 1, (
        "Coupler spectroscopy is only available for one qubit pair at a time."
    )
    pair = targets[0]
    sequence = PulseSequence()
    qubit = pair[1]
    natives = platform.natives.single_qubit[qubit]
    qd_channel, qd_pulse = natives.RX()[0]
    ro_channel, ro_pulse = natives.MZ()[0]

    qd_pulse = replace(qd_pulse, duration=params.drive_duration)
    if params.drive_amplitude is not None:
        qd_pulse = replace(qd_pulse, amplitude=params.drive_amplitude)

    sequence.append((qd_channel, qd_pulse))
    sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
    sequence.append((ro_channel, ro_pulse))

    # define the parameters to sweep and their range:
    freq_sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=platform.config(qd_channel).frequency + params.frequency_span,
        channels=[qd_channel],
    )

    coupler_channel = platform.couplers[
        list(platform.couplers)[platform.pairs.index(pair)]
    ].flux
    offset0 = platform.config(coupler_channel).offset
    bias_sweeper = Sweeper(
        parameter=Parameter.offset,
        values=offset0 + params.bias_span,
        channels=[coupler_channel],
    )

    data = CouplerSpectroscopyData(
        frequencies_swept={pair: freq_sweeper.values.tolist()},
        biases_swept={pair: bias_sweeper.values.tolist()},
    )
    results = platform.execute(
        [sequence],
        [[bias_sweeper], [freq_sweeper]],
        updates=[
            {
                platform.qubits[qubit].probe: {
                    "frequency": readout_frequency(qubit, platform)
                }
            }
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data.data[pair] = results[ro_pulse.id]
    return data


def _fit(data: CouplerSpectroscopyData) -> CouplerSpectroscopyResults:
    """Fitting routine for Coupler readout experiment."""

    return CouplerSpectroscopyResults()


def _plot(
    data: CouplerSpectroscopyData,
    target: QubitPairId,
    fit: CouplerSpectroscopyResults = None,
):
    """Plotting function for CouplerSpectroscopy Experiment."""
    figures = []
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=np.array(data.frequencies_swept[target]) * HZ_TO_GHZ,
            y=data.biases_swept[target],
            z=magnitude(data.data[target]),
            colorbar=dict(
                title=dict(
                    text="P_1",
                    side="top",
                ),
            ),
        )
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Qubit Frequency [GHz]",
        yaxis_title="Coupler Bias [V]",
    )
    figures.append(fig)
    fitting_report = ""
    return figures, fitting_report


coupler_spectroscopy = Routine(_acquisition, _fit, _plot)
"""CouplerSpectroscopy Routine object."""
