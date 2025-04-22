from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Sweeper,
)

from .....auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from .....calibration import CalibrationPlatform
from .....result import probability
from ..utils import Basis, SetControl, cr_sequence

HamiltonianTomographyCRAmplitudeType = np.dtype(
    [
        ("prob_target", np.float64),
        ("prob_control", np.float64),
        ("amp", np.float64),
    ]
)
"""Custom dtype for CR length."""


@dataclass
class HamiltonianTomographyCRAmplitudeParameters(Parameters):
    """HamiltonianTomographyCRAmplitude runcard inputs."""

    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""
    pulse_duration: int
    """CR pulse duration in ns."""
    echo: bool = False
    """Apply echo sequence or not."""

    @property
    def amplitude_range(self):
        return np.arange(self.min_amp, self.max_amp, self.step_amp)


@dataclass
class HamiltonianTomographyCRAmplitudeResults(Results):
    """HamiltonianTomographyCRAmplitude outputs."""


@dataclass
class HamiltonianTomographyCRAmplitudeData(Data):
    """Data structure for CR length."""

    anharmonicity: dict[QubitPairId, float] = field(default_factory=dict)
    detuning: dict[QubitPairId, float] = field(default_factory=dict)
    data: dict[
        tuple[QubitId, QubitId, str], npt.NDArray[HamiltonianTomographyCRAmplitudeType]
    ] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: HamiltonianTomographyCRAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRAmplitudeData:
    """Data acquisition for cross resonance protocol."""

    data = HamiltonianTomographyCRAmplitudeData()

    for pair in targets:
        control, target = pair
        pair = (control, target)
        data.detuning[pair] = (
            platform.config(platform.qubits[control].drive).frequency
            - platform.config(platform.qubits[target].drive).frequency
        )
        data.anharmonicity[pair] = platform.calibration.single_qubits[
            control
        ].qubit.anharmonicity
        for basis in Basis:
            for setup in SetControl:
                sequence, cr_pulses, delays = cr_sequence(
                    platform=platform,
                    control=control,
                    target=target,
                    setup=setup,
                    amplitude=params.min_amp,
                    duration=params.pulse_duration,
                    echo=params.echo,
                    basis=basis,
                )

                sweeper = Sweeper(
                    parameter=Parameter.amplitude,
                    values=params.amplitude_range,
                    pulses=cr_pulses,
                )

                updates = []
                updates.append(
                    {
                        platform.qubits[control].drive_extra[target]: {
                            "frequency": platform.config(
                                platform.qubits[target].drive
                            ).frequency
                        }
                    }
                )
                # execute the sweep
                results = platform.execute(
                    [sequence],
                    [[sweeper]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.SINGLESHOT,
                    updates=updates,
                )
                target_acq_handle = list(
                    sequence.channel(platform.qubits[target].acquisition)
                )[-1].id
                control_acq_handle = list(
                    sequence.channel(platform.qubits[control].acquisition)
                )[-1].id
                exp_target = 1 - 2 * probability(results[target_acq_handle], state=1)
                prob_control = probability(results[control_acq_handle], state=1)
                data.register_qubit(
                    HamiltonianTomographyCRAmplitudeType,
                    (control, target, basis, setup),
                    dict(
                        amp=sweeper.values,
                        prob_target=exp_target,
                        prob_control=prob_control,
                    ),
                )
    # finally, save the remaining data
    return data


def _fit(
    data: HamiltonianTomographyCRAmplitudeData,
) -> HamiltonianTomographyCRAmplitudeResults:
    """Post-processing function for HamiltonianTomographyCRAmplitude."""

    return HamiltonianTomographyCRAmplitudeResults()


def _plot(
    data: HamiltonianTomographyCRAmplitudeData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRAmplitudeResults,
):
    """Plotting function for HamiltonianTomographyCRAmplitude."""
    fig = make_subplots(
        rows=3,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
        shared_xaxes=True,
    )
    for i, basis in enumerate(Basis):
        for setup in SetControl:
            pair_data = data.data[target[0], target[1], basis, setup]
            fig.add_trace(
                go.Scatter(
                    x=pair_data.amp,
                    y=pair_data.prob_target,
                    name=f"Target <{basis.name}> when Control at {setup.name}",
                    legendgrouptitle_text=setup.name,
                    legendgroup=str(i),
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        yaxis1=dict(range=[-1.2, 1.2]),
        yaxis2=dict(range=[-1.2, 1.2]),
        yaxis3=dict(range=[-1.2, 1.2]),
        height=600,
        legend_tracegroupgap=130,
        xaxis3_title="CR pulse amplitude [a.u.]",
    )
    fig.update_yaxes(title_text="<X(t)>", row=1, col=1)
    fig.update_yaxes(title_text="<Y(t)>", row=2, col=1)
    fig.update_yaxes(title_text="<Z(t)>", row=3, col=1)
    return [fig], ""


hamiltonian_tomography_cr_amplitude = Routine(_acquisition, _fit, _plot)
"""HamiltonianTomography Routine object."""
