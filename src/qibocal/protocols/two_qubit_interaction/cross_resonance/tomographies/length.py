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

HamiltonianTomographyCRLengthType = np.dtype(
    [
        ("prob_target", np.float64),
        ("prob_control", np.float64),
        ("length", np.int64),
    ]
)
"""Custom dtype for CR length."""


@dataclass
class HamiltonianTomographyCRLengthParameters(Parameters):
    """HamiltonianTomographyCRLength runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    pulse_amplitude: float
    """CR pulse amplitude"""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""
    echo: bool = False
    """Apply echo sequence or not."""

    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class HamiltonianTomographyCRLengthResults(Results):
    """HamiltonianTomographyCRLength outputs."""


@dataclass
class HamiltonianTomographyCRLengthData(Data):
    """Data structure for CR length."""

    anharmonicity: dict[QubitPairId, float] = field(default_factory=dict)
    detuning: dict[QubitPairId, float] = field(default_factory=dict)
    data: dict[
        tuple[QubitId, QubitId, str], npt.NDArray[HamiltonianTomographyCRLengthType]
    ] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: HamiltonianTomographyCRLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> HamiltonianTomographyCRLengthData:
    """Data acquisition for cross resonance protocol."""

    data = HamiltonianTomographyCRLengthData()

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
                    amplitude=params.pulse_amplitude,
                    duration=params.pulse_duration_end,
                    interpolated_sweeper=params.interpolated_sweeper,
                    echo=params.echo,
                    basis=basis,
                )

                if params.interpolated_sweeper:
                    sweeper = Sweeper(
                        parameter=Parameter.duration_interpolated,
                        values=params.duration_range,
                        pulses=cr_pulses,
                    )
                else:
                    sweeper = Sweeper(
                        parameter=Parameter.duration,
                        values=params.duration_range,
                        pulses=cr_pulses + delays,
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
                    HamiltonianTomographyCRLengthType,
                    (control, target, basis, setup),
                    dict(
                        length=sweeper.values,
                        prob_target=exp_target,
                        prob_control=prob_control,
                    ),
                )
    # finally, save the remaining data
    return data


def _fit(
    data: HamiltonianTomographyCRLengthData,
) -> HamiltonianTomographyCRLengthResults:
    """Post-processing function for HamiltonianTomographyCRLength."""

    return HamiltonianTomographyCRLengthResults()


def _plot(
    data: HamiltonianTomographyCRLengthData,
    target: QubitPairId,
    fit: HamiltonianTomographyCRLengthResults,
):
    """Plotting function for HamiltonianTomographyCRLength."""
    # idle_data = data.data[target[0], target[1], SetControl.Id]
    # excited_data = data.data[target[0], target[1], SetControl.X]
    # fitting_report = ""
    fig = make_subplots(
        rows=3,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
        shared_xaxes=True,
        shared_yaxes=True,
    )
    for i, basis in enumerate(Basis):
        for setup in SetControl:
            pair_data = data.data[target[0], target[1], basis, setup]
            fig.add_trace(
                go.Scatter(
                    x=pair_data.length,
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
        xaxis3_title="CR pulse length [ns]",
    )
    fig.update_yaxes(title_text="<X(t)>", row=1, col=1)
    fig.update_yaxes(title_text="<Y(t)>", row=2, col=1)
    fig.update_yaxes(title_text="<Z(t)>", row=3, col=1)
    return [fig], ""


hamiltonian_tomography_cr_length = Routine(_acquisition, _fit, _plot)
"""HamiltonianTomography Routine object."""
