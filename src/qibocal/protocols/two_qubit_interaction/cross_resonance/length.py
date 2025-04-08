from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Sweeper,
)

from ....auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from ....calibration import CalibrationPlatform
from ....config import log
from ....result import probability
from ...rabi.utils import fit_length_function, rabi_length_function
from ...utils import fallback_period, guess_period
from .utils import SetControl, cr_sequence

CrossResonanceLengthType = np.dtype(
    [
        ("prob_target", np.float64),
        ("prob_control", np.float64),
        ("length", np.int64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class CrossResonanceLengthParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

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

    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class CrossResonanceLengthResults(Results):
    """ResonatorSpectroscopy outputs."""

    fitted_parameters: dict[tuple[QubitPairId, str], list] = field(default_factory=dict)

    def __contains__(self, pair: QubitPairId):
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class CrossResonanceLengthData(Data):
    """Data structure for resonator spectroscopy with attenuation."""

    data: dict[tuple[QubitId, QubitId, str], npt.NDArray[CrossResonanceLengthType]] = (
        field(default_factory=dict)
    )
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceLengthParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CrossResonanceLengthData:
    """Data acquisition for cross resonance protocol."""

    data = CrossResonanceLengthData()

    for pair in targets:
        control, target = pair
        pair = (control, target)
        for setup in SetControl:
            sequence, cr_pulse, delays = cr_sequence(
                platform=platform,
                control=control,
                target=target,
                setup=setup,
                amplitude=params.pulse_amplitude,
                duration=params.pulse_duration_end,
                interpolated_sweeper=params.interpolated_sweeper,
            )

            if params.interpolated_sweeper:
                sweeper = Sweeper(
                    parameter=Parameter.duration_interpolated,
                    values=params.duration_range,
                    pulses=[cr_pulse],
                )
            else:
                sweeper = Sweeper(
                    parameter=Parameter.duration,
                    values=params.duration_range,
                    pulses=[cr_pulse] + delays,
                )

            updates = []
            updates.append(
                {
                    platform.qubit_pairs[pair].drive: {
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
            prob_target = probability(results[target_acq_handle], state=1)
            prob_control = probability(results[control_acq_handle], state=1)
            data.register_qubit(
                CrossResonanceLengthType,
                (control, target, setup),
                dict(
                    length=sweeper.values,
                    prob_target=prob_target,
                    prob_control=prob_control,
                ),
            )
    # finally, save the remaining data
    return data


def _fit(
    data: CrossResonanceLengthData,
) -> CrossResonanceLengthResults:
    """Post-processing function for ResonatorSpectroscopy."""

    fitted_parameters = {}

    for pair in data.pairs:
        for setup in SetControl:
            pair_data = data[pair[0], pair[1], setup]
            raw_x = pair_data.length
            min_x = np.min(raw_x)
            max_x = np.max(raw_x)
            y = pair_data.prob_target
            x = (raw_x - min_x) / (max_x - min_x)

            period = fallback_period(guess_period(x, y))
            pguess = [0.5, 0.5, period, 0, 0]

            try:
                popt, _, _ = fit_length_function(
                    x,
                    y,
                    pguess,
                    # sigma=qubit_data.error,
                    signal=False,
                    x_limits=(min_x, max_x),
                )
                fitted_parameters[pair[0], pair[1], setup] = popt

            except Exception as e:
                log.warning(f"CR length fit failed for pair {pair} due to {e}.")
    return CrossResonanceLengthResults(fitted_parameters=fitted_parameters)


def _plot(
    data: CrossResonanceLengthData,
    target: QubitPairId,
    fit: CrossResonanceLengthResults,
):
    """Plotting function for CrossResonanceLength."""
    idle_data = data.data[target[0], target[1], "I"]
    excited_data = data.data[target[0], target[1], "X"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idle_data.length,
            y=idle_data.prob_control,
            name="Control at 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=excited_data.length,
            y=excited_data.prob_control,
            name="Control at 1",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=idle_data.length,
            y=idle_data.prob_target,
            name="Target when Control at 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=excited_data.length,
            y=excited_data.prob_target,
            name="Target when Control at 1",
        ),
    )

    if fit is not None:
        for setup in SetControl:
            fit_data = idle_data if setup == "Id" else excited_data
            x = np.linspace(fit_data.length.min(), fit_data.length.max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=rabi_length_function(
                        x, *fit.fitted_parameters[target[0], target[1], setup]
                    ),
                    name=f"Fit target when control at {0 if setup == 'Id' else 1}",
                )
            )

    fig.update_layout(
        xaxis_title="Cross resonance pulse duration [ns]",
        yaxis_title="Excited state population",
    )
    return [fig], ""


cross_resonance_length = Routine(_acquisition, _fit, _plot)
"""CrossResonance Routine object."""
