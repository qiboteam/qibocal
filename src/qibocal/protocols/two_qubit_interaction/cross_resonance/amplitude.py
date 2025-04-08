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
from ...rabi.utils import fit_amplitude_function, rabi_amplitude_function
from ...utils import fallback_period, guess_period
from .utils import SetControl, cr_sequence

CrossResonanceAmplitudeType = np.dtype(
    [
        ("prob_target", np.float64),
        ("prob_control", np.float64),
        ("amp", np.float64),
    ]
)
"""Custom dtype for cross resonance amplitude."""


@dataclass
class CrossResonanceAmplitudeParameters(Parameters):
    """CrossResonanceAmplitude runcard inputs."""

    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""
    pulse_duration: int
    """CR pulse duration in ns."""

    @property
    def amplitude_range(self):
        return np.arange(self.min_amp, self.max_amp, self.step_amp)


@dataclass
class CrossResonanceAmplitudeResults(Results):
    """CrossResonanceAmplitude outputs."""

    fitted_parameters: dict[tuple[QubitPairId, str], list] = field(default_factory=dict)

    def __contains__(self, pair: QubitPairId):
        return all(key[:2] == pair for key in list(self.fitted_parameters))


@dataclass
class CrossResonanceAmplitudeData(Data):
    """Data structure for resonator spectroscopy with attenuation."""

    data: dict[
        tuple[QubitId, QubitId, str], npt.NDArray[CrossResonanceAmplitudeType]
    ] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CrossResonanceAmplitudeData:
    """Data acquisition for resonator spectroscopy."""

    data = CrossResonanceAmplitudeData()

    for pair in targets:
        control, target = pair
        pair = (control, target)
        for setup in SetControl:
            sequence, cr_pulse, _ = cr_sequence(
                platform=platform,
                control=control,
                target=target,
                setup=setup,
                amplitude=params.min_amp,
                duration=params.pulse_duration,
            )

            sweeper = Sweeper(
                parameter=Parameter.amplitude,
                values=params.amplitude_range,
                pulses=[cr_pulse],
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

            # store the results
            target_acq_handle = list(
                sequence.channel(platform.qubits[target].acquisition)
            )[-1].id
            control_acq_handle = list(
                sequence.channel(platform.qubits[control].acquisition)
            )[-1].id
            prob_target = probability(results[target_acq_handle], state=1)
            prob_control = probability(results[control_acq_handle], state=1)
            data.register_qubit(
                CrossResonanceAmplitudeType,
                (control, target, setup),
                dict(
                    amp=sweeper.values,
                    prob_target=prob_target,
                    prob_control=prob_control,
                ),
            )
    # finally, save the remaining data
    return data


def _fit(
    data: CrossResonanceAmplitudeData,
) -> CrossResonanceAmplitudeResults:
    """Post-processing function for CrossResonanceAmplitude."""

    fitted_parameters = {}

    for pair in data.pairs:
        for setup in SetControl:
            pair_data = data[pair[0], pair[1], setup]
            y = pair_data.prob_target
            x = pair_data.amp

            period = fallback_period(guess_period(x, y))
            pguess = [0.5, 0.5, period, np.pi]

            try:
                popt, _, _ = fit_amplitude_function(
                    x,
                    y,
                    pguess,
                    # sigma=qubit_data.error,
                    signal=False,
                )
                fitted_parameters[pair[0], pair[1], setup] = popt.tolist()

            except Exception as e:
                log.warning(f"CR amplitude fit failed for pair {pair} due to {e}.")
    return CrossResonanceAmplitudeResults(fitted_parameters=fitted_parameters)


def _plot(
    data: CrossResonanceAmplitudeData,
    target: QubitPairId,
    fit: CrossResonanceAmplitudeResults,
):
    """Plotting function for CrossResonanceAmplitude."""
    idle_data = data.data[target[0], target[1], "I"]
    excited_data = data.data[target[0], target[1], "X"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idle_data.amp,
            y=idle_data.prob_control,
            name="Control at 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=excited_data.amp,
            y=excited_data.prob_control,
            name="Control at 1",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=idle_data.amp,
            y=idle_data.prob_target,
            name="Target when Control at 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=excited_data.amp,
            y=excited_data.prob_target,
            name="Target when Control at 1",
        ),
    )

    if fit is not None:
        for setup in SetControl:
            fit_data = idle_data if setup == "Id" else excited_data
            x = np.linspace(fit_data.amp.min(), fit_data.amp.max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=rabi_amplitude_function(
                        x, *fit.fitted_parameters[target[0], target[1], setup]
                    ),
                    name=f"Fit target when control at {0 if setup == 'Id' else 1}",
                )
            )
    fig.update_layout(
        xaxis_title="Cross resonance pulse amplitude [a.u.]",
        yaxis_title="Excited state population",
    )
    return [fig], ""


cross_resonance_amplitude = Routine(_acquisition, _fit, _plot)
"""CrossResonance Routine object."""
