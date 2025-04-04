from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
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
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.result import probability
from qibocal.update import replace

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
    """ResonatorSpectroscopy runcard inputs."""

    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""

    pulse_duration: Optional[int] = None

    @property
    def amplitude_range(self):
        return np.arange(self.min_amp, self.max_amp, self.step_amp)


@dataclass
class CrossResonanceAmplitudeResults(Results):
    """ResonatorSpectroscopy outputs."""


@dataclass
class CrossResonanceAmplitudeData(Data):
    """Data structure for resonator spectroscopy with attenuation."""

    data: dict[QubitId, npt.NDArray[CrossResonanceAmplitudeType]] = field(
        default_factory=dict
    )
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
        for setup in ["I", "X"]:
            sequence = PulseSequence()
            natives_control = platform.natives.single_qubit[control]
            natives_target = platform.natives.single_qubit[target]
            cr_channel, cr_drive_pulse = platform.natives.two_qubit[pair].CNOT()[0]
            control_drive_channel, control_drive_pulse = natives_control.RX()[0]
            ro_channel, ro_pulse = natives_target.MZ()[0]
            ro_channel_control, ro_pulse_control = natives_control.MZ()[0]
            if setup == "X":
                sequence.append((control_drive_channel, control_drive_pulse))
                sequence.append(
                    (ro_channel, Delay(duration=control_drive_pulse.duration))
                )
                sequence.append(
                    (ro_channel_control, Delay(duration=control_drive_pulse.duration))
                )
                sequence.append(
                    (cr_channel, Delay(duration=control_drive_pulse.duration))
                )

            if params.pulse_duration is not None:
                cr_drive_pulse = replace(
                    cr_drive_pulse, amplitude=params.pulse_duration
                )

            sequence.append((cr_channel, cr_drive_pulse))

            delay1 = Delay(duration=cr_drive_pulse.duration)
            delay2 = Delay(duration=cr_drive_pulse.duration)

            sequence.append((ro_channel, delay1))
            sequence.append((ro_channel_control, delay2))
            sequence.append((ro_channel, ro_pulse))
            sequence.append((ro_channel_control, ro_pulse_control))

            sweeper = Sweeper(
                parameter=Parameter.amplitude,
                values=params.amplitude_range,
                pulses=[cr_drive_pulse],
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
            for q in [target]:
                prob_target = probability(results[ro_pulse.id], state=1)
                prob_control = probability(results[ro_pulse_control.id], state=1)
                data.register_qubit(
                    CrossResonanceAmplitudeType,
                    (q, setup),
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
    """Post-processing function for ResonatorSpectroscopy."""
    return CrossResonanceAmplitudeResults()


def _plot(
    data: CrossResonanceAmplitudeData,
    target: QubitPairId,
    fit: CrossResonanceAmplitudeResults,
):
    """Plotting function for ResonatorSpectroscopy."""
    # TODO: share colorbar

    target_idle_data = data.data[target[1], "I"]
    target_excited_data = data.data[target[1], "X"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=target_idle_data.amp,
            y=target_idle_data.prob_control,
            name="Control at 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=target_idle_data.amp,
            y=target_excited_data.prob_control,
            name="Control at 1",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=target_idle_data.amp,
            y=target_idle_data.prob_target,
            name="Target when Control at 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=target_excited_data.amp,
            y=target_excited_data.prob_target,
            name="Target when Control at 1",
        ),
    )

    return [fig], ""


cross_resonance_amplitude = Routine(_acquisition, _fit, _plot)
"""CrossResonance Routine object."""
