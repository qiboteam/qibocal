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

CrossResonanceType = np.dtype(
    [
        ("prob_target", np.float64),
        ("prob_control", np.float64),
        ("length", np.int64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class CrossResonanceParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""

    pulse_amplitude: Optional[float] = None
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""

    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class CrossResonanceResults(Results):
    """ResonatorSpectroscopy outputs."""


@dataclass
class CrossResonanceData(Data):
    """Data structure for resonator spectroscopy with attenuation."""

    data: dict[QubitId, npt.NDArray[CrossResonanceType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CrossResonanceData:
    """Data acquisition for resonator spectroscopy."""

    data = CrossResonanceData()

    for pair in targets:
        control, target = pair
        print("CONTROL", control)
        print("TARGET", target)
        pair = (control, target)
        for setup in ["I", "X"]:
            sequence = PulseSequence()
            natives_control = platform.natives.single_qubit[control]
            natives_target = platform.natives.single_qubit[target]
            cr_channel, cr_drive_pulse = platform.natives.two_qubit[pair].CNOT()[0]
            control_drive_channel, control_drive_pulse = natives_control.RX()[0]
            target_drive_channel, target_drive_pulse = natives_target.RX()[0]
            ro_channel, ro_pulse = natives_target.MZ()[0]
            ro_channel_control, ro_pulse_control = natives_control.MZ()[0]
            if setup == "X":
                sequence.append((control_drive_channel, control_drive_pulse))
                # sequence.append((target_drive_channel, target_drive_pulse))
                sequence.append(
                    (ro_channel, Delay(duration=control_drive_pulse.duration))
                )
                sequence.append(
                    (ro_channel_control, Delay(duration=control_drive_pulse.duration))
                )
                sequence.append(
                    (cr_channel, Delay(duration=control_drive_pulse.duration))
                )

            if params.pulse_amplitude is not None:
                cr_drive_pulse = replace(
                    cr_drive_pulse, amplitude=params.pulse_amplitude
                )

            sequence.append((cr_channel, cr_drive_pulse))

            delay1 = Delay(duration=0)
            delay2 = Delay(duration=0)
            if params.interpolated_sweeper:
                sequence.align([cr_channel, ro_channel, ro_pulse_control])
            else:
                sequence.append((ro_channel, delay1))
                sequence.append((ro_channel_control, delay2))
            sequence.append((ro_channel, ro_pulse))
            sequence.append((ro_channel_control, ro_pulse_control))

            # sequence, qd_pulses, delays, ro_pulses, amplitudes = sequence_length(
            # [target],
            # params,
            # platform,
            # use_align=params.interpolated_sweeper, cross_resonance=control if setup == "X" else None
            # )

            sweep_range = (
                params.pulse_duration_start,
                params.pulse_duration_end,
                params.pulse_duration_step,
            )
            if params.interpolated_sweeper:
                sweeper = Sweeper(
                    parameter=Parameter.duration_interpolated,
                    range=sweep_range,
                    pulses=[cr_drive_pulse],
                )
            else:
                sweeper = Sweeper(
                    parameter=Parameter.duration,
                    range=sweep_range,
                    pulses=[cr_drive_pulse, delay1, delay2],
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
                    CrossResonanceType,
                    (q, setup),
                    dict(
                        length=sweeper.values,
                        prob_target=prob_target,
                        prob_control=prob_control,
                        # error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
                    ),
                )
    # finally, save the remaining data
    return data


def _fit(
    data: CrossResonanceData,
) -> CrossResonanceResults:
    """Post-processing function for ResonatorSpectroscopy."""
    return CrossResonanceResults()


def _plot(data: CrossResonanceData, target: QubitPairId, fit: CrossResonanceResults):
    """Plotting function for ResonatorSpectroscopy."""
    # TODO: share colorbar

    target_idle_data = data.data[target[1], "I"]
    target_excited_data = data.data[target[1], "X"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=target_idle_data.length,
            y=target_idle_data.prob_control,
            name="Control at 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=target_idle_data.length,
            y=target_excited_data.prob_control,
            name="Control at 1",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=target_idle_data.length,
            y=target_idle_data.prob_target,
            name="Target when Control at 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=target_excited_data.length,
            y=target_excited_data.prob_target,
            name="Target when Control at 1",
        ),
    )

    return [fig], ""


cross_resonance = Routine(_acquisition, _fit, _plot)
"""CrossResonance Routine object."""
