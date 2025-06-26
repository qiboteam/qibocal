from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
import plotly.graph_objects as go
from qibolab import Delay, Platform, Pulse, PulseSequence, Rectangular, VirtualZ

from ....auto.operation import QubitId, QubitPairId
from ....config import log
from ....update import replace
from ...utils import fallback_period, guess_period


class SetControl(str, Enum):
    """Helper to create sequence with control set to X or I."""

    Id = "Id"
    X = "X"


class Basis(str, Enum):
    """Measurement basis."""

    X = "X"
    Y = "Y"
    Z = "Z"


def cr_sequence(
    platform: Platform,
    control: QubitId,
    target: QubitId,
    setup: SetControl,
    amplitude: float,
    duration: int,
    target_amplitude: float = 0,
    target_phase: float = 0,
    interpolated_sweeper: bool = False,
    echo: bool = False,
    basis: Basis = Basis.Z,
    phase: float = 0,
) -> tuple[PulseSequence, list[Pulse], list[Pulse], list[Delay]]:
    """Creates sequence for CR experiment on ``control`` and ``target`` qubits.

    With ``setup`` it is possible to set the control qubit to 1 or keep it at 0.
    If ``echo`` is set to ``True`` a ECR gate will be played.
    With ``basis`` it is possible to set the measurement basis. If it is not provided
    the default is Z."""

    cr_pulses = []
    cr_target_pulses = []
    sequence = PulseSequence()
    natives_control = platform.natives.single_qubit[control]
    natives_target = platform.natives.single_qubit[target]
    cr_channel = platform.qubits[control].drive_extra[target]
    cr_drive_pulse = Pulse(
        duration=duration,
        amplitude=amplitude,
        relative_phase=phase,
        # envelope=GaussianSquare(rel_sigma=0.2, risefall=15),
        envelope=Rectangular(),
    )
    target_drive_pulse = Pulse(
        duration=duration,
        amplitude=target_amplitude,
        relative_phase=target_phase,
        # envelope=GaussianSquare(rel_sigma=0.2, risefall=15),
        envelope=Rectangular(),
    )
    cr_pulses.append(cr_drive_pulse)
    cr_target_pulses.append(target_drive_pulse)
    control_drive_channel, control_drive_pulse = natives_control.RX()[0]
    target_drive_channel, _ = natives_target.RX()[0]
    ro_channel, ro_pulse = natives_target.MZ()[0]
    ro_channel_control, ro_pulse_control = natives_control.MZ()[0]
    if setup == SetControl.X:
        control_delay = Delay(duration=control_drive_pulse.duration)
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((target_drive_channel, control_delay))
        sequence.append((ro_channel, control_delay))
        sequence.append((ro_channel_control, control_delay))
        sequence.append((cr_channel, control_delay))

    if echo:
        delays = 6 * [Delay(duration=cr_drive_pulse.duration)]
        control_delay = Delay(duration=control_drive_pulse.duration)
        cr_pulse_minus = replace(cr_drive_pulse, relative_phase=np.pi)
        target_pulse_minus = replace(target_drive_pulse, relative_phase=np.pi)
        cr_pulses.append(cr_pulse_minus)
        cr_target_pulses.append(target_pulse_minus)
        sequence.append((cr_channel, cr_drive_pulse))
        sequence.append((control_drive_channel, delays[-1]))
        sequence.append((target_drive_channel, target_drive_pulse))
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((cr_channel, control_delay))
        sequence.append((target_drive_channel, control_delay))
        sequence.append((cr_channel, cr_pulse_minus))
        sequence.append((control_drive_channel, delays[-2]))
        sequence.append((target_drive_channel, target_pulse_minus))
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((target_drive_channel, control_delay))

    else:
        delays = 2 * [Delay(duration=cr_drive_pulse.duration)]
        sequence.append((cr_channel, cr_drive_pulse))
        sequence.append((target_drive_channel, target_drive_pulse))

    if interpolated_sweeper:
        sequence.align(
            [
                cr_channel,
                target_drive_channel,
                control_drive_channel,
                ro_channel,
                ro_channel_control,
            ]
        )
    else:
        sequence.append((ro_channel, delays[0]))
        sequence.append((ro_channel_control, delays[1]))
        if echo:
            sequence.append((ro_channel, delays[2]))
            sequence.append((ro_channel_control, delays[3]))
            sequence.append(
                (ro_channel, Delay(duration=2 * control_drive_pulse.duration))
            )
            sequence.append(
                (ro_channel_control, Delay(duration=2 * control_drive_pulse.duration))
            )

    if basis == Basis.X:
        # H
        sequence.append((target_drive_channel, VirtualZ(phase=np.pi)))
        sequence.append(
            (
                target_drive_channel,
                natives_target.R(theta=np.pi / 2, phi=np.pi / 2)[0][1],
            )
        )
    elif basis == Basis.Y:
        # SDG
        sequence.append((target_drive_channel, VirtualZ(phase=np.pi / 2)))
        # H
        sequence.append((target_drive_channel, VirtualZ(phase=np.pi)))
        sequence.append(
            (
                target_drive_channel,
                natives_target.R(theta=np.pi / 2, phi=np.pi / 2)[0][1],
            )
        )

    target_delay = Delay(
        duration=natives_target.R(theta=np.pi / 2, phi=np.pi / 2)[0][1].duration
    )
    sequence.append((ro_channel, target_delay))
    sequence.append((ro_channel_control, target_delay))
    sequence.append((ro_channel, ro_pulse))
    sequence.append((ro_channel_control, ro_pulse_control))
    return sequence, cr_pulses, cr_target_pulses, delays


def cr_fit(
    data: Union[
        "CrossResonanceLengthData",  # noqa: F821
        "CrossResonanceAmplitudeData",  # noqa: F821
    ],
    fitting_function: Callable,
) -> dict[tuple[QubitId, QubitId, SetControl], list]:
    """Perform fitting on CR data for probabilities.

    We fit oscillations observed in the target qubit. Using a cosine function.
    When on the x axis we change the duration of the CR pulse we include an exponential
    term to address the relaxation time of the qubit.
    """
    fitted_parameters = {}
    for pair in data.pairs:
        for setup in SetControl:
            pair_data = data[pair[0], pair[1], setup]
            pair = (pair[0], pair[1])
            raw_x = pair_data.x
            min_x = np.min(raw_x)
            max_x = np.max(raw_x)
            y = pair_data.prob_target
            x = (raw_x - min_x) / (max_x - min_x)

            period = fallback_period(guess_period(x, y))
            pguess = (
                [0, 0.5, period, 0, 0]
                if fitting_function.__name__ == "fit_length_function"
                else [0, 0.5, period, 0]
            )

            try:
                popt, _, _ = fitting_function(
                    x,
                    y,
                    pguess,
                    sigma=pair_data.error_target,
                    signal=False,
                    x_limits=(min_x, max_x),
                )
                fitted_parameters[pair[0], pair[1], setup] = popt
            except Exception as e:  # pragma: no cover
                log.warning(f"CR fit failed for pair {pair} due to {e}.")
    return fitted_parameters


def cr_plot(
    data: Union[
        "CrossResonanceLengthData",  # noqa: F821
        "CrossResonanceAmplitudeData",  # noqa: F821
    ],
    target: QubitPairId,
    fit: Optional[
        Union[
            "CrossResonanceLengthResults",  # noqa: F821
            "CrossResonanceAmplitudeResults",  # noqa: F821
        ]
    ] = None,
    fitting_function: Optional[Callable] = None,
) -> tuple[list[go.Figure], str]:
    """Plotting function for CR protocols."""
    fig = go.Figure()
    for setup in SetControl:
        target = target if target in data.pairs else (target[1], target[0])
        pair_data = data.data[target[0], target[1], setup]
        fig.add_trace(
            go.Scatter(
                x=pair_data.x,
                y=pair_data.prob_target,
                name=f"Target when Control at {0 if setup is SetControl.Id else 1}",
                showlegend=True,
                legendgroup=f"Target when Control at {0 if setup is SetControl.Id else 1}",
                mode="markers",
                marker=dict(color="blue" if setup is SetControl.Id else "red"),
                error_y=dict(
                    type="data",
                    array=pair_data.error_target,
                    visible=True,
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pair_data.x,
                y=pair_data.prob_control,
                name=f"Control at {0 if setup is SetControl.Id else 1}",
                showlegend=True,
                legendgroup=f"Control at {0 if setup is SetControl.Id else 1}",
                mode="markers",
                marker=dict(color="green" if setup is SetControl.Id else "orange"),
                error_y=dict(
                    type="data",
                    array=pair_data.error_control,
                    visible=True,
                ),
            )
        )
        if fit is not None:
            if (target[0], target[1], setup) in fit.fitted_parameters:
                x = np.linspace(pair_data.x.min(), pair_data.x.max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=fitting_function(
                            x,
                            *fit.fitted_parameters[target[0], target[1], setup],
                        ),
                        name=f"Fit target when control at {0 if setup is SetControl.Id else 1}",
                        showlegend=True,
                        legendgroup=f"Fit target when control at {0 if setup is SetControl.Id else 1}",
                        mode="lines",
                        line=dict(
                            color="blue" if setup is SetControl.Id else "red",
                        ),
                    )
                )

    fig.update_layout(
        yaxis1=dict(range=[-0.1, 1.1]),
        yaxis2=dict(range=[-0.1, 1.1]),
        yaxis3=dict(range=[-0.1, 1.1]),
        height=600,
    )
    return [fig], ""
