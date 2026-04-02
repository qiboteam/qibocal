from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
import plotly.graph_objects as go
from qibolab import Delay, Platform, Pulse, PulseSequence, Rectangular

from ....auto.operation import QubitId, QubitPairId
from ....config import log
from ....update import replace
from ...utils import angle_wrap, fallback_period, guess_period


class SetControl(str, Enum):
    """Helper to create sequence with control set to X or I."""

    Id = "Id"
    X = "X"


class Basis(str, Enum):
    """Measurement basis."""

    X = "X"
    Y = "Y"
    Z = "Z"


def retrieve_cr_parameters(
    platform: Platform,
    control: QubitId,
    target: QubitId,
) -> tuple[Optional[dict[str, float]], Optional[dict[str, float]]]:
    """Retrieve cross-resonance (CR) pulse parameters from platform calibration.
    This function extracts the CR pulse and its corresponding cancellation pulse
    from the platform's native gates configuration for a given control-target qubit pair.
    """

    cr_params = None
    canc_params = None
    if len(platform.parameters.native_gates.two_qubit[(control, target)].CNOT) != 0:
        for p in platform.parameters.native_gates.two_qubit[(control, target)].CNOT[2:]:
            if (
                cr_params is None
                and str(control) == str(p[0]).split("/")[0]
                and isinstance(p[1], Pulse)
            ):
                cr_params = {
                    "amplitude": p[1].amplitude,
                    "duration": p[1].duration,
                    "relative_phase": p[1].relative_phase,
                }
            if (
                canc_params is None
                and str(target) == str(p[0]).split("/")[0]
                and isinstance(p[1], Pulse)
            ):
                canc_params = {
                    "amplitude": p[1].amplitude,
                    "duration": p[1].duration,
                    "relative_phase": p[1].relative_phase,
                }

            if cr_params is not None and canc_params is not None:
                # here we identified both the CR and cancellation pulse,
                # we can stop looking at the remaining pulses
                break

    return cr_params, canc_params


def cross_res_sequence(
    platform: Platform,
    control: QubitId,
    target: QubitId,
    duration: int,
    control_amplitude: float,
    control_phase: float,
    target_amplitude: float,
    target_phase: float,
    echo: bool,
    interpolated_sweeper: bool = False,
) -> tuple[PulseSequence, list[Pulse], list[Pulse], list[Delay]]:
    """Creates sequence for CR experiment on ``control`` and ``target`` qubits.

    With ``setup`` it is possible to set the control qubit to 1 or keep it at 0.
    If ``echo`` is set to ``True`` a ECR gate will be played.
    With ``basis`` it is possible to set the measurement basis. If it is not provided
    the default is Z."""

    cr_control_pulses = []
    cr_target_pulses = []
    # delays introduced by the cross resonance sequence
    cr_delays = []

    sequence = PulseSequence()

    natives_control = platform.natives.single_qubit[control]
    natives_target = platform.natives.single_qubit[target]

    # qubit channels necessary for the cross resonance sequence
    cr_channel = platform.qubits[control].drive_extra[target]
    control_drive_channel, control_drive_pulse = natives_control.RX()[0]
    target_drive_channel, _ = natives_target.RX()[0]

    cr_drive_pulse = Pulse(
        duration=duration,
        amplitude=control_amplitude,
        relative_phase=angle_wrap(control_phase),
        envelope=Rectangular(),
    )
    target_drive_pulse = Pulse(
        duration=duration,
        amplitude=target_amplitude,
        relative_phase=angle_wrap(target_phase),
        envelope=Rectangular(),
    )
    cr_control_pulses.append(cr_drive_pulse)
    cr_target_pulses.append(target_drive_pulse)

    # delays introduced by cross resonance pulses
    cr_delays.append(Delay(duration=cr_drive_pulse.duration))
    # first cross resonance pulse
    sequence.append((cr_channel, cr_drive_pulse))
    sequence.append((target_drive_channel, target_drive_pulse))
    if interpolated_sweeper:
        _ = sequence.align([control_drive_channel, cr_channel, target_drive_channel])
    else:
        sequence.append((control_drive_channel, cr_delays[-1]))

    if echo:
        # phase-flipped cross resonance pulses
        cr_drive_pulse_flipped = replace(
            cr_drive_pulse.new(),
            relative_phase=angle_wrap(np.pi + cr_drive_pulse.relative_phase),
        )
        cr_control_pulses.append(cr_drive_pulse_flipped)

        target_pulse_flipped = replace(
            target_drive_pulse.new(),
            relative_phase=angle_wrap(np.pi + target_drive_pulse.relative_phase),
        )
        cr_target_pulses.append(target_pulse_flipped)

        # delay introduced by echo sequence
        echo_delay = Delay(duration=control_drive_pulse.duration)

        # first echo pulse
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((cr_channel, echo_delay))
        sequence.append((target_drive_channel, echo_delay))

        # delays introduced by cross resonance pulses
        cr_delays.append(Delay(duration=cr_drive_pulse.duration))
        # second cross resonance pulse with flipped phase
        sequence.append((cr_channel, cr_drive_pulse_flipped))
        sequence.append((target_drive_channel, target_pulse_flipped))
        if interpolated_sweeper:
            _ = sequence.align(
                [control_drive_channel, cr_channel, target_drive_channel]
            )
        else:
            sequence.append((control_drive_channel, cr_delays[-1]))

        # second echo pulse
        sequence.append((control_drive_channel, control_drive_pulse))
        sequence.append((cr_channel, echo_delay))
        sequence.append((target_drive_channel, echo_delay))

    return sequence, cr_control_pulses, cr_target_pulses, cr_delays


def appending_ro_sequence(
    platform: Platform,
    control: QubitId,
    target: QubitId,
    exp_sequence: PulseSequence,
    basis: Basis,
    interpolated_sweeper: bool,
) -> PulseSequence:
    """Append a readout pulse sequence for two qubits with a specified delay.

    This function constructs a pulse sequence that applies readout operations to both
    control and target qubits with an initial delay on each readout channel.
    """
    natives_control = platform.natives.single_qubit[control]
    natives_target = platform.natives.single_qubit[target]

    # platform acquisition channels and pulses for control and target qubits
    ro_channel_control, ro_pulse_control = natives_control.MZ()[0]
    ro_channel_target, ro_pulse_target = natives_target.MZ()[0]

    # switching measurement basis for target line and align all the others
    target_drive_channel, _ = natives_target.RX()[0]
    # delay
    ro_delays = 2 * [Delay(duration=exp_sequence.duration)]

    if interpolated_sweeper:
        # if using interpolated_sweepers I need all the channels to be aligned
        control_drive_channel, _ = natives_control.RX()[0]
        cr_channel = platform.qubits[control].drive_extra[target]

        # align all the channels of the sequence
        _ = exp_sequence.align(
            [
                cr_channel,
                target_drive_channel,
                control_drive_channel,
                ro_channel_control,
                ro_channel_target,
            ]
        )
    else:
        # switching measurement basis for target line and align all the others
        target_drive_channel, _ = natives_target.RX()[0]

        # delay to align the readout pulses of control and target qubits
        # done in every case, even when measurinz Z-basis (no additional pulse on target)
        # to have the same timing for all the measurements
        target_delay = Delay(
            duration=natives_target.R(theta=3 * np.pi / 2, phi=np.pi / 2)[0][1].duration
        )
        exp_sequence.append((ro_channel_control, target_delay))
        exp_sequence.append((ro_channel_target, target_delay))

        # wait the whole cr-sequence duration before starting the readout pulses
        exp_sequence.append((ro_channel_target, ro_delays[0]))
        exp_sequence.append((ro_channel_control, ro_delays[1]))

    if basis == Basis.X:
        exp_sequence.append(
            (
                target_drive_channel,
                natives_target.R(theta=np.pi / 2, phi=np.pi / 2)[0][1],
            )
        )
    elif basis == Basis.Y:
        exp_sequence.append(
            (
                target_drive_channel,
                natives_target.R(theta=np.pi / 2, phi=0)[0][1],
            )
        )

    # adding readout pulses
    exp_sequence.append((ro_channel_target, ro_pulse_target))
    exp_sequence.append((ro_channel_control, ro_pulse_control))

    return exp_sequence, ro_delays


def cross_resonance_experiment(
    platform: Platform,
    control: QubitId,
    target: QubitId,
    duration: int,
    control_amplitude: float,
    control_phase: float,
    target_amplitude: float,
    target_phase: float,
    basis: Basis = Basis.Z,
    setup: SetControl = SetControl.Id,
    echo: bool = False,
    interpolated_sweeper: bool = False,
) -> tuple[PulseSequence, list[Pulse], list[Pulse], list[Delay], list[Delay]]:
    """Creates sequence for CR experiment on ``control`` and ``target`` qubits.

    With ``setup`` it is possible to set the control qubit to 1 or keep it at 0.
    If ``echo`` is set to ``True`` a ECR gate will be played.
    With ``basis`` it is possible to set the measurement basis. If it is not provided
    the default is Z."""

    # adding pi-pulse if we want to set control to 1
    cntl_setup_sequence = PulseSequence()
    if setup == SetControl.X:
        control_drive_channel, control_drive_pulse = platform.natives.single_qubit[
            control
        ].RX()[0]
        target_drive_channel, _ = platform.natives.single_qubit[target].RX()[0]
        cr_channel = platform.qubits[control].drive_extra[target]

        control_delay = Delay(duration=control_drive_pulse.duration)

        cntl_setup_sequence.append((control_drive_channel, control_drive_pulse))
        cntl_setup_sequence.append((target_drive_channel, control_delay))
        cntl_setup_sequence.append((cr_channel, control_delay))

    cr_sequence, cr_pulses, cr_target_pulses, cr_delays = cross_res_sequence(
        platform=platform,
        control=control,
        target=target,
        amplitude=control_amplitude,
        duration=duration,
        phase=control_phase,
        target_amplitude=target_amplitude,
        target_phase=target_phase,
        echo=echo,
        setup=setup,
        interpolated_sweeper=interpolated_sweeper,
    )
    cr_sequence = cntl_setup_sequence | cr_sequence

    total_sequence, ro_delays = appending_ro_sequence(
        platform=platform,
        control=control,
        target=target,
        exp_sequence=cr_sequence,
        basis=basis,
        interpolated_sweeper=interpolated_sweeper,
    )

    return total_sequence, cr_pulses, cr_target_pulses, cr_delays, ro_delays


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
