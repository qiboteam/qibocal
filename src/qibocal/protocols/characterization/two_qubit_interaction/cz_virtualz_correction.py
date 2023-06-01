from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class CzVirtualZCorrection(Parameters):
    """CzVirtualZCorrection runcard inputs."""

    theta_start: float
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float
    """Step size for the theta sweep in radians."""
    nshots: Optional[int] = None
    """Number of shots per point."""
    relaxation_time: Optional[float] = None
    """Relaxation time."""
    dt: Optional[float] = None
    """Time delay between the two flux pulses if enabled."""


@dataclass
class CzVirtualZCorrectionResults(Results):
    """CzVirtualZCorrection outputs when fitting will be done."""

    amplitude: Dict[List[Tuple], float]


@plot("Landscape 2-qubit gate", plots.landscape_2q_gate)
def tune_landscape(
    platform,
    qubits: dict,
    theta_start,
    theta_end,
    theta_step,
    nshots=1024,
    relaxation_time=None,
    dt=1,
):
    """Check the two-qubit landscape created by a flux pulse of a given duration
    and amplitude.
    The system is initialized with a Y90 pulse on the low frequency qubit and either
    an Id or an X gate on the high frequency qubit. Then the flux pulse is applied to
    the high frequency qubit in order to perform a two-qubit interaction. The Id/X gate
    is undone in the high frequency qubit and a theta90 pulse is applied to the low
    frequency qubit before measurement. That is, a pi-half pulse around the relative phase
    parametereized by the angle theta.

    Measurements on the low frequency qubit yield the the 2Q-phase of the gate and the
    remnant single qubit Z phase aquired during the execution to be corrected.
    Population of the high frequency qubit yield the leakage to the non-computational states
    during the execution of the flux pulse.

    Args:
        platform: platform where the experiment is meant to be run.
        qubit (int): qubit that will interact with center qubit 2.
        theta_start (float): initial angle for the low frequency qubit measurement in radians.
        theta_end (float): final angle for the low frequency qubit measurement in radians.
        theta_step, (float): step size for the theta sweep in radians.
        dt (int): time delay between the two flux pulses if enabled.

    Returns:
        data (DataSet): Measurement data for both the high and low frequency qubits for the two setups of Id/X.

    """
    from qibolab.pulses import PulseSequence

    # TODO: generalize this for more qubits?
    if len(qubits) > 1:
        raise NotImplementedError

    qubit = list(qubits.keys())[0]
    platform.reload_settings()

    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    data = DataUnits(
        name=f"data_q{lowfreq}{highfreq}",
        quantities={
            "theta": "rad",
        },
        options=["target_qubit", "qubit", "setup"],
    )

    for target_qubit, control_qubit in ((lowfreq, highfreq), (highfreq, lowfreq)):
        sequence = PulseSequence()
        Y90_pulse = {}
        RX_pulse_start = {}
        flux_sequence = {}
        theta_pulse = {}
        RX_pulse_end = {}
        measure_target = {}
        measure_control = {}
        start = 0

        for setup in ("I", "X"):
            Y90_pulse[setup] = platform.create_RX90_pulse(
                target_qubit, start=start, relative_phase=np.pi / 2
            )
            RX_pulse_start[setup] = platform.create_RX_pulse(
                control_qubit, start=start, relative_phase=0
            )

            flux_sequence[setup], virtual_z_phase = platform.create_CZ_pulse_sequence(
                (highfreq, lowfreq), start=Y90_pulse[setup].finish
            )

            theta_pulse[setup] = platform.create_RX90_pulse(
                target_qubit,
                start=flux_sequence[setup].finish,
                relative_phase=virtual_z_phase[target_qubit],
            )

            RX_pulse_end[setup] = platform.create_RX_pulse(
                control_qubit,
                start=flux_sequence[setup].finish,
                relative_phase=virtual_z_phase[control_qubit],
            )

            measure_target[setup] = platform.create_qubit_readout_pulse(
                target_qubit, start=theta_pulse[setup].finish
            )
            measure_control[setup] = platform.create_qubit_readout_pulse(
                control_qubit, start=theta_pulse[setup].finish
            )

            sequence.add(
                Y90_pulse[setup],
                flux_sequence[setup],
                theta_pulse[setup],
                measure_target[setup],
                measure_control[setup],
            )
            if setup == "X":
                sequence.add(
                    RX_pulse_start[setup],
                    RX_pulse_end[setup],
                )
            start = sequence.finish + relaxation_time

        thetas = (
            np.arange(theta_start, theta_end, theta_step)
            + virtual_z_phase[target_qubit]
        )
        sweeper = Sweeper(Parameter.relative_phase, thetas, theta_pulse.values())

        results = platform.sweep(
            sequence, sweeper, nshots=nshots, relaxation_time=relaxation_time
        )

        for setup in ("I", "X"):
            result_target = results[measure_target[setup].serial].raw
            result_target.update(
                {
                    "theta[rad]": thetas,
                    "target_qubit": len(thetas) * [target_qubit],
                    "qubit": len(thetas) * [target_qubit],
                    "setup": len(thetas) * [setup],
                }
            )
            data.add_data_from_dict(result_target)

            result_control = results[measure_control[setup].serial].raw
            result_control.update(
                {
                    "theta[rad]": thetas,
                    "target_qubit": len(thetas) * [target_qubit],
                    "qubit": len(thetas) * [control_qubit],
                    "setup": len(thetas) * [setup],
                }
            )
            data.add_data_from_dict(result_control)

    yield data
    yield landscape_fit(
        data=data,
        x="theta[rad]",
        y="MSR[uV]",
        qubits=(lowfreq, highfreq),
        resonator_type=platform.resonator_type,
    )


def landscape_fit(data, x, y, qubits, resonator_type):
    r"""
    Fitting routine for T1 experiment. The used model is

    .. math::

        y = p_0 sin\Big(2 \pi x + p_2\Big) + p_1.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the flipping model
        y (str): name of the output values for the flipping model
        qubit (int): ID qubit number

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2

    """
    data_fit = Data(
        name="fits",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "qubit",
            "setup",
        ],
    )

    for qubit in qubits:
        qubit_data = (
            data.df[(data.df["target_qubit"] == qubit) & (data.df["qubit"] == qubit)]
            .drop(columns=["target_qubit", "qubit"])
            .groupby(["setup", "theta"], as_index=False)
            .mean()
        )
        thetas_keys = parse(x)
        voltages_keys = parse(y)
        thetas = qubit_data[thetas_keys[0]].pint.to(thetas_keys[1]).pint.magnitude
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude
        for setup in ("I", "X"):
            setup_voltages = voltages[qubit_data["setup"] == setup]
            setup_thetas = thetas[qubit_data["setup"] == setup]

            if resonator_type == "3D":
                pguess = [
                    np.max(setup_voltages) - np.min(setup_voltages),
                    np.mean(setup_voltages),
                    3.14,
                ]
            else:
                pguess = [
                    np.max(setup_voltages) - np.min(setup_voltages),
                    np.mean(setup_voltages),
                    3.14,
                ]

            try:
                popt, pcov = curve_fit(
                    landscape,
                    setup_thetas,
                    setup_voltages,
                    p0=pguess,
                    bounds=((0, 0, 0), (2.5e6, 2.5e6, 2 * np.pi)),
                )
                data_fit.add(
                    {
                        "popt0": popt[0],
                        "popt1": popt[1],
                        "popt2": popt[2],
                        "qubit": qubit,
                        "setup": setup,
                    }
                )
            except:
                log.warning("landscape_fit: the fitting was not succesful")
                data_fit.add(
                    {"popt0": 0, "popt1": 0, "popt2": 0, "qubit": qubit, "setup": setup}
                )

    return data_fit


def landscape_2q_gate(folder, routine, qubit, format):
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    subfolder = get_data_subfolders(folder)[0]
    data = DataUnits.load_data(
        folder, subfolder, routine, format, f"data_q{lowfreq}{highfreq}"
    )

    try:
        data_fit = load_data(folder, subfolder, routine, format, "fits")
    except:
        data_fit = Data(
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "qubit",
                "setup",
            ]
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (uV) - Low Frequency",  # TODO: change this to <Z>
            "MSR (uV) - High Frequency",
        ),
    )

    fitting_report = ""
    column = 0
    for qubit in (lowfreq, highfreq):
        filter = (data.df["target_qubit"] == qubit) & (data.df["qubit"] == qubit)
        thetas = data.df[filter]["theta"].unique()
        column += 1
        color = 0
        offset = {}
        for setup in ("I", "X"):
            color += 1
            fig.add_trace(
                go.Scatter(
                    x=data.get_values("theta", "rad")[filter][
                        data.df["setup"] == setup
                    ].to_numpy(),
                    y=data.get_values("MSR", "uV")[filter][
                        data.df["setup"] == setup
                    ].to_numpy(),
                    name=f"q{qubit} {setup} Data",
                    marker_color=get_color(2 * column + color),
                ),
                row=1,
                col=column,
            )

            angle_range = np.linspace(thetas[0], thetas[-1], 100)
            params = data_fit.df[
                (data_fit.df["qubit"] == qubit) & (data_fit.df["setup"] == setup)
            ].to_dict(orient="records")[0]
            if (params["popt0"], params["popt1"], params["popt2"]) != (0, 0, 0):
                fig.add_trace(
                    go.Scatter(
                        x=angle_range,
                        y=landscape(
                            angle_range,
                            float(params["popt0"]),
                            float(params["popt1"]),
                            float(params["popt2"]),
                        ),
                        name=f"q{qubit} {setup} Fit",
                        line=go.scatter.Line(dash="dot"),
                        marker_color=get_color(2 * column + color),
                    ),
                    row=1,
                    col=column,
                )
                offset[setup] = params["popt2"]
                fitting_report += (
                    f"q{qubit} {setup} | offset: {offset[setup]:,.3f} rad<br>"
                )
        if "X" in offset and "I" in offset:
            fitting_report += (
                f"q{qubit} | Z rotation: {offset['X'] - offset['I']:,.3f} rad<br>"
            )
        fitting_report += "<br>"

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="theta (rad)",
        yaxis_title="MSR (uV)",
        xaxis2_title="theta (rad)",
        yaxis2_title="MSR (uV)",
    )

    return [fig], fitting_report


def landscape(x, p0, p1, p2):
    #
    # Amplitude                     : p[0]
    # Offset                        : p[1]
    # Phase offset                  : p[2]
    return np.sin(x + p2) * p0 + p1
