from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo.config import log, raise_error
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.sweeper import Parameter, Sweeper
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from .utils import landscape, parse


@dataclass
class CzVirtualZCorrectionParameters(Parameters):
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

    amplitude: Dict[Union[str, int], float]
    """Amplitude of the fit."""
    offset: Dict[Union[str, int], float]
    """Offset of the fit."""
    phase_offset: Dict[Union[str, int], float]
    """Phase offset of the fit."""
    setup: Dict[Union[str, int], str]
    """Setup of the fit."""


class CzVirtualZCorrectionData(DataUnits):
    """CzVirtualZCorrection data."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={
                "theta": "rad",
            },
            options=["target_qubit", "qubit", "setup"],
        )


def _acquisition(
    params: CzVirtualZCorrectionParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> CzVirtualZCorrectionData:
    r"""
    Acquisition for CzVirtualZCorrection.

    Check the two-qubit landscape created by a flux pulse of a given duration
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
    """

    # FIXME: make general for multiple qubits
    if 2 in qubits:
        qubits.pop(2)
    lowfreq = list(qubits.values())[0].name
    highfreq = 2

    data = CzVirtualZCorrectionData()

    for target_qubit, control_qubit in ((lowfreq, highfreq), (highfreq, lowfreq)):
        sequence = {}
        sweeper = {}
        Y90_pulse = {}
        RX_pulse_start = {}
        flux_sequence = {}
        theta_pulse = {}
        RX_pulse_end = {}
        measure_target = {}
        measure_control = {}
        start = 0

        for setup in ("I", "X"):
            sequence[setup] = PulseSequence()

            Y90_pulse[setup] = platform.create_RX90_pulse(
                target_qubit, start=start, relative_phase=np.pi / 2
            )
            RX_pulse_start[setup] = platform.create_RX_pulse(
                control_qubit, start=start, relative_phase=0
            )

            flux_sequence[setup], virtual_z_phase = platform.create_CZ_pulse_sequence(
                (highfreq, lowfreq),
                start=max(Y90_pulse[setup].finish, RX_pulse_start[setup].finish),
            )

            theta_pulse[setup] = platform.create_RX90_pulse(
                target_qubit,
                start=flux_sequence[setup].finish,
                # relative_phase=virtual_z_phase[target_qubit],
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

            sequence[setup].add(
                Y90_pulse[setup],
                flux_sequence[setup],
                theta_pulse[setup],
                measure_target[setup],
                # measure_control[setup],
            )
            if setup == "X":
                sequence[setup].add(
                    RX_pulse_start[setup],
                    RX_pulse_end[setup],
                )

            thetas = (
                np.arange(params.theta_start, params.theta_end, params.theta_step)
                + virtual_z_phase[target_qubit]
            )
            sweeper = Sweeper(
                Parameter.relative_phase, thetas, pulses=[theta_pulse[setup]]
            )

            results = platform.sweep(
                sequence[setup],
                ExecutionParameters(
                    nshots=params.nshots,
                    averaging_mode=AveragingMode.CYCLIC,
                    acquisition_type=AcquisitionType.INTEGRATION,
                ),
                sweeper,
            )
            result_target = results[measure_target[setup].serial].serialize
            result_target["MSR[V]"] = np.abs(
                results[measure_target[setup].serial].voltage_i
                + 1j * results[measure_target[setup].serial].voltage_q
                - complex(platform.qubits[measure_target[setup].qubit].mean_gnd_states)
            ) / np.abs(
                complex(platform.qubits[measure_target[setup].qubit].mean_exc_states)
                - complex(platform.qubits[measure_target[setup].qubit].mean_gnd_states)
            )
            result_target.update(
                {
                    "theta[rad]": thetas,
                    "target_qubit": len(thetas) * [target_qubit],
                    "qubit": len(thetas) * [target_qubit],
                    "setup": len(thetas) * [setup],
                }
            )
            data.add_data_from_dict(result_target)

            # result_control = results[measure_control[setup].serial].serialize
            # result_control["MSR[V]"] = np.abs(
            #     results[measure_control[setup].serial].voltage_i
            #     + 1j * results[measure_control[setup].serial].voltage_q
            #     - complex(platform.qubits[measure_control[setup].qubit].mean_gnd_states)
            # ) / np.abs(
            #     complex(platform.qubits[measure_control[setup].qubit].mean_exc_states)
            #     - complex(platform.qubits[measure_control[setup].qubit].mean_gnd_states)
            # )
            # result_control.update(
            #     {
            #         "theta[rad]": thetas,
            #         "target_qubit": len(thetas) * [target_qubit],
            #         "qubit": len(thetas) * [control_qubit],
            #         "setup": len(thetas) * [setup],
            #     }
            # )
            # data.add_data_from_dict(result_control)

    return data


def _fit(
    data: CzVirtualZCorrectionData,
) -> CzVirtualZCorrectionResults:
    r"""
    Fitting routine for T1 experiment. The used model is

    .. math::

        y = p_0 sin\Big(2 \pi x + p_2\Big) + p_1.
    """
    qubits = data.df["qubit"].unique()

    amplitude = {qubit: [] for qubit in qubits}
    offset = {qubit: [] for qubit in qubits}
    phase_offset = {qubit: [] for qubit in qubits}
    setups = {qubit: [] for qubit in qubits}

    for qubit in qubits:
        qubit_data = (
            data.df[(data.df["target_qubit"] == qubit) & (data.df["qubit"] == qubit)]
            .drop(columns=["target_qubit", "qubit"])
            .groupby(["setup", "theta"], as_index=False)
            .mean()
        )
        thetas_keys = parse("theta[rad]")
        voltages_keys = parse("MSR[V]")

        thetas = qubit_data[thetas_keys[0]].pint.to(thetas_keys[1]).pint.magnitude
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude
        for setup in ("I", "X"):
            setup_voltages = voltages[qubit_data["setup"] == setup]
            setup_thetas = thetas[qubit_data["setup"] == setup]

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
                amplitude[qubit] += [popt[0]]
                offset[qubit] += [popt[1]]
                phase_offset[qubit] += [popt[2]]
                setups[qubit] += [setup]
            except:
                log.warning("landscape_fit: the fitting was not succesful")
                print(setup)
                amplitude[qubit] += [None]
                offset[qubit] += [None]
                phase_offset[qubit] += [None]
                setups[qubit] += [setup]

    return CzVirtualZCorrectionResults(
        amplitude=amplitude,
        offset=offset,
        phase_offset=phase_offset,
        setup=setups,
    )


def _plot(data: CzVirtualZCorrectionData, data_fit: CzVirtualZCorrectionResults, qubit):
    r"""
    Plot routine for CzVirtualZCorrection.
    """

    highfreq = 2
    lowfreq = qubit

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
        thetas = data.df[filter]["theta"].pint.to("rad").pint.magnitude.unique()
        column += 1
        offset = {}
        for setup in ("I", "X"):
            fig.add_trace(
                go.Scatter(
                    x=data.get_values("theta", "rad")[filter][
                        data.df["setup"] == setup
                    ].to_numpy(),
                    y=data.get_values("MSR", "V")[filter][
                        data.df["setup"] == setup
                    ].to_numpy(),
                    name=f"q{qubit} {setup} Data",
                ),
                row=1,
                col=column,
            )

            angle_range = np.linspace(thetas[0], thetas[-1], 100)
            if (
                (
                    data_fit.amplitude[qubit][data_fit.setup[qubit].index(setup)]
                    is not None
                )
                and (
                    data_fit.offset[qubit][data_fit.setup[qubit].index(setup)]
                    is not None
                )
                and (
                    data_fit.phase_offset[qubit][data_fit.setup[qubit].index(setup)]
                    is not None
                )
            ):
                fig.add_trace(
                    go.Scatter(
                        x=angle_range,
                        y=landscape(
                            angle_range,
                            data_fit.amplitude[qubit][
                                data_fit.setup[qubit].index(setup)
                            ],
                            data_fit.offset[qubit][data_fit.setup[qubit].index(setup)],
                            data_fit.phase_offset[qubit][
                                data_fit.setup[qubit].index(setup)
                            ],
                        ),
                        name=f"q{qubit} {setup} Fit",
                        line=go.scatter.Line(dash="dot"),
                    ),
                    row=1,
                    col=column,
                )
                offset[setup] = data_fit.offset[qubit][
                    data_fit.setup[qubit].index(setup)
                ]
                fitting_report += (
                    f"q{qubit} {setup} | offset: {offset[setup]:,.3f} rad<br>"
                )
        if "X" in offset and "I" in offset:
            fitting_report += f"q{qubit} | Z rotation: {data_fit.offset[qubit][data_fit.setup[qubit].index('X')] - data_fit.offset[qubit][data_fit.setup[qubit].index('I')]:,.3f} rad<br>"
        fitting_report += "<br>"

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="theta (rad)",
        yaxis_title="Probability",
        xaxis2_title="theta (rad)",
        yaxis2_title="Probability",
    )

    return [fig], fitting_report


cz_virtualz_correction = Routine(_acquisition, _fit, _plot)
"""CZ virtual Z correction routine."""
