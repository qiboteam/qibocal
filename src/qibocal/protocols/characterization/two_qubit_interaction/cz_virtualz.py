"""CZ virtual correction experiment for two qubit gates, tune landscape"""
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo.config import log
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import Pulse, PulseSequence
from qibolab.qubits import Qubit, QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits
from qibocal.protocols.characterization.two_qubit_interaction.chevron import order_pairs


@dataclass
class CZVirtualZParameters(Parameters):
    """CzVirtualZ runcard inputs."""

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
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class CZVirtualZResults(Results):
    """CzVirtualZ outputs when fitting will be done."""

    amplitude: dict[Union[str, int], float]
    """Amplitude of the fit."""
    offset: dict[Union[str, int], float]
    """Offset of the fit."""
    phase_offset: dict[Union[str, int], float]
    """Phase offset of the fit."""
    setup: dict[Union[str, int], str]
    """Setup of the fit."""


class CZVirtualZData(DataUnits):
    """CZVirtualZ data."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={
                "theta": "rad",
            },
            options=["probability", "target_qubit", "qubit", "setup", "pair"],
        )


def create_sequence(
    platform: Platform,
    setup: str,
    target_qubit: QubitId,
    control_qubit: QubitId,
    ord_pair: list[QubitId, QubitId],
    parking: bool,
) -> tuple[
    PulseSequence, dict[QubitId, Pulse], dict[QubitId, Pulse], dict[QubitId, Pulse]
]:
    """Create the experiment PulseSequence"""
    lowfreq = ord_pair[0]
    highfreq = ord_pair[1]

    sequence = PulseSequence()

    Y90_pulse = platform.create_RX90_pulse(
        target_qubit, start=0, relative_phase=np.pi / 2
    )
    RX_pulse_start = platform.create_RX_pulse(control_qubit, start=0, relative_phase=0)

    flux_sequence, virtual_z_phase = platform.create_CZ_pulse_sequence(
        (highfreq, lowfreq),
        start=max(Y90_pulse.finish, RX_pulse_start.finish),
    )

    theta_pulse = platform.create_RX90_pulse(
        target_qubit,
        start=flux_sequence.finish,
        relative_phase=virtual_z_phase[target_qubit],
    )

    RX_pulse_end = platform.create_RX_pulse(
        control_qubit,
        start=flux_sequence.finish,
        relative_phase=virtual_z_phase[control_qubit],
    )

    measure_target = platform.create_qubit_readout_pulse(
        target_qubit, start=theta_pulse.finish
    )
    measure_control = platform.create_qubit_readout_pulse(
        control_qubit, start=theta_pulse.finish
    )

    sequence.add(
        Y90_pulse,
        flux_sequence,
        theta_pulse,
        measure_target,
        # measure_control,
    )
    if setup == "X":
        sequence.add(
            RX_pulse_start,
            RX_pulse_end,
        )

    if parking:
        # if parking is true, create a cz pulse from the runcard and
        # add to the sequence all parking pulses
        cz_sequence, _ = platform.pairs[
            tuple(sorted([target_qubit, control_qubit]))
        ].native_gates.CZ.sequence(start=0)
        for pulse in cz_sequence:
            if pulse.qubit not in {target_qubit, control_qubit}:
                pulse.start = flux_sequence[setup].start
                pulse.duration = flux_sequence[setup].duration
                sequence.add(pulse)

    return sequence, measure_target, virtual_z_phase, theta_pulse


def _acquisition(
    params: CZVirtualZParameters,
    platform: Platform,
    qubits: Qubits,
) -> CZVirtualZData:
    r"""
    Acquisition for CZVirtualZ.

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
    # create a DataUnits object to store the results,
    data = CZVirtualZData()

    for pair in qubits:
        # order the qubits so that the low frequency one is the first
        ord_pair = order_pairs(pair, platform.qubits)

        for target_q, control_q in (
            (ord_pair[0], ord_pair[1]),
            (ord_pair[1], ord_pair[0]),
        ):
            for setup in ("I", "X"):
                (
                    sequence,
                    measure_target,
                    virtual_z_phase,
                    theta_pulse,
                ) = create_sequence(
                    platform, setup, target_q, control_q, ord_pair, params.parking
                )

                thetas = (
                    np.arange(params.theta_start, params.theta_end, params.theta_step)
                    + virtual_z_phase[target_q]
                )
                sweeper = Sweeper(
                    Parameter.relative_phase,
                    thetas,
                    pulses=[theta_pulse],
                    type=SweeperType.ABSOLUTE,
                )
                results = platform.sweep(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        acquisition_type=AcquisitionType.DISCRIMINATION,
                        averaging_mode=AveragingMode.CYCLIC,
                    ),
                    sweeper,
                )

                result = results[measure_target.serial]
                prob = result.statistical_frequency

                r = {
                    "probability": prob.flatten(),
                    "theta[rad]": thetas,
                    "target_qubit": len(thetas) * [target_q],
                    "qubit": len(thetas) * [target_q],
                    "setup": len(thetas) * [setup],
                    "pair": len(thetas) * [str(pair)],
                }
                data.add_data_from_dict(r)

    return data


def fit_function(x, p0, p1, p2):
    # TODO maybe x=2pi*x
    return np.sin(x + p2) * p0 + p1


def _fit(
    data: CZVirtualZData,
) -> CZVirtualZResults:
    r"""
    Fitting routine for the experiment. The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    amplitude = {}
    offset = {}
    phase_offset = {}
    setups = {}

    pairs = data.df["pair"].unique()
    for pair in pairs:
        data_pair = data.df[data.df["pair"] == pair]
        qubits = data_pair["qubit"].unique()

        for qubit in qubits:
            amplitude[(qubit, pair)] = []
            offset[(qubit, pair)] = []
            phase_offset[(qubit, pair)] = []
            setups[(qubit, pair)] = []

        for qubit in qubits:
            qubit_data = (
                data_pair[
                    (data_pair["target_qubit"] == qubit) & (data_pair["qubit"] == qubit)
                ]
                .drop(columns=["target_qubit", "qubit"])
                .groupby(["setup", "theta", "probability"], as_index=False)
                .mean()
            )

            thetas = qubit_data["theta"].pint.to("rad").pint.magnitude
            probabilities = qubit_data["probability"]

            for setup in ("I", "X"):
                setup_probabilities = probabilities[qubit_data["setup"] == setup]
                setup_thetas = thetas[qubit_data["setup"] == setup]

                pguess = [
                    np.max(setup_probabilities) - np.min(setup_probabilities),
                    np.mean(setup_probabilities),
                    3.14,
                ]

                try:
                    popt, pcov = curve_fit(
                        fit_function,
                        setup_thetas,
                        setup_probabilities,
                        p0=pguess,
                        bounds=((0, 0, 0), (2.5e6, 2.5e6, 2 * np.pi)),
                    )
                    amplitude[(qubit, pair)] += [popt[0]]
                    offset[(qubit, pair)] += [popt[1]]
                    phase_offset[(qubit, pair)] += [popt[2]]
                    setups[(qubit, pair)] += [setup]
                except:
                    log.warning("landscape_fit: the fitting was not succesful")
                    amplitude[(qubit, pair)] += [None]
                    offset[(qubit, pair)] += [None]
                    phase_offset[(qubit, pair)] += [None]
                    setups[(qubit, pair)] += [setup]

    return CZVirtualZResults(
        amplitude=amplitude,
        offset=offset,
        phase_offset=phase_offset,
        setup=setups,
    )


def _plot(data: CZVirtualZData, data_fit: CZVirtualZResults, qubits):
    r"""
    Plot routine for CZVirtualZ.
    """
    qubits = tuple(qubits)
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "P(0) - Low Frequency",
            "P(0) - High Frequency",
        ),
    )

    fitting_report = ""
    column = 0
    for qubit in qubits:
        df_filter = (
            (data.df["target_qubit"] == qubit)
            & (data.df["qubit"] == qubit)
            & (data.df["pair"] == str(qubits))
        )
        thetas = data.df[df_filter]["theta"].pint.to("rad").pint.magnitude.unique()
        column += 1
        offset = {}
        for setup in ("I", "X"):
            fig.add_trace(
                go.Scatter(
                    x=data.get_values("theta", "rad")[df_filter][
                        data.df["setup"] == setup
                    ].to_numpy(),
                    y=data.df[df_filter][data.df["setup"] == setup]["probability"],
                    name=f"q{qubit} {setup} Data",
                ),
                row=1,
                col=column,
            )

            angle_range = np.linspace(thetas[0], thetas[-1], 100)
            if (
                (
                    data_fit.amplitude[(qubit, str(qubits))][
                        data_fit.setup[(qubit, str(qubits))].index(setup)
                    ]
                    is not None
                )
                and (
                    data_fit.offset[(qubit, str(qubits))][
                        data_fit.setup[(qubit, str(qubits))].index(setup)
                    ]
                    is not None
                )
                and (
                    data_fit.phase_offset[(qubit, str(qubits))][
                        data_fit.setup[(qubit, str(qubits))].index(setup)
                    ]
                    is not None
                )
            ):
                fig.add_trace(
                    go.Scatter(
                        x=angle_range,
                        y=fit_function(
                            angle_range,
                            data_fit.amplitude[(qubit, str(qubits))][
                                data_fit.setup[(qubit, str(qubits))].index(setup)
                            ],
                            data_fit.offset[(qubit, str(qubits))][
                                data_fit.setup[(qubit, str(qubits))].index(setup)
                            ],
                            data_fit.phase_offset[(qubit, str(qubits))][
                                data_fit.setup[(qubit, str(qubits))].index(setup)
                            ],
                        ),
                        name=f"q{qubit} {setup} pair {qubits} Fit",
                        line=go.scatter.Line(dash="dot"),
                    ),
                    row=1,
                    col=column,
                )
                offset[setup] = data_fit.offset[(qubit, str(qubits))][
                    data_fit.setup[(qubit, str(qubits))].index(setup)
                ]
                fitting_report += (
                    f"q{qubit} {setup} {qubits}| offset: {offset[setup]:,.3f} rad<br>"
                )
        if "X" in offset and "I" in offset:
            fitting_report += f"q{qubit} pair {qubits} |Z rotation: {data_fit.offset[(qubit, str(qubits))][data_fit.setup[(qubit, str(qubits))].index('X')] - data_fit.offset[(qubit, str(qubits))][data_fit.setup[(qubit, str(qubits))].index('I')]:,.3f} rad<br>"
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


cz_virtualz = Routine(_acquisition, _fit, _plot)
"""CZ virtual Z correction routine."""
