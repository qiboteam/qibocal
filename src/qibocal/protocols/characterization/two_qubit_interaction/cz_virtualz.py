"""CZ virtual correction experiment for two qubit gates, tune landscape."""
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import Pulse, PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
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


CZVirtualZType = np.dtype([("target", np.float64), ("control", np.float64)])


@dataclass
class CZVirtualZData(Data):
    """CZVirtualZ data."""

    data: dict[tuple[QubitId, QubitId, str], npt.NDArray[CZVirtualZType]] = field(
        default_factory=dict
    )

    thetas: list = field(default_factory=list)

    def register_qubit(self, target, control, setup, prob_target, prob_control):
        ar = np.empty(prob_target.shape, dtype=CZVirtualZType)
        ar["target"] = prob_target
        ar["control"] = prob_control
        self.data[target, control, setup] = np.rec.array(ar)


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
    """Create the experiment PulseSequence."""
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
        start=flux_sequence.finish + 20,
        relative_phase=virtual_z_phase[target_qubit],
    )
    RX_pulse_end = platform.create_RX_pulse(
        control_qubit,
        start=flux_sequence.finish + 20,
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
        measure_control,
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

    return sequence, virtual_z_phase, theta_pulse


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
    if not isinstance(list(qubits.keys())[0], tuple):
        raise ValueError("You need to specify a list of pairs.")

    # create a DataUnits object to store the results,

    thetas = np.arange(params.theta_start, params.theta_end, params.theta_step)

    data = CZVirtualZData(thetas=thetas.tolist())
    for pair in qubits:
        # order the qubits so that the low frequency one is the first
        ord_pair = order_pairs(pair, platform.qubits)

        for target_q, control_q in (
            (ord_pair[0], ord_pair[1]),
            (ord_pair[1], ord_pair[0]),
        ):
            for setup in ("I", "X"):
                theta = np.arange(
                    params.theta_start, params.theta_end, params.theta_step
                )

                (
                    sequence,
                    virtual_z_phase,
                    theta_pulse,
                ) = create_sequence(
                    platform, setup, target_q, control_q, ord_pair, params.parking
                )

                theta += virtual_z_phase[target_q]
                sweeper = Sweeper(
                    Parameter.relative_phase,
                    theta,
                    pulses=[theta_pulse],
                    type=SweeperType.ABSOLUTE,
                )
                results = platform.sweep(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        acquisition_type=AcquisitionType.INTEGRATION,
                        averaging_mode=AveragingMode.CYCLIC,
                    ),
                    sweeper,
                )

                result_target = results[target_q].magnitude
                result_control = results[control_q].magnitude

                data.register_qubit(
                    target=target_q,
                    control=control_q,
                    setup=setup,
                    prob_target=result_target,
                    prob_control=result_control,
                )

    return data


def fit_function(x, p0, p1, p2):
    """Sinusoidal fit function."""
    # return p0 + p1 * np.sin(2*np.pi*p2 * x + p3)
    return np.sin(x + p2) * p0 + p1


def _fit(
    data: CZVirtualZData,
) -> CZVirtualZResults:
    r"""Fitting routine for the experiment.

    The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    amplitude = {}
    offset = {}
    phase_offset = {}
    setups = {}
    pairs = data.pairs

    # for target, control, setup in data:

    #     amplitude[str(qubit_pair)] = []
    #     offset[str(qubit_pair)] = []
    #     phase_offset[str(qubit_pair)] = []
    #     setups[str(qubit_pair)] = []

    #     # for qubit in pair:
    #     # qubit_data = (
    #     #     data_pair[
    #     #         (data_pair["target_qubit"] == qubit) & (data_pair["qubit"] == qubit)
    #     #     ]
    #     #     .drop(columns=["target_qubit", "qubit"])
    #     #     .groupby(["setup", "theta", "probability"], as_index=False)
    #     #     .mean()
    #     # )

    #     # thetas = qubit_data["theta"].pint.to("rad").pint.magnitude
    #     # probabilities = qubit_data["probability"]

    #     # for setup in ("I", "X"):
    #     #     print(pair[0], pair[1], qubit, setup)
    #     prob_target = data[target, control, setup][0]
    #     prob_control = data[target, control, setup][1]
    #     pguess = [
    #         np.max(prob_target) - np.min(prob_target),
    #         np.mean(prob_target),
    #         3.14,
    #     ]

    #     try:
    #         popt, _ = curve_fit(
    #             fit_function,
    #             thetas,
    #             prob,
    #             p0=pguess,
    #             bounds=((0, 0, 0), (2.5e6, 2.5e6, 2 * np.pi)),
    #         )
    #         amplitude[str(qubit_pair)] += [popt[0]]
    #         offset[str(qubit_pair)] += [popt[1]]
    #         phase_offset[str(qubit_pair)] += [popt[2]]
    #         setups[str(qubit_pair)] += [index[3]]
    #     except:
    #         log.warning("landscape_fit: the fitting was not succesful")
    #         amplitude[str(qubit_pair)] += [None]
    #         offset[str(qubit_pair)] += [None]
    #         phase_offset[str(qubit_pair)] += [None]
    #         setups[str(qubit_pair)] += [index[3]]

    return CZVirtualZResults(
        amplitude=amplitude,
        offset=offset,
        phase_offset=phase_offset,
        setup=setups,
    )


def _plot(data: CZVirtualZData, data_fit: CZVirtualZResults, qubits):
    """Plot routine for CZVirtualZ."""
    qubits = tuple(qubits)

    fig1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {qubits[0]}",
            f"Qubit {qubits[1]}",
        ),
    )

    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {qubits[0]}",
            f"Qubit {qubits[1]}",
        ),
    )

    fitting_report = "No fitting data"
    thetas = data.thetas
    for target, control, setup in data.data:
        target_prob = data[target, control, setup].target
        control_prob = data[target, control, setup].control
        fig = fig1 if (target, control) == qubits else fig2
        fig.add_trace(
            go.Scatter(
                x=thetas, y=target_prob, name=f"{setup} sequence", legendgroup=setup
            ),
            row=1,
            col=1 if fig == fig1 else 2,
        )

        fig.add_trace(
            go.Scatter(
                x=thetas, y=control_prob, name=f"{setup} sequence", legendgroup=setup
            ),
            row=1,
            col=2 if fig == fig1 else 1,
        )

        # angle_range = np.linspace(thetas[0], thetas[-1], 100)
        # qubit_pair = measure, target, control
        # if (
        #     (
        #         data_fit.amplitude[str(qubit_pair)][
        #             data_fit.setup[str(qubit_pair)].index(setup)
        #         ]
        #         is not None
        #     )
        #     and (
        #         data_fit.offset[str(qubit_pair)][
        #             data_fit.setup[str(qubit_pair)].index(setup)
        #         ]
        #         is not None
        #     )
        #     and (
        #         data_fit.phase_offset[str(qubit_pair)][
        #             data_fit.setup[str(qubit_pair)].index(setup)
        #         ]
        #         is not None
        #     )
        # ):
        #     fig.add_trace(
        #         go.Scatter(
        #             x=angle_range,
        #             y=fit_function(
        #                 angle_range,
        #                 data_fit.amplitude[str(qubit_pair)][
        #                     data_fit.setup[str(qubit_pair)].index(setup)
        #                 ],
        #                 data_fit.offset[str(qubit_pair)][
        #                     data_fit.setup[str(qubit_pair)].index(setup)
        #                 ],
        #                 data_fit.phase_offset[str(qubit_pair)][
        #                     data_fit.setup[str(qubit_pair)].index(setup)
        #                 ],
        #             ),
        #             name=f"q{measure} {setup} pair {qubits} Fit",
        #             line=go.scatter.Line(dash="dot"),
        #         ),
        #         row=1,
        #         col=1 if ([target, control] == pair).all() else 2,
        #     )
        #     offset[setup] = data_fit.offset[str(qubit_pair)][
        #         data_fit.setup[str(qubit_pair)].index(setup)
        #     ]
        #     fitting_report += (
        #         f"q{measure} {setup} {qubits}| offset: {offset[setup]:,.3f} rad<br>"
        #     )
        # if "X" in offset and "I" in offset:
        #     fitting_report += f"q{measure} pair {qubits} |"
        #     ang_X = data_fit.offset[str(qubit_pair)][
        #         data_fit.setup[str(qubit_pair)].index("X")
        #     ]
        #     ang_I = data_fit.offset[str(qubit_pair)][
        #         data_fit.setup[str(qubit_pair)].index("I")
        #     ]
        #     fitting_report += f"Z rotation: {round(ang_X - ang_I, 3)} rad<br>"
        # fitting_report += "<br>"

    fig1.update_layout(
        title_text=f"Phase correction Qubit {qubits[0]}",
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis1_title="theta [rad]",
        xaxis2_title="theta [rad]",
        yaxis_title="MSR[V]",
    )

    fig2.update_layout(
        title_text=f"Phase correction Qubit {qubits[1]}",
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis1_title="theta [rad]",
        xaxis2_title="theta [rad]",
        yaxis_title="MSR[V]",
    )

    return [fig1, fig2], fitting_report


cz_virtualz = Routine(_acquisition, _fit, _plot)
"""CZ virtual Z correction routine."""
