from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    ParallelSweepers,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import QubitId, QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from .acquisition import (
    RamseyData,
    RamseyParameters,
    RamseyResults,
    single_qubit_ramsey_sequence,
)
from .processing import (
    MAXIMUM_FIT_POINTS,
    fitting,
    process_fit,
    ramsey_fit,
    ramsey_update,
)

__all__ = ["ramsey_zz"]


EPS = 1  # Hz
"""we add 1Hz when computing Delta frequency between the two qubit frequencies in order to avoid numerical error."""


class AnharmonicityError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def compute_coupling_strength(
    omega1: float,
    omega2: float,
    anharmonicity1: float,
    anharmonicity2: float,
    zz: list[float],
) -> list[float]:
    """Compute the ZZ coupling from the difference in frequency and anharmonicity.

    coupling computing by inverting the following formula
    delta_q = omega1 - omega2
    xi = 2 g**2 (1 / (delta_q - alpha_2) - 1 / (delta_q + alpha_1))
    where delta_q is the difference in frequency and alpha_i is the anharmonicity
    """

    if anharmonicity1 == 0 or anharmonicity2 == 0:
        raise AnharmonicityError

    # adding an eps to avoid numerical issues
    delta_qubit_freq = omega1 - omega2 + EPS
    denominator = 1 / (delta_qubit_freq - anharmonicity2) - 1 / (
        delta_qubit_freq + anharmonicity1
    )

    # here we compute coupling as a frequency and do error propagation
    return [
        float(np.sqrt(np.abs(zz[0] / 2 / denominator))),
        float(zz[1] / 2 / np.sqrt(2 * abs(denominator * zz[0]))),
    ]


@dataclass
class RamseyZZParameters(RamseyParameters):
    """RamseyZZ runcard inputs."""


@dataclass
class RamseyZZResults(RamseyResults):
    """RamseyZZ outputs."""

    zz: dict[QubitPairId, list[float]] = field(default_factory=dict)
    coupling: dict[QubitPairId, list[float]] = field(default_factory=dict)

    def __contains__(self, pair: QubitPairId):
        return pair in self.zz


RamseyZZType = np.dtype(
    [("wait", np.float64), ("targ_prob", np.float64), ("spect_prob", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class RamseyZZData(RamseyData):
    """RamseyZZ acquisition outputs."""

    anharmonicity: dict[QubitId, float] = field(default_factory=dict)
    """Targets qubit anharmonicity."""
    data: dict[tuple[QubitPairId, str], npt.NDArray[RamseyZZType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _add_spectator_readout_define_parsweepers(
    platform: CalibrationPlatform,
    pair: QubitPairId,
    delay_range: tuple[float, float, float],
) -> tuple[PulseSequence, PulseSequence, ParallelSweepers]:
    """Build the Ramsey sequence and sweepers including spectator readout.

    The target qubit is driven through a Ramsey sequence and the spectator
    qubit readout is appended with a sweep over the spectator delay that
    includes the duration of the target Ramsey sequence.
    """

    init_t, fin_t, step_t = delay_range

    sequence, targets_ro_seq, target_delays = single_qubit_ramsey_sequence(
        platform=platform,
        target=pair[0],
        wait=init_t,
    )

    target_sweeper = Sweeper(
        parameter=Parameter.duration,
        range=delay_range,
        pulses=target_delays,
    )

    # adding sequence for measuring also spectator qubit
    spect_ro_seq = PulseSequence()
    spect_natives = platform.natives.single_qubit[pair[1]]
    spect_ro_ch, spect_ro_pulse = spect_natives.MZ()[0]

    ramsey_duration = sequence.duration
    # delay computed by the duration of the ramsey sequence
    spect_ro_delay = Delay(duration=ramsey_duration)
    spect_ro_seq.append((spect_ro_ch, spect_ro_delay))
    spect_ro_seq.append((spect_ro_ch, spect_ro_pulse))

    # sweeping over spectator delay before measuring it
    spectator_sweeper = Sweeper(
        parameter=Parameter.duration,
        # here we are adding to the spectator sweeper the duration of the PI/2 pulses
        # durations for all target qubita
        range=(ramsey_duration, fin_t + ramsey_duration - init_t, step_t),
        pulses=[spect_ro_delay],
    )

    return (
        (sequence + targets_ro_seq + spect_ro_seq),
        spect_natives.RX(),
        ParallelSweepers([target_sweeper, spectator_sweeper]),
    )


def _execute_ramsey_zz(
    platform: CalibrationPlatform,
    pair_list: list[QubitPairId],
    params: RamseyZZParameters,
    data: RamseyZZData,
    ramsey_sequence: PulseSequence,
    spectators_flip_sequence: PulseSequence,
    full_parsweepers: ParallelSweepers,
    updates=list,
) -> RamseyZZData:
    """Execute Ramsey ZZ coupling measurement.

    Performs Ramsey sequences on target qubits with optional X flip on spectator
    qubits. Registers measurement probabilities for both target and spectator qubits.
    """

    for setup in ["I", "X"]:
        if setup == "X":
            # adding X gate on spectator qubit
            experiment_sequence = spectators_flip_sequence | ramsey_sequence
        else:
            experiment_sequence = ramsey_sequence

        # execute the sweep
        results = platform.execute(
            [experiment_sequence],
            [full_parsweepers],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=updates,
        )

        for pair in pair_list:
            targ_ro_pulse = list(
                experiment_sequence.channel(platform.qubits[pair[0]].acquisition)
            )[-1]
            targ_probs = results[targ_ro_pulse.id]
            spect_ro_pulse = list(
                experiment_sequence.channel(platform.qubits[pair[1]].acquisition)
            )[-1]
            spect_probs = results[spect_ro_pulse.id]

            data.register_qubit(
                RamseyZZType,
                (pair, setup),
                dict(
                    wait=np.arange(
                        params.delay_between_pulses_start,
                        params.delay_between_pulses_end,
                        params.delay_between_pulses_step,
                    ),
                    targ_prob=targ_probs,
                    spect_prob=spect_probs,
                ),
            )

    return data


def _acquisition(
    params: RamseyZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RamseyZZData:
    """Data acquisition for RamseyZZ Experiment.
    Targets is a list of qubit pair in the order: (target, spectator).

    Standard Ramsey experiment repeated twice.
    In the second execution one qubit is brought to the excited state.
    """

    qubits_list = list(chain.from_iterable(targets))
    qubits_set = set(qubits_list)

    data = RamseyZZData(
        detuning=params.detuning,
        qubit_freqs={
            q: platform.config(platform.qubits[q].drive).frequency for q in qubits_set
        },
        anharmonicity={
            q: platform.calibration.single_qubits[q].qubit.anharmonicity
            for q in qubits_set
        },
    )

    if len(qubits_set) != len(qubits_list):
        log.warning(
            "In the pair list there are repeated qubits: "
            "Parallel execution is not possible."
        )
        parallel_execution = False
    else:
        parallel_execution = True

    # creating the RamseyZZ pulse sequence to run in parallel and also define the sweepers
    ramsey_spectator_seq = PulseSequence()
    spectators_flip_seq = PulseSequence()
    full_parsweepers = ParallelSweepers([])
    if parallel_execution:
        for pair in targets:
            updates = []
            if params.detuning is not None:
                channel = platform.qubits[pair[0]].drive
                f0 = platform.config(channel).frequency
                updates.append({channel: {"frequency": f0 + params.detuning}})

            seq, spect_flip, parsweep = _add_spectator_readout_define_parsweepers(
                platform=platform,
                pair=pair,
                delay_range=params.delay_range,
            )

            ramsey_spectator_seq += seq
            spectators_flip_seq += spect_flip
            full_parsweepers += parsweep

        data = _execute_ramsey_zz(
            platform=platform,
            pair_list=targets,
            params=params,
            data=data,
            ramsey_sequence=ramsey_spectator_seq,
            spectators_flip_sequence=spectators_flip_seq,
            full_parsweepers=full_parsweepers,
            updates=updates,
        )

    else:
        for pair in targets:
            ramsey_spectator_seq, spectators_flip_seq, full_parsweepers = (
                _add_spectator_readout_define_parsweepers(
                    platform=platform,
                    pair=pair,
                    delay_range=params.delay_range,
                )
            )

            if params.detuning is not None:
                channel = platform.qubits[pair[0]].drive
                f0 = platform.config(channel).frequency
                updates = [{channel: {"frequency": f0 + params.detuning}}]

            data = _execute_ramsey_zz(
                platform=platform,
                pair_list=[pair],
                params=params,
                data=data,
                ramsey_sequence=ramsey_spectator_seq,
                spectators_flip_sequence=spectators_flip_seq,
                full_parsweepers=full_parsweepers,
                updates=updates,
            )

    return data


def _fit(data: RamseyZZData) -> RamseyZZResults:
    """Fitting procedure for RamseyZZ protocol.

    Standard Ramsey fitting procedure is applied for both version of
    the experiment.

    """
    waits = data.waits
    popts: dict[QubitId, list[float]] = {}
    freq_measure: dict[QubitId, list[float]] = {}
    t2_measure: dict[QubitId, list[float]] = {}
    delta_phys_measure: dict[QubitId, list[float]] = {}
    delta_fitting_measure: dict[QubitId, list[float]] = {}
    zz: dict[QubitId, list[float]] = {}
    coupling: dict[QubitId, list[float]] = {}
    for pair in data.qubits:
        target, spectator = pair

        for setup in ["I", "X"]:
            qubit_freq = data.qubit_freqs[target]
            try:
                popt, perr = fitting(waits, data[pair, setup]["targ_prob"])
                (
                    freq_measure[pair, setup],
                    t2_measure[pair, setup],
                    delta_phys_measure[pair, setup],
                    delta_fitting_measure[pair, setup],
                    popts[pair, setup],
                ) = process_fit(popt, perr, qubit_freq, data.detuning)
            except Exception as e:
                log.warning(f"Ramsey fitting failed for qubit pair {pair} due to {e}.")
        # compute zz and qq coupling
        # zz the difference in frequency between the two measurement
        zz[pair] = [
            float(freq_measure[pair, "X"][0] - freq_measure[pair, "I"][0]),
            float(
                np.sqrt(
                    freq_measure[pair, "X"][1] ** 2 + freq_measure[pair, "I"][1] ** 2
                )
            ),
        ]

        # here we compute coupling as a frequency
        try:
            coupling[pair] = compute_coupling_strength(
                omega1=data.qubit_freqs[target],
                omega2=data.qubit_freqs[spectator],
                anharmonicity1=data.anharmonicity[target],
                anharmonicity2=data.anharmonicity[spectator],
                zz=zz[pair],
            )
        except AnharmonicityError:
            log.warning(
                f"e-f transitions are not calibrated for qubit pair {pair}, cannot compute coupling strength"
            )

    return RamseyZZResults(
        delta_phys=delta_phys_measure,
        fitted_parameters=popts,
        zz=zz,
        coupling=coupling,
    )


def zz_fit_plot(
    target: QubitId,
    spect_qubit: QubitId,
    fit: RamseyZZResults,
    waits: npt.NDArray,
    fig: go.Figure,
) -> str:
    fit_waits = np.linspace(min(waits), max(waits), MAXIMUM_FIT_POINTS)

    fig.add_traces(
        [
            go.Scatter(
                x=fit_waits,
                y=ramsey_fit(fit_waits, *fit.fitted_parameters[target, "I"]),
                name="Fit I",
                mode="lines",
            ),
            go.Scatter(
                x=fit_waits,
                y=ramsey_fit(fit_waits, *fit.fitted_parameters[target, "X"]),
                name="Fit X",
                mode="lines",
            ),
        ],
        rows=1,
        cols=1,
    )

    fitting_report = table_html(
        table_dict(
            target,
            [
                f"ZZ  with {spect_qubit} [kHz]",
                f"Coupling with {spect_qubit} [MHz]",
            ],
            [
                np.round(
                    fit.zz[target] * 1e-3,
                    0,
                ),
                np.round(
                    fit.coupling[target] * 1e-6,
                    2,
                )
                if target in fit.coupling
                else None,
            ],
        )
    )

    return fitting_report


def _plot(
    data: RamseyZZData, target: QubitId, fit: RamseyZZResults | None = None
) -> tuple[list[go.Figure], str]:
    """Plotting function for Ramsey Experiment."""

    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        subplot_titles=(
            f"Target qubit {target[0]}",
            f"Spectator qubit {target[1]}",
        ),
    )

    waits = data.waits
    fig.add_traces(
        [
            go.Scatter(
                x=waits,
                y=data.data[target, "I"].targ_prob,
                opacity=1,
                name="I",
                showlegend=True,
                legendgroup="I ",
                mode="markers",
            ),
            go.Scatter(
                x=waits,
                y=data.data[target, "X"].targ_prob,
                opacity=1,
                name="X",
                showlegend=True,
                legendgroup="X",
                mode="markers",
            ),
        ],
        rows=1,
        cols=1,
    )

    fig.add_traces(
        [
            go.Scatter(
                x=waits,
                y=data.data[target, "I"].spect_prob,
                opacity=1,
                name="I",
                showlegend=True,
                legendgroup="I ",
                mode="markers",
            ),
            go.Scatter(
                x=waits,
                y=data.data[target, "X"].spect_prob,
                opacity=1,
                name="X",
                showlegend=True,
                legendgroup="X",
                mode="markers",
            ),
        ],
        rows=1,
        cols=2,
    )

    if fit is not None:
        fitting_report = zz_fit_plot(
            target=target,
            spect_qubit=data.target_qubit,
            fit=fit,
            waits=waits,
            fig=fig,
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    return [fig], fitting_report


ramsey_zz = Routine(_acquisition, _fit, _plot, ramsey_update)
"""Ramsey ZZ Routine object.

This protocol measures the state-dependent frequency shift (ZZ interaction)
between a selected target qubit and one spectator qubit. It
performs two Ramsey experiments for each qubit in the target list:

- "I" setup: spectator qubit remain in the ground state.
- "X" setup: spectator qubit is excited before the Ramsey sequence.

The fitted Ramsey frequencies from these two experiments are compared to
extract the conditional ZZ shift experienced by each measured qubit. The
difference between the two fitted frequencies is reported as the
ZZ interaction strength. Using the measured qubit frequencies, the target
qubit frequency, and the anharmonicities, the routine also estimates the
effective coupling strength between each qubit and the target qubit.
This protocol is useful for characterizing residual static coupling and
frequency shifts induced by an excited neighboring qubit. The plot output
shows Ramsey traces in probability for both the I and X setups.
"""
