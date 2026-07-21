from dataclasses import dataclass, field
from itertools import chain

import numpy as np
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseLike,
    PulseSequence,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal.auto.operation import (
    Protocol,
    QubitId,
    QubitPairId,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import GHZ_TO_HZ, quinn_fernandes_algorithm

from .utils import (
    ZZInteractionData,
    ZZInteractionParameters,
    ZZInteractionResults,
    ZZIntType,
    coupling_strength,
    create_report_table,
    signal_plot,
    zz_update,
)

DAMPED_CONSTANT = 1.5
"""See :const:`rabi.utils.QUANTILE_CONSTANT` for details."""

__all__ = ["jazz"]


def jazz_fit(x, offset, amplitude, delta, decay) -> np.typing.NDArray | float:
    """Dumped sinusoidal fit."""
    return offset + amplitude * np.sin(x * delta) * np.exp(-x * decay)


@dataclass
class JAZZData(ZZInteractionData):
    data: dict[QubitPairId, ZZIntType] = field(default_factory=dict)
    """Raw data acquired."""


@dataclass
class JAZZResults(ZZInteractionResults):
    fitted_parameters: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Parameters fitted during the execution."""


def _acquisition(
    params: ZZInteractionParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> JAZZData:
    """Data acquisition for JAZZ"""

    qubits_list = list(chain.from_iterable(targets))
    qubits_set = set(qubits_list)

    if len(qubits_set) != len(qubits_list):
        raise ValueError(
            "In the pair list there are repeated qubits: "
            "Parallel execution is not possible."
        )

    data = JAZZData(
        qubit_freqs={
            q: platform.config(platform.qubits[q].drive).frequency for q in qubits_set
        },
        anharmonicity={
            q: platform.calibration.single_qubits[q].qubit.anharmonicity
            for q in qubits_set
        },
    )

    complete_sequence = PulseSequence()
    delay = Delay(duration=0)
    targets_ro_pulses: dict[QubitId, PulseLike] = {}
    spectator_ro_pulses: dict[QubitId, PulseLike] = {}
    for pair in targets:
        target, spectator = pair

        # target qubit channels and native pulses
        target_drive_channel = platform.qubits[target].drive
        target_natives = platform.natives.single_qubit[target]
        target_ro_channel, target_ro_pulse = target_natives.MZ()[0]

        # spectator qubit channnels and native pulses
        spectator_drive_channel = platform.qubits[spectator].drive
        spectator_natives = platform.natives.single_qubit[spectator]
        spectator_ro_channel, spectator_ro_pulse = spectator_natives.MZ()[0]

        pair_sequence = PulseSequence()

        # delay
        pair_sequence += [(target_drive_channel, delay)]
        pair_sequence += [(spectator_drive_channel, delay)]
        pair_sequence += [(target_ro_channel, delay)]
        pair_sequence += [(spectator_ro_channel, delay)]

        # X(pi) on both spectator and target qubits
        _, target_pi_pulse = target_natives.RX()[0]
        pair_sequence += [
            (target_drive_channel, target_pi_pulse),
            (target_ro_channel, Delay(duration=target_pi_pulse.duration)),
        ]
        pair_sequence += spectator_natives.RX()

        # delay
        pair_sequence += [(target_drive_channel, delay)]
        pair_sequence += [(target_ro_channel, delay)]
        pair_sequence += [(spectator_ro_channel, delay)]

        # Y(pi/2)
        pair_sequence += target_natives.R(theta=np.pi / 2, phi=np.pi / 2)

        # measuring target qubit
        pair_sequence |= [
            (target_ro_channel, target_ro_pulse),
            (spectator_ro_channel, spectator_ro_pulse),
        ]

        # adding the initial X(pi/2) on target
        pair_sequence = target_natives.R(theta=np.pi / 2) | pair_sequence

        # adding both qubits ro pulses in the dictionary
        targets_ro_pulses |= {target: target_ro_pulse}
        spectator_ro_pulses |= {spectator: spectator_ro_pulse}

        complete_sequence += pair_sequence

    sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.delay_range,
        pulses=[delay],
    )

    results = platform.execute(
        [complete_sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for pair in targets:
        q_targ, q_spect = pair
        targ_prob = results[targets_ro_pulses[q_targ].id]
        targ_error = np.sqrt(targ_prob * (1 - targ_prob) / params.nshots)
        spect_prob = results[spectator_ro_pulses[q_spect].id]
        spect_error = np.sqrt(spect_prob * (1 - spect_prob) / params.nshots)
        data.register_qubit(
            ZZIntType,
            pair,
            dict(
                delay=sweeper.values,
                targ_prob=np.array(targ_prob),
                targ_error=targ_error,
                spect_prob=np.array(spect_prob),
                spect_error=spect_error,
            ),
        )

    return data


def _fit(data: JAZZData) -> JAZZResults:
    """Post-processing for JAZZ."""

    delays = data.delays
    zz: dict[QubitPairId, list[float]] = {}
    coupling: dict[QubitPairId, list[float]] = {}
    fit_params: dict[QubitPairId, list[float]] = {}
    for pair in data.pairs:
        target, spectator = pair
        try:
            probs = data.data[pair]["targ_prob"]
            err = data.data[pair]["targ_error"]

            # performing a min-max scaling on x and y arrays
            probs_max = np.max(probs)
            probs_min = np.min(probs)
            d_max = np.max(delays)
            d_min = np.min(delays)
            delta_probs = probs_max - probs_min
            delta_delay = d_max - d_min
            min_max_probs = (probs - probs_min) / delta_probs
            min_max_delays = (delays - d_min) / delta_delay
            if err is not None:
                err = err / delta_probs

            omega = quinn_fernandes_algorithm(
                min_max_probs, min_max_delays, speedup_flag=True
            )
            median_sig = np.median(min_max_probs)
            q80 = np.quantile(min_max_probs, 0.8)
            q20 = np.quantile(min_max_probs, 0.2)
            amplitude_guess = abs(q80 - q20) / DAMPED_CONSTANT

            p0 = [
                median_sig,
                amplitude_guess,
                omega,
                1,
            ]

            popt, perr = curve_fit(
                jazz_fit,
                min_max_delays,
                min_max_probs,
                p0=p0,
                maxfev=5000,
                bounds=(
                    [0, 0, -np.inf, 0],
                    [1, 1, np.inf, np.inf],
                ),
                sigma=err,
            )

            # inverting the scaling
            popt = [
                delta_probs * popt[0] + probs_min,
                delta_probs * popt[1] * np.exp(d_min * popt[3] / delta_delay),
                popt[2] / delta_delay,
                popt[3] / delta_delay,
            ]

            perr = np.sqrt(np.diag(perr))

            fit_params |= {pair: popt}
            zz_zeta = [
                popt[2] * GHZ_TO_HZ / (2 * np.pi),
                # error propagating the error for delta and then converyting into frequency
                perr[2] / delta_delay * GHZ_TO_HZ / (2 * np.pi),
            ]
            zz |= {pair: zz_zeta}

            # here we compute coupling as a frequency
            coupling[pair] = coupling_strength(
                omega1=data.qubit_freqs[target],
                omega2=data.qubit_freqs[spectator],
                anharmonicity1=data.anharmonicity[target],
                anharmonicity2=data.anharmonicity[spectator],
                zz=zz_zeta,
            )

        except Exception as e:
            log.warning(f"JAZZ fitting failed for pair {pair} due to {e}.")

    return JAZZResults(zz=zz, coupling=coupling, fitted_parameters=fit_params)


def _plot(
    data: JAZZData,
    target: QubitPairId,
    fit: JAZZResults | None = None,
):
    """Plotting function for JAZZ Experiment."""

    targ, spect = target
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        subplot_titles=(
            f"Target qubit {targ}",
            f"Spectator qubit {spect}",
        ),
    )
    fitting_report = ""

    if fit is not None and target in fit.fitted_parameters:
        target_traces, spectator_trace = signal_plot(
            signal=data.data[target],
            module=jazz_fit,
            fit_params=fit.fitted_parameters[target],
        )
        fig.add_traces(
            target_traces + spectator_trace,
            rows=1,
            cols=[1] * len(target_traces) + [2],
        )
        fig.update_layout(showlegend=True)
        fig.update_xaxes(title_text="Time [ns]", col=1)
        fig.update_xaxes(title_text="Time [ns]", col=2)
        fig.update_yaxes(title_text="Excited state probability", col=1)
        fig.update_yaxes(title_text="Excited state probability", col=2)
        fig.update_yaxes(range=[0, 1.1], col=2)

        fitting_report = create_report_table(target, fit)

    return [fig], fitting_report


jazz = Protocol(_acquisition, _fit, _plot, zz_update)
"""JAZZ Protocol object."""
