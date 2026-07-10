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

from qibocal.auto.operation import (
    Protocol,
    QubitId,
    QubitPairId,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.ramsey.processing import fitting as ramsey_fitting
from qibocal.protocols.utils import GHZ_TO_HZ

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

__all__ = ["jazz"]


def _acquisition(
    params: ZZInteractionParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ZZInteractionData:
    """Data acquisition for JAZZ"""

    qubits_list = list(chain.from_iterable(targets))
    qubits_set = set(qubits_list)

    if len(qubits_set) != len(qubits_list):
        raise ValueError(
            "In the pair list there are repeated qubits: "
            "Parallel execution is not possible."
        )

    data = ZZInteractionData(
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

    for q_targ, q_spect in targets:
        targ_prob = results[targets_ro_pulses[q_targ].id]
        targ_error = np.sqrt(targ_prob * (1 - targ_prob) / params.nshots)
        spect_prob = results[spectator_ro_pulses[q_spect].id]
        spect_error = np.sqrt(spect_prob * (1 - spect_prob) / params.nshots)
        data.register_qubit(
            ZZIntType,
            q_targ,
            q_spect,
            dict(
                delay=sweeper.values,
                targ_prob=np.array(targ_prob),
                targ_error=targ_error,
                spect_prob=np.array(spect_prob),
                spect_error=spect_error,
            ),
        )

    return data


def _fit(data: ZZInteractionData) -> ZZInteractionResults:
    """Post-processing for JAZZ."""

    delays = data.delays
    zz: dict[QubitPairId, list[float]] = {}
    coupling: dict[QubitPairId, list[float]] = {}
    fit_params: dict[QubitPairId, list[float]] = {}
    for pair in data.pairs:
        target, spectator = pair
        pair_data = data[pair]
        try:
            popt, perr = ramsey_fitting(
                delays, pair_data["targ_prob"], pair_data["targ_error"]
            )

            fit_params |= {pair: popt}
            zz_zeta = [popt[2] * GHZ_TO_HZ, perr[2] * GHZ_TO_HZ]
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
            log.warning(f"Ramsey fitting failed for pair {pair} due to {e}.")

    return ZZInteractionResults(zz=zz, coupling=coupling, fitted_parameters=fit_params)


def _plot(
    data: ZZInteractionData,
    target: QubitPairId,
    fit: ZZInteractionResults | None = None,
):
    """Plotting function for JAZZ Experiment."""

    # casting target as a tuple
    target = tuple(target)

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
