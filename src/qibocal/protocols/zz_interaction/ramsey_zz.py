from dataclasses import dataclass, field
from itertools import chain
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.ramsey.acquisition import single_qubit_ramsey_sequence
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

__all__ = ["ramsey_zz"]


EPS = 1  # Hz
"""we add 1Hz when computing Delta frequency between the two qubit frequencies in order to avoid numerical error."""


@dataclass
class RamseyZZParameters(ZZInteractionParameters):
    """RamseyZZ acquisition outputs."""

    detuning: float | None = None
    """Frequency detuning [Hz]."""


@dataclass
class RamseyZZData(ZZInteractionData):
    """RamseyZZ acquisition outputs."""

    detuning: float | None = None
    """Frequency detuning [Hz]."""
    data: dict[tuple[QubitPairId, Literal["I", "X"]], ZZIntType] = field(
        default_factory=dict
    )
    """Raw data acquired."""


@dataclass
class RamseyZZResults(ZZInteractionResults):
    """Container for RamseyZZ experiment results."""

    fitted_parameters: dict[QubitPairId, dict[Literal["I", "X"], list[float]]] = field(
        default_factory=dict
    )


def _acquisition(
    params: RamseyZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> RamseyZZData:
    """Data acquisition for RamseyZZ Experiment.
    Targets is a list of qubit pair in the order: (target, spectator).

    Standard Ramsey experiment repeated twice.
    In the second execution one qubit is brought to the excited state.
    """

    qubits_list = list(chain.from_iterable(targets))
    qubits_set = set(qubits_list)

    if len(qubits_set) != len(qubits_list):
        raise ValueError(
            "In the pair list there are repeated qubits: "
            "Parallel execution is not possible."
        )

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

    sequences: dict[Literal["I", "X"], PulseSequence] = {}
    parsweepers: dict[Literal["I", "X"], Sweeper] = {}
    for setup in ["I", "X"]:
        setup_sequence = PulseSequence()
        setup_delays: list[Delay] = []
        for pair in targets:
            targ, spect = pair

            updates = {}
            if params.detuning is not None:
                channel = platform.qubits[targ].drive
                f0 = platform.config(channel).frequency
                if channel not in updates:
                    updates[channel] = {"frequency": f0 + params.detuning}

            qubit_ramsey, qubit_ro_seq, target_delays = single_qubit_ramsey_sequence(
                platform=platform,
                target=targ,
                wait=0,
            )

            # adding sequence for measuring also spectator qubit
            spect_natives = platform.natives.single_qubit[spect]
            spect_ro_ch, spect_ro_pulse = spect_natives.MZ()[0]
            spect_ro_delay = Delay(duration=0)

            qubit_ro_seq.extend(
                (
                    (
                        spect_ro_ch,
                        Delay(
                            duration=2
                            * platform.natives.single_qubit[targ]
                            .R(theta=np.pi / 2)
                            .duration
                        ),
                    ),
                    (spect_ro_ch, spect_ro_delay),
                    (spect_ro_ch, spect_ro_pulse),
                )
            )

            if setup == "X":
                # adding X gate on spectator qubit
                qubit_sequence = spect_natives.RX() | (qubit_ramsey + qubit_ro_seq)
            else:
                qubit_sequence = qubit_ramsey + qubit_ro_seq

            setup_sequence += qubit_sequence
            setup_delays += target_delays + [spect_ro_delay]

        sequences[setup] = setup_sequence
        parsweepers[setup] = Sweeper(
            parameter=Parameter.duration,
            range=params.delay_range,
            pulses=setup_delays,
        )

    # execute the sweep
    results = platform.execute(
        list(sequences.values()),
        [list(parsweepers.values())],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        updates=[updates],
    )

    for pair in targets:
        targ, spect = pair
        for setup in ("I", "X"):
            targ_ro_pulse = list(
                sequences[setup].channel(platform.qubits[targ].acquisition)
            )[-1]
            targ_probs = results[targ_ro_pulse.id]
            spect_ro_pulse = list(
                sequences[setup].channel(platform.qubits[spect].acquisition)
            )[-1]
            spect_probs = results[spect_ro_pulse.id]

            data.register_qubit(
                ZZIntType,
                (pair, setup),
                dict(
                    delay=parsweepers[setup].values,
                    targ_prob=np.array(targ_probs),
                    targ_error=np.sqrt(targ_probs * (1 - targ_probs) / params.nshots),
                    spect_prob=np.array(spect_probs),
                    spect_error=np.sqrt(
                        spect_probs * (1 - spect_probs) / params.nshots
                    ),
                ),
            )

    return data


def _fit(data: RamseyZZData) -> RamseyZZResults:
    """Fitting procedure for RamseyZZ protocol.

    Standard Ramsey fitting procedure is applied for both version of
    the experiment.

    """
    delays = data.delays
    popts: dict[QubitId, list[float]] = {}
    delta_fitting_measure: dict[QubitId, list[float]] = {}
    zz: dict[QubitId, list[float]] = {}
    coupling: dict[QubitId, list[float]] = {}
    for pair in data.qubits:
        target, spectator = pair

        try:
            setup_param_dict = {}
            for setup in ["I", "X"]:
                setup_data = data[pair, setup]
                popt, perr = ramsey_fitting(
                    delays, setup_data["targ_prob"], setup_data["targ_error"]
                )
                delta_fitting_measure[pair, setup] = [
                    -popt[2] / (2 * np.pi) * GHZ_TO_HZ,
                    perr[2] * GHZ_TO_HZ / (2 * np.pi),
                ]
                setup_param_dict[setup] = popt

            popts[pair] = setup_param_dict
            # compute zz and qq coupling
            # zz the difference in frequency between the two measurement
            zz[pair] = [
                float(
                    delta_fitting_measure[pair, "X"][0]
                    - delta_fitting_measure[pair, "I"][0]
                ),
                float(
                    np.sqrt(
                        delta_fitting_measure[pair, "X"][1] ** 2
                        + delta_fitting_measure[pair, "I"][1] ** 2
                    )
                ),
            ]

            # here we compute coupling as a frequency
            coupling[pair] = coupling_strength(
                omega1=data.qubit_freqs[target],
                omega2=data.qubit_freqs[spectator],
                anharmonicity1=data.anharmonicity[target],
                anharmonicity2=data.anharmonicity[spectator],
                zz=zz[pair],
            )

        except Exception as e:
            log.warning(f"Ramsey fitting failed for qubit pair {pair} due to {e}.")

    return RamseyZZResults(
        fitted_parameters=popts,
        zz=zz,
        coupling=coupling,
    )


def _plot(
    data: RamseyZZData, target: QubitPairId, fit: RamseyZZResults | None = None
) -> tuple[list[go.Figure], str]:
    """Plotting function for Ramsey Experiment."""

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

    if fit is None or any(
        term not in fit.fitted_parameters[target] for term in ("I", "X")
    ):
        fit_params = None
    else:
        fit_params = fit.fitted_parameters[target]

    for s in ("I", "X"):
        target_traces, spectator_trace = signal_plot(
            signal=data[target, s],
            fit_params=fit_params[s],
            label=s,
        )
        fig.add_traces(
            target_traces + spectator_trace,
            rows=1,
            cols=[1] * len(target_traces) + [2],
        )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    if fit_params is not None:
        fitting_report = create_report_table(target, fit)

    return [fig], fitting_report


ramsey_zz = Protocol(_acquisition, _fit, _plot, zz_update)
"""Ramsey ZZ Protocol object.

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
