from dataclasses import dataclass
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import SNZ, FluxPulse
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from . import utils


@dataclass
class SnzTuningParameters(Parameters):
    """SnzTuning runcard inputs."""

    amplitude_factor_start: float
    """Starting amplitude for the flux pulse."""
    amplitude_factor_end: float
    """Ending amplitude for the flux pulse."""
    amplitude_factor_step: float
    """Step amplitude for the flux pulse."""
    b_amplitude_factor_start: float
    """Starting amplitude for the flux pulse."""
    b_amplitude_factor_end: float
    """Ending amplitude for the flux pulse."""
    b_amplitude_factor_step: float
    """Step amplitude for the flux pulse."""
    detuning_start: float
    """Starting detuning for the flux pulse."""
    detuning_end: float
    """Ending detuning for the flux pulse."""
    detuning_step: float
    """Step detuning for the flux pulse."""
    pairs: list[list[QubitId, QubitId]]
    """List of qubit pairs to be used in the experiment."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class SnzTuningResults(Results):
    """SnzTuning outputs."""

    pulse_amplitude: dict[
        tuple[QubitId, QubitId], float
    ]  # = field(metadata=dict(update="snz_pulse_amplitude"))
    """CZ pulse amplitude."""
    snz_ratio: dict[
        tuple[QubitId, QubitId], float
    ]  # = field(metadata=dict(update="snz_ratio"))
    """CZ pulse ratio, A/B in the pulse description."""
    snz_leakage: dict[
        tuple[QubitId, QubitId], float
    ]  # = field(metadata=dict(update="snz_leakage"))
    """CZ leakage measured on the control qubit."""
    data_fit: DataUnits
    """Data fit used in plotting to debug the fitting.
        quantities={
            "flux_pulse_amplitude": "dimensionless",
            "flux_pulse_ratio": "dimensionless",
            "initial_phase_ON": "degree",
            "initial_phase_OFF": "degree",
            "phase_difference": "degree",
            "leakage": "dimensionless",
        },
        options=["controlqubit", "targetqubit"],
    """


class SnzTuningData(DataUnits):
    """SnzTuning acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={},
            options=[
                "detuning",
                "flux_pulse_amplitude",
                "flux_pulse_ratio",
                "controlqubit",
                "targetqubit",
                "on_off",
                "result_qubit",
                "iq_distance",
            ],
        )


def _acquisition(
    params: SnzTuningParameters,
    platform: Platform,
    qubits: Qubits,
) -> SnzTuningData:
    r"""Acquisition routine for SnzTuning.

    This experiment is used to find the optimal amplitude for the flux pulse for
    CZ gate.

    Two sets of sequences are played:
    1. The first set of sequences is applying a RX90 pulse, a flux pulse, a RX90
    pulse and a readout pulse on the target qubit, while only applying a readout
    pulse on the control qubit.
    2. The second set of sequences is applying a RX90 pulse, a flux pulse, a
    RX90 pulse and a readout pulse on the target qubit, and a RX pulse and
    readout pulse on the control qubit.

    Each set of sequences is played while varying the phase of the second RX90
    pulse on the control qubit. This is a Ramsey like experiment which allows to
    find the detuning acquired by the flux pulse. The phase of both ON and OFF
    sequences are measured and the difference between the two is used to find
    the optimal amplitude (positive half of SNZ pulse) and optimal ratio
    (positive/negative half of SNZ pulse) that will minimize |02> leakage while
    performing CZ gate (180 phase difference).
    """

    # Get the high frequency index in the pairs
    indices_highfreq = []
    for pair in params.pairs:
        if (
            platform.qubits[pair[0]].drive_frequency
            > platform.qubits[pair[1]].drive_frequency
        ):
            indices_highfreq.append(0)
        else:
            indices_highfreq.append(1)

    # Create the data object
    data = SnzTuningData()

    # Variables
    amplitudes = np.arange(
        params.amplitude_factor_start,
        params.amplitude_factor_end,
        params.amplitude_factor_step,
    )
    ratios = np.arange(
        params.b_amplitude_factor_start,
        params.b_amplitude_factor_end,
        params.b_amplitude_factor_step,
    )
    detuning = np.arange(
        params.detuning_start,
        params.detuning_end,
        params.detuning_step,
    )
    # generate the flattened mesh grid of the nested sweep of amplitude_sweep, b_amplitude_sweep and detuning_sweep
    amplitude_mesh, b_amplitude_mesh, detuning_mesh = np.meshgrid(
        amplitudes, ratios, detuning, indexing="ij"
    )
    amplitude_mesh = amplitude_mesh.flatten()
    b_amplitude_mesh = b_amplitude_mesh.flatten()
    detuning_mesh = detuning_mesh.flatten()

    for pair, index_highfreq in zip(params.pairs, indices_highfreq):
        q_target = pair[1]
        q_control = pair[0]

        # Target sequence RX90 - CPhi - RX90 - MZ
        initial_RX90_pulse = platform.create_RX90_pulse(
            q_target, start=0, relative_phase=0
        )

        # Creating the SNZ sequence which might contain parking pulses
        sequence, virtual = platform.pairs[
            tuple(sorted([q_target, q_control]))
        ].native_gates.CZ.sequence(start=initial_RX90_pulse.se_finish)

        pulses = []
        for pulse in sequence:
            if (
                isinstance(pulse, FluxPulse)
                and isinstance(pulse.shape, SNZ)
                and pulse.qubit == pair[index_highfreq]
            ):
                pulses.append(pulse)

        if len(pulses) != 2:
            raise ValueError(f"Unvalid sequence for SNZ tuning: \n {sequence}")
        else:
            # Snz_a is the pulse with the longest duration
            snz_a = max(pulses, key=lambda x: x.duration)
            # Snz_b is the pulse with the shortest duration
            snz_b = min(pulses, key=lambda x: x.duration)

        RX90_pulse = platform.create_RX90_pulse(
            q_target, start=snz_a.se_finish, relative_phase=virtual[q_target]
        )
        ro_pulse_target = platform.create_qubit_readout_pulse(
            q_target, start=RX90_pulse.se_finish
        )

        # Creating different measurment
        sequence_target = initial_RX90_pulse + sequence + RX90_pulse + ro_pulse_target

        # Control pulses
        initial_RX_pulse = platform.create_RX_pulse(q_control, start=0)
        RX_pulse = platform.create_RX_pulse(q_control, start=RX90_pulse.se_start)
        ro_pulse_control = platform.create_qubit_readout_pulse(
            q_control, start=RX90_pulse.se_finish
        )

        # Creating the two different sequences ON and OFF
        sequences = {
            "ON": sequence_target + initial_RX_pulse + RX_pulse + ro_pulse_control,
            "OFF": sequence_target + ro_pulse_control,
        }

        # Create the Sweepers
        amplitude_sweep = Sweeper(Parameter.amplitude, amplitudes, pulses=[snz_a])
        b_amplitude_sweep = Sweeper(Parameter.amplitude, ratios, pulses=[snz_b])
        detuning_sweep = Sweeper(
            Parameter.relative_phase, detuning, pulses=[RX90_pulse]
        )

        for on_off in ["ON", "OFF"]:
            results = platform.sweep(
                sequences[on_off],
                ExecutionParameters(
                    nshots=params.nshots,
                    acquisition_type=AcquisitionType.INTEGRATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
                amplitude_sweep,
                b_amplitude_sweep,
                detuning_sweep,
            )

            for ro_pulse in sequences[on_off].ro_pulses:
                result = results[ro_pulse.serial]
                r = result.serialize
                r.update(
                    {
                        "iq_distance": np.abs(
                            result.voltage_i
                            + 1j * result.voltage_q
                            - complex(platform.qubits[ro_pulse.qubit].mean_gnd_states)
                        )
                        / np.abs(
                            complex(platform.qubits[ro_pulse.qubit].mean_exc_states)
                            - complex(platform.qubits[ro_pulse.qubit].mean_gnd_states)
                        ).flatten(),
                        "controlqubit": len(amplitude_mesh) * [q_control],
                        "targetqubit": len(amplitude_mesh) * [q_target],
                        "result_qubit": len(amplitude_mesh) * [ro_pulse.qubit],
                        "on_off": len(amplitude_mesh) * [on_off],
                        "detuning[degree]": detuning_mesh,
                        "flux_pulse_amplitude": amplitude_mesh,
                        "flux_pulse_ratio": b_amplitude_mesh,
                    }
                )
                data.add_data_from_dict(r)

    return data


def _fit(data: SnzTuningData) -> SnzTuningResults:
    r"""Post-processing function for the SNZ tuning experiment."""
    min_leakage = {}
    amplitude = {}
    ratio = {}

    data_fit = utils.fit_amplitude_balance_cz(data)

    pairs = utils.unique_combination(data)
    for pair in pairs:
        q_control = pair[0]
        q_target = pair[1]

        # Find the minimum leakage for a phase difference being 180 +- 1 degree
        min_leakage[tuple(sorted(pair))] = data_fit.df[
            (data_fit.df["controlqubit"] == q_control)
            & (data_fit.df["targetqubit"] == q_target)
            & (data_fit.df["phase_difference"] > 179)
            & (data_fit.df["phase_difference"] < 181)
        ]["leakage"].min()

        amplitude[tuple(sorted(pair))] = data_fit.df[
            (min_leakage[tuple(sorted(pair))] == data_fit.df["leakage"])
            & (data_fit.df["controlqubit"] == q_control)
            & (data_fit.df["targetqubit"] == q_target)
        ]["flux_pulse_amplitude"]

        ratio[tuple(sorted(pair))] = data_fit.df[
            (min_leakage[tuple(sorted(pair))] == data_fit.df["leakage"])
            & (data_fit.df["controlqubit"] == q_control)
            & (data_fit.df["targetqubit"] == q_target)
        ]["flux_pulse_ratio"]

    return SnzTuningResults(
        pulse_amplitude=amplitude,
        snz_ratio=ratio,
        data_fit=data_fit,
        snz_leakage=min_leakage,
    )


def _plot(data: SnzTuningData, results: SnzTuningResults, qubit):
    r"""Plotting function for the SNZ tuning experiment."""

    figures = [
        utils.amplitude_balance_cz_raw_data(data, results.data_fit, qubit),
        utils.amplitude_balance_cz_acquired_phase(data, results.data_fit, qubit),
        utils.amplitude_balance_cz_leakage(data, results.data_fit, qubit),
        utils.amplitude_balance_cz_acquired_phase(data, results.data_fit, qubit),
    ]

    # fitting_report = (
    #     f"{qubit} | SNZ amplitude: {results.pulse_amplitude[qubit]:.4f} <br>"
    #     + f"{qubit} | SNZ ratio: {results.snz_ratio[qubit]:.4f} <br>"
    # )
    fitting_report = "No fitting data."
    return figures, fitting_report


snz_tuning = Routine(_acquisition, _fit, _plot)
