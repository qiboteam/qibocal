from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import networkx as nx
import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import SNZ, FluxPulse, Rectangular
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
    dt_spacing: Optional[float] = 0
    """Time spacing between the two halves of the SNZ pulse."""
    snz_b_half_duration: Optional[float] = 1
    """Duration of the lowered amplitude (B) of the SNZ pulse."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class SnzTuningResults(Results):
    """SnzTuning outputs."""

    pulse_amplitude: Dict[Union[str, int], float] = field(
        metadata=dict(update="snz_pulse_amplitude")
    )
    """CZ pulse amplitude."""
    snz_ratio: Dict[Union[str, int], float] = field(metadata=dict(update="snz_ratio"))
    """CZ pulse ratio, A/B in the pulse description."""
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
            quantities={
                "detuning": "degree",
                "flux_pulse_amplitude": "dimensionless",
                "flux_pulse_ratio": "dimensionless",
            },
            options=[
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

    G = nx.Graph()
    G.add_nodes_from(platform.qubits)
    G.add_edges_from(platform.topology)

    # Find unique pairs of qubits
    unique_pairs = []
    for qubit in qubits:
        neighbors = list(G.neighbors(qubit))
        for neighbor in neighbors:
            if (neighbor, qubit) not in unique_pairs and (
                qubit,
                neighbor,
            ) not in unique_pairs:
                if (
                    platform.qubits[qubit].drive_frequency
                    > platform.qubits[neighbor].drive_frequency
                ):
                    unique_pairs.append((qubit, neighbor))
                else:
                    unique_pairs.append((neighbor, qubit))

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

    for q_highfreq, q_lowfreq in unique_pairs:
        # Target sequence RX90 - CPhi - RX90 - MZ
        initial_RX90_pulse = platform.create_RX90_pulse(
            q_highfreq, start=0, relative_phase=0
        )

        # Creating the deconstructed SNZ pulse and pulses for the target qubit
        # The Create_CZ_pulse_sequence function must return the square CZ pulse
        sequence, virtual = platform.create_CZ_pulse_sequence(
            tuple(sorted([q_highfreq, q_lowfreq])),
        )
        for pulse in sequence:
            if (
                isinstance(pulse, FluxPulse)
                and isinstance(pulse.shape, Rectangular)
                and pulse.qubit == q_highfreq
            ):
                half_flux_pulse_duration = pulse.duration / 2
                snz_amplitude = pulse.amplitude

        snz_a = FluxPulse(
            start=initial_RX90_pulse.finish,
            duration=half_flux_pulse_duration * 2 + params.dt_spacing,
            amplitude=snz_amplitude,
            shape=SNZ(
                dt_spacing=params.dt_spacing,
            ),
            qubit=q_highfreq,
            channel=platform.qubits[q_highfreq].flux.name,
        )
        snz_b = FluxPulse(
            start=initial_RX90_pulse.finish
            + half_flux_pulse_duration
            - params.snz_b_half_duration,
            duration=params.snz_b_half_duration * 2 + params.dt_spacing,
            amplitude=snz_amplitude,
            shape=SNZ(
                dt_spacing=params.dt_spacing,
            ),
            qubit=q_highfreq,
            channel=platform.qubits[q_highfreq].flux.name,
        )

        RX90_pulse = platform.create_RX90_pulse(
            q_highfreq, start=snz_a.se_finish, relative_phase=virtual[q_highfreq]
        )
        ro_pulse_target = platform.create_qubit_readout_pulse(
            q_highfreq, start=RX90_pulse.se_finish
        )

        # Creating different measurment
        sequence_target = (
            initial_RX90_pulse + snz_a + snz_b + RX90_pulse + ro_pulse_target
        )

        # Control pulses
        initial_RX_pulse = platform.create_RX_pulse(q_lowfreq, start=0)
        RX_pulse = platform.create_RX_pulse(q_lowfreq, start=RX90_pulse.se_start)
        ro_pulse_control = platform.create_qubit_readout_pulse(
            q_lowfreq, start=RX90_pulse.se_finish
        )

        # Creating the two different sequences ON and OFF
        sequences = {
            "ON": sequence_target + initial_RX_pulse + RX_pulse + ro_pulse_control,
            "OFF": sequence_target + ro_pulse_control,
        }

        # Create the Sweepers
        amplitude_sweep = Sweeper("amplitude", amplitudes, pulses=[snz_a])
        b_amplitude_sweep = Sweeper("amplitude", ratios, pulses=[snz_b])
        detuning_sweep = Sweeper("relative_phase", detuning, pulses=[RX90_pulse])

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
                        "iq_distance[dimensionless]": np.abs(
                            result.voltage_i
                            + 1j * result.voltage_q
                            - complex(platform.qubits[ro_pulse.qubit].mean_gnd_states)
                        )
                        / np.abs(
                            complex(platform.qubits[ro_pulse.qubit].mean_exc_states)
                            - complex(platform.qubits[ro_pulse.qubit].mean_gnd_states)
                        ).flatten(),
                        "controlqubit": len(amplitude_mesh) * [q_lowfreq],
                        "targetqubit": len(amplitude_mesh) * [q_highfreq],
                        "result_qubit": len(amplitude_mesh) * [ro_pulse.qubit],
                        "on_off": len(amplitude_mesh) * [on_off],
                        "detuning[degree]": detuning_mesh,
                        "flux_pulse_amplitude[dimensionless]": amplitude_mesh,
                        "flux_pulse_ratio[dimensionless]": b_amplitude_mesh,
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
    qubits = data_fit.df["result_qubit"].unique()
    for qubit in qubits:
        # Find the minimum leakage for a phase difference being 180 +- 1 degree
        min_leakage[qubit] = data_fit.df[
            (data_fit.df["result_qubit"] == qubit)
            & (data_fit.get_values("phase_difference", "degree") > 179)
            & (data_fit.get_values("phase_difference", "degree") < 181)
        ]["leakage"].min()
        amplitude[qubit] = data_fit.df[(min_leakage[qubit] == data_fit.df["leakage"])][
            "flux_pulse_amplitude"
        ]
        ratio[qubit] = data_fit.df[(min_leakage[qubit] == data_fit.df["leakage"])][
            "flux_pulse_ratio"
        ]

    return SnzTuningResults(
        pulse_amplitude=amplitude,
        snz_ratio=ratio,
        data_fit=data_fit,
    )


def _plot(data: SnzTuningData, results: SnzTuningResults, qubit):
    r"""Plotting function for the SNZ tuning experiment."""

    figures = [
        utils.amplitude_balance_cz_raw_data(data, results.data_fit, qubit),
        utils.amplitude_balance_cz_acquired_phase(data, results.data_fit, qubit),
        utils.amplitude_balance_cz_leakage(data, results.data_fit, qubit),
        utils.amplitude_balance_cz_acquired_phase(data, results.data_fit, qubit),
    ]

    fitting_report = (
        f"{qubit} | SNZ amplitude: {results.pulse_amplitude[qubit]:.4f} <br>"
        + f"{qubit} | SNZ ratio: {results.snz_ratio[qubit]:.4f} <br>"
    )
    return figures, fitting_report


snz_tuning = Routine(_acquisition, _fit, _plot)
