from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Parameters, Qubits, Results, Routine

from . import utils


# TODO: implement cross-talk
@dataclass
class QubitFluxParameters(Parameters):
    """QubitFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the qubit frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep (V)."""
    drive_amplitude: float
    """Drive pulse amplitude. Same for all qubits."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    transition: Optional[str] = "0->1"
    """Flux spectroscopy transition type ("0->1" or "0->2")."""


@dataclass
class QubitFluxResults(Results):
    """QubitFlux outputs."""

    sweetspot: Dict[QubitId, float] = field(metadata=dict(update="sweetspot"))
    """Sweetspot for each qubit."""
    frequency: Dict[QubitId, float] = field(metadata=dict(update="drive_frequency"))
    """Drive frequency for each qubit."""

    # TODO: After testing fitting, decide which fitted params should be saved/updated in the runcard
    fitted_parameters: Dict[QubitId, Dict[str, float]]
    """Raw fitting output."""


class QubitFluxData(DataUnits):
    """QubitFlux acquisition outputs."""

    def __init__(self, resonator_type):
        super().__init__(
            "data",
            {"frequency": "Hz", "bias": "V"},
            options=["qubit", "Ec", "Ej"],
        )
        self._resonator_type = resonator_type

    @property
    def resonator_type(self):
        """Type of resonator"""
        return self._resonator_type


def _acquisition(
    params: QubitFluxParameters,
    platform: Platform,
    qubits: Qubits,
) -> QubitFluxData:
    """Data acquisition for QubitFlux Experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )

        if transition == "0->2":
            qd_pulses[qubit].frequency -= (
                qubits[qubit].anharmonicity / 2
            )  # TODO: add anharmonicity to platform runcard - single qubit gates settings

        qd_pulses[qubit].amplitude = params.drive_amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    bias_sweeper = Sweeper(
        Parameter.bias,
        delta_bias_range,
        qubits=list(qubits.values()),
        type=SweeperType.ABSOLUTE,
    )
    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and flux bias
    data = QubitFluxData(platform.resonator_type)

    # repeat the experiment as many times as defined by software_averages
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        bias_sweeper,
        freq_sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]

        biases = np.repeat(delta_bias_range, len(delta_frequency_range))
        freqs = np.array(
            len(delta_bias_range)
            * list(delta_frequency_range + qd_pulses[qubit].frequency)
        ).flatten()
        # store the results
        r = {k: v.ravel() for k, v in result.serialize.items()}
        r.update(
            {
                "frequency[Hz]": freqs,
                "bias[V]": biases,
                "qubit": len(freqs) * [qubit],
                "Ec": len(freqs)
                * [
                    platform.qubits[qubit].Ec
                ],  # TODO: add Ec to platform runcard - single qubit gates settings
                "Ej": len(freqs)
                * [
                    platform.qubits[qubit].Ej
                ],  # TODO: add Ej to platform runcard - single qubit gates settings
                "fluxline": len(freqs) * [platform.qubits[qubit].flux],  # fluxline
            }
        )
        data.add_data_from_dict(r)

    return data


def _fit(data: QubitFluxData) -> QubitFluxResults:
    """
    Post-processing for QubitFlux Experiment.
    Fit frequency as a function of current for the flux qubit spectroscopy
    data (QubitFluxData): data object with information on the feature response at each current point.
    """

    qubits = data.df["qubit"].unique()
    frequency = {}
    sweetspot = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit]
        bias_keys = qubit_data["bias"].pint.to("V").pint.magnitude.unique()
        frequency_keys = qubit_data["frequency"].pint.to("Hz").pint.magnitude.unique()

        fluxlines = (
            qubit_data["fluxline"].pint.to("dimensionless").pint.magnitude.unique()
        )
        Ec = qubit_data["Ec"].pint.to("dimensionless").pint.magnitude.unique()
        Ej = qubit_data["Ej"].pint.to("dimensionless").pint.magnitude.unique()

        for fluxline in fluxlines:
            qubit_data = qubit_data[qubit_data["fluxline"] == fluxline]
            qubit_data[bias_keys[0]] = (
                qubit_data[bias_keys[0]].pint.to(bias_keys[1]).pint.magnitude
            )
            qubit_data[frequency_keys[0]] = (
                qubit_data[frequency_keys[0]].pint.to(frequency_keys[1]).pint.magnitude
            )
            if data.resonator_type == "2D":
                qubit_data["MSR"] = -qubit_data["MSR"]

            biases = qubit_data[bias_keys[0]]
            frequencies = qubit_data[frequency_keys[0]]
            msr = qubit_data["MSR"] * 1e6

            frequencies, biases = image_to_curve(frequencies, biases, msr)
            if fluxline == qubit:
                scaler = 10**9
                try:
                    max_c = biases[np.argmax(frequencies)]
                    min_c = biases[np.argmin(frequencies)]
                    xi = 1 / (2 * abs(max_c - min_c))  # Convert bias to flux.

                    # First order approximation: Ec and Ej NOT provided
                    if (Ec and Ej) is None:
                        f_q_0 = np.max(
                            frequencies
                        )  # Initial estimation for qubit frequency at sweet spot.
                        popt = curve_fit(
                            freq_q_transmon,
                            biases,
                            frequencies / scaler,
                            p0=[max_c, xi, 0, f_q_0 / scaler],
                        )[0]
                        popt[3] *= scaler
                        f_qs = popt[3]  # Qubit frequency at sweet spot.
                        f_q_offset = freq_q_transmon(
                            0, *popt
                        )  # Qubit frequenct at zero current.
                        C_ii = (f_qs - f_q_offset) / popt[
                            0
                        ]  # Corresponding flux matrix element.

                        frequency[qubit] = f_qs
                        sweetspot[qubit] = popt[0]
                        # fitted_parameters = xi, d, f_q_offset, C_ii
                        fitted_parameters[qubit] = (
                            popt[1],
                            abs(popt[2]),
                            f_q_offset,
                            C_ii,
                        )

                    # Second order approximation: Ec and Ej provided
                    elif (Ec and Ej) is not None:
                        freq_q_mathieu1 = partial(freq_q_mathieu, p7=0.4999)
                        popt = curve_fit(
                            freq_q_mathieu1,
                            biases,
                            frequencies / scaler,
                            p0=[max_c, xi, 0, Ec / scaler, Ej / scaler],
                            method="dogbox",
                        )[0]
                        popt[3] *= scaler
                        popt[4] *= scaler
                        f_qs = freq_q_mathieu(
                            popt[0], *popt
                        )  # Qubit frequency at sweet spot.
                        f_q_offset = freq_q_mathieu(
                            0, *popt
                        )  # Qubit frequenct at zero current.
                        C_ii = (f_qs - f_q_offset) / popt[
                            0
                        ]  # Corresponding flux matrix element.

                        frequency[qubit] = f_qs
                        sweetspot[qubit] = popt[0]
                        # fitted_parameters = xi, d, Ec, Ej, f_q_offset, C_ii
                        fitted_parameters[qubit] = (
                            popt[1],
                            abs(popt[2]),
                            popt[3],
                            popt[4],
                            f_q_offset,
                            C_ii,
                        )

                    else:
                        log.warning(
                            "qubit_flux_fit: the fitting was not succesful. Not enought guess parameters provided"
                        )

                except:
                    log.warning("qubit_flux_fit: the fitting was not succesful")

            else:
                try:
                    freq_min = np.min(frequencies)
                    freq_max = np.max(frequencies)
                    freq_norm = (frequencies - freq_min) / (freq_max - freq_min)
                    popt = curve_fit(line, biases, freq_norm)[0]
                    popt[0] = popt[0] * (freq_max - freq_min)
                    popt[1] = popt[1] * (freq_max - freq_min) + freq_min  # C_ij

                    frequency[qubit] = None
                    sweetspot[qubit] = None
                    fitted_parameters[qubit] = popt[0], popt[1]
                except:
                    log.warning("qubit_flux_fit: the fitting was not succesful")

    return QubitFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: QubitFluxData, fit: QubitFluxResults, qubit):
    """Plotting function for QubitFlux Experiment."""
    return utils.flux_dependence_plot(data, fit, qubit)


qubit_flux = Routine(_acquisition, _fit, _plot)
"""QubitFlux Routine object."""
