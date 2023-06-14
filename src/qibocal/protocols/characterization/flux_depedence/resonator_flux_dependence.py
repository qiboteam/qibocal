from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from . import utils


# TODO: implement cross-talk (maybe separate routine?)
@dataclass
class ResonatorFluxParameters(Parameters):
    """ResonatorFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep (V)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ResonatorFluxResults(Results):
    """ResonatoFlux outputs."""

    sweetspot: Dict[QubitId, float] = field(metadata=dict(update="sweetspot"))
    """Sweetspot for each qubit."""
    frequency: Dict[QubitId, float] = field(metadata=dict(update="readout_frequency"))
    """Readout frequency for each qubit."""

    #TODO: After testing fitting, decide which fitted params should be saved/updated in the runcard
    fitted_parameters: Dict[QubitId, Dict[str, float]]
    """Raw fitting output."""


class ResonatorFluxData(DataUnits):
    """ResonatorFlux acquisition outputs."""

    def __init__(self, resonator_type):
        super().__init__(
            "data",
            {"frequency": "Hz", "bias": "V"},
            options=["qubit", "Ec", "Ej", "g", "f_rh": "Hz" ],
        )
        self._resonator_type = resonator_type

    @property
    def resonator_type(self):
        """Type of resonator"""
        return self._resonator_type

def _acquisition(
    params: ResonatorFluxParameters, platform: Platform, qubits: Qubits
) -> ResonatorFluxData:
    """Data acquisition for ResonatorFlux experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
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
    data = ResonatorFluxData(platform.resonator_type)

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
            * list(delta_frequency_range + ro_pulses[qubit].frequency)
        ).flatten()
        # store the results
        r = {k: v.ravel() for k, v in result.serialize.items()}
        r.update(
            {
                "frequency[Hz]": freqs,
                "bias[V]": biases,
                "qubit": len(freqs) * [qubit],
                "Ec": len(freqs) * [platform.qubits[qubit].Ec], #TODO: add to platform runcard - single qubit gates settings
                "Ej": len(freqs) * [platform.qubits[qubit].Ej], #TODO: add to platform runcard - single qubit gates settings
                "g": len(freqs) * [platform.qubits[qubit].g] #TODO: g in the readout coupling - add to platform runcard - single qubit gates settings
                "f_rh": len(freqs) * [platform.qubits[qubit].bare_resonator_frequency] #TODO: Resonator frequency at high power - add to platform runcard - single qubit gates settings
                "fluxline": len(freqs) * [platform.qubits[qubit].flux] #fluxline
            }
        )
        data.add_data_from_dict(r)

    return data


def _fit(data: ResonatorFluxData) -> ResonatorFluxResults:
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

        fluxlines = qubit_data["fluxline"].pint.to("dimensionless").pint.magnitude.unique()
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
                    scaler = 10**9
                    f_rh = qubit_data["f_rh"].pint.to("Hz").pint.magnitude.unique()  # Resonator frequency at high power.
                    g = qubit_data["g"].pint.to("dimensionless").pint.magnitude.unique()  # Readout coupling.
                    max_c = biases[np.argmax(frequencies)]
                    min_c = biases[np.argmin(frequencies)]
                    xi = 1 / (2 * abs(max_c - min_c))  # Convert bias to flux.

                    # First order approximation: f_rh, g provided
                    if ( (Ec and Ej) is None ) and ( (f_rh and g) is not None ):
                        f_r_0 = np.max(frequencies)  # Initial estimation for resonator frequency at sweet spot.
                        f_q_0 = f_rh - g**2 / (f_r_0 - f_rh)  # Initial estimation for qubit frequency at sweet spot.
                        popt = curve_fit(
                            freq_r_transmon,
                            biases,
                            frequencies / scaler,
                            p0=[max_c, xi, 0, f_q_0 / f_rh, g / scaler, f_rh / scaler],
                        )[0]
                        popt[4] *= scaler
                        popt[5] *= scaler
                        f_qs = popt[3] * popt[5]  # Qubit frequency at sweet spot.
                        f_rs = freq_r_transmon(popt[0], *popt)  # Resonator frequency at sweet spot.
                        f_r_offset = freq_r_transmon(0, *popt)  # Resonator frequency at zero current.
                        C_ii = (f_rs - f_r_offset) / popt[0]  # Corresponding flux matrix element.

                        frequency[qubit] = f_rs
                        sweetspot[qubit] = popt[0]
                        #fitted_parameters = xi, d, f_q/f_rh, g, f_rh, f_qs, f_r_offset, C_ii
                        fitted_parameters[qubit] = popt[1], abs(popt[2]), popt[3], g, f_rh, f_qs, f_r_offset, C_ii

                    # Second order approximation: f_rh, g, Ec, Ej provided
                    elif (Ec and Ej and f_rh and g) is not None:
                        freq_r_mathieu1 = partial(freq_r_mathieu, p7=0.4999)
                        popt = curve_fit(
                            freq_r_mathieu1,
                            biases,
                            frequencies / scaler,
                            p0=[
                                f_rh / scaler,
                                g / scaler,
                                max_c,
                                xi,
                                0,
                                Ec / scaler,
                                Ej / scaler,
                            ],
                            method="dogbox",
                        )[0]
                        popt[0] *= scaler
                        popt[1] *= scaler
                        popt[5] *= scaler
                        popt[6] *= scaler
                        f_qs = freq_q_mathieu(popt[2], *popt[2::])  # Qubit frequency at sweet spot.
                        f_rs = freq_r_mathieu(popt[2], *popt)  # Resonator frequency at sweet spot.
                        f_r_offset = freq_r_mathieu(0, *popt)  # Resonator frequenct at zero current.
                        C_ii = (f_rs - f_r_offset) / popt[2]  # Corresponding flux matrix element.

                        frequency[qubit] = f_rs
                        sweetspot[qubit] = popt[2]
                        #fitted_parameters = xi, d, g, Ec, Ej, f_rh, f_qs, f_r_offset, C_ii
                        fitted_parameters[qubit] = popt[3], abs(popt[4]), popt[1], popt[5], popt[6], popt[0], f_qs, f_r_offset, C_ii

                    else:
                        log.warning("resonator_flux_fit: the fitting was not succesful. Not enought guess parameters provided")

                except:
                    log.warning("resonator_flux_fit: the fitting was not succesful")

            else:
                try:
                    freq_min = np.min(frequencies)
                    freq_max = np.max(frequencies)
                    freq_norm = (frequencies - freq_min) / (freq_max - freq_min)
                    popt = curve_fit(line, biases, freq_norm)[0]
                    popt[0] = popt[0] * (freq_max - freq_min)
                    popt[1] = popt[1] * (freq_max - freq_min) + freq_min # C_ij

                    frequency[qubit] = None
                    sweetspot[qubit] = None
                    fitted_parameters[qubit] = popt[0], popt[1]
                except:
                    log.warning("resonator_flux_fit: the fitting was not succesful")

    return ResonatorFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, qubit):
    """Plotting function for ResonatorFlux Experiment."""
    return utils.flux_dependence_plot(data, fit, qubit)


resonator_flux = Routine(_acquisition, _fit, _plot)
"""ResonatorFlux Routine object."""
