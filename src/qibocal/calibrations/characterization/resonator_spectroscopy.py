import numpy as np
from qibo.config import log
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.calibrations.characterization.utils import variable_resolution_scanrange
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase)
def resonator_spectroscopy(
    platform: AbstractPlatform,
    qubits: list,
    lowres_width,
    lowres_step,
    highres_width,
    highres_step,
    software_averages,
    points=10,
):

    r"""
    Perform spectroscopy on the 2D or 3D readout resonator.
    This routine executes a variable resolution scan around the expected resonator frequency indicated
    in the platform runcard. Afterthat, a final sweep with more precision is executed centered in the new
    resonator frequency found.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        lowres_width (int): Width frequenecy in HZ to perform the low resolution sweep
        lowres_step (int): Step frequenecy in HZ for the low resolution sweep
        highres_width (int): Width frequenecy in HZ to perform the high resolution sweep
        highres_step (int): Step frequenecy in HZ for the high resolution sweep
        precision_width (int): Width frequenecy in HZ to perform the precision resolution sweep
        precision_step (int): Step frequenecy in HZ for the precission resolution sweep
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz

        - A DataUnits object with the fitted data obtained with the following keys

            - **resonator_freq**: frequency
            - **peak_voltage**: peak voltage
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
    """

    platform.reload_settings()

    # create pulse sequence
    sequence = PulseSequence()

    # collect readout pulses and resonator frequencies for all qubits
    resonator_frequencies = {}
    ro_pulses = {}
    qrm_LOs = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])
        resonator_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "resonator_freq"
        ]
        qrm_LOs[qubit] = platform.qrm[qubit].ports["o1"].lo_frequency

    delta_frequency_range = variable_resolution_scanrange(
        lowres_width, lowres_step, highres_width, highres_step
    )

    fast_sweep_data = DataUnits(
        name=f"fast_sweep", quantities={"frequency": "Hz"}, options=["qubit"]
    )

    count = 0
    for _ in range(software_averages):
        for delta_freq in delta_frequency_range:
            if count % points == 0 and count > 0:
                yield fast_sweep_data
                for qubit in qubits:
                    yield lorentzian_fit(
                        fast_sweep_data.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["resonator_freq", "peak_voltage", "MZ_freq"],
                        qrm_lo=qrm_LOs[qubit],
                    )

            # TODO: move to qibolab platform.set_lo_frequencies(qubits, frequencies)
            for qubit in qubits:
                platform.ro_port[qubit].lo_frequency = (
                    delta_freq
                    + resonator_frequencies[qubit]
                    - ro_pulses[qubit].frequency
                )

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]

                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": delta_freq + resonator_frequencies[qubit],
                    "qubit": qubit,
                }
                fast_sweep_data.add(results)
            count += 1
    yield fast_sweep_data


@plot("Frequency vs Attenuation", plots.frequency_attenuation_msr_phase)
@plot("MSR vs Frequency", plots.frequency_attenuation_msr_phase__cut)
# Does not add much value unless one could select the attenuation of the section.
def resonator_punchout(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    min_att,
    max_att,
    step_att,
    software_averages,
    points=10,
):

    r"""
    Perform spectroscopy on the readout resonator decreasing the attenuation applied to
    the read-out pulse, producing an increment of the power sent to the resonator.
    That shows the two regimes of a given resonator, low and high-power regimes.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        freq_width (int): Width frequenecy in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequenecy in HZ for the spectroscopy sweep
        min_att (int): Minimum value in db for the attenuation sweep
        max_att (int): Minimum value in db for the attenuation sweep
        step_att (int): Step attenuation in db for the attenuation sweep
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz
            - **attenuation[dB]**: attenuation value in db applied to the flux line

    """

    platform.reload_settings()

    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "attenuation": "dB"},
        options=["qubit"],
    )

    sequence = PulseSequence()
    ro_pulses = {}
    resonator_frequencies = {}

    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])
        resonator_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "resonator_freq"
        ]
    delta_frequency_range = np.arange(-freq_width, freq_width, freq_step) - (
        freq_width / 4
    )

    # TODO: move this explicit instruction to the platform

    attenuation_range = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    for _ in range(software_averages):
        for att in attenuation_range:
            for delta_freq in delta_frequency_range:
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                for qubit in qubits:
                    platform.ro_port[qubit].lo_frequency = (
                        delta_freq
                        + resonator_frequencies[qubit]
                        - ro_pulses[qubit].frequency
                    )
                    platform.ro_port[qubit].attenuation = att

                result = platform.execute_pulse_sequence(sequence)

                for qubit in qubits:
                    msr, phase, i, q = result[ro_pulses[qubit].serial]
                    results = {
                        "MSR[V]": msr * (np.exp(att / 10)),
                        "i[V]": i,
                        "q[V]": q,
                        "phase[rad]": phase,
                        "frequency[Hz]": delta_freq + resonator_frequencies[qubit],
                        "attenuation[dB]": att,
                        "qubit": qubit,
                    }
                    # TODO: implement normalization
                    data.add(results)
                count += 1

    yield data


@plot("MSR and Phase vs Frequency", plots.dispersive_frequency_msr_phase)
def dispersive_shift(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    software_averages,
    points=10,
):

    r"""
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        freq_width (int): Width frequenecy in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequenecy in HZ for the spectroscopy sweep
        software_averages (int): Number of executions of the routine for averaging results
        fluxlines (list): List of flux control lines associated to different qubits to sweep current
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the normal and shifted sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz

    """

    platform.reload_settings()

    # create pulse sequence
    sequence = PulseSequence()

    # collect readout pulses and resonator frequencies for all qubits
    resonator_frequencies = {}
    frequency_ranges = {}
    ro_pulses = {}
    qrm_LOs = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])
        resonator_frequencies[qubit] = platform.characterization["single_qubit"][qubit][
            "resonator_freq"
        ]
        frequency_ranges[qubit] = (
            np.arange(-freq_width, freq_width, freq_step) + resonator_frequencies[qubit]
        )

        qrm_LOs[qubit] = platform.qrm[qubit].ports["o1"].lo_frequency

    data_spec = DataUnits(
        name=f"data", quantities={"frequency": "Hz"}, options=["qubit"]
    )
    count = 0
    for _ in range(software_averages):
        for freq in range(len(frequency_ranges[qubit])):
            if count % points == 0 and count > 0:
                yield data_spec
                for qubit in qubits:
                    yield lorentzian_fit(
                        data_spec.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["resonator_freq", "peak_voltage", "MZ_freq"],
                        qrm_lo=qrm_LOs[qubit],
                    )

            for qubit in qubits:
                platform.ro_port[qubit].lo_frequency = (
                    frequency_ranges[qubit][freq] - ro_pulses[qubit].frequency
                )

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": frequency_ranges[qubit][freq],
                    "qubit": qubit,
                }
                data_spec.add(results)
                count += 1
    yield data_spec

    # Shifted Spectroscopy
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        RX_pulse = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulse.finish
        )
        sequence.add(RX_pulse)
        sequence.add(ro_pulses[qubit])

    data_shifted = DataUnits(
        name=f"data_shifted", quantities={"frequency": "Hz"}, options=["qubit"]
    )
    count = 0
    for _ in range(software_averages):
        for freq in range(len(frequency_ranges[qubit])):
            if count % points == 0 and count > 0:
                yield data_shifted
                for qubit in qubits:
                    yield lorentzian_fit(
                        data_shifted.get_column("qubit", qubit),
                        x="frequency[GHz]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=["resonator_freq", "peak_voltage", "MZ_freq"],
                        fit_file_name="fit_shifted",
                        qrm_lo=qrm_LOs[qubit],
                    )
            for qubit in qubits:
                platform.ro_port[qubit].lo_frequency = (
                    frequency_ranges[qubit][freq] - ro_pulses[qubit].frequency
                )

            result = platform.execute_pulse_sequence(sequence)

            for qubit in qubits:
                msr, phase, i, q = result[ro_pulses[qubit].serial]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": frequency_ranges[qubit][freq],
                    "qubit": qubit,
                }
                data_shifted.add(results)
                count += 1
    yield data_shifted
