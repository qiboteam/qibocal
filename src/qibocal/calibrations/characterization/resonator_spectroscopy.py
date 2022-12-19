import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.calibrations.characterization.utils import variable_resolution_scanrange
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__fast_precision)
def resonator_spectroscopy(
    platform: AbstractPlatform,
    qubit: int,
    lowres_width,
    lowres_step,
    highres_width,
    highres_step,
    precision_width,
    precision_step,
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
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys:
            - "MSR[V]": Resonator signal voltage mesurement in volts
            - "i[V]": Resonator signal voltage mesurement for the component I in volts
            - "q[V]": Resonator signal voltage mesurement for the component Q in volts
            - "phase[rad]": Resonator signal phase mesurement in radians
            - "frequency[Hz]": Resonator frequency value in Hz

        A DataUnits object with the fitted data obtained with the following keys:
            - resonator_freq: frequency
            - peak_voltage: peak voltage
            - *popt0*: Lorentzian's amplitude
            - *popt1*: Lorentzian's center
            - *popt2*: Lorentzian's sigma
            - *popt3*: Lorentzian's offset
    """

    platform.reload_settings()
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        variable_resolution_scanrange(
            lowres_width, lowres_step, highres_width, highres_step
        )
        + resonator_frequency
    )
    fast_sweep_data = DataUnits(
        name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield fast_sweep_data
                yield lorentzian_fit(
                    fast_sweep_data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                )

            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            fast_sweep_data.add(results)
            count += 1
    yield fast_sweep_data

    if platform.resonator_type == "3D":
        resonator_frequency = fast_sweep_data.get_values("frequency", "Hz")[
            np.argmax(fast_sweep_data.get_values("MSR", "V"))
        ]
        avg_voltage = (
            np.mean(
                fast_sweep_data.get_values("MSR", "V")[: (lowres_width // lowres_step)]
            )
            * 1e6
        )
    else:
        resonator_frequency = fast_sweep_data.get_values("frequency", "Hz")[
            np.argmin(fast_sweep_data.get_values("MSR", "V"))
        ]
        avg_voltage = (
            np.mean(
                fast_sweep_data.get_values("MSR", "V")[: (lowres_width // lowres_step)]
            )
            * 1e6
        )

    precision_sweep__data = DataUnits(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    freqrange = (
        np.arange(-precision_width, precision_width, precision_step)
        + resonator_frequency
    )

    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0 and count > 0:
                yield precision_sweep__data
                yield lorentzian_fit(
                    fast_sweep_data + precision_sweep__data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                )

            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            precision_sweep__data.add(results)
            count += 1
    yield precision_sweep__data


@plot("Frequency vs Attenuation", plots.frequency_attenuation_msr_phase)
@plot("MSR vs Frequency", plots.frequency_attenuation_msr_phase__cut)
def resonator_punchout(
    platform: AbstractPlatform,
    qubit: int,
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
        A DataUnits object with the raw data obtained with the following keys:
            - "MSR[V]": Resonator signal voltage mesurement in volts
            - "i[V]": Resonator signal voltage mesurement for the component I in volts
            - "q[V]": Resonator signal voltage mesurement for the component Q in volts
            - "phase[rad]": Resonator signal phase mesurement in radians
            - "frequency[Hz]": Resonator frequency value in Hz
            - "attenuation[dB]": attenuation value in db applied to the flux line

    """

    platform.reload_settings()

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence = PulseSequence()
    sequence.add(ro_pulse)

    # TODO: move this explicit instruction to the platform
    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step)
        + resonator_frequency
        - (freq_width / 4)
    )
    attenuation_range = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    for _ in range(software_averages):
        for att in attenuation_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.ro_port[qubit].attenuation = att
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr * (np.exp(att / 10)),
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "attenuation[dB]": att,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data


@plot("MSR and Phase vs Flux Current", plots.frequency_flux_msr_phase)
def resonator_spectroscopy_flux(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    current_max,
    current_min,
    current_step,
    software_averages,
    fluxline=0,
    points=10,
):

    r"""
    Perform spectroscopy on the readout resonator modifying the current applied in the flux control line.
    This routine works for quantum devices flux controlled.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        freq_width (int): Width frequenecy in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequenecy in HZ for the spectroscopy sweep
        current_max (int): Minimum value in mV for the flux current sweep
        current_min (int): Minimum value in mV for the flux current sweep
        current_step (int): Step attenuation in mV for the flux current sweep
        software_averages (int): Number of executions of the routine for averaging results
        fluxline (int): Flux line associated to the target qubit. If it is set to "qubit", the platform
                        automatically obtain the flux line number of the target qubit.
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained with the following keys:
            - "MSR[V]": Resonator signal voltage mesurement in volts
            - "i[V]": Resonator signal voltage mesurement for the component I in volts
            - "q[V]": Resonator signal voltage mesurement for the component Q in volts
            - "phase[rad]": Resonator signal phase mesurement in radians
            - "frequency[Hz]": Resonator frequency value in Hz
            - "current[A]": Current value in mA applied to the flux line

    """

    platform.reload_settings()

    if fluxline == "qubit":
        fluxline = qubit

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    qubit_biasing_current = platform.characterization["single_qubit"][qubit][
        "sweetspot"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )
    current_range = (
        np.arange(current_min, current_max, current_step) + qubit_biasing_current
    )

    count = 0
    for _ in range(software_averages):
        for curr in current_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.qf_port[fluxline].current = curr
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "current[A]": curr,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data
    # TODO: automatically extract the sweet spot current
    # TODO: add a method to generate the matrix


@plot("MSR row 1 and Phase row 2", plots.frequency_flux_msr_phase__matrix)
def resonator_spectroscopy_flux_matrix(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    current_min,
    current_max,
    current_step,
    fluxlines,
    software_averages,
    points=10,
):

    r"""
    Perform spectroscopy on the readout resonator modifying the current applied in the given flux control lines associated
    to a different qubits. As a result we obtain a matrix of plots where the flux dependence is shown for a list of qubits.
    This routine works for quantum devices flux controlled.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubit (int): Target qubit to perform the action
        freq_width (int): Width frequenecy in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequenecy in HZ for the spectroscopy sweep
        current_max (int): Minimum value in mV for the flux current sweep
        current_min (int): Minimum value in mV for the flux current sweep
        current_step (int): Step attenuation in mV for the flux current sweep
        software_averages (int): Number of executions of the routine for averaging results
        fluxlines (list): List of flux control lines associated to different qubits to sweep current
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained with the following keys:
            - "MSR[V]": Resonator signal voltage mesurement in volts
            - "i[V]": Resonator signal voltage mesurement for the component I in volts
            - "q[V]": Resonator signal voltage mesurement for the component Q in volts
            - "phase[rad]": Resonator signal phase mesurement in radians
            - "frequency[Hz]": Resonator frequency value in Hz
            - "current[A]": Current value in mA applied to the flux line

    """

    platform.reload_settings()

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )
    current_range = np.arange(current_min, current_max, current_step)

    count = 0
    for fluxline in fluxlines:
        fluxline = int(fluxline)
        print(fluxline)
        data = DataUnits(
            name=f"data_q{qubit}_f{fluxline}",
            quantities={"frequency": "Hz", "current": "A"},
        )
        for _ in range(software_averages):
            for curr in current_range:
                for freq in frequency_range:
                    if count % points == 0:
                        yield data
                    platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                    platform.qf_port[fluxline].current = curr
                    msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                        ro_pulse.serial
                    ]
                    results = {
                        "MSR[V]": msr,
                        "i[V]": i,
                        "q[V]": q,
                        "phase[rad]": phase,
                        "frequency[Hz]": freq,
                        "current[A]": curr,
                    }
                    # TODO: implement normalization
                    data.add(results)
                    count += 1

    yield data


@plot("MSR and Phase vs Frequency", plots.dispersive_frequency_msr_phase)
def dispersive_shift(
    platform: AbstractPlatform,
    qubit: int,
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
        A DataUnits object with the raw data obtained for the normal and shifted sweeps with the following keys:
            - "MSR[V]": Resonator signal voltage mesurement in volts
            - "i[V]": Resonator signal voltage mesurement for the component I in volts
            - "q[V]": Resonator signal voltage mesurement for the component Q in volts
            - "phase[rad]": Resonator signal phase mesurement in radians
            - "frequency[Hz]": Resonator frequency value in Hz

    """

    platform.reload_settings()

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )

    data_spec = DataUnits(name=f"data_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield data_spec
                yield lorentzian_fit(
                    data_spec,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                )
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data_spec.add(results)
            count += 1
    yield data_spec

    # Shifted Spectroscopy
    sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.finish)
    sequence.add(RX_pulse)
    sequence.add(ro_pulse)

    data_shifted = DataUnits(
        name=f"data_shifted_q{qubit}", quantities={"frequency": "Hz"}
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield data_shifted
                yield lorentzian_fit(
                    data_shifted,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                    fit_file_name="fit_shifted",
                )
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data_shifted.add(results)
            count += 1
    yield data_shifted
