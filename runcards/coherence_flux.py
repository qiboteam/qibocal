from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from qibocal.protocols.flux_dependence.utils import (
    transmon_frequency,
    transmon_readout_frequency,
)

biases = np.arange(-0.025, 0.025, 0.002)
"bias points to sweep"

# Qubit spectroscopy
freq_width = 10_000_000
"""Width [Hz] for frequency sweep relative  to the qubit frequency."""
freq_step = 500_000
"""Frequency [Hz] step for sweep."""
drive_duration = 1000
"""Drive pulse duration [ns]. Same for all qubits."""

# Rabi amp signal
min_amp_factor = 0.0
"""Minimum amplitude multiplicative factor."""
max_amp_factor = 1.5
"""Maximum amplitude multiplicative factor."""
step_amp_factor = 0.01
"""Step amplitude multiplicative factor."""

# Flipping
nflips_max = 200
"""Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
nflips_step = 10
"""Flip step."""

# T1 signal
delay_before_readout_start = 16
"""Initial delay before readout [ns]."""
delay_before_readout_end = 100_000
"""Final delay before readout [ns]."""
delay_before_readout_step = 4_000
"""Step delay before readout [ns]."""

#  Ramsey signal
detuning = 3_000_000
"""Frequency detuning [Hz]."""
delay_between_pulses_start = 16
"""Initial delay between RX(pi/2) pulses in ns."""
delay_between_pulses_end = 1_000
"""Final delay between RX(pi/2) pulses in ns."""
delay_between_pulses_step = 5
"""Step delay between RX(pi/2) pulses in ns."""

# T2 and Ramsey signal
delay_between_pulses_start_T2 = 16
"""Initial delay between RX(pi/2) pulses in ns."""
delay_between_pulses_end_T2 = 80_000
"""Final delay between RX(pi/2) pulses in ns."""
delay_between_pulses_step_T2 = 500
"""Step delay between RX(pi/2) pulses in ns."""
single_shot_T2: bool = False
"""If ``True`` save single shot signal data."""

# Optional qubit spectroscopy
drive_amplitude: Optional[float] = 0.1
"""Drive pulse amplitude (optional). Same for all qubits."""
hardware_average: bool = True
"""By default hardware average will be performed."""
# Optional rabi amp signal
pulse_length: Optional[float] = 40
"""RX pulse duration [ns]."""
# Optional T1 signal
single_shot_T1: bool = False
"""If ``True`` save single shot signal data."""


parser = ArgumentParser()
parser.add_argument("--target", nargs="+", required=True, help="Target qubit")
parser.add_argument("--platform", type=str, required=True, help="Platform name")
parser.add_argument(
    "--path", type=str, default="TTESTCoherenceFlux1", help="Path for the output"
)
args = parser.parse_args()

targets = args.target
path = args.path


fit_function = transmon_frequency
platform = create_platform(args.platform)

for target in targets:
    params_qubit = {
        "w_max": platform.qubits[target].drive_frequency,
        "xj": 0,
        "d": platform.qubits[target].asymmetry,
        "normalization": platform.qubits[target].crosstalk_matrix[target],
        "offset": -platform.qubits[target].sweetspot
        * platform.qubits[target].crosstalk_matrix[target],
        "crosstalk_element": 1,
        "charging_energy": platform.qubits[target].Ec,
    }

    # NOTE: Center around the sweetspot
    centered_biases = biases + platform.qubits[target].sweetspot

    for i, bias in enumerate(centered_biases):
        with Executor.open(
            f"myexec_{i}",
            path=args.path / Path(f"flux_{bias}"),
            platform=args.platform,
            targets=[target],
            update=True,
            force=True,
        ) as e:

            # Change the flux
            e.platform.qubits[target].flux.offset = bias

            # Change the qubit frequency
            qubit_frequency = fit_function(bias, **params_qubit)  # * 1e9

            res_frequency = transmon_readout_frequency(
                bias,
                **params_qubit,
                g=platform.qubits[target].g,
                resonator_freq=platform.qubits[target].bare_resonator_frequency,
            )
            e.platform.qubits[target].drive_frequency = qubit_frequency
            e.platform.qubits[target].native_gates.RX.frequency = qubit_frequency

            res_spectroscopy_output = e.resonator_spectroscopy(
                freq_width=freq_width,
                freq_step=freq_step,
                power_level="low",
                relaxation_time=2000,
                nshots=1024,
            )
            rabi_output = e.rabi_amplitude_signal(
                min_amp_factor=min_amp_factor,
                max_amp_factor=max_amp_factor,
                step_amp_factor=step_amp_factor,
                pulse_length=e.platform.qubits[target].native_gates.RX.duration,
            )

            if rabi_output.results.amplitude[target] > 0.5:
                print(
                    f"Rabi fit has pi pulse amplitude {rabi_output.results.amplitude[target]}, greater than 0.5 not possible for QM. Skipping to next bias point."
                )
                e.platform.qubits[target].native_gates.RX.amplitude = (
                    0.5  # FIXME: For QM this should be 0.5
                )

                report(e.path, e.history)
                continue

            classification_output = e.single_shot_classification(nshots=5000)
            ramsey_output = e.ramsey(
                delay_between_pulses_start=delay_between_pulses_start,
                delay_between_pulses_end=delay_between_pulses_end,
                delay_between_pulses_step=delay_between_pulses_step,
                detuning=detuning,
            )

            ramsey_output = e.ramsey(
                delay_between_pulses_start=delay_between_pulses_start,
                delay_between_pulses_end=delay_between_pulses_end,
                delay_between_pulses_step=delay_between_pulses_step,
                detuning=detuning,
            )
            classification_output = e.single_shot_classification(nshots=5000)
            t1_output = e.t1(
                delay_before_readout_start=delay_before_readout_start,
                delay_before_readout_end=delay_before_readout_end,
                delay_before_readout_step=delay_before_readout_step,
                single_shot=single_shot_T1,
            )

            ramsey_t2_output = e.ramsey(
                delay_between_pulses_start=delay_between_pulses_start_T2,
                delay_between_pulses_end=delay_between_pulses_end_T2,
                delay_between_pulses_step=delay_between_pulses_step_T2,
                detuning=0,
            )
            readout_characterization_out = e.readout_characterization(
                delay=1000,
                nshots=5000,
            )
            rb_out = e.rb_ondevice(
                num_of_sequences=10000,
                max_circuit_depth=1000,
                delta_clifford=20,
                n_avg=1,
                save_sequences=False,
                apply_inverse=True,
            )
            report(e.path, e.history)
