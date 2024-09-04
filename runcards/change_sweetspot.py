import argparse

from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from qibocal.protocols.flux_dependence.utils import transmon_frequency

MAXIMUM_FREQUENCIES = {
    "D1": 4.9578e9,
    "D2": 5.5637e9,
    "D3": 5.6522e9,
    "D4": 6.249116e9,
    "D5": 5.4218869e9,
}
# QUBIT_PARAMS = {}
BIAS = {"D1": 0.122, "D2": -0.4192, "D3": -0.2074, "D4": -0.433, "D5": -0.04}
# QUBIT_FREQUENCIES = {}
# NEW_BIAS = {
#     "D1": 0.1,
# }


def main(targets: list[QubitId], platform_name: str, output: str):

    with Executor.open(
        "myexec",
        path=output,
        platform=platform_name,
        targets=targets,
        update=False,
        force=True,
    ) as e:
        platform = e.platform

        for qubit in targets:
            e.platform.qubits[qubit].flux.offset = BIAS[qubit]
            params_qubit = {
                "w_max": MAXIMUM_FREQUENCIES[qubit],
                "xj": 0,
                "d": 0,
                "normalization": platform.qubits[qubit].crosstalk_matrix[qubit],
                "offset": -platform.qubits[qubit].sweetspot
                * platform.qubits[qubit].crosstalk_matrix[qubit],
                "crosstalk_element": 1,
                "charging_energy": platform.qubits[qubit].Ec * 1e-9,
            }
            new_frequency = transmon_frequency(BIAS[qubit], **params_qubit)
            platform.qubits[qubit].native_gates.RX.frequency = new_frequency
            print(new_frequency)
            platform.qubits[qubit].drive_frequency = new_frequency

        # resonator_spectroscopy_output = e.resonator_spectroscopy(
        #     freq_width=5_000_000,
        #     freq_step=100_000,
        #     relaxation_time=5000,
        #     nshots=1024,
        #     power_level="low",
        # )

        # resonator_spectroscopy_output.update_platform(platform)

        qubit_spectroscopy_output = e.qubit_spectroscopy(
            freq_width=50_000_000,
            freq_step=500_000,
            drive_duration=2000,
            drive_amplitude=0.01,
            relaxation_time=5000,
            nshots=1024,
        )

        qubit_spectroscopy_output.update_platform(platform)

        rabi_output = e.rabi_amplitude_signal(
            min_amp_factor=0.5,
            max_amp_factor=2,
            step_amp_factor=0.03,
            pulse_length=40,
        )

        rabi_output.update_platform(platform)

        classification_output = e.single_shot_classification(
            nshots=5000,
        )

        classification_output.update_platform(platform)

        rabi_output = e.rabi_amplitude(
            min_amp_factor=0.1,
            max_amp_factor=2,
            step_amp_factor=0.03,
            pulse_length=40,
        )

        rabi_output.update_platform(platform)

        ramsey = e.ramsey(
            delay_between_pulses_start=10,
            delay_between_pulses_end=1_000,
            delay_between_pulses_step=20,
            detuning=10_000_000,
        )

        ramsey.update_platform(platform)

        ramsey = e.ramsey(
            delay_between_pulses_start=10,
            delay_between_pulses_end=1_000,
            delay_between_pulses_step=20,
            detuning=10_000_000,
        )

        ramsey.update_platform(platform)

        classification_output = e.single_shot_classification(
            nshots=5000,
        )

        classification_output.update_platform(platform)

        t1_output = e.t1(
            delay_before_readout_start=10,
            delay_before_readout_end=100_000,
            delay_before_readout_step=1_000,
        )

        ramsey = e.ramsey(
            delay_between_pulses_start=10,
            delay_between_pulses_end=100_000,
            delay_between_pulses_step=1_000,
        )

        report(e.path, e.history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Qubit recalibration")
    parser.add_argument("--platform", type=str, help="Qibo platform")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument(
        "--targets", nargs="+", help="Target qubit to recalibrate", required=True
    )

    args = parser.parse_args()
    main(targets=args.targets, platform_name=args.platform, output=args.output)
