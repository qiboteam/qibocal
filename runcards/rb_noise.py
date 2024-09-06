import argparse

from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

# Change bias points here
# in this case I am addressing D1 and I am modifying the bias
# point of the D2 and D3
NEW_BIAS = {
    "D2": -0.219,
    "D3": -0.0074,
}


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

        # changing bias of neighbor qubit
        for qubit, new_bias in NEW_BIAS.items():
            e.platform.qubits[qubit].flux.offset = new_bias

        rb_output = e.rb_ondevice(
            apply_inverse=True,
            delta_clifford=10,
            max_circuit_depth=500,
            n_avg=1,
            num_of_sequences=10000,
            save_sequences=True,
            state_discrimination=True,
        )

        # qubit_spectroscopy_output = e.qubit_spectroscopy(
        #     freq_width=40_000_000,
        #     freq_step=500_000,
        #     drive_duration=2000,
        #     drive_amplitude=0.01,
        #     relaxation_time=5000,
        #     nshots=1024,
        # )

        # qubit_spectroscopy_output.update_platform(platform)

        # rabi_output = e.rabi_amplitude_signal(
        #     min_amp_factor=0.1,
        #     max_amp_factor=2,
        #     step_amp_factor=0.03,
        #     pulse_length=40,
        # )

        # rabi_output.update_platform(platform)

        # classification_output = e.single_shot_classification(
        #     nshots=5000,
        # )

        # classification_output.update_platform(platform)

        # rabi_output = e.rabi_amplitude(
        #     min_amp_factor=0.1,
        #     max_amp_factor=2,
        #     step_amp_factor=0.03,
        #     pulse_length=40,
        # )

        # rabi_output.update_platform(platform)

        ramsey = e.ramsey(
            delay_between_pulses_start=10,
            delay_between_pulses_end=1_000,
            delay_between_pulses_step=10,
            detuning=5_000_000,
        )

        ramsey.update_platform(platform)

        ramsey = e.ramsey(
            delay_between_pulses_start=10,
            delay_between_pulses_end=1_000,
            delay_between_pulses_step=10,
            detuning=5_000_000,
        )

        ramsey.update_platform(platform)

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

        classification_output = e.single_shot_classification(
            nshots=5000,
        )

        classification_output.update_platform(platform)

        rb_output = e.rb_ondevice(
            apply_inverse=True,
            delta_clifford=10,
            max_circuit_depth=500,
            n_avg=1,
            num_of_sequences=10000,
            save_sequences=True,
            state_discrimination=True,
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
