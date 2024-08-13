import argparse

from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.task import Completed
from qibocal.cli.report import report

MAX_CHI2 = 10
"""Maximum chi2 to allow platform update."""


def check_chi2(
    node: Completed,
    platform: Platform,
    targets: list[QubitId],
    max_chi2: float = MAX_CHI2,
):
    """Chi2 control and update."""
    for target in targets:
        try:
            getattr(node.results, "chi2")
            if node.results.chi2[target][0] > max_chi2:
                print(
                    f"{node.task.id} has chi2 {node.results.chi2[target][0]} greater than {max_chi2}. Stopping."
                )
            else:
                node.update_platform(platform)
        except AttributeError:
            node.update_platform(platform)


def main(targets: list[QubitId], platform_name: str, output: str):

    with Executor.open(
        "myexec",
        path=output,
        platform=platform_name,
        targets=targets,
        update=False,
        force=True,
    ) as e:

        # flux_dependence_output = e.qubit_flux(bias_step=0.05,
        #                             bias_width=1,
        #                             drive_amplitude=0.05,
        #                             drive_duration=4000,
        #                             freq_step=1_000_000,
        #                             freq_width=50_000_000,
        #                             relaxation_time=10_000,
        #                             nshots=1024)

        # flux_dependence_output = e.qubit_flux(bias_step=0.005,
        #                             bias_width=0.1,
        #                             drive_amplitude=0.05,
        #                             drive_duration=4000,
        #                             freq_step=1_000_000,
        #                             freq_width=30_000_000,
        #                             relaxation_time=10_000,
        #                             nshots=1024)

        # flux_dependence_output.update_platform(platform)

        rabi_signal_output = e.rabi_amplitude_signal(
            min_amp_factor=0.1,
            max_amp_factor=1.5,
            step_amp_factor=0.01,
            nshots=1024,
        )
        check_chi2(rabi_signal_output, platform=e.platform, targets=targets)

        classification_output = e.single_shot_classification(
            nshots=5000,
        )

        check_chi2(classification_output, platform=e.platform, targets=targets)

        rabi_output = e.rabi_amplitude(
            min_amp_factor=0.1,
            max_amp_factor=1.5,
            step_amp_factor=0.01,
            nshots=1024,
        )

        check_chi2(rabi_output, platform=e.platform, targets=targets)

        ramsey_output = e.ramsey(
            delay_between_pulses_start=10,
            delay_between_pulses_end=1000,
            delay_between_pulses_step=25,
            detuning=5_000_000,
            nshots=1024,
        )

        check_chi2(ramsey_output, platform=e.platform, targets=targets)

        drag_output = e.drag_tuning(beta_start=-2, beta_end=2, beta_step=0.2)

        check_chi2(drag_output, platform=e.platform, targets=targets, max_chi2=50)

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
