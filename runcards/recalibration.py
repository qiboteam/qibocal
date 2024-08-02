import argparse
from pathlib import Path

from qibo.backends import construct_backend
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.history import History
from qibocal.auto.output import Metadata, Output
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

    folder = Path(f"{output}_{'_'.join(targets)}")

    backend = construct_backend(backend="qibolab", platform=platform_name)
    platform = backend.platform
    if platform is None:
        raise ValueError("Qibocal requires a Qibolab platform to run.")

    executor = Executor(
        name="myexec",
        history=History(),
        platform=platform,
        targets=targets,
        update=False,
    )

    # generate output folder
    path = Output.mkdir(folder, force=True)

    # generate meta
    meta = Metadata.generate(path.name, backend)
    output = Output(History(), meta, platform)
    output.dump(path)

    from myexec import (
        drag_tuning,
        rabi_amplitude,
        rabi_amplitude_signal,
        ramsey,
        single_shot_classification,
    )

    # connect and initialize platform
    platform.connect()

    # run
    meta.start()

    # flux_dependence_output = qubit_flux(bias_step=0.01,
    #                             bias_width=0.1,
    #                             drive_amplitude=0.05,
    #                             drive_duration=4000,
    #                             freq_step=500_000,
    #                             freq_width=20_000_000,
    #                             relaxation_time=10_000,
    #                             nshots=1024)

    # flux_dependence_output.update_platform(platform)

    rabi_signal_output = rabi_amplitude_signal(
        min_amp_factor=0.1,
        max_amp_factor=1.5,
        step_amp_factor=0.01,
        nshots=1024,
    )
    check_chi2(rabi_signal_output, platform=platform, targets=targets)

    classification_output = single_shot_classification(
        nshots=5000,
    )

    check_chi2(classification_output, platform=platform, targets=targets)

    rabi_output = rabi_amplitude(
        min_amp_factor=0.1,
        max_amp_factor=1.5,
        step_amp_factor=0.01,
        nshots=1024,
    )

    check_chi2(rabi_output, platform=platform, targets=targets)

    ramsey_output = ramsey(
        delay_between_pulses_start=10,
        delay_between_pulses_end=2000,
        delay_between_pulses_step=20,
        detuning=5_000_000,
        nshots=1024,
    )

    check_chi2(ramsey_output, platform=platform, targets=targets)

    drag_output = drag_tuning(beta_start=-1, beta_end=1, beta_step=0.1)

    check_chi2(drag_output, platform=platform, targets=targets, max_chi2=50)

    meta.end()

    # stop and disconnect platform
    platform.disconnect()

    history = executor.history
    # dump history, metadata, and updated platform
    output.history = history
    output.dump(path)

    report(path, history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Qubit recalibration")
    parser.add_argument("--platform", type=str, help="Qibo platform")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument(
        "--targets", nargs="+", help="Target qubit to recalibrate", required=True
    )

    args = parser.parse_args()
    main(targets=args.targets, platform_name=args.platform, output=args.output)
