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

MAX_CHI2 = 5
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
            getattr(node.results[target], "chi2")
            try:
                node.results.chi2[target]

                if node.results.chi2[target][0] > max_chi2:
                    print(
                        f"{node.task.id} has chi2 {node.results.chi2[target][0]} greater than {max_chi2}. Stopping."
                    )
                else:
                    node.update_platform(platform)
            except KeyError:
                pass
        except AttributeError:
            node.update_platform(platform)


def main(targets: list, platform_name: str, output: str):

    folder = Path(f"{output}_{'_'.join(targets)}")
    force = True

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
    path = Output.mkdir(folder, force)

    # generate meta
    meta = Metadata.generate(path.name, backend)
    output = Output(History(), meta, platform)
    output.dump(path)

    from myexec import ramsey, readout_characterization, t1

    # connect and initialize platform
    platform.connect()

    # run
    meta.start()

    t1_output = t1(
        delay_before_readout_start=10,
        delay_before_readout_end=100000,
        delay_before_readout_step=500,
        nshots=1024,
    )

    check_chi2(t1_output, platform=platform, targets=targets)

    ramsey_output = ramsey(
        delay_between_pulses_start=10,
        delay_between_pulses_end=100000,
        delay_between_pulses_step=500,
        nshots=1024,
    )

    check_chi2(ramsey_output, platform=platform, targets=targets)

    # TODO: long without sweepers
    # t2_echo_output = spin_echo(delay_between_pulses_start=10,
    #                delay_between_pulses_end=100000,
    #                delay_between_pulses_step=500,
    #                unrolling=True,
    #                nshots=1024)

    ro_char_output = readout_characterization(nshots=5000, delay=1000)

    check_chi2(ro_char_output, platform=platform, targets=targets)
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
    parser.add_argument(
        "--output", default="monitoring", type=str, help="Output folder"
    )
    parser.add_argument(
        "--targets", nargs="+", help="Target qubit to recalibrate", required=True
    )

    args = parser.parse_args()
    main(targets=args.targets, platform_name=args.platform, output=args.output)
