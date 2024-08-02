import argparse
from pathlib import Path

from qibo.backends import construct_backend

from qibocal.auto.execute import Executor
from qibocal.auto.history import History
from qibocal.auto.output import Metadata, Output
from qibocal.cli.report import report


def main(pair: list, platform_name: str, output: str):

    folder = Path(f"{output}_{'_'.join(pair)}")

    backend = construct_backend(backend="qibolab", platform=platform_name)
    platform = backend.platform
    if platform is None:
        raise ValueError("Qibocal requires a Qibolab platform to run.")

    executor = Executor(
        name="myexec",
        history=History(),
        platform=platform,
        targets=[pair],
        update=False,
    )

    # generate output folder
    path = Output.mkdir(folder, force=True)

    # generate meta
    meta = Metadata.generate(path.name, backend)
    output = Output(History(), meta, platform)
    output.dump(path)

    from myexec import cz_sweep

    amplitude = platform.pairs[tuple(pair)].native_gates.CZ.pulses[0].amplitude
    duration = platform.pairs[tuple(pair)].native_gates.CZ.pulses[0].duration
    # connect and initialize platform
    platform.connect()

    # run
    meta.start()

    cz_sweep_output = cz_sweep(
        flux_pulse_amplitude_min=0.9 * amplitude,
        flux_pulse_amplitude_max=1.1 * amplitude,
        flux_pulse_amplitude_step=amplitude * 0.01,
        duration_max=duration + 1,
        duration_min=duration - 1,
        duration_step=1,
        theta_start=0,
        theta_end=7,
        theta_step=0.5,
        relaxation_time=50000,
    )

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
        "--pair", nargs="+", help="Target pair to recalibrate", required=True
    )
    args = parser.parse_args()
    main(pair=args.pair, platform_name=args.platform, output=args.output)
