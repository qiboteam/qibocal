import argparse

from qibocal.auto.execute import Executor
from qibocal.cli.report import report


def main(pair: list, platform_name: str, output: str):

    with Executor.open(
        "myexec",
        path=output,
        platform=platform_name,
        targets=[pair],
        update=False,
        force=True,
    ) as e:

        platform = e.platform

        amplitude = platform.pairs[tuple(pair)].native_gates.CZ.pulses[0].amplitude
        duration = platform.pairs[tuple(pair)].native_gates.CZ.pulses[0].duration

        # TODO: chage once main is merged
        cz_sweep_output = e.cz_sweep(
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

        report(e.path, e.history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Qubit recalibration")
    parser.add_argument("--platform", type=str, help="Qibo platform")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument(
        "--pair", nargs="+", help="Target pair to recalibrate", required=True
    )
    args = parser.parse_args()
    main(pair=args.pair, platform_name=args.platform, output=args.output)
