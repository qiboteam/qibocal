import argparse

import numpy as np
from qibolab.qubits import QubitId
from scipy.optimize import minimize

from qibocal.auto.execute import Executor


def rb_infidelity(x, e, target):

    # print(f'trying amplitude: {float(amplitude)}...\n')

    amplitude = float(x[0])
    drag_param = float(x[1])

    e.platform.qubits[target].native_gates.RX.amplitude = float(amplitude)
    e.platform.qubits[target].native_gates.RX.shape = f"Drag(5, {drag_param})"

    rb_output = e.rb_ondevice(
        apply_inverse=True,
        delta_clifford=10,
        max_circuit_depth=200,
        n_avg=1,
        num_of_sequences=10000,
        save_sequences=False,
        state_discrimination=True,
    )

    one_minus_p = 1 - rb_output.results.pars.get(target)[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / 1.875

    if r_g < 1e-5:
        r_g = 1e-2

    print()
    print()
    print(f"trying amplitude: {float(amplitude)}...\n")
    print(f"trying drag param: {float(drag_param)}...\n")
    print(f"           reached infidelity: {r_g}\n")
    print()

    return r_g


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

        amplitude0 = 0.040
        drag_param0 = 0.05

        bnds = [(0.03, 0.05), (-0.1, 0.1)]

        x0 = np.array([amplitude0, drag_param0])

        target = targets[0]

        res = minimize(
            rb_infidelity,
            x0,
            args=(e, target),
            method="Nelder-Mead",
            tol=1e-7,
            bounds=bnds,
            options={"maxfev": 50, "disp": True},
        )

        print(res)
        print()
        print(res.x)

        # report(e.path, e.history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Qubit recalibration")
    parser.add_argument("--platform", type=str, help="Qibo platform")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument(
        "--targets", nargs="+", help="Target qubit to recalibrate", required=True
    )

    args = parser.parse_args()
    main(targets=args.targets, platform_name=args.platform, output=args.output)
