from pathlib import Path

import numpy as np
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Data, DataUnits
from qibocal.decorators import plot
from qibocal.fitting.classifier import run
from qibocal.fitting.classifier.qubit_fit import QubitFit

MESH_SIZE = 50
MARGIN = 0


@plot("Qubit States", plots.qubit_states)
def calibrate_qubit_states(
    platform: Platform, qubits: dict, nshots: int, classifiers, save_dir: str
):
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.Platform`): custom abstract platform on which we perform the calibration.
        qubits (dict): Dict of target Qubit objects to perform the action
        nshots (int): number of times the pulse sequence will be repeated.
        classifiers (list): list of classifiers
        save_dir (str): save path

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages
            - **state**: qubit state

    """

    # reload instrument settings from runcard
    platform.reload_settings()
    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    # create a DataUnits object to store the results
    data = DataUnits(name="data", options=["qubit", "iteration", "state"])

    # execute the first pulse sequence
    state0_results = platform.execute_pulse_sequence(state0_sequence, nshots=nshots)

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state0_results[ro_pulse.serial].raw
        r.update(
            {
                "qubit": [ro_pulse.qubit] * nshots,
                "iteration": np.arange(nshots),
                "state": [0] * nshots,
            }
        )
        data.add_data_from_dict(r)

    # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots)

    # retrieve and store the results for every qubit
    for ro_pulse in ro_pulses.values():
        r = state1_results[ro_pulse.serial].raw
        r.update(
            {
                "qubit": [ro_pulse.qubit] * nshots,
                "iteration": np.arange(nshots),
                "state": [1] * nshots,
            }
        )
        data.add_data_from_dict(r)

    parameters = Data(
        name=f"parameters",
        quantities={
            "model_name",
            "rotation_angle",  # in degrees
            "threshold",
            "fidelity",
            "assignment_fidelity",
            "average_state0",
            "average_state1",
            "accuracy",
            "predictions",
            "grid",
            "y_test",
            "y_pred",
            "qubit",
        },
    )
    classifiers_dict = {}
    for qubit in qubits:
        benchmark_table, y_test, x_test, models, names, hpars_list = run.train_qubit(
            Path(save_dir), qubits[qubit], qubits_data=data.df, classifiers=classifiers
        )

        classifiers_dict = {
            **classifiers_dict,
            qubit: {names[j]: hpars_list[j] for j in range(len(names))},
        }
        y_test = y_test.astype(np.int64)
        state0_data = data.df[(data.df["qubit"] == qubit) & (data.df["state"] == 0)]
        state1_data = data.df[(data.df["qubit"] == qubit) & (data.df["state"] == 1)]
        # Build the grid for the contour plots
        max_x = (
            max(
                0,
                state0_data["i"].max().magnitude,
                state1_data["i"].max().magnitude,
            )
            + MARGIN
        )
        max_y = (
            max(
                0,
                state0_data["q"].max().magnitude,
                state1_data["q"].max().magnitude,
            )
            + MARGIN
        )
        min_x = (
            min(
                0,
                state0_data["i"].min().magnitude,
                state1_data["i"].min().magnitude,
            )
            - MARGIN
        )
        min_y = (
            min(
                0,
                state0_data["q"].min().magnitude,
                state1_data["q"].min().magnitude,
            )
            - MARGIN
        )
        i_values, q_values = np.meshgrid(
            np.linspace(min_x, max_x, num=MESH_SIZE),
            np.linspace(min_y, max_y, num=MESH_SIZE),
        )
        grid = np.vstack([i_values.ravel(), q_values.ravel()]).T

        for i, model in enumerate(models):
            grid_pred = np.round(
                np.reshape(model.predict(grid), q_values.shape)
            ).astype(np.int64)

            try:
                y_pred = model.predict_proba(x_test)[:, 1]
            except AttributeError:
                y_pred = model.predict(x_test)

            # Useful for NN that return as predictions the probability
            y_pred = y_pred.astype(np.float64)
            benchmarks = benchmark_table.iloc[i].to_dict()
            results1 = {}
            if isinstance(model, QubitFit):
                results1 = {
                    "rotation_angle": model.angle,
                    "threshold": model.threshold,
                    "fidelity": model.fidelity,
                    "assignment_fidelity": model.assignment_fidelity,
                    "average_state0": complex(*model.iq_mean0),  # transform in complex
                    "average_state1": complex(*model.iq_mean1),  # transform in complex
                }
            results2 = {
                "model_name": names[i],
                "predictions": grid_pred.tobytes(),
                "grid": grid.tobytes(),
                "y_test": y_test.tobytes(),
                "y_pred": y_pred.tobytes(),
                "qubit": qubit,
            }

            parameters.add({**results1, **results2, **benchmarks})
    platform.update({"classifiers_hpars": classifiers_dict})
    yield data
    yield parameters
