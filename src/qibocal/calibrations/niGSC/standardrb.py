"""
"""


from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import SingleCliffordsFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig


class moduleFactory(SingleCliffordsFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        self.name = "SingleCliffordsInv"

    def build_circuit(self, depth: int):
        circuit = Circuit(len(self.qubits))
        for _ in range(depth):
            circuit.add(self.gate_layer())
        # If there is at least one gate in the circuit, add an inverse.
        if depth > 0:
            # Build a gate out of the unitary of the whole circuit and
            # take the daggered version of that.
            circuit.add(
                gates.Unitary(circuit.unitary(), *range(len(self.qubits))).dagger()
            )
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


class moduleExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int | None = None,
        data: list | None = None,
        noisemodel: NoiseModel = None,
    ) -> None:
        super().__init__(circuitfactory, data, nshots, noisemodel)
        self.name = "StandardRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        """Executes a circuit, returns the single shot results

        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """
        datadict = super().execute(circuit, datarow)
        # Substract 1 for sequence length to not count the inverse gate and
        # substract the measurement gate.
        datadict["depth"] = (circuit.depth - 2) if circuit.depth > 1 else 0
        return datadict


class moduleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Standard Randomized Benchmarking"


def groundstate_probabilities(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the groundstate probability with data from single shot measurements.

    Args:
        circuit (Circuit): Not needed here.
        datarow (dict): The dictionary holding the samples.

    Returns:
        dict: The updated dictionary.
    """
    # Get the samples data from the dictionary
    samples = datarow["samples"]
    # Create the ground state as it would look like in a single shot measurement.
    ground = np.array([0] * len(samples[0]))
    # Calculate the probability of the samples being in the ground state
    # by counting the number of samples that are equal to the ground state
    # and dividing it by the total number of samples.
    datarow["groundstate probability"] = np.sum(
        np.product(samples == ground, axis=1)
    ) / len(samples)
    # Return the updated dictionary.
    return datarow


def post_processing_sequential(experiment: Experiment):
    # Compute and add the ground state probabilities row by row.
    experiment.perform(groundstate_probabilities)


def get_aggregational_data(experiment: Experiment) -> pd.DataFrame:
    depths, ydata = experiment.extract("groundstate probability", "depth", "mean")
    _, ydata_std = experiment.extract("groundstate probability", "depth", "std")

    popt, perr = fitting_methods.fit_exp1B_func(depths, ydata)
    data = [
        {
            "depth": depths,
            "data": ydata,
            "2sigma": 2 * ydata_std,
            "fit_func": "exp1_func",
            "popt": {"A": popt[0], "p": popt[1], "B": popt[2]},
            "perr": {"A_err": perr[0], "p_err": perr[1], "B_err": perr[2]},
        }
    ]
    df = pd.DataFrame(data, index=["groundstate probability"])
    return df


def build_report(experiment: Experiment, df_aggr: pd.DataFrame):
    report = moduleReport()
    report.info_dict["Number of qubits"] = len(experiment.data[0]["samples"][0])
    report.info_dict["Number of shots"] = len(experiment.data[0]["samples"])
    report.info_dict["runs"] = experiment.extract("samples", "depth", "count")[1][0]
    report.info_dict["Fitting daviations"] = "".join(
        [
            "{}:{:.3f} ".format(key, df_aggr.iloc[0]["perr"][key])
            for key in df_aggr.iloc[0]["perr"]
        ]
    )
    report.all_figures.append(
        scatter_fit_fig(experiment, df_aggr, "depth", "groundstate probability")
    )
    return report.build()
