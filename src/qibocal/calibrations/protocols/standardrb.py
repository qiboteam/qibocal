"""
"""


from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.fitting.rb_methods as fitting_methods
from qibocal.calibrations.protocols.abstract import (
    Experiment,
    Report,
    SingleCliffordsFactory,
    scatter_fit_fig,
)
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.config import raise_error


class moduleFactory(SingleCliffordsFactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits)
        self.name = "SingleCliffordsInv"

    def build_circuit(self, depth: int):
        circuit = Circuit(len(self.qubits))
        for _ in range(depth):
            circuit.add(self.gates())
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
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
    ) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel)
        self.name = "StandardRB"

    def execute(self, circuit: Circuit, datarow: dict) -> None:
        """Executes a circuit, returns the single shot results

        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """
        datadict = super().execute(circuit, datarow)
        # Substract 1 for sequence length to not count the inverse gate and
        # substract the measurement gate.
        datadict["depth"] = circuit.ngates - 2 if circuit.ngates > 1 else 0
        return datadict


class moduleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Standard Randomized Benchmarking"


def groundstate_probabilities(circuit: Circuit, datarow: dict) -> dict:
    """_summary_

    Args:
        circuit (Circuit): _description_
        datarow (dict): _description_

    Returns:
        dict: _description_

    Yields:
        Iterator[dict]: _description_
    """
    samples = datarow["samples"]
    # This is how
    ground = np.array([0] * len(samples[0]))
    datarow["groundstate probability"] = np.sum(
        np.product(samples == ground, axis=1)
    ) / len(samples)
    return datarow


def theoretical_outcome(noisemodel: NoiseModel) -> float:
    """Take the used noise model acting on unitaries and calculates the
    effective depolarizing parameter.

    Args:
        experiment (Experiment): Experiment which executed the simulation.
        noisemddel (NoiseModel): Applied noise model.

    Returns:
        (float): The effective depolarizing parameter of given error.
    """
    # Check for correctness of noise model and gate independence.
    errorkeys = noisemodel.errors.keys()
    if len(errorkeys) == 1 and list(errorkeys)[0] == gates.Unitary:
        # Extract the noise acting on unitaries and turn it into the associated
        # error channel.
        error = noisemodel.errors[gates.Unitary][0]
        errorchannel = error.channel(0, *error.options)
    else:
        raise_error(ValueError, "Wrong noisemodel given.")
    # Calculate the effective depolarizing parameter.
    return effective_depol(errorchannel)


def post_processing_sequential(experiment: Experiment):
    # Compute and add the ground state probabilities row by row.
    experiment.perform(groundstate_probabilities)


def get_aggregational_data(experiment: Experiment) -> pd.DataFrame:
    depths, ydata = experiment.extract("depth", "groundstate probability", "mean")
    _, ydata_std = experiment.extract("depth", "groundstate probability", "std")

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
    report.info_dict["runs"] = experiment.extract("depth", "samples", "count")[1][0]
    print(df_aggr)
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
