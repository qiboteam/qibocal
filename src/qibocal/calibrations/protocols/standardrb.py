"""
"""


from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel, PauliError
from qibolab.platforms.abstract import AbstractPlatform

import qibocal.calibrations.protocols.noisemodels as noisemodels
from qibocal.calibrations.protocols.abstract import (
    Experiment,
    Result,
    SingleCliffordsFactory,
)
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.config import raise_error
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.fitting.rb_methods import fit_exp1_func, fit_exp1B_func
from qibocal.plots.rb import standardrb_plot


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

    @property
    def depths(self) -> np.ndarray:
        """Extracts the used circuits depths.

        Returns:
            np.ndarray: Used depths for every data row.
        """
        try:
            return self.dataframe["depth"].to_numpy()
        except KeyError:
            raise_error(KeyError, "No depths. Execute experiment first.")


class moduleResult(Result):
    def __init__(self, dataframe: pd.DataFrame, fitting_func) -> None:
        super().__init__(dataframe)
        self.fitting_func = fitting_func
        self.title = "Standard Randomized Benchmarking"

    def single_fig(self, dataframe):
        xdata_scatter = self.df["depth"].to_numpy()
        ydata_scatter = self.df["groundstate_probabilities"].to_numpy()
        xdata, ydata = self.extract("depth", "groundstate_probabilities", "mean")
        self.scatter_fit_fig(xdata_scatter, ydata_scatter, xdata, ydata)


def groundstate_probability(experiment: Experiment):
    """Computes sequential the ground state probabilities of an executed
    experiment and stores them in the experiment.

    Args:
        experiment (Experiment): Executed experiment for which the ground state
        probabilities for each data row are calculated.
    """
    probs = experiment.probabilities[:, 0]
    experiment._append_data("groundstate_probabilities", list(probs))
    return experiment


def groundstate_probs(circuit: Circuit, datarow: dict) -> dict:
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
    datarow["groundstate_probabilities"] = np.sum(
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


def analyze(
    experiment: Experiment, noisemodel: NoiseModel = None, **kwargs
) -> go._figure.Figure:
    # Compute and add the ground state probabilities.
    # experiment = groundstate_probability(experiment)
    experiment.perform(groundstate_probs)
    result = moduleResult(experiment.dataframe, fit_exp1B_func)
    result.single_fig()
    result.info_dict["effective depol"] = np.around(theoretical_outcome(noisemodel), 3)
    report = result.report()
    return report


def perform(
    nqubits: int,
    depths: list,
    runs: int,
    nshots: int,
    qubits: list = None,
    noise_model: NoiseModel = None,
):
    # Initiate the circuit factory and the faulty Experiment object.
    factory = moduleFactory(nqubits, depths, runs, qubits=qubits)
    experiment = moduleExperiment(factory, nshots, noisemodel=noise_model)
    # Execute the experiment.
    experiment.perform(experiment.execute)
    analyze(experiment, noisemodel=noise_model).show()


@plot("Randomized benchmarking", standardrb_plot)
def qqperform_standardrb(
    platform: AbstractPlatform,
    qubit: list,
    depths: list,
    runs: int,
    nshots: int,
    nqubit: int = None,
    noise_model: str = None,
    noise_params: list = None,
):
    # Check if noise should artificially be added.
    if noise_model is not None:
        # Get the wanted noise model class.
        noise_model = getattr(noisemodels, noise_model)(noise_params)
        validation = Data("validation", quantities=["effective_depol"])
        validation.add({"effective_depol": theoretical_outcome(noise_model)})
        yield validation
    # Initiate the circuit factory and the Experiment object.
    factory = moduleFactory(nqubit, depths, runs, qubits=qubit)
    experiment = moduleExperiment(factory, nshots, noisemodel=noise_model)
    # Execute the experiment.
    experiment.perform(experiment.execute)
    data = Data()
    data.df = experiment.dataframe
    yield data
