"""
"""


from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel, PauliError
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.calibrations.protocols.abstract import (
    Experiment,
    Result,
    SingleCliffordsFactory,
)
from qibocal.calibrations.protocols.fitting_methods import fit_exp1_func
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.plots.rb import standardrb_plot


class SingleCliffordsInvFactory(SingleCliffordsFactory):
    def __init__(
        self, nqubits: list, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits)

    def build_circuit(self, depth: int):
        circuit = Circuit(len(self.qubits))
        for _ in range(depth):
            circuit.add(self.gates())
        # If there is at least one gate in the circuit, add an inverse.
        if depth > 0:
            # Build a gate out of the unitary of the whole circuit and
            # take the daggered version of that.
            circuit.add(
                gates.Unitary(
                    circuit.unitary(), *range(len(self.qubits))
                    ).dagger()
                )
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


class StandardRBExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
    ) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel)

    def single_task(self, circuit: Circuit, datarow: dict) -> None:
        """Executes a circuit, returns the single shot results
        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """
        datadict = super().single_task(circuit, datarow)
        # Substract 1 for sequence length to not count the inverse gate
        # FIXME and on the measurement branch of qibo the measurement is
        # counted as one gate on the master branch not.
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
            print("No depths. Execute experiment first.")
            return None


class StandardRBResult(Result):
    def __init__(self, dataframe: pd.DataFrame, fitting_func) -> None:
        super().__init__(dataframe)
        self.fitting_func = fitting_func

    def single_fig(self):
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


def analyze(experiment: Experiment, noisemodel: NoiseModel = None, **kwargs) -> None:
    # Compute and add the ground state probabilities.
    experiment.apply_task(groundstate_probability)
    result = StandardRBResult(experiment.dataframe, fit_exp1_func)
    result.single_fig()
    report = result.report()
    return report


def perform(
    nqubits: int,
    depths: list,
    runs: int,
    nshots: int,
    qubits: list = None,
    noise_params: list = None,
):
    if noise_params is not None:
        # Define the noise model.
        paulinoise = PauliError(*noise_params)
        noise = NoiseModel()
        noise.add(paulinoise, gates.Unitary)
        depol = effective_depol(paulinoise)
    else:
        noise = None
    # Initiate the circuit factory and the faulty Experiment object.
    factory = SingleCliffordsInvFactory(nqubits, depths, runs, qubits=qubits)
    experiment = StandardRBExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    experiment.execute()
    analyze(experiment, noisemodel=noise).show()


@plot("Randomized benchmarking", standardrb_plot)
def qqperform_standardrb(
    platform: AbstractPlatform,
    qubit: list,
    depths: list,
    runs: int,
    nshots: int,
    nqubit: int = None,
    noise_params: list = None,
):
    # Check if noise should artificially be added.
    if noise_params is not None:
        # Define the noise model.
        paulinoise = PauliError(*noise_params)
        noise = NoiseModel()
        noise.add(paulinoise, gates.Unitary)
        data_depol = Data("effectivedepol", quantities=["effective_depol"])
        data_depol.add({"effective_depol": effective_depol(paulinoise)})
        yield data_depol
    else:
        noise = None
    # Initiate the circuit factory and the Experiment object.
    factory = SingleCliffordsInvFactory(nqubit, depths, runs, qubits=qubit)
    experiment = StandardRBExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    experiment.execute()
    data = Data()
    data.df = experiment.dataframe
    yield data
