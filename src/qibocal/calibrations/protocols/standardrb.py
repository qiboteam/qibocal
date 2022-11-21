from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
from qibocal.plots.scatters import standardrb_plot


class SingleCliffordsInvFactory(SingleCliffordsFactory):
    def __init__(self, qubits: list, depths: list, runs: int) -> None:
        super().__init__(qubits, depths, runs)

    def build_circuit(self, depth: int):
        circuit = Circuit(len(self.qubits))
        for _ in range(depth):
            circuit.add(self.gate())
        # If there is at least one gate in the circuit, add an inverse.
        if depth > 0:
            # Build a gate out of the unitary of the whole circuit and
            # take the daggered version of that.
            circuit.add(gates.Unitary(circuit.unitary(), *self.qubits).dagger())
        circuit.add(gates.M(*self.qubits))
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
        # Substract 1 for sequence length to not count the inverse gate.
        # FIXME and on the measurement branch of qibo the measurement is
        # counted as one gate.
        datadict["depth"] = len(circuit.queue) - 2
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
        myfigs = []
        xdata_scatter = self.df["depth"].to_numpy()
        ydata_scatter = self.df["groundstate_probabilities"].to_numpy()
        xdata, ydata = self.extract("depth", "groundstate_probabilities", "mean")
        popt, pcov, x_fit, y_fit = self.fitting_func(xdata, ydata)
        fig = go.Scatter(
            x=xdata_scatter,
            y=ydata_scatter,
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
        )
        myfigs.append(fig)
        fig = go.Scatter(
            x=xdata, y=ydata, line=dict(color="#aa6464"), mode="markers", name="average"
        )
        myfigs.append(fig)
        fig = go.Scatter(
            x=x_fit,
            y=y_fit,
            name="A: {:.3f}, p: {:.3f}, B: {:.3f}".format(popt[0], popt[1], popt[2]),
            line=go.scatter.Line(dash="dot"),
        )
        myfigs.append(fig)
        self.all_figures.append(myfigs)


def groundstate_probability(experiment: Experiment):
    """Computes sequential the ground state probabilities of an executed
    experiment and stores them in the experiment.

    Args:
        experiment (Experiment): Executed experiment for which the ground state
        probabilities for each data row are calculated.
    """
    probs = experiment.probabilities[:, 0]
    experiment._append_data("groundstate_probabilities", list(probs))


def analyze(experiment: Experiment, **kwargs) -> None:
    # Compute and add the ground state probabilities.
    experiment.apply_task(groundstate_probability)
    result = StandardRBResult(experiment.dataframe, fit_exp1_func)
    result.single_fig()
    report = result.report()
    return report


def perform_qhardware(qubits: list, depths: list, runs: int, nshots: int):
    # Initiate the circuit factory and the Experiment object.
    factory = SingleCliffordsInvFactory(qubits, depths, runs)
    experiment = StandardRBExperiment(factory, nshots)
    # Execute the experiment.
    experiment.execute()
    analyze(experiment).show()


def perform_simulation(
    qubits: list, depths: list, runs: int, nshots: int, noise_params: list
):
    # Define the noise model.
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Initiate the circuit factory and the faulty Experiment object.
    factory = SingleCliffordsInvFactory(qubits, depths, runs)
    faultyexperiment = StandardRBExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    faultyexperiment.execute()
    analyze(faultyexperiment).show()


@plot("Hardware Randomized benchmarking", standardrb_plot)
def qqperform_qhardware(
    platform: AbstractPlatform, qubit: list, depths: list, runs: int, nshots: int
):
    # Initiate the circuit factory and the Experiment object.
    factory = SingleCliffordsInvFactory(qubit, depths, runs)
    experiment = StandardRBExperiment(factory, nshots)
    # Execute the experiment.
    experiment.execute()
    data = Data("data", quantities=["dataframe"])
    data.add({"dataframe": experiment.dataframe})
    yield data


@plot("Simulation Randomized benchmarking", standardrb_plot)
def qqperform_simulation(
    platform: AbstractPlatform,
    qubit: list,
    depths: list,
    runs: int,
    nshots: int,
    noise_params: list,
):
    # Define the noise model.
    paulinoise = PauliError(*noise_params)
    noise = NoiseModel()
    noise.add(paulinoise, gates.Unitary)
    # Initiate the circuit factory and the Experiment object.
    factory = SingleCliffordsInvFactory(qubit, depths, runs)
    experiment = StandardRBExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    experiment.execute()
    data = Data("data", quantities=["dataframe"])
    data.add({"dataframe": experiment.dataframe})
    yield data
    data_depol = Data("effectivedepol", quantities=["effective_depol"])
    data_depol.add({"effective_depol": effective_depol(paulinoise)})
