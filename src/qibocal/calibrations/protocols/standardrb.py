from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel, PauliError

from qibocal.calibrations.protocols.abstract import (
    Experiment,
    Result,
    SingleCliffordsFactory,
)
from qibocal.calibrations.protocols.fitting_methods import fit_exp1_func


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
            name="",
        )
        myfigs.append(fig)
        fig = go.Scatter(
            x=xdata, y=ydata, line=dict(color="#aa6464"), mode="markers", name="average"
        )
        myfigs.append(fig)
        fig = go.Scatter(
            x=x_fit,
            y=y_fit,
            name="A: {:.3f}, f: {:.3f}, B: {:.3f}".format(popt[0], popt[1], popt[2]),
            line=go.scatter.Line(dash="dot"),
        )
        myfigs.append(fig)
        self.all_figures.append(myfigs)


def embed_circuit(circuit: Circuit, nqubits: list, support: list) -> Circuit:
    """Takes a circuit and redistributes the gates to the support of
    a new circuit with ``nqubits`` qubits.

    Args:
        circuit (Circuit): The circuit with len(``support``) many qubits.
        nqubits (list): Qubits of new circuit.
        support (list): The qubits were the gates should be places.

    Returns:
        Circuit: Circuit with redistributed gates.
    """

    idxmap = np.vectorize(lambda idx: support[idx])
    newcircuit = Circuit(nqubits)
    for gate in circuit.queue:
        newcircuit.add(gate._class_(*idxmap(np.array(gate.init_args))))
    return newcircuit


def groundstate_probability(experiment: Experiment):
    """Computes sequential the ground state probabilities of an executed
    experiment and stores them in the experiment.

    Args:
        experiment (Experiment): Executed experiment for which the ground state
        probabilities for each data row are calculated.
    """
    probs = experiment.probabilities[:, 0]
    experiment._append_data("groundstate_probabilities", list(probs))


def analyze(experiment: Experiment, **kwargs) -> Result:
    # Compute and add the ground state probabilities.
    experiment.apply_task(groundstate_probability)
    result = Result(experiment.dataframe, fit_exp1_func)


def perform_qhardware(qubits: list, depths: list, runs: int, nshots: int):
    pass


def perform_simulation(
    qubits: list, depths: list, runs: int, nshots: int, noise_params: tuple
):
    pass


def plot(experiment: StandardRBExperiment):

    import matplotlib.pyplot as plt

    from qibocal.calibrations.protocols.fitting_methods import exp1_func, fit_exp1_func

    colorfunc = plt.get_cmap("inferno")

    experiment.postprocess()

    # Take the ground state population.
    ydata_scattered = experiment.probabilities[:, 0]
    xdata_scattered = experiment.depths
    plt.scatter(
        xdata_scattered,
        ydata_scattered,
        marker="_",
        linewidths=5,
        s=100,
        color=colorfunc(100),
        alpha=0.4,
        label="each run",
    )
    gdf = experiment.dataframe.groupby("depth")["probabilities"].agg("mean")
    xdata = np.array(gdf.index)
    ydata = gdf.values
    plt.scatter(xdata, ydata, marker=5, label="averaged")
    xfit, yfit, popt = fit_exp1_func(xdata, ydata)
    # import pdb
    # pdb.set_trace()
    # ydata_scattered =


def postprocess(experiment: StandardRBExperiment) -> dict:
    """Takes an experiment object, calculates the probabilities from the
    samples, fits an exponential to the probabilities and returns the parameters.

    Args:
        experiment (StandardRBExperiment): _description_

    """
    from qibocal.calibrations.protocols.fitting_methods import fit_exp1_func
    from qibocal.calibrations.protocols.utils import effective_depol

    df = experiment.dataframe
    probs = experiment.probabilities
    df["probabilities"] = list(probs)

    gdf = df.groupby("depth")["probabilities"].agg("mean")
    xdata = np.array(gdf.index)
    ydata = gdf.values
    popt, pcov = fit_exp1_func(xdata, ydata)

    pauli = PauliError(*df["noise_params"][0])
    depol = effective_depol(pauli)
    result = {
        "effective_depol": depol,
        "fitting_params": (popt, pcov),
        "mean_probabilities": ydata,
    }
    return result
