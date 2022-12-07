from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from itertools import product

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
from qibocal.fitting.rb_methods import fit_exp1_func
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.plots.rb import crosstalkrb_plot
from qibocal.config import raise_error


class CrosstalkRBExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
    ) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel)

    def single_task(self, circuit: Circuit, datarow: dict) -> dict:
        """Executes a circuit, returns the single shot results
        
        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """
        # First save the unexecuted circuit. Else there will be a
        # pickle conflict.
        justcircuitdict = {"circuit": deepcopy(circuit)}
        datadict = super().single_task(circuit, datarow)
        # FIXME and on the measurement branch of qibo the measurement is
        # counted as one gate on the master branch not.
        datadict["depth"] = int(
            (circuit.ngates - 1 if circuit.ngates > 1 else 0)
            / len(datadict["samples"][0])
        )
        return {**datadict, **justcircuitdict}

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


class CrosstalkRBResult(Result):
    def __init__(self, dataframe: pd.DataFrame, fitting_func) -> None:
        super().__init__(dataframe)
        self.fitting_func = fitting_func
        self.title = "Crosstalk Filtered Randomized Benchmarking"

    def cross_figs(self):
        xdata_scatter = self.df["depth"].to_numpy()
        allydata_scatter = np.array(self.df["crosstalk"].tolist())
        xdata, allydata = self.extract("depth", "crosstalk", "mean")
        lambdas = iter(product([0, 1], repeat=int(np.log2(len(allydata_scatter[0])))))
        for count in range(len(allydata_scatter[0])):
            self.scatter_fit_fig(
                xdata_scatter, allydata_scatter[:, count], xdata, allydata[:, count]
            )
            self.all_figures[-1]["subplot_title"] = f"Irrep {next(lambdas)}"


def filter_function(experiment: CrosstalkRBExperiment):
    """Calculates the filtered signal for every cross talk irrep.

    Every irrep has a projector charactarized with a bit string
    :math:`\\boldsymbol{\\lambda}\\in\\mathbb{F}_2^N` where :math:`N` is the
    number of qubits.
    The experimental outcome for each qubit is denoted as
    :math:`\\ket{i_k}` with :math:`i_k=0, 1` with :math:`d=2`.

    .. math::
        f_{\\boldsymbol{\\lambda}}(i,g) = \\frac{1}{2^{N-|\\boldsymbol{\\lambda}|}}\\sum_{\\mathbf b\\in\\mathbb F_2^N}(-1)^{|\\boldsymbol{\\lambda}\\wedge\\mathbf b|}\\frac{1}{d^N}\\left(\\prod_{k=1}^N(d|\\bra{i_k} U_{g_{(k)}} \\ket{0}|^2)^{\\lambda_k-\\lambda_kb_k}\\right)

    Args:
        experiment (CrosstalkRBExperiment): The executed (crosstalk) experiment. 
            The circuits must be stored in the experiment object.
    """
    # Extract amount of used qubits and used shots.
    nqubits = len(experiment.data[0]["samples"][0])
    nshots = len(experiment.data[0]["samples"])
    d = 2
    # For each data row calculate the filtered signals.
    biglist = []
    for datarow in experiment.data:
        samples = datarow["samples"]
        # Fuse the gates for each qubit.
        fused_circuit = datarow["circuit"].fuse(max_qubits = 1)
        # Extract for each qubit the ideal state.
        ideal_states = np.array(
            [
                fused_circuit.queue[k].matrix[:, 0] for k in range(nqubits)
            ]
        )
        # Go through every irrep.
        f_list = []
        for l in np.array(list(product([False, True], repeat=nqubits))):
            # Check if the trivial irrep is calculated
            if not sum(l):
                # In the end every value will be divided by ``nshots``.
                a = nshots
            else:
                # Get the supported ideal outcomes and samples
                # for this irreps projector.
                suppl = ideal_states[l]
                suppsamples = samples[:, l]
                a = 0
                # Go through all ``nshots`` samples
                for s in suppsamples:
                    # Go through all combinations of (0,1) on the support
                    # of lambda ``l``.
                    for b in np.array(list(product([False, True], repeat=sum(l)))):
                        # Calculate the sign depending on how many times the nontrivial
                        # projector was used.
                        # Take the product of all probabilities chosen by the experimental
                        # outcome which are supported by the inverse of b.
                        a += (-1) ** sum(b) * np.prod(
                            d * np.abs(suppl[~b][np.eye(2, dtype=bool)[s[~b]]]) ** 2)
            # Normalize with inverse of effective measuremetn.
            a_norm = a * (d + 1) ** sum(l) / d ** nqubits
            f_list.append(a_norm)
        biglist.append(np.array(f_list) / nshots)
    experiment._append_data("crosstalk", biglist)

def analyze(experiment: CrosstalkRBExperiment, noisemodel: NoiseModel = None, **kwargs):
    experiment.apply_task(filter_function)
    result = CrosstalkRBResult(experiment.dataframe, fit_exp1_func)
    result.cross_figs()
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
        # depol = effective_depol(paulinoise)
    else:
        noise = None
    # Initiate the circuit factory and the (faulty) Experiment object.
    factory = SingleCliffordsFactory(nqubits, depths, runs, qubits=qubits)
    experiment = CrosstalkRBExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    experiment.execute()
    analyze(experiment, noisemodel=noise).show()


@plot("Randomized benchmarking", crosstalkrb_plot)
def qqperform_crosstalkrb(
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
    factory = SingleCliffordsFactory(nqubit, depths, runs, qubits=qubit)
    experiment = CrosstalkRBExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    experiment.execute()
    data = Data()
    data.df = experiment.dataframe
    yield data
