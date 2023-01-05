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

import qibocal.calibrations.protocols.noisemodels as noisemodels
from qibocal.calibrations.protocols.abstract import Experiment, Result
from qibocal.calibrations.protocols.abstract import (
    SingleCliffordsFactory as moduleFactory,
)
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.config import raise_error
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.fitting.rb_methods import fit_exp1_func
from qibocal.plots.rb import crosstalkrb_plot


class moduleExperiment(Experiment):
    """Inherits from abstract ``Experiment`` class.

    Store ``circuitfactory`` as list such that after executing the used
    circuitsare still there.

    """

    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
    ) -> None:
        """Calles the parent ``__init__`` method and additionally prebuilds
        the circuit factory making it a list stored in memory.

        Args:
            circuitfactory (Iterable): _description_
            nshots (int, optional): _description_. Defaults to None.
            data (list, optional): _description_. Defaults to None.
            noisemodel (NoiseModel, optional): _description_. Defaults to None.
        """

        super().__init__(circuitfactory, nshots, data, noisemodel)
        # Make the circuitfactory a list. That way they will be stored when
        # calling the save method and the circuits are not lost once executed.
        self.prebuild()
        self.name = "CrosstalkRB"

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

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        """Executes a circuit, returns the single shot results and depth.

        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """

        datadict = super().execute(circuit, datarow)
        # Measurement gate should not contribute to depth, therefore -1.
        # Take the amount of qubits into account.
        datadict["depth"] = int((circuit.ngates - 1) / len(datadict["samples"][0]))
        return datadict


class moduleResult(Result):
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


def filter_function(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for every crosstalk irrep.

    Every irrep has a projector charactarized with a bit string
    :math:`\\boldsymbol{\\lambda}\\in\\mathbb{F}_2^N` where :math:`N` is the
    number of qubits.
    The experimental outcome for each qubit is denoted as
    :math:`\\ket{i_k}` with :math:`i_k=0, 1` with :math:`d=2`.

    .. math::
        f_{\\boldsymbol{\\lambda}}(i,g)
        = \\frac{1}{2^{N-|\\boldsymbol{\\lambda}|}}
            \\sum_{\\mathbf b\\in\\mathbb F_2^N}
            (-1)^{|\\boldsymbol{\\lambda}\\wedge\\mathbf b|}
            \\frac{1}{d^N}\\left(\\prod_{k=1}^N(d|\\bra{i_k} U_{g_{(k)}}
            \\ket{0}|^2)^{\\lambda_k-\\lambda_kb_k}\\right)

    Args:
        circuit (Circuit): The circuit used to produce the samples in ``datarow``.
        datarow (dict): Dictionary with samples produced by given ``circuit``.

    Returns:
        datarow (dict):  Filtered signals are stored additionally.
    """

    # Extract amount of used qubits and used shots.
    nshots, nqubits = datarow["samples"].shape
    # For qubits the local dimension is 2.
    d = 2
    # Fuse the gates for each qubit.
    fused_circuit = circuit.fuse(max_qubits=1)
    # Extract for each qubit the ideal state.
    # TODO if depth = 0 there is only a measurement circuit and it does
    # not have an implemented matrix. This exception has to be dealt with.
    ideal_states = np.array(
        [fused_circuit.queue[k].matrix[:, 0] for k in range(nqubits)]
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
            suppsamples = datarow["samples"][:, l]
            a = 0
            # Go through all ``nshots`` samples
            for s in suppsamples:
                # Go through all combinations of (0,1) on the support
                # of lambda ``l``.
                for b in np.array(list(product([False, True], repeat=sum(l)))):
                    # Calculate the sign depending on how many times the
                    # nontrivial projector was used.
                    # Take the product of all probabilities chosen by the
                    # experimental outcome which are supported by the
                    # inverse of b.
                    a += (-1) ** sum(b) * np.prod(
                        d * np.abs(suppl[~b][np.eye(2, dtype=bool)[s[~b]]]) ** 2
                    )
        # Normalize with inverse of effective measuremetn.
        f_list.append(a * (d + 1) ** sum(l) / d**nqubits)
    datarow["crosstalk"] = np.array(f_list) / nshots
    return datarow


def theoretical_outcome(noisemodel: NoiseModel) -> float:
    return 0


def analyze(experiment: moduleExperiment, noisemodel: NoiseModel = None):
    # Apply the fiterfunction via matmul operator.
    experiment.perform(filter_function)
    result = moduleResult(experiment.dataframe, fit_exp1_func)
    result.cross_figs()
    result.info_dict["effective depol"] = np.around(theoretical_outcome(noisemodel), 3)
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
    factory = moduleFactory(nqubits, depths, runs, qubits=qubits)
    experiment = moduleExperiment(factory, nshots, noisemodel=noise)
    # Execute the experiment.
    experiment @ experiment.execute
    analyze(experiment, noisemodel=noise).show()


@plot("Randomized benchmarking", crosstalkrb_plot)
def qqperform_crosstalkrb(
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
    experiment @ experiment.execute
    data = Data()
    data.df = experiment.dataframe
    yield data
