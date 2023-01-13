from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np
import pandas as pd
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.fitting.rb_methods as fitting_methods
from qibocal.calibrations.protocols.abstract import Experiment, Report
from qibocal.calibrations.protocols.abstract import (
    SingleCliffordsFactory as moduleFactory,
)
from qibocal.calibrations.protocols.abstract import scatter_fit_fig


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
        self.name = "CorrelatedRB"

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


class moduleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Correlated Filtered Randomized Benchmarking"

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
    for kk in range(len(f_list)):
        datarow[f"irrep{kk}"] = f_list[kk] / nshots
    return datarow


def theoretical_outcome(noisemodel: NoiseModel) -> float:
    return 0


def post_processing_sequential(experiment: Experiment):
    # Compute and add the ground state probabilities row by row.
    experiment.perform(filter_function)


def get_aggregational_data(experiment: Experiment) -> pd.DataFrame:
    nqubits = len(experiment.data[0]["samples"][0])
    data_list, index = [], []
    for kk in range(2**nqubits):
        ylabel = f"irrep{kk}"
        depths, ydata = experiment.extract("depth", ylabel, "mean")
        _, ydata_std = experiment.extract("depth", ylabel, "std")
        popt, perr = fitting_methods.fit_exp1_func(depths, ydata)
        data_list.append(
            {
                "depth": depths,
                "data": ydata,
                "2sigma": 2 * ydata_std,
                "fit_func": "exp1_func",
                "popt": {"A": popt[0], "p": popt[1], "B": popt[2]},
                "perr": {"A_err": perr[0], "p_err": perr[1], "B_err": perr[2]},
            }
        )
        index.append(ylabel)

    df = pd.DataFrame(data_list, index=index)
    return df


def build_report(experiment: Experiment, df_aggr: pd.DataFrame):
    report = moduleReport()
    report.info_dict["Number of qubits"] = len(experiment.data[0]["samples"][0])
    report.info_dict["Number of shots"] = len(experiment.data[0]["samples"])
    report.info_dict["runs"] = experiment.extract("depth", "samples", "count")[1][0]
    lambdas = iter(product([0, 1], repeat=int(report.info_dict["Number of qubits"])))
    print(df_aggr)
    for kk, l in enumerate(lambdas):
        report.info_dict[f"Fitting daviations irrep {l}"] = "".join(
            [
                "{}:{:.3f} ".format(key, df_aggr.loc[f"irrep{kk}"]["perr"][key])
                for key in df_aggr.loc[f"irrep{kk}"]["perr"]
            ]
        )

        figdict = scatter_fit_fig(experiment, df_aggr, "depth", f"irrep{kk}")
        figdict["subplot_title"] = f"Irrep {l}"
        report.all_figures.append(figdict)
    return report.build()
