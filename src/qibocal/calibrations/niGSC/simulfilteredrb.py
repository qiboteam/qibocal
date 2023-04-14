# TODO simulfilteredrb

from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from qibo.models import Circuit
from qibo.noise import NoiseModel
from qibo.quantum_info import comp_basis_to_pauli

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import SingleCliffordsFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig, update_fig
from qibocal.calibrations.niGSC.basics.rb_validation import filtered_decay_parameters
from qibocal.config import raise_error


class ModuleFactory(SingleCliffordsFactory):
    pass


class ModuleExperiment(Experiment):
    """Inherits from abstract ``Experiment`` class."""

    def __init__(
        self,
        circuitfactory: Iterable,
        data: Iterable | None = None,
        nshots: int | None = None,
        noise_model: NoiseModel = None,
    ) -> None:
        """Calles the parent method and additionally prebuilds the circuit factory
        making it a list stored in memory and saves if ``save()`` method is called.

        Args:
            circuitfactory (Iterable): Gives a certain amount of circuits when
                iterated over.
            nshots (int): For execution of circuit, indicates how many shots.
            data (Iterable): If filled, ``data`` can be used to specifying parameters
                     while executing a circuit or deciding how to process results.
                     It is used to store all relevant data.
        """

        super().__init__(circuitfactory, data, nshots, noise_model)
        # Make the circuitfactory a list. That way they will be stored when
        # calling the save method and the circuits are not lost once executed.
        self.prebuild()
        self.name = "simulfilteredrb"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        """Overwrited parents method. Executes a circuit, returns the single shot results and depth.

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


class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Correlated Filtered Randomized Benchmarking"


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
    # If depth = 0 there is only a measurement circuit and it does
    # not have an implemented matrix. Set the ideal states to ground states.
    if circuit.depth == 1:
        ideal_states = np.tile(np.array([1, 0]), nqubits).reshape(nqubits, 2)
    else:
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


def post_processing_sequential(experiment: Experiment):
    """Perform sequential tasks needed to analyze the experiment results.

    The data is added/changed in the experiment, nothign has to be returned.

    Args:
        experiment (Experiment): Experiment object after execution of the experiment itself.
    """

    # Compute and add the ground state probabilities row by row.
    experiment.perform(filter_function)


def get_aggregational_data(experiment: Experiment, ndecays: int = None) -> pd.DataFrame:
    """Computes aggregational tasks, fits data and stores the results in a data frame.

    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.
        ndecays (int): Number of decay parameters to fit. Default is 1.

    Returns:
        pd.DataFrame: The summarized data.
    """
    ndecays = ndecays if ndecays is not None else 1
    # Needed for the amount of plots in the report
    nqubits = len(experiment.data[0]["samples"][0])
    data_list, index = [], []
    # Go through every irreducible representation projector used in the filter function.
    for kk in range(2**nqubits):
        # This has to match the label chosen in ``filter_function``.
        ylabel = f"irrep{kk}"
        depths, ydata = experiment.extract(ylabel, "depth", "mean")
        _, ydata_std = experiment.extract(ylabel, "depth", "std")
        # Fit an exponential without linear offset.
        if ndecays == 1:
            popt, perr = fitting_methods.fit_exp1_func(depths, ydata)
            popt_dict = {"A": popt[0], "p": popt[1]}
            perr_dict = {"A_err": perr[0], "p_err": perr[1]}
        else:
            fitting_methods.fit_expn_func(depths, ydata, ndecays)
            popt_labels = np.array(
                [[f"A{k+1}", f"p{k+1}"] for k in range(len(popt) // 2)]
            ).ravel()
            popt_dict = dict(zip(popt_labels, popt[::2] + popt[1::2]))
            perr_labels = np.array(
                [[f"A{k+1}_err", f"p{k+1}_err"] for k in range(len(popt) // 2)]
            ).ravel()
            perr_dict = dict(zip(perr_labels, perr))
        data_list.append(
            {
                "depth": depths,  # The x-axis.
                "data": ydata,  # The mean of ground state probability for each depth.
                "2sigma": 2 * ydata_std,  # The standard deviation error for each depth.
                "fit_func": "exp1_func",  # Which function was used to fit.
                "popt": popt_dict,  # The fitting paramters.
                "perr": perr_dict,  # The estimated errors.
            }
        )
        # Store the name to set is as row name for the data.
        index.append(ylabel)
    # Create a data frame out of the list with dictionaries.
    df = pd.DataFrame(data_list, index=index)
    return df


def gate_group(nqubits=1):
    """
    Local Cliffords gate group
    """
    from itertools import product

    from qibo import gates
    from qibo.quantum_info.random_ensembles import _clifford_unitary
    from qibo.quantum_info.utils import ONEQUBIT_CLIFFORD_PARAMS

    cliffords_1q = [_clifford_unitary(*p) for p in ONEQUBIT_CLIFFORD_PARAMS]
    if nqubits == 1:
        return [gates.Unitary(c, *range(nqubits)) for c in cliffords_1q]
    cliffords = list(product(cliffords_1q, repeat=nqubits))
    cliffords_unitaries = [
        gates.Unitary(np.kron(*clifford_layer), *range(nqubits))
        for clifford_layer in cliffords
    ]
    return cliffords_unitaries


def irrep_info(nqubits=1):
    """
    Infromation about the irreducible representation of the Clifford group.

    Returns:
        tuple: (basis, index, size, multiplicity) of the sign irrep
    """
    basis_c2p_1q = comp_basis_to_pauli(1, normalize=True)
    return (basis_c2p_1q, 1, 3, 1)


def add_validation(
    experiment: Experiment, dataframe: pd.DataFrame | dict, N: int | None = None
) -> pd.DataFrame:
    """Computes theoretical values of coefficients and decay parameters of a given experiment
    and add validation data to the dataframe.
    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.
        dataframe (pd.DataFrame): The data where the validation should be added.

    Returns:
        pd.DataFrame: The summarized data.
    """
    nqubits = len(experiment.data[0]["samples"][0])
    if nqubits > 1:
        raise_error(
            NotImplementedError,
            "Theoretical validation for multiple qubits is not implemented.",
        )

    data = dataframe.to_dict("records")
    coefficients, decay_parameters = filtered_decay_parameters(
        experiment.name, nqubits, experiment.noise_model, with_coefficients=True, N=N
    )
    ndecays = len(coefficients)
    validation_keys = [f"A{k+1}" for k in range(ndecays)]
    validation_keys += [f"p{k+1}" for k in range(ndecays)]
    validation_dict = dict(
        zip(validation_keys, np.concatenate((coefficients, decay_parameters)))
    )

    data[1].update({"validation": validation_dict, "validation_func": "expn_func"})
    # The row name will be displayed as y-axis label.
    df = pd.DataFrame(data, index=dataframe.index)
    return df


def build_report(experiment: Experiment, df_aggr: pd.DataFrame) -> Figure:
    """Use data and information from ``experiment`` and the aggregated data dataframe to
    build a reprot as plotly figure.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.
        df_aggr (pd.DataFrame): Normally build with ``get_aggregational_data`` function.

    Returns:
        (Figure): A plotly.graphical_object.Figure object.
    """

    # Initiate a report object.
    report = ModuleReport()
    # Add general information to the object.
    report.info_dict["Number of qubits"] = len(experiment.data[0]["samples"][0])
    report.info_dict["Number of shots"] = len(experiment.data[0]["samples"])
    report.info_dict["runs"] = experiment.extract("samples", "depth", "count")[1][0]
    lambdas = iter(product([0, 1], repeat=int(report.info_dict["Number of qubits"])))
    for kk, l in enumerate(lambdas):
        # Add the fitting errors which will be displayed in a box under the plots.
        report.info_dict[f"Fitting daviations irrep {l}"] = "".join(
            [
                "{}:{:.3f} ".format(key, df_aggr.loc[f"irrep{kk}"]["perr"][key])
                for key in df_aggr.loc[f"irrep{kk}"]["perr"]
            ]
        )
        # Use the predefined ``scatter_fit_fig`` function from ``basics.utils`` to build the wanted
        # plotly figure with the scattered filter function points and then mean per depth.
        figdict = scatter_fit_fig(experiment, df_aggr, "depth", f"irrep{kk}")
        # Add a subplot title for each irrep.
        figdict["subplot_title"] = f"Irrep {l}"
        report.all_figures.append(figdict)
        if "validation" in df_aggr:
            report.all_figures[-1] = update_fig(report.all_figures[-1], df_aggr)
    return report.build()
