from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

from qibocal.calibrations.niGSC.basics.circuitfactory import ZkFilteredCircuitFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig, update_fig


# Define the circuit factory class for this specific module.
class ModuleFactory(ZkFilteredCircuitFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits, size=3)

    @property
    def gate_group(self):
        return [gates.I(0), gates.RX(0, 2 * np.pi / 3), gates.RX(0, 4 * np.pi / 3)]


# Define the experiment class for this specific module.
class ModuleExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        data: Iterable | None = None,
        nshots: int | None = None,
        noise_model: NoiseModel = None,
    ) -> None:
        super().__init__(circuitfactory, data, nshots, noise_model)
        self.prebuild()
        self.name = "Z3rb"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        # Find sum of k where each gate of a circuit is RX(k*pi/2)
        rx_k = 0
        for gate in circuit.gates_of_type("rx"):
            rx_k += int(gate[-1].parameters[0] * 3 / (2 * np.pi))
        datadict["sumK"] = rx_k
        return datadict


# Define the result class for this specific module.
class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Z3 Benchmarking"


# The filter functions/post processing functions always dependent on circuit and data row!
# It is executed row by row when used on an experiment object.
def filter_irrep(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for the gate group :math:`\\{Id, R_x(2\\pi/3), R_x(4\\pi/3)\\}`.

    Each gate from the circuit with gates :math:`g` can be written as :math:`g_j=R_x(k_j\\cdot 2\\pi/3)`
    and :math`i` the outcome which is either ground state :math:`0`
    or exited state :math:`1`.

    .. math::
        f_{\\lambda}(i,g)
        = (-1)^i\\left(\\frac\\{-1+\\sqrt3i\\}\\{2\\}\\right)^{\\sum k_j}/2


    Args:
        circuit (Circuit): Not needed here.
        datarow (dict): The dictionary with the samples from executed circuits and :math:`\\sum k_j`
                        of all the gates :math:`R_x(k_j\\pi/2)` in the executed circuit.

    Returns:
        dict: _description_
    """
    samples = datarow["samples"]
    sumK = datarow["sumK"]
    filtervalue = 0
    for s in samples:
        filtervalue += (-1) ** s[0] * np.conj(
            ((-1 + np.sqrt(3) * 1j) / 2) ** (sumK % 3) / 2.0
        )

    datarow["filter"] = filtervalue / len(samples)
    return datarow


# All the predefined sequential postprocessing / filter functions are bundled together here.
def post_processing_sequential(experiment: Experiment):
    """Perform sequential tasks needed to analyze the experiment results.

    The data is added/changed in the experiment, nothing has to be returned.

    Args:
        experiment (Experiment): Experiment object after execution of the experiment itself.
    """

    # Compute and add the ground state probabilities row by row.
    experiment.perform(filter_irrep)


# After the row by row execution of tasks comes the aggregational task. Something like calculation
# of means, deviations, fitting data, tasks where the whole data as to be looked at, and not just
# one instance of circuit + other information.
def get_aggregational_data(experiment: Experiment, ndecays: int = None) -> pd.DataFrame:
    """Computes aggregational tasks, fits data and stores the results in a data frame.

    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.
        ndecays (int): Number of decay parameters to fit. Default is 2.

    Returns:
        pd.DataFrame: The summarized data.
    """
    from qibocal.calibrations.niGSC.XIdrb import get_aggregational_data as gad_xidrb

    df = gad_xidrb(experiment, ndecays)
    return df


def gate_group(nqubits=1):
    """
    Z3 gate group
    """
    return [gates.I(0), gates.RX(0, 2 * np.pi / 3), gates.RX(0, 4 * np.pi / 3)]


def irrep_info(nqubits=1):
    """
    Infromation about the irreducible representation of the Z3 gate group.
    
    Returns:
        tuple: (basis, index, size, multiplicity) of the irrep
    """
    zk_basis = np.array(
        [
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [-1j / 2, 1j / 2, -1j / 2, 1j / 2],
            [-0.5, -0.5, 0.5, 0.5],
        ]
    )
    return (zk_basis, 3, 1, 1)


def add_validation(
    experiment: Experiment, dataframe: pd.DataFrame | dict, N: int | None = None
) -> pd.DataFrame:
    """Computes theoretical values of coefficients and decay parameters of a given experiment
    and add validation data to the dataframe.
    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.

    Returns:
        pd.DataFrame: The summarized data.
    """
    from qibocal.calibrations.niGSC.XIdrb import add_validation as addv_xid

    df = addv_xid(experiment, dataframe, N)
    return df


# This is highly individual. The only important thing for the qq module is that a plotly figure is
# returned, if qq is not used any type of figure can be build.
def build_report(
    experiment: Experiment, df_aggr: pd.DataFrame, validate: bool = False, N: int = None
) -> Figure:
    """Use data and information from ``experiment`` and the aggregated data data frame to
    build a report as plotly figure.

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
    report.info_dict["Fitting daviations"] = "".join(
        [
            "{}:{:.3f} ".format(key, df_aggr.loc["filter"]["perr"][key])
            for key in df_aggr.loc["filter"]["perr"]
        ]
    )

    # Check if there are imaginary values in the data
    is_imag = "popt_imag" in df_aggr
    fittingparam_label = "popt_imag" if is_imag else "popt"
    validation_label = "validation_imag" if is_imag else "validation"
    # Use the predefined ``scatter_fit_fig`` function from ``basics.utils`` to build the wanted
    # plotly figure with the scattered filtered data along with the mean for
    # each depth and the exponential fit for the means.
    report.all_figures.append(
        scatter_fit_fig(
            experiment,
            df_aggr,
            "depth",
            "filter",
            fittingparam_label=fittingparam_label,
        )
    )

    # If there is validation, add it to the figure
    if validation_label in df_aggr.loc["filter"]:
        report.all_figures[-1] = update_fig(
            report.all_figures[-1],
            df_aggr,
            param_label=validation_label,
        )

    # If there are imaginary values in the data, create another figure
    if is_imag:
        report.all_figures.append(
            scatter_fit_fig(
                experiment,
                df_aggr,
                "depth",
                "filter",
                fittingparam_label="popt_imag",
                is_imag=True,
            )
        )

        if validation_label in df_aggr.loc["filter"]:
            report.all_figures[-1] = update_fig(
                report.all_figures[-1],
                df_aggr,
                param_label=validation_label,
                is_imag=True,
            )
    # Return the figure the report object builds out of all figures added to the report.
    return report.build()
