from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import qibo
from plotly.graph_objects import Figure
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import CircuitFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig

qibo.set_backend("numpy")


# Define the circuit factory class for this specific module.
class ModuleFactory(CircuitFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        assert (
            len(self.qubits) == 1
        ), """
        This class is written for gates acting on only one qubit, not {} qubits.""".format(
            len(self.qubits)
        )
        self.name = "Z4"

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(1, density_matrix=True)
        # There are only two gates to choose from for every qubit.
        a = [gates.I(0), gates.RX(0, np.pi / 2), gates.X(0), gates.RX(0, 3 * np.pi / 2)]
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 4, size=depth)
        # Get the gates with random_ints as indices.
        gate_lists = np.take(a, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(0))
        return circuit


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
        self.name = "Z4RB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        # Find sum of k where each gate of a circuit is RX(k*pi/2)
        rx_k = 0
        for gate in circuit.gates_of_type("rx"):
            rx_k += 1 if (gate[-1].parameters[0] == np.pi / 2) else 3
        datadict["sumK"] = rx_k + (circuit.gate_types["x"] * 2)
        return datadict


# Define the result class for this specific module.
class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Z4 Benchmarking"


# The filter functions/post processing functions always dependent on circuit and data row!
# It is executed row by row when used on an experiment object.
def filter_irrep(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for the gate group :math:`\\{Id, R_x(\\pi/2), X, R_x(3\\pi/2)\\}`.

    Each gate from the circuit with gates :math:`g` can be written as :math:`g_j=R_x(k_j\\pi/2)`
    and :math`i` the outcome which is either ground state :math:`0`
    or exited state :math:`1`.

    .. math::
        f_{\\lambda}(i,g)
        = (-1)^{\\sum k_j + i}/2


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
        filtervalue += np.conj(((-1j) ** (sumK + 2 * s[0])) / 2.0)

    datarow["filter"] = np.real(filtervalue / len(samples))
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
def get_aggregational_data(experiment: Experiment) -> pd.DataFrame:
    """Computes aggregational tasks, fits data and stores the results in a data frame.

    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.

    Returns:
        pd.DataFrame: The summarized data.
    """
    # Has to fit the column describtion from ``filter_irrep``.
    depths, ydata = experiment.extract("filter", "depth", "mean")
    _, ydata_std = experiment.extract("filter", "depth", "std")
    # Fit the filtered signal for each depth, there could be two overlaying exponential functions.
    popt, perr = fitting_methods.fit_exp2_func(depths, ydata)
    # Build a list of dictionaries with the aggregational information.
    data = [
        {
            "depth": depths,  # The x-axis.
            "data": ydata,  # The filtred signal.
            "2sigma": 2 * ydata_std,  # The 2 * standard deviation error for each depth.
            "fit_func": "exp2_func",  # Which function was used to fit.
            "popt": {
                "A1": popt[0],
                "A2": popt[1],
                "p1": popt[2],
                "p2": popt[3],
            },  # The fitting parameters.
            "perr": {
                "A1_err": perr[0],
                "A2_err": perr[1],
                "p1_err": perr[2],
                "p2_err": perr[3],
            },  # The estimated errors.
        }
    ]
    df = pd.DataFrame(data, index=["filter"])
    return df


# This is highly individual. The only important thing for the qq module is that a plotly figure is
# returned, if qq is not used any type of figure can be build.
def build_report(experiment: Experiment, df_aggr: pd.DataFrame) -> Figure:
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
    # Use the predefined ``scatter_fit_fig`` function from ``basics.utils`` to build the wanted
    # plotly figure with the scattered filtered data along with the mean for
    # each depth and the exponential fit for the means.
    report.all_figures.append(scatter_fit_fig(experiment, df_aggr, "depth", "filter"))
    # Return the figure the report object builds out of all figures added to the report.
    return report.build()
