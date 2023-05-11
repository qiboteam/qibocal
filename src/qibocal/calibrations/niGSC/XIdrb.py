""" This script implements an easy 1-qubit protocol with only X-gates and identities.
It is a great example on how to write an own niGSC protocol. The single functions above have
little descriptions for the purpose of that function and what is important to include.

1. Step:
  Define the two module specific classes which are used in defining and executing an experiment,
  the circuit factory and experiment class.
  They can also just inherit everything from another module.
2. Step:
  Write the sequential post processing functions.
3. Step:
  Write the aggregational post processing function.
4. Step:
  Write the function to build a report. When using the qq module, a plotly figure has to be returned.
"""

# These libraries should be enough when starting a new protocol.
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import CircuitFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig
from qibocal.calibrations.niGSC.basics.utils import number_to_str
from qibocal.config import raise_error


# Define the circuit factory class for this specific module.
class ModuleFactory(CircuitFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        if not len(self.qubits) == 1:
            raise_error(
                ValueError,
                f"This class is written for gates acting on only one qubit, not {len(self.qubits)} qubits.",
            )
        self.name = "XId"

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(1)
        # Draw sequence length many I and X gates.
        gate_lists = np.random.choice([gates.I(0), gates.X(0)], size=depth)
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
        # Make the circuitfactory a list be able to store them using save_circuits method.
        self.prebuild()
        self.name = "XIdRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        # TODO change that.
        datadict["countX"] = len(circuit.gates_of_type("x"))
        return datadict


# Define the result class for this specific module.
class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "X-Id Benchmarking"


# The filter functions/post processing functions always dependent on circuit and data row!
# It is executed row by row when used on an experiment object.
def filter_sign(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for the XId.

    :math:`n_X` denotes the amount of :math:`X` gates in the circuit with gates
    :math:`g` and :math`i` the outcome which is either ground state :math:`0`
    or exited state :math:`1`.

    .. math::
        f_{\\text{sign}}(i,g)
        = (-1)^{n_X\\%2 + i}/2


    Args:
        circuit (Circuit): Not needed here.
        datarow (dict): The dictionary with the samples from executed circuits and amount of
                        X gates in the executed circuit.

    Returns:
        dict: _description_
    """
    samples = datarow["samples"]
    countX = datarow["countX"]
    filtersign = 0
    for s in samples:
        filtersign += (-1) ** (countX % 2 + s[0]) / 2.0
    datarow["filter"] = filtersign / len(samples)
    return datarow


# All the predefined sequential postprocessing / filter functions are bundled together here.
def post_processing_sequential(experiment: Experiment):
    """Perform sequential tasks needed to analyze the experiment results.

    The data is added/changed in the experiment, nothing has to be returned.

    Args:
        experiment (Experiment): Experiment object after execution of the experiment itself.
    """

    # Compute and add the ground state probabilities row by row.
    experiment.perform(filter_sign)


# After the row by row execution of tasks comes the aggregational task. Something like calculation
# of means, deviations, fitting data, tasks where the whole data as to be looked at, and not just
# one instance of circuit + other information.
def get_aggregational_data(experiment: Experiment, ndecays: int = 2) -> pd.DataFrame:
    """Computes aggregational tasks, fits data and stores the results in a data frame.

    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.
        ndecays (int): Number of decays to be fitted. Default is 2.

    Returns:
        pd.DataFrame: The summarized data.
    """
    # Has to fit the column describtion from ``filter_sign``.
    depths, ydata = experiment.extract("filter", "depth", "mean")
    _, ydata_std = experiment.extract("filter", "depth", "std")
    # Fit the filtered signal for each depth, there could be two overlaying exponential functions.
    popt, perr = fitting_methods.fit_expn_func(depths, ydata, n=ndecays)
    # Create dictionaries with fitting parameters and estimated errors
    popt_keys = [f"A{k+1}" for k in range(ndecays)]
    popt_keys += [f"p{k+1}" for k in range(ndecays)]
    popt_dict = dict(zip(popt_keys, popt))
    perr_keys = [f"A{k+1}_err" for k in range(ndecays)]
    perr_keys += [f"p{k+1}_err" for k in range(ndecays)]
    perr_dict = dict(zip(perr_keys, perr))
    # Build a list of dictionaries with the aggregational information.
    data = [
        {
            "depth": depths,  # The x-axis.
            "data": ydata,  # The filtred signal.
            "2sigma": 2 * ydata_std,  # The 2 * standard deviation error for each depth.
            "fit_func": "expn_func",  # Which function was used to fit.
            "popt": popt_dict,  # The real fitting parameters.
            "perr": perr_dict,  # The estimated errors.
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
    # Add general information to the table.
    report.info_dict["Number of qubits"] = len(experiment.data[0]["samples"][0])
    report.info_dict["Number of shots"] = len(experiment.data[0]["samples"])
    report.info_dict["runs"] = experiment.extract("samples", "depth", "count")[1][0]
    dfrow = df_aggr.loc["filter"]
    popt_pairs = list(dfrow["popt"].items())[::2] + list(dfrow["popt"].items())[1::2]
    report.info_dict["Fit"] = "".join(
        [f"{key}={number_to_str(value)} " for key, value in popt_pairs]
    )
    perr_pairs = list(dfrow["perr"].items())[::2] + list(dfrow["perr"].items())[1::2]
    report.info_dict["Fitting deviations"] = "".join(
        [f"{key}={number_to_str(value)} " for key, value in perr_pairs]
    )
    # Use the predefined ``scatter_fit_fig`` function from ``basics.utils`` to build the wanted
    # plotly figure with the scattered filtered data along with the mean for
    # each depth and the exponential fit for the means.
    report.all_figures.append(scatter_fit_fig(experiment, df_aggr, "depth", "filter"))

    # Return the figure of the report object and the corresponding table.
    return report.build()
