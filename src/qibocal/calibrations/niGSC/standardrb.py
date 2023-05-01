""" Here the standard randomized benchmarking is implemented using the
niGSC (non-interactive gate set characterization) architecture.
"""


from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import SingleCliffordsFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig
from qibocal.calibrations.niGSC.basics.utils import gate_fidelity, number_to_str


class ModuleFactory(SingleCliffordsFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        self.name = "SingleCliffordsInv"

    def build_circuit(self, depth: int) -> Circuit:
        """Overwrites parent method. Add an inverse gate before the measurement.

        Args:
            depth (int): How many gate layers.

        Returns:
            (Circuit): A circuit with single qubit Clifford gates with ``depth`` many layers
            and an inverse gate before the measurement gate.
        """

        # Initiate a ``Circuit`` object with as many qubits as is indicated with the list
        # of qubits on which the gates should act on.
        circuit = Circuit(len(self.qubits))
        # Add ``depth`` many gate layers.
        for _ in range(depth):
            circuit.add(self.gate_layer())
        # If there is at least one gate in the circuit, add an inverse.
        if depth > 0:
            # Build a gate out of the unitary of the whole circuit and
            # take the daggered version of that.
            circuit.add(
                gates.Unitary(circuit.unitary(), *range(len(self.qubits))).dagger()
            )
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


class ModuleExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        data: Iterable | None = None,
        nshots: int | None = None,
        noise_model: NoiseModel = None,
    ) -> None:
        """Calls the parent method, sets name.

        Args:
            circuitfactory (Iterable): Gives a certain amount of circuits when
                iterated over.
            nshots (int): For execution of circuit, indicates how many shots.
            data (Iterable): If filled, ``data`` can be used to specifying parameters
                        while executing a circuit or deciding how to process results.
                        It is used to store all relevant data.
        """
        super().__init__(circuitfactory, data, nshots, noise_model)
        self.name = "StandardRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        """Overwrites parent class method. Executes a circuit, adds the single shot results
        and depth of the circuit to the data row.

        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.

        Returns:
            datarow (dict):
        """

        # Execute parent class method.
        datarow = super().execute(circuit, datarow)
        # Substract 1 for sequence length to not count the inverse gate and
        # substract the measurement gate.
        datarow["depth"] = (circuit.depth - 2) if circuit.depth > 1 else 0
        return datarow


class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Standard Randomized Benchmarking"


def groundstate_probabilities(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the ground state probability with data from single shot measurements.

    Args:
        circuit (Circuit): Not needed here.
        datarow (dict): The dictionary holding the samples.

    Returns:
        dict: The updated dictionary.
    """

    # Get the samples data from the dictionary
    samples = datarow["samples"]
    # Create the ground state as it would look like in a single shot measurement.
    ground = np.array([0] * len(samples[0]))
    # Calculate the probability of the samples being in the ground state
    # by counting the number of samples that are equal to the ground state
    # and dividing it by the total number of samples.
    datarow["groundstate probability"] = np.sum(
        np.product(samples == ground, axis=1)
    ) / len(samples)
    # Return the updated dictionary.
    return datarow


def post_processing_sequential(experiment: Experiment):
    """Perform sequential tasks needed to analyze the experiment results.

    The data is added/changed in the experiment, nothing has to be returned.

    Args:
        experiment (Experiment): Experiment object after execution of the experiment itself.
    """

    # Compute and add the ground state probabilities row by row.
    experiment.perform(groundstate_probabilities)


def get_aggregational_data(experiment: Experiment) -> pd.DataFrame:
    """Computes aggregational tasks, fits data and stores the results in a data frame.

    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.

    Returns:
        pd.DataFrame: The summarized data.
    """

    # Has to fit the column description from ``groundstate_probabilities``.
    depths, ydata = experiment.extract("groundstate probability", "depth", "mean")
    _, ydata_std = experiment.extract("groundstate probability", "depth", "std")
    # Fit the ground state probabilies mean for each depth.
    popt, perr = fitting_methods.fit_exp1B_func(depths, ydata)
    # Build a list of dictionaries with the aggregational information.
    data = [
        {
            "depth": depths,  # The x-axis.
            "data": ydata,  # The mean of ground state probability for each depth.
            "2sigma": 2 * ydata_std,  # The 2 * standard deviation error for each depth.
            "fit_func": "exp1B_func",  # Which function was used to fit.
            "popt": {
                "A": popt[0],
                "p": popt[1],
                "B": popt[2],
            },  # The fitting paramters.
            "perr": {
                "A_err": perr[0],
                "p_err": perr[1],
                "B_err": perr[2],
            },  # The estimated errors.
        }
    ]
    # The row name will be displayed as y-axis label.
    df = pd.DataFrame(data, index=["groundstate probability"])
    return df


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
    report.info_dict["Gate fidelity"] = "{:.4f}".format(
        gate_fidelity(df_aggr.iloc[0]["popt"]["p"])
    )
    report.info_dict["Gate fidelity primitive"] = "{:.4f}".format(
        gate_fidelity(df_aggr.iloc[0]["popt"]["p"], primitive=True)
    )
    dfrow = df_aggr.loc["groundstate probability"]
    report.info_dict["Fit"] = "".join(
        [f"{key}={number_to_str(value)} " for key, value in dfrow["popt"].items()]
    )
    report.info_dict["Fitting deviations"] = "".join(
        [f"{key}={number_to_str(value)} " for key, value in dfrow["perr"].items()]
    )
    # Use the predefined ``scatter_fit_fig`` function from ``basics.plot`` to build the wanted
    # plotly figure with the scattered ground state probability data along with the mean for
    # each depth and the exponential fit for the means.
    report.all_figures.append(
        scatter_fit_fig(experiment, df_aggr, "depth", "groundstate probability")
    )
    # Return the figure of the report object and the corresponding table.
    return report.build()
