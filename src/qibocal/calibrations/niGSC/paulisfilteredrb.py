from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import qibo
from plotly.graph_objects import Figure
from qibo import gates, matrices
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import CircuitFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig, update_fig
from qibocal.calibrations.niGSC.basics.rb_validation import filtered_decay_parameters
from qibocal.config import raise_error

qibo.set_backend("numpy")

# matrices = QiboMatrices()
pauli_list = []
for p in [matrices.I, matrices.X, matrices.Y, matrices.Z]:
    pauli_list.append(gates.Unitary(p, 0))
    pauli_list.append(gates.Unitary(-p, 0))
    pauli_list.append(gates.Unitary(1j * p, 0))
    pauli_list.append(gates.Unitary(-1j * p, 0))


def is_xy(g):
    """Check if a Unitary gate is not diagonal"""
    return not np.allclose(np.abs(g.parameters), np.eye(2))


# Define the circuit factory class for this specific module.
class ModuleFactory(CircuitFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        if len(self.qubits) != 1:
            raise_error(
                ValueError,
                f"This class is written for gates acting on only one qubit, not {len(self.qubits)} qubits.",
            )
        self.name = "PauliGroup"

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(1, density_matrix=True)
        # There are only two gates to choose from for every qubit.
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, len(pauli_list), size=depth)
        # Get the gates with random_ints as indices.
        gate_lists = np.take(pauli_list, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(0))
        return circuit

    def gate_group(self):
        return pauli_list


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
        self.name = "PaulisFilteredRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        # Find the number of X and Y gates in the circuit
        countXY = circuit.gate_types["x"] + circuit.gate_types["y"]
        for gate in circuit.gates_of_type("Unitary"):
            countXY += int(is_xy(gate[-1]))
        datadict["countXY"] = countXY
        return datadict


# Define the result class for this specific module.
class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Pauli Filtered Randomized Benchmarking"


# The filter functions/post processing functions always dependent on circuit and data row!
# It is executed row by row when used on an experiment object.
def filter_irrep(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for the gate group :math:`\\{i^k\\sigma_j\\}_\\{j, k=0\\}^\\{3\\}`.

    :math:`n_X`, :math:`n_Y` denote the amount of :math:`X` and :math:`Y` gates in the circuit with gates
    :math:`g` and :math`i` the outcome which is either ground state :math:`0`
    or exited state :math:`1`.

    .. math::
        f_{\\lambda}(i,g)
        = (-1)^{m_x+m_y + i}/2


    Args:
        circuit (Circuit): Not needed here.
        datarow (dict): The dictionary with the samples from executed circuits and :math:`\\sum k_j`
                        of all the gates :math:`R_x(k_j\\pi/2)` in the executed circuit.

    Returns:
        dict: _description_
    """
    samples = datarow["samples"]
    countXY = datarow["countXY"]
    filtervalue = 0
    for s in samples:
        filtervalue += np.conj(((-1) ** (countXY % 2 + s[0])) / 2.0)

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
def get_aggregational_data(experiment: Experiment, ndecays: int = 1) -> pd.DataFrame:
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
    # Build a list of dictionaries with the aggregational information.
    data = [
        {
            "depth": depths,  # The x-axis.
            "data": ydata,  # The filtred signal.
            "2sigma": 2 * ydata_std,  # The 2 * standard deviation error for each depth.
            "fit_func": "exp1_func",  # Which function was used to fit.
            "popt": popt_dict,  # The fitting parameters.
            "perr": perr_dict,  # The estimated errors.
        }
    ]

    df = pd.DataFrame(data, index=["filter"])
    return df


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

    data = dataframe.to_dict("records")
    coefficients, decay_parameters = filtered_decay_parameters(
        experiment.circuitfactory, experiment.noise_model, with_coefficients=True, N=N
    )
    validation_dict = {"A": coefficients[0], "p": decay_parameters[0]}
    data[0].update({"validation": validation_dict})
    # The row name will be displayed as y-axis label.
    df = pd.DataFrame(data, index=dataframe.index)
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
    if "validation" in df_aggr:
        report.all_figures[-1] = update_fig(report.all_figures[-1], df_aggr)
    # Return the figure the report object builds out of all figures added to the report.
    return report.build()


def execute_simulation(
    depths: list,
    nshots: int = 500,
    noise_model: NoiseModel = None,
    ndecays: int = 1,
    validate: bool = False,
):
    """Execute simulation of Paulis Filtered RB experiment
    and generate an html report with the validation of the results

    Args:
        depths (list): list of depths for circuits
        nshots (int): number of shots per measurement
        noise_model (:class:`qibo.noise.NoiseModel`): noise model applied to the circuits in the simulation
        ndecays (int): number of decay parameters to fit. Default is 1.
        validate (bool): adds theoretical RB signal to the report when `True`. Dafault is `False`.

    Example:
        .. testcode::
            from qibocal.calibrations.niGSC.paulisfilteredrb import execute_simulation
            from qibocal.calibrations.niGSC.basics import noisemodels
            # Build the noise model.
            noise_params = [0.01, 0.02, 0.05]
            pauli_noise_model = noisemodels.PauliErrorOnX(*noise_params)
            # Generate the list of depths repeating 20 times
            runs = 20
            depths = list(range(1, 31)) * runs
            # Run the simulation
            execute_simulation(depths, 500, pauli_noise_model)
    """

    # Execute an experiment.
    nqubits = 1
    factory = ModuleFactory(nqubits, depths)
    experiment = ModuleExperiment(factory, nshots=nshots, noise_model=noise_model)
    experiment.perform(experiment.execute)

    # Build a report with validation of the results
    post_processing_sequential(experiment)
    aggr_df = get_aggregational_data(experiment, ndecays)
    if validate:
        aggr_df = add_validation(experiment, aggr_df)
    report_figure = build_report(experiment, aggr_df)
    report_figure.show()
