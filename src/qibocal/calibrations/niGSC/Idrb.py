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
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig, update_fig
from qibocal.config import raise_error

qibo.set_backend("numpy")


# Define the circuit factory class for this specific module.
class ModuleFactory(CircuitFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        if not len(self.qubits) == 1:
            raise_error(
                ValueError,
                "This class is written for gates acting on only one qubit, not {} qubits.".format(
                    len(self.qubits)
                ),
            )
        self.name = "Id"

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(1, density_matrix=True)
        # Create a list with Id gates of size depth.
        gate_lists = [gates.I(0)] * depth
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(0))
        return circuit

    def gate_group(self):
        return [gates.I(0)]

    def irrep_info(self):
        from qibo.quantum_info import comp_basis_to_pauli

        basis = np.eye(4)  # comp_basis_to_pauli(self.nqubits, normalize=True)

        return (basis, 0, 1, 4)


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
        self.name = "IdRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        return datadict


# Define the result class for this specific module.
class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Id Benchmarking"


# The filter functions/post processing functions always dependent on circuit and data row!
# It is executed row by row when used on an experiment object.
def filter_trivial(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for the Id.

    The filter function is calculated for the circuit with gates
    :math:`g` and the outcome :math`i` which is either ground state :math:`0`
    or exited state :math:`1`.

    .. math::
        f_{\\text{sign}}(i,g)
        = 1-i


    Args:
        circuit (Circuit): Not needed here.
        datarow (dict): The dictionary with the samples from executed circuits.

    Returns:
        dict: _description_
    """
    samples = datarow["samples"]
    filter_trivial = 0
    for s in samples:
        filter_trivial += 1 - s[0]
    datarow["filter"] = filter_trivial / len(samples)
    return datarow


# All the predefined sequential postprocessing / filter functions are bundled together here.
def post_processing_sequential(experiment: Experiment):
    """Perform sequential tasks needed to analyze the experiment results.

    The data is added/changed in the experiment, nothing has to be returned.

    Args:
        experiment (Experiment): Experiment object after execution of the experiment itself.
    """

    # Compute and add the ground state probabilities row by row.
    experiment.perform(filter_trivial)


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
    # Has to fit the column description from ``filter_sign``.
    depths, ydata = experiment.extract("filter", "depth", "mean")
    _, ydata_std = experiment.extract("filter", "depth", "std")
    # Fit the filtered signal for each depth, there could be two overlaying exponential functions.
    popt, perr = fitting_methods.fit_exp4_func(depths, ydata)
    # Check if there can be non-zero imaginary values in the data.
    is_imaginary = np.any(np.iscomplex(ydata))
    popt_key = "popt_imag" if is_imaginary else "popt"
    # Build a list of dictionaries with the aggregational information.
    data = [
        {
            "depth": depths,  # The x-axis.
            "data": ydata,  # The filtred signal.
            "2sigma": 2 * ydata_std,  # The 2 * standard deviation error for each depth.
            "fit_func": "expn_func",  # Which function was used to fit.
            popt_key: {
                "A1": popt[0],
                "p1": popt[4],
                "A2": popt[1],
                "p2": popt[5],
                "A3": popt[2],
                "p3": popt[6],
                "A4": popt[3],
                "p4": popt[7],
            },  # The real fitting parameters.
            "perr": {
                "A1_err": perr[0],
                "p1_err": perr[4],
                "A2_err": perr[1],
                "p2_err": perr[5],
                "A3_err": perr[2],
                "p3_err": perr[6],
                "A4_err": perr[3],
                "p4_err": perr[7],
            },  # The estimated errors.
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
    # If there is validation, add it to the ``report.info_dict`` as it might be too long for th plot legend
    validation_label = "validation_imag" if is_imag else "validation"
    if validation_label in df_aggr.loc["filter"]:
        from qibocal.calibrations.niGSC.basics import utils

        report.info_dict["Validation"] = "".join(
            [
                "{}: {} ".format(
                    key,
                    utils.number_to_str(df_aggr.loc["filter"][validation_label][key]),
                )
                for key in df_aggr.loc["filter"][validation_label]
            ]
        )
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
            name="Validation",
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
                name="Validation",
                is_imag=True,
            )
    # Return the figure the report object builds out of all figures added to the report.
    return report.build()


def execute_simulation(
    depths: list,
    nshots: int = 500,
    noise_model: NoiseModel = None,
    validate: bool = False,
):
    """Execute simulation of Id Radomized Benchmarking experiment and generate an html report with the validation of the results

    Args:
        depths (list): list of depths for circuits
        nshots (int): number of shots per measurement
        noise_model (:class:`qibo.noise.NoiseModel`): noise model applied to the circuits in the simulation

    Example:
        .. testcode::
            from qibocal.calibrations.niGSC.Idrb.py import execute_simulation
            from qibocal.calibrations.niGSC.basics import noisemodels
            # Build the noise model.
            noise_params = [0.01, 0.02, 0.05]
            pauli_noise_model = noisemodels.PauliErrorOnX(*noise_params)
            # Generate the list of depths repeating 20 times
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
    aggr_df = get_aggregational_data(experiment)
    if validate:
        aggr_df = add_validation(experiment, aggr_df)
    report_figure = build_report(experiment, aggr_df)
    report_figure.show()