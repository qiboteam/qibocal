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
import qibo
from plotly.graph_objects import Figure
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import ZkFilteredCircuitFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig, update_fig
from qibocal.calibrations.niGSC.basics.rb_validation import filtered_decay_parameters

qibo.set_backend("numpy")


# Define the circuit factory class for this specific module.
class ModuleFactory(ZkFilteredCircuitFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits, size=2)
        self.name = "XId"

    def gate_group(self):
        return [gates.I(0), gates.X(0)]


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
        self.name = "XIdRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        # TODO change that.
        datadict["countX"] = circuit.gate_types["x"]
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
    g
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

    Returns:
        pd.DataFrame: The summarized data.
    """
    # Has to fit the column description from ``filter_sign``.
    depths, ydata = experiment.extract("filter", "depth", "mean")
    _, ydata_std = experiment.extract("filter", "depth", "std")
    # Fit the filtered signal for each depth, there could be two overlaying exponential functions.
    popt, perr = fitting_methods.fit_expn_func(depths, ydata, n=ndecays)
    # Create dictionaries with fitting parameters and estimated errors in the form {A1: ..., p1: ..., A2: ..., p2: ...}
    popt_keys = [f"A{k+1}" for k in range(ndecays)]
    popt_keys += [f"p{k+1}" for k in range(ndecays)]
    popt_dict = dict(zip(popt_keys, popt))
    perr_keys = [f"A{k+1}_err" for k in range(ndecays)]
    perr_keys += [f"p{k+1}_err" for k in range(ndecays)]
    perr_dict = dict(zip(perr_keys, perr))
    # Check if there are any imaginary values in the data.
    is_imaginary = np.any(np.iscomplex(ydata))
    popt_key = "popt_imag" if is_imaginary else "popt"
    # Build a list of dictionaries with the aggregational information.
    data = [
        {
            "depth": depths,  # The x-axis.
            "data": ydata,  # The filtred signal.
            "2sigma": 2 * ydata_std,  # The 2 * standard deviation error for each depth.
            "fit_func": "expn_func",  # Which function was used to fit.
            popt_key: popt_dict,  # The real fitting parameters.
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
        dataframe (pd.DataFrame): The data where the validation should be added.

    Returns:
        pd.DataFrame: The summarized data.
    """
    data = dataframe.to_dict("records")
    validation_label = "validation_imag" if "popt_imag" in data[0] else "validation"
    coefficients, decay_parameters = filtered_decay_parameters(
        experiment.circuitfactory, experiment.noise_model, with_coefficients=True, N=N
    )
    ndecays = len(coefficients)
    validation_keys = [f"A{k+1}" for k in range(ndecays)]
    validation_keys += [f"p{k+1}" for k in range(ndecays)]
    validation_dict = dict(
        zip(validation_keys, np.concatenate((coefficients, decay_parameters)))
    )

    data[0].update({validation_label: validation_dict})
    # The row name will be displayed as y-axis label.
    df = pd.DataFrame(data, index=dataframe.index)
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


def execute_simulation(
    depths: list,
    nshots: int = 500,
    noise_model: NoiseModel = None,
    ndecays: int = 2,
    validate: bool = False,
):
    """Execute simulation of XId Radomized Benchmarking experiment and generate an html report.

    Args:
        depths (list): list of depths for circuits
        nshots (int): number of shots per measurement
        noise_model (:class:`qibo.noise.NoiseModel`): noise model applied to the circuits in the simulation
        ndecays (int): number of decay parameters to fit. Default is 2.
        validate (bool): adds theoretical RB signal to the report when `True`. Dafault is `False`.

    Example:
        .. testcode::
            from qibocal.calibrations.niGSC.XIdrb import execute_simulation
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
    aggr_df = get_aggregational_data(experiment, ndecays=ndecays)
    if validate:
        aggr_df = add_validation(experiment, aggr_df)
    report_figure = build_report(experiment, aggr_df)
    report_figure.show()
