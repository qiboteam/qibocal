# 1. Step:
#   Define the two module specific classes which are used in defining and executing an experiment,
#   the circuit factory and experiment class.
#   They can also just inherit everything from another module.
# 2. Step:
#   Write the analzye function.
# 3. Step:
#   Write the result class which uses the modified data (modified by the analyze function)
#   from the experiment object and displays the results module specific.
# 4. Step:
#
# Load to __init__.py file in calibrations/
# Make a jupyter notebook with the single steps with 'checks'
# -> create a factory, check the factory
# -> create an experiment, check the experiment

# For typing
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.fitting.rb_methods as fitting_methods
from qibocal.calibrations.protocols.abstract import (
    Circuitfactory,
    Experiment,
    Report,
    scatter_fit_fig,
)
from qibocal.calibrations.protocols.utils import gate_adjoint_action_to_pauli_liouville


# Define the circuit factory class for this specific module.
class moduleFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits)
        self.name = "XId"

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(len(self.qubits), density_matrix=True)
        # There are only two gates to choose from.
        a = [gates.I(0), gates.X(0)]
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 2, size=depth)
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = np.take(a, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


# Define the experiment class for this specific module.
class moduleExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
    ) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel)
        self.name = "XIdRB"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        # TODO change that.
        datadict["countX"] = circuit.draw().count("X")
        return datadict


# Define the result class for this specific module.
class moduleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "X-Id Benchmarking"


# filter functions always dependent on circuit and datarow!
def filter_sign(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for the XId.

    :math:`n_X` denotes the amount of :math:`X` gates in the circuit with gates
    :math:`g` and :math`i` the outcome which is either ground state :math:`0`
    or exited state :math:`1`.

    .. math::
        f_{\\text{sign}}(i,g)
        = (-1)^{n_X\\%2 + i}/2


    Args:
        circuit (Circuit): _description_
        datarow (dict): _description_

    Returns:
        dict: _description_
    """
    samples = datarow["samples"]
    countX = datarow["countX"]
    filtersign = 0
    for s in samples:
        filtersign += (-1) ** (countX % 2 + s[0]) / 2.0
    datarow["filter"] = filtersign / len(samples)


def post_processing_sequential(experiment: Experiment):
    # Compute and add the ground state probabilities row by row.
    experiment.perform(filter_sign)


def get_aggregational_data(experiment: Experiment) -> pd.DataFrame:
    depths, ydata = experiment.extract("depth", "filter", "mean")
    _, ydata_std = experiment.extract("depth", "filter", "std")

    popt, perr = fitting_methods.fit_exp2_func(depths, ydata)
    data = [
        {
            "depth": depths,
            "data": ydata,
            "2sigma": 2 * ydata_std,
            "fit_func": "exp2_func",
            "popt": {"A1": popt[0], "A2": popt[1], "p1": popt[2], "p2": popt[2]},
            "perr": {
                "A1_err": perr[0],
                "A2_err": perr[1],
                "p1_err": perr[2],
                "p2_err": perr[3],
            },
        }
    ]
    df = pd.DataFrame(data, index=["filter"])
    return df


def build_report(experiment: Experiment, df_aggr: pd.DataFrame):
    report = moduleReport()
    report.info_dict["Number of qubits"] = len(experiment.data[0]["samples"][0])
    report.info_dict["Number of shots"] = len(experiment.data[0]["samples"])
    report.info_dict["runs"] = experiment.extract("depth", "samples", "count")[1][0]
    print(df_aggr)
    report.info_dict["Fitting daviations"] = "".join(
        [
            "{}:{:.3f} ".format(key, df_aggr.loc["filter"]["perr"][key])
            for key in df_aggr.loc["filter"]["perr"]
        ]
    )
    report.all_figures.append(scatter_fit_fig(experiment, df_aggr, "depth", "filter"))
    return report.build()


# def theoretical_outcome(noisemodel: NoiseModel) -> list:
#     """Only for one qubit and Pauli Error noise!

#     Args:
#         experiment (Experiment): _description_
#         noisemodel (NoiseModel): _description_

#     Returns:
#         float: _description_

#     Yields:
#         Iterator[float]: _description_
#     """
#     from qibocal.calibrations.protocols import validate_simulations

#     # Check for correctness of noise model and gate independence.
#     errorkeys = noisemodel.errors.keys()
#     if len(errorkeys) == 1 and list(errorkeys)[0] == gates.X:
#         # Extract the noise acting on unitaries and turn it into the associated
#         # error channel.
#         error = noisemodel.errors[gates.X][0]
#         errorchannel = error.channel(0, *error.options)
#         if isinstance(
#             errorchannel, (gates.PauliNoiseChannel, gates.ThermalRelaxationChannel)
#         ):
#             liouvillerep = errorchannel.to_pauli_liouville(normalize=True)
#             phi = (
#                 np.eye(4)
#                 - liouvillerep @ gate_adjoint_action_to_pauli_liouville(gates.X(0))
#             ) / 2
#     return validate_simulations.validation(phi)
