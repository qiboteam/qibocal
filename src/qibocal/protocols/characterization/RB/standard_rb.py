from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from qibo.noise import NoiseModel

from qibocal.auto.operation import Routine
from qibocal.calibrations.niGSC.standardrb import ModuleFactory as StandardRBScan
from qibocal.protocols.characterization.RB.result import DecayWithOffsetResult
from qibocal.protocols.characterization.RB.utils import extract_from_data

from .data import RBData
from .params import RBParameters

NoneType = type(None)
NPULSES_PER_CLIFFORD = 1.875

@dataclass
class StandardRBResult(DecayWithOffsetResult):
    """Inherits from `DecayWithOffsetResult`, a result class storing data and parameters
    of a single decay with statistics.

    Adds the method of calculating a fidelity out of the fitting parameters.
    TODO calculate SPAM errors with A and B
    TODO calculate the error of the fidelity

    """

    def calculate_fidelities(self):
        """Takes the fitting parameter of the decay and calculates a fidelity. Stores the
        primitive fidelity, the fidelity and the average gate error in an attribute dictionary.
        """
        infidelity = (1 - self.p) / 2
        self.fidelity_dict = {
            "fidelity": 1 - infidelity,
            "pulse fidelity": 1 - infidelity / NPULSES_PER_CLIFFORD,
        }


def setup_scan(params: RBParameters) -> Iterable:
    """An iterator building random Clifford sequences with an inverse in the end.

    Args:
        params (RBParameters): The needed parameters.

    Returns:
        Iterable: The iterator of circuits.
    """

    return StandardRBScan(params.nqubits, params.depths * params.niter, params.qubits)


def execute(
    scan: Iterable,
    nshots: Union[int, NoneType] = None,
    noise_model: Union[NoiseModel, NoneType] = None,
) -> List[dict]:
    """Execute a given scan with the given number of shots and if its a simulation with the given
    noise model.

    Args:
        scan (Iterable): The ensemble of experiments (here circuits)
        nshots (Union[int, NoneType], optional): Number of shots per circuit. Defaults to None.
        noise_model (Union[NoiseModel, NoneType], optional): If its a simulation a noise model
            can be applied. Defaults to None.

    Returns:
        List[dict]: A list with one dictionary for each executed circuit where the data is stored.
    """

    data_list = []
    # Iterate through the scan and execute each circuit.
    for c in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (c.depth - 2) if c.depth > 1 else 0
        if noise_model is not None:
            c = noise_model.apply(c)
        samples = c.execute(nshots=nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append({"depth": depth, "samples": samples})
    return data_list


def aggregate(data: RBData) -> StandardRBResult:
    """Takes a data frame, processes it and aggregates data in order to create
    a routine result object.

    Args:
        data (RBData): Actually a data frame from where the data is processed.

    Returns:
        StandardRBResult: The aggregated data.
    """

    def p0s(samples_list):
        ground = np.array([0] * len(samples_list[0][0]))
        my_p0s = []
        for samples in samples_list:
            my_p0s.append(np.sum(np.product(samples == ground, axis=1)) / len(samples))
        return my_p0s

    # The signal is here the survival probability.
    data_agg = data.assign(signal=lambda x: p0s(x.samples.to_list()))
    return StandardRBResult(*extract_from_data(data_agg, "signal", "depth", list))


def acquire(params: RBParameters, *args) -> RBData:
    """The data acquisition stage of standard rb.

    1. Set up the scan
    2. Execute the scan
    3. Put the acquired data in a standard rb data object.

    Args:
        params (RBParameters): All parameters in one object.

    Returns:
        RBData: _description_
    """

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    scan = setup_scan(params)
    # For simulations, a noise model can be added.
    if params.noise_model:
        from qibocal.calibrations.niGSC.basics import noisemodels

        noise_model = getattr(noisemodels, params.noise_model)(*params.noise_params)
    else:
        noise_model = None
    # Execute the scan.
    data = execute(scan, params.nshots, noise_model)
    # Build the data object which will be returned and later saved.
    standardrb_data = RBData(data)
    standardrb_data.attrs = params.__dict__
    return standardrb_data


def extract(data: RBData) -> StandardRBResult:
    """Takes a data frame and extracts the depths,
    average values of the survival probability and histogram

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """

    result = aggregate(data)
    result.semi_parametric_bootstrap(100, data.attrs['nshots'])
    result.calculate_fidelities()
    return result


def plot(data: RBData, result: StandardRBResult, qubit) -> Tuple[List[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        result (StandardRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        Tuple[List[go.Figure], str]:
    """

    table_str = "".join(
        [
            f" | {key}: {value}<br>"
            for key, value in data.attrs.items()
        ]
    )
    table_str += "".join(
        f" | {key}: {value:.4f}\u00B1{result.perr/2:.4f}<br>"
            for key, value in result.fidelity_dict.items()
    )
    fig = result.plot()
    return [fig], table_str


# Build the routine object which is used by qq-auto.
standard_rb = Routine(acquire, extract, plot)
