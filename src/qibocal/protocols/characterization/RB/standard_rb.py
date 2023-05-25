from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo.noise import NoiseModel
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.auto.operation import Parameters, Qubits, Routine
from qibocal.calibrations.niGSC.standardrb import ModuleFactory as StandardRBScan
from qibocal.protocols.characterization.RB.result import (
    DecayWithOffsetResult,
    get_hists_data,
)
from qibocal.protocols.characterization.RB.utils import extract_from_data

NoneType = type(None)


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    nqubits: int
    """The amount of qubits on the chip """
    qubits: list
    """A list of indices which qubit(s) should be benchmarked """
    depths: Union[list, dict]
    """A list of depths/sequence lengths. If a dictionary is given the list will be build."""
    niter: int
    """Sets how many iterations over the same depth value."""
    nshots: int
    """For each sequence how many shots for statistics should be performed."""
    noise_model: str = ""
    """For simulation purposes, string has to match what is in qibocal. ... basics.noisemodels"""
    noise_params: list = field(default_factory=list)
    """With this the noise model will be initialized, if not given random values will be used."""

    def __post_init__(self):
        if isinstance(self.depths, dict):
            self.depths: list = list(
                range(self.depths["start"], self.depths["stop"], self.depths["step"])
            )


class StandardRBData(pd.DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # FIXME this is necessary because the auto builder calls .to_csv(path_to_directory).
        # But the DataFrame object from pandas needs a path to a file.
        self.save_func = self.to_csv
        self.to_csv = self.to_csv_helper

    def to_csv_helper(self, path):
        self.save_func(f"{path}/{self.__class__.__name__}.csv")


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

        # Divide infidelity by magic number
        magic_number = 1.875
        infidelity = (1 - self.p) / magic_number
        self.fidelity_dict = {
            "fidelity_primitive": 1 - ((1 - self.p) / 2),
            "fidelity": 1 - infidelity,
            "average_error_gate": infidelity * 100,
        }


def setup_scan(params: StandardRBParameters) -> Iterable:
    """An iterator building random Clifford sequences with an inverse in the end.

    Args:
        params (StandardRBParameters): The needed parameters.

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


def aggregate(data: StandardRBData) -> StandardRBResult:
    """Takes a data frame, processes it and aggregates data in order to create
    a routine result object.

    Args:
        data (StandardRBData): Actually a data frame from where the data is processed.

    Returns:
        StandardRBResult: The aggregated data.
    """

    # The signal is here the survival probability.
    data_agg = data.assign(signal=lambda x: 1 - np.mean(x.samples.to_list(), axis=1))
    # Histogram
    hists = get_hists_data(data_agg)
    # Build the result object
    return StandardRBResult(
        *extract_from_data(data_agg, "signal", "depth", "mean"), hists=hists
    )


def acquire(
    params: StandardRBParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> StandardRBData:
    """The data acquisition stage of standard rb.

    1. Set up the scan
    2. Execute the scan
    3. Put the acquired data in a standard rb data object.

    Args:
        params (StandardRBParameters): All parameters in one object.
        platform (AbstractPlatform): Not used yet.
        qubits (Qubits): Not used yet.

    Returns:
        StandardRBData: _description_
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
    standardrb_data = StandardRBData(data)
    standardrb_data.attrs = params.__dict__
    return standardrb_data


def extract(data: StandardRBData) -> StandardRBResult:
    """Takes a data frame and extracts the depths,
    average values of the survival probability and histogram

    Args:
        data (StandardRBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """

    result = aggregate(data)
    result.fit()
    result.calculate_fidelities()
    return result


def plot(
    data: StandardRBData, result: StandardRBResult, qubit
) -> Tuple[List[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (StandardRBData): Data object used for the table.
        result (StandardRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        Tuple[List[go.Figure], str]:
    """

    table_str = "".join(
        [
            f" | {key}: {value}<br>"
            for key, value in {**data.attrs, **result.fidelity_dict}.items()
        ]
    )
    fig = result.plot()
    return [fig], table_str


standard_rb = Routine(acquire, extract, plot)
