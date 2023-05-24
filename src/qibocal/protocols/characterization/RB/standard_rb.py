# from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Union

import numpy as np
import pandas as pd
from qibo.noise import NoiseModel
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.calibrations.niGSC.standardrb import ModuleFactory as StandardRBScan
from qibocal.protocols.characterization.RB.result import DecayResult, get_hists_data
from qibocal.protocols.characterization.RB.utils import extract_from_data

NoneType = type(None)


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    nqubits: int
    qubits: list
    depths: list
    niter: int
    nshots: int
    noise_model: str = ""
    noise_params: list = field(default_factory=list)

    def __post_init__(self):
        # TODO should the noise_model be already build here?
        if self.noise_model is not None:
            pass


class StandardRBData(pd.DataFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_func = self.to_csv
        self.to_csv = self.to_csv_helper

    def to_csv_helper(self, path):
        self.save_func(f"{path}/{self.__class__.__name__}.csv")


class StandardRBResult(DecayResult):
    # def __init__(self, m, y, A=None, p=None, B=None, Aerr=None, perr=None, Berr=None, hists=None):
    # super().__init__(m, y, A, p, B, Aerr, perr, Berr, hists)
    def calculate_fidelities(self):
        # Divide infidelity by magic number
        magic_number = 1.875
        infidelity = (1 - self.p) / magic_number
        self.fidelity_dict = {
            "fidelity_primitive": 1 - ((1 - self.p) / 2),
            "fidelity": 1 - infidelity,
            "average_error_gate": infidelity * 100,
        }


def setup_scan(params: StandardRBParameters) -> Iterable:
    return StandardRBScan(params.nqubits, params.depths * params.niter, params.qubits)


def execute(
    scan: Iterable,
    nshots: Union[int, NoneType] = None,
    noise_model: Union[NoiseModel, NoneType] = None,
):
    # Execute
    data_list = []
    for c in scan:
        depth = (c.depth - 2) if c.depth > 1 else 0
        if noise_model is not None:
            c = noise_model.apply(c)
        samples = c.execute(nshots=nshots).samples()
        data_list.append({"depth": depth, "samples": samples})
    return data_list


def aggregate(data: StandardRBData):
    # The signal is here the survival probability.
    data_agg = data.assign(signal=lambda x: 1 - np.mean(x.samples.to_list(), axis=1))
    # Histogram
    hists = get_hists(data_agg)
    # Build the result object
    return StandardRBResult(
        *extract_from_data(data_agg, "signal", "depth", "mean"), hists=hists
    )


def aquire(
    params: StandardRBParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> StandardRBData:
    scan = setup_scan(params)
    if params.noise_model:
        from qibocal.calibrations.niGSC.basics import noisemodels

        noise_model = getattr(noisemodels, params.noise_model)(*params.noise_params)
    else:
        noise_model = None
    # Here the platform can be connected
    data = execute(scan, params.nshots, noise_model)
    standardrb_data = StandardRBData(data)
    standardrb_data.attrs = params.__dict__
    return standardrb_data


def extract(data: StandardRBData):
    result = aggregate(data)
    result.fit()
    result.calculate_fidelities()
    return result


def plot(data: StandardRBData, result: StandardRBResult, qubit):
    table_str = "".join(
        [
            f" | {key}: {value}<br>"
            for key, value in {**data.attrs, **result.fidelity_dict}.items()
        ]
    )
    fig = result.plot()
    return [fig], table_str


standard_rb = Routine(aquire, extract, plot)
