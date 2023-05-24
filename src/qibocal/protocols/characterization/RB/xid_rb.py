# from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Union

import numpy as np
import pandas as pd
from qibo.gates import X
from qibo.noise import NoiseModel
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.calibrations.niGSC.XIdrb import ModuleFactory as XIdScan
from qibocal.protocols.characterization.RB.result import (
    NoOffsetDecayResult,
    get_hists_data,
)
from qibocal.protocols.characterization.RB.utils import extract_from_data

NoneType = type(None)


@dataclass
class XIdParameters(Parameters):
    """X-Id Randomized Benchmarking runcard inputs."""

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


class XIdData(pd.DataFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_func = self.to_csv
        self.to_csv = self.to_csv_helper

    def to_csv_helper(self, path):
        self.save_func(f"{path}/{self.__class__.__name__}.csv")


class XIdResult(NoOffsetDecayResult):
    pass


def setup_scan(params: XIdParameters) -> Iterable:
    return XIdScan(params.nqubits, params.depths * params.niter, params.qubits)


def execute(
    scan: Iterable,
    nshots: Union[int, NoneType] = None,
    noise_model: Union[NoiseModel, NoneType] = None,
):
    # Execute
    data_list = []
    for c in scan:
        depth = c.depth
        nx = c.gate_types[X]
        if noise_model is not None:
            c = noise_model.apply(c)
        samples = c.execute(nshots=nshots).samples()
        data_list.append({"depth": depth, "samples": samples, "nx": nx})
    return data_list


def aggregate(data: XIdData):
    def filter(nx_list, samples_list):
        return [
            sum((-1) ** (nx % 2 + s[0]) / 2.0 for s in samples)
            for nx, samples in zip(nx_list, samples_list)
        ]

    data_agg = data.assign(signal=lambda x: filter(x.nx.to_list(), x.samples.to_list()))
    # Histogram
    hists = get_hists_data(data_agg)
    # Build the result object
    return XIdResult(
        *extract_from_data(data_agg, "signal", "depth", "mean"), hists=hists
    )


def aquire(
    params: XIdParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> XIdData:
    scan = setup_scan(params)
    if params.noise_model:
        from qibocal.calibrations.niGSC.basics import noisemodels

        noise_model = getattr(noisemodels, params.noise_model)(*params.noise_params)
    else:
        noise_model = None
    # Here the platform can be connected
    data = execute(scan, params.nshots, noise_model)
    xid_data = XIdData(data)
    xid_data.attrs = params.__dict__
    return xid_data


def extract(data: XIdData):
    result = aggregate(data)
    result.fit()
    return result


def plot(data: XIdData, result: XIdResult, qubit):
    table_str = "".join(
        [f" | {key}: {value}<br>" for key, value in {**data.attrs}.items()]
    )
    fig = result.plot()
    return [fig], table_str


xid_rb = Routine(aquire, extract, plot)
