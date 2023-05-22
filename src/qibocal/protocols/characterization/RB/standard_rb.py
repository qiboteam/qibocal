from dataclasses import dataclass, field
from typing import List, NoneType, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame
from qibo.noise import NoiseModel
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.auto.operation import Parameters, Qubits, Results, Routine

# from types import NoneType
from qibocal.calibrations.niGSC.basics.fitting import exp1B_func, fit_exp1B_func
from qibocal.calibrations.niGSC.basics.plot import plot_qq
from qibocal.calibrations.niGSC.standardrb import (
    ModuleExperiment,
    ModuleFactory,
    build_report,
    get_aggregational_data,
    post_processing_sequential,
)

real_numeric = Union[int, float, np.number]
numeric = Union[int, float, complex, np.number]


class DecayResult(Results):
    """
    y[i] = (A +- Aerr) (p +- perr)^m[i] + (B +- Berr)
    # for later: y= sum_i A_i p_i^m (needs m integer)
    """

    def __init__(self, m, y, A=None, p=None, B=None, Aerr=None, perr=None, Berr=None):
        self.m: List[numeric] = m
        self.y: List[numeric] = y
        #    model: SingleDecayModel
        self.A: numeric = A if A is not None else np.max(y) - np.mean(y)
        self.Aerr: Union[numeric, NoneType] = Aerr
        self.p: numeric = p if p is not None else 0.9
        self.perr: Union[numeric, NoneType] = perr
        self.B: numeric = B if B is not None else np.mean(y)
        self.Berr: Union[numeric, NoneType] = Berr
        hist: Tuple[List[numeric], List[numeric], List[numeric]] = field(
            default_factory=lambda: (list(), list(), list())
        )
        fig: go.Figure = field(default_factory=go.Figure)

    def __post_init__(self):
        if len(self.m) == 0:
            self.m = list(range(len(self.y)))
        if len(self.y) != len(self.m):
            raise ValueError(
                "Lenght of y and m must agree. len(m)={} != len(y)={}".format(
                    len(self.m), len(self.y)
                )
            )

    def __str__(self):
        return "DecayResult: y = A p^m + B"

    def _fit(self):
        params, errs = fit_exp1B_func(self.m, self.y, p0=(self.A, self.p, self.B))
        self.A, self.p, self.B = params
        self.Aerr, self.perr, self.Berr = errs

    # can be defined by quicker by dataclass
    def _plot(self):
        self.fig = plot_decay_result(self)
        return self.fig

    def __str__(self):
        return (
            "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m + ({:.3f}\u00B1{:.3f})".format(
                self.A, self.Aerr, self.p, self.perr, self.B, self.Berr
            )
        )

    def get_tables(self):
        return ["{} | {} ".format(self.name, str(self))]

    def get_figures(self):
        return [self.fig]


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    nqubits: int
    qubits: list
    depths: list
    runs: int
    nshots: int
    noise_model: NoiseModel = field(default_factory=NoiseModel)
    noise_params: list = field(default_factory=list)


@dataclass
class StandardRBResults(Results):
    """Standard RB outputs."""

    df: DataFrame

    def save(self, path):
        self.df.to_pickle(path)


class StandardRBData:
    """Standard RB data acquisition."""

    def __init__(self, experiment: ModuleExperiment):
        self.experiment = experiment

    def save(self, path):
        self.experiment.save(path)

    def to_csv(self, path):
        self.save(path)

    def load(self, path):
        self.experiment.load(path)


def _acquisition(
    params: StandardRBParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> StandardRBData:
    factory = ModuleFactory(
        params.nqubits, params.depths * params.runs, qubits=params.qubits
    )
    experiment = ModuleExperiment(
        factory, nshots=params.nshots, noise_model=params.noise_model
    )
    experiment.perform(experiment.execute)
    post_processing_sequential(experiment)
    return StandardRBData(experiment)


def _fit(data: StandardRBData) -> StandardRBResults:
    df = get_aggregational_data(data.experiment)
    return StandardRBResults(df)


def _plot(data: StandardRBData, fit: StandardRBResults, qubit):
    """Plotting function for StandardRB."""
    return build_report(data.experiment, fit.df)


standardrb = Routine(_acquisition, _fit, _plot)
