from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from qibocal.auto.operation import Results
from qibocal.calibrations.niGSC.basics.fitting import (
    exp1_func,
    exp1B_func,
    fit_exp1_func,
    fit_exp1B_func,
)
from qibocal.protocols.characterization.RB.utils import extract_from_data

real_numeric = Union[int, float, np.number]
numeric = Union[int, float, complex, np.number]
NoneType = type(None)


@dataclass
class DecayResult(Results):
    """ """

    m: Union[List[numeric], np.ndarray]
    y: Union[List[numeric], np.ndarray]
    A: Optional[numeric] = field(default=None)
    Aerr: Optional[numeric] = field(default=None)
    p: Optional[numeric] = field(default=None)
    perr: Optional[numeric] = field(default=None)
    hists: Tuple[List[numeric], List[numeric]] = field(
        default_factory=lambda: (list(), list())
    )

    def __post_init__(self):
        if len(self.y) != len(self.m):
            raise ValueError(
                "Lenght of y and m must agree. len(m)={} != len(y)={}".format(
                    len(self.m), len(self.y)
                )
            )
        if self.A is None:
            self.A = np.max(self.y) - np.mean(self.y)
        if self.p is None:
            self.p = 0.9
        self.func = exp1_func

    @property
    def fitting_params(self):
        return (self.A, self.p)

    def reset_fitting_params(self, A=None, p=None):
        self.A: Optional[numeric] = (
            A if A is not None else np.max(self.y) - np.mean(self.y)
        )
        self.p: Optional[numeric] = p if p is not None else 0.9

    def fit(self, **kwargs):
        kwargs.setdefault("bounds", ((0, 0), (1, 1)))
        kwargs.setdefault("p0", (self.A, self.p))
        params, errs = fit_exp1_func(self.m, self.y, **kwargs)
        self.A, self.p = params
        self.Aerr, self.perr = errs

    def plot(self):
        if self.hists is not None:
            self.fig = plot_hists_result(self)
        self.fig = plot_decay_result(self, self.fig)
        return self.fig

    def __str__(self):
        if self.perr is not None:
            return "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m".format(
                self.A, self.Aerr, self.p, self.perr
            )
        else:
            return "DecayResult: Ap^m"

    def get_tables(self):
        pass

    def get_figures(self):
        return [self.fig]


@dataclass
class DecayWithOffsetResult(DecayResult):
    """
    y[i] = (A +- Aerr) (p +- perr)^m[i] + (B +- Berr)
    # for later: y= sum_i A_i p_i^m (needs m integer)
    """

    B: Optional[numeric] = field(default=None)
    Berr: Optional[numeric] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.B is None:
            self.B = np.mean(np.array(self.y))
        self.func = exp1B_func

    @property
    def fitting_params(self):
        return (*super().fitting_params, self.B)

    def reset_fitting_params(self, A=None, p=None, B=None):
        super().reset_fitting_params(A, p)
        self.B: Optional[numeric] = B if B is not None else np.mean(self.y)

    def fit(self, **kwargs):
        kwargs.setdefault("bounds", ((0, 0, 0), (1, 1, 1)))
        kwargs.setdefault("p0", (self.A, self.p, self.B))
        params, errs = fit_exp1B_func(self.m, self.y, **kwargs)
        self.A, self.p, self.B = params
        self.Aerr, self.perr, self.Berr = errs

    def __str__(self):
        if self.perr is not None:
            return "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m + ({:.3f}\u00B1{:.3f})".format(
                self.A, self.Aerr, self.p, self.perr, self.B, self.Berr
            )
        else:
            return "DecayResult: Ap^m+B"


def plot_decay_result(
    result: DecayResult, fig: Optional[go.Figure] = None
) -> go.Figure:
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.m,
            y=result.y,
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    m_fit = np.linspace(min(result.m), max(result.m), 100)
    y_fit = result.func(m_fit, *result.fitting_params)
    fig.add_trace(
        go.Scatter(x=m_fit, y=y_fit, name=str(result), line=go.scatter.Line(dash="dot"))
    )
    return fig


def plot_hists_result(result: DecayResult) -> go.Figure:
    count_array, bins_array = np.array(result.hists[0]), np.array(result.hists[1])
    if bins_array.shape[1] - count_array.shape[1]:
        bins_array = (
            bins_array[::, :-1]
            + (np.array(bins_array[::, 1] - bins_array[::, 0]).reshape(-1, 1)) / 2
        )
    fig_hist = px.scatter(
        x=np.repeat(result.m, bins_array.shape[-1]),
        y=bins_array.flatten(),
        color=count_array.flatten() if not np.all(count_array == 1) else None,
        color_continuous_scale=px.colors.sequential.Tealgrn,
    )
    fig_hist.update_traces(marker=dict(symbol="square"))
    fig_hist.update_layout(
        coloraxis_colorbar_x=-0.15, coloraxis_colorbar_title_text="count"
    )
    return fig_hist


def choose_bins(niter):
    if niter <= 10:
        return niter
    else:
        return int(np.log10(niter) * 10)


def get_hists_data(data_agg: pd.DataFrame):
    signal = extract_from_data(data_agg, "signal", "depth")[1].reshape(
        -1, data_agg.attrs["niter"]
    )
    print(data_agg.attrs["niter"])
    if data_agg.attrs["niter"] > 10:
        nbins = choose_bins(data_agg.attrs["niter"])
        counts_list, bins_list = zip(*[np.histogram(x, bins=nbins) for x in signal])
    else:
        counts_list, bins_list = np.ones(signal.shape), signal
    return counts_list, bins_list
