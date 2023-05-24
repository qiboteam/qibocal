from typing import List, Tuple, Union

import numpy as np
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


class DecayResult(Results):
    """
    y[i] = (A +- Aerr) (p +- perr)^m[i] + (B +- Berr)
    # for later: y= sum_i A_i p_i^m (needs m integer)
    """

    def __init__(
        self, m, y, A=None, p=None, B=None, Aerr=None, perr=None, Berr=None, hists=None
    ):
        if len(y) != len(m):
            raise ValueError(
                "Lenght of y and m must agree. len(m)={} != len(y)={}".format(
                    len(self.m), len(self.y)
                )
            )
        self.m: List[numeric] = m
        self.y: List[numeric] = y
        self.A: numeric = A if A is not None else np.max(y) - np.mean(y)
        self.Aerr: Union[numeric, NoneType] = Aerr
        self.p: numeric = p if p is not None else 0.9
        self.perr: Union[numeric, NoneType] = perr
        self.B: numeric = B if B is not None else np.mean(y)
        self.Berr: Union[numeric, NoneType] = Berr
        self.hists: Union[Tuple[List[numeric], List[numeric]], NoneType] = hists
        self.fig = None
        self.func = exp1B_func

    @property
    def fitting_params(self):
        return (self.A, self.p, self.B)

    def reset_fittingparams(
        self,
        A=None,
        p=None,
        B=None,
        Aerr=None,
        perr=None,
        Berr=None,
    ):
        self.A: numeric = A if A is not None else np.max(self.y) - np.mean(self.y)
        self.Aerr: Union[numeric, NoneType] = Aerr
        self.p: numeric = p if p is not None else 0.9
        self.perr: Union[numeric, NoneType] = perr
        self.B: numeric = B if B is not None else np.mean(self.y)
        self.Berr: Union[numeric, NoneType] = Berr

    def fit(self, **kwargs):
        kwargs.setdefault("bounds", ((0, 0, 0), (1, 1, 1)))
        kwargs.setdefault("p0", (self.A, self.p, self.B))
        params, errs = fit_exp1B_func(
            self.m, self.y, **kwargs
        )  # , bounds = (([0,0,0]),([10,1,10])))
        self.A, self.p, self.B = params
        self.Aerr, self.perr, self.Berr = errs

    def plot(self):
        if self.hists is not None:
            self.fig = plot_hists_result(self)
        self.fig = plot_decay_result(self, self.fig)
        return self.fig

    def __str__(self):
        if self.perr is not None:
            return "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m + ({:.3f}\u00B1{:.3f})".format(
                self.A, self.Aerr, self.p, self.perr, self.B, self.Berr
            )
        else:
            return "DecayResult: Ap^m+B"

    def get_tables(self):
        pass

    def get_figures(self):
        return [self.fig]


class NoOffsetDecayResult(DecayResult):
    def __init__(
        self, m, y, A=None, p=None, B=None, Aerr=None, perr=None, Berr=None, hists=None
    ):
        super().__init__(m, y, A, p, 0, Aerr, perr, None, hists)
        self.func = exp1_func

    @property
    def fitting_params(self):
        return (self.A, self.p)

    def fit(self, **kwargs):
        kwargs.setdefault("bounds", ((0, 0), (1, 1)))
        kwargs.setdefault("p0", (self.A, self.p))
        params, errs = fit_exp1_func(
            self.m, self.y, **kwargs
        )  # , bounds = (([0,0,0]),([10,1,10])))
        self.A, self.p = params
        self.Aerr, self.perr = errs

    def __str__(self):
        if self.perr is not None:
            return "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m ".format(
                self.A, self.Aerr, self.p, self.perr
            )
        else:
            return "DecayResult: Ap^m"


def plot_decay_result(
    result: DecayResult, fig: Union[go.Figure, None] = None
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
    # print(fig)
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


def get_hists_data(data_agg: DecayResult):
    signal = extract_from_data(data_agg, "signal", "depth")[1].reshape(
        -1, data_agg.attrs["niter"]
    )
    if data_agg.attrs["niter"] > 10:
        nbins = choose_bins(data_agg.attrs["niter"])
        counts_list, bins_list = zip(*[np.histogram(x, bins=nbins) for x in signal])
    else:
        counts_list, bins_list = np.ones(signal.shape), signal
    return counts_list, bins_list
