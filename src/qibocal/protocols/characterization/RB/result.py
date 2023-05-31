from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame

from qibocal.auto.operation import Results
from qibocal.calibrations.niGSC.basics.fitting import (
    exp1_func,
    exp1B_func,
    fit_exp1_func,
    fit_exp1B_func,
)
from qibocal.protocols.characterization.RB.utils import extract_from_data

numeric = Union[int, float, complex, np.number]


@dataclass
class DecayResult(Results):
    """Data being described by a single decay, Ap^x."""

    # x and y data.
    m: Union[List[numeric], np.ndarray]
    y: Union[List[numeric], np.ndarray]
    # Fitting parameters.
    A: Optional[numeric] = field(default=None)
    Aerr: Optional[numeric] = field(default=None)
    p: Optional[numeric] = field(default=None)
    perr: Optional[numeric] = field(default=None)
    hists: Tuple[List[numeric], List[numeric]] = field(
        default_factory=lambda: (list(), list())
    )

    def __post_init__(self):
        """Do some checks if the data given is correct. If no initial fitting parameters are given,
        choose a wise guess based on the given data.
        """
        if len(self.y) != len(self.m):
            raise ValueError(
                "Lenght of y and m must agree. len(m)={} != len(y)={}".format(
                    len(self.m), len(self.y)
                )
            )
        if self.A is None:
            self.A = np.max(self.y) - np.min(self.y)
        if self.p is None:
            self.p = 0.9
        self.func = exp1_func

    @property
    def fitting_params(self):
        return (self.A, self.p)

    def fit(self, **kwargs):
        """Fits the data, all parameters given through kwargs will be passed on to the optimization function."""

        kwargs.setdefault("bounds", ((0, 0), (1, 1)))
        kwargs.setdefault("p0", (self.A, self.p))
        params, errs = fit_exp1_func(self.m, self.y, **kwargs)
        self.A, self.p = params
        self.Aerr, self.perr = errs

    def plot(self):
        """Plots the histogram data for each point and the averges plus the fit."""

        if self.hists is not None:
            self.fig = plot_hists_result(self)
        self.fig = plot_decay_result(self, self.fig)
        return self.fig

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters if there are any."""

        if self.perr is not None:
            return "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m".format(
                self.A, self.Aerr, self.p, self.perr
            )
        else:
            return "DecayResult: Ap^m"


@dataclass
class DecayWithOffsetResult(DecayResult):
    """Data being described by a single decay with offset, Ap^x + B."""

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

    def fit(self, xdata, ydata, **kwargs):
        """Fits the data, all parameters given through kwargs will be passed on to the optimization function."""

        kwargs.setdefault("bounds", ((0, 0, 0), (1, 1, 1)))
        kwargs.setdefault("p0", (self.A, self.p, self.B))
        params, errs = fit_exp1B_func(xdata, ydata, **kwargs)
        return params, errs
        # self.A, self.p, self.B = params
        # self.Aerr, self.perr, self.Berr = errs

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters if there are any."""

        if self.perr is not None:
            return "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m + ({:.3f}\u00B1{:.3f})".format(
                self.A, self.Aerr, self.p, self.perr, self.B, self.Berr
            )
        else:
            return "DecayResult: Ap^m+B"
    
    def semi_parametric_bootstrap(self, n_bootstrap=10, niter=20, sample_size=1024, **kwargs):
        def samples_to_p0(samples):
            ground = np.array([0] * len(samples[0]))
            return np.sum(np.product(samples == ground, axis=1)) / len(samples)
        
        popt, _ = self.fit(self.m, [np.mean(y) for y in self.y], **kwargs)
        bootstrap_estimates = [popt]
        for _ in range(n_bootstrap):
            fit_x = []
            fit_y = []
            for x, y in zip(self.m, self.y):
                # Non-parametric bootstrap: Resample sequences with replacement
                bootstrap_y = np.random.choice(y, size=sample_size, replace=True)

                # Parametrically sample the number of "correct" measurement results using binomial distribution
                bootstrap_y_corrected = []

                for y_prob in bootstrap_y:
                    samples_corrected = np.random.binomial(n=1, p=1-y_prob, size=(sample_size, 1))
                    bootstrap_y_corrected.append(samples_to_p0(samples_corrected))

                bootstrap_y_corrected = np.array(bootstrap_y_corrected)

                # Fit the resampled data to extract parameters
                fit_x.append(x)
                fit_y.append(np.mean(bootstrap_y_corrected))

            # Fit new values
            params, _ = self.fit(fit_x, fit_y, **kwargs)
            bootstrap_estimates.append(params)
        
        self.bootstrap_estimates = bootstrap_estimates
        perr = np.std(np.array(bootstrap_estimates), axis=0)
        self.A, self.p, self.B = popt
        self.Aerr, self.perr, self.Berr = 3 * perr
        self.y = [np.mean(y) for y in self.y]
        return popt, perr


def plot_decay_result(
    result: DecayResult, fig: Optional[go.Figure] = None
) -> go.Figure:
    """Plots the average and the fitting data from a `DecayResult`.

    Args:
        result (DecayResult): Data to plot.
        fig (Optional[go.Figure], optional): If given, traces. Defaults to None.

    Returns:
        go.Figure: Figure with at least two traces, one for the data, one for the fit.
    """

    # Initiate an empty figure if none was given.
    if fig is None:
        fig = go.Figure()
    # Plot the x and y data from the result, they are (normally) the averages.
    fig.add_trace(
        go.Scatter(
            x=result.m,
            y=result.y,
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    # Build the fit and plot the fit.
    m_fit = np.linspace(min(result.m), max(result.m), 100)
    y_fit = result.func(m_fit, *result.fitting_params)
    fig.add_trace(
        go.Scatter(x=m_fit, y=y_fit, name=str(result), line=go.scatter.Line(dash="dot"))
    )
    return fig


def plot_hists_result(result: DecayResult) -> go.Figure:
    """Plots the distribution of data around each point.

    Args:
        result (DecayResult): Where the histogramm data comes from.

    Returns:
        go.Figure: A plotly figure with one single trace with the distribution of
    """

    counts_list, bins_list = result.hists
    counts_list = sum(counts_list, [])
    fig_hist = go.Figure(
        go.Scatter(
            x=np.repeat(result.m, [len(bins) for bins in bins_list]),
            y=sum(bins_list, []),
            mode="markers",
            marker={"symbol": "square"},
            marker_color=[
                f"rgba(101, 151, 170, {count/max(counts_list)})"
                for count in counts_list
            ],
            text=[count for count in counts_list],
            hovertemplate="<br>x:%{x}<br>y:%{y}<br>count:%{text}",
            name="iterations",
        )
    )

    return fig_hist


def get_hists_data(
    data_agg: DataFrame, xlabel: str = "depth", ylabel: str = "signal"
) -> Tuple[list, list]:
    """From a dataframe extract for each point the histogram data.

    Args:
        data_agg (DataFrame): The raw data for the histogram.
        xlabel (str, optional): The label where the x data is stored in the data frame. Defaults to 'depth'.
        ylabel (str, optional): The label where the y data is stored in the data frame. Defaults to 'signal'.

    Returns:
        Tuple[list, list]: Counts and bin for each point.
    """

    signal = extract_from_data(data_agg, ylabel, xlabel)[1].reshape(
        -1, data_agg.attrs["niter"]
    )
    # Get the exact number of occurences
    counters = [Counter(np.round(x, 3)) for x in signal]
    bins_list = [list(counter_x.keys()) for counter_x in counters]
    counts_list = [list(counter_x.values()) for counter_x in counters]

    return counts_list, bins_list
