from collections import Counter
from dataclasses import dataclass, field
from numbers import Number
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from qibocal.auto.operation import Results
from qibocal.calibrations.niGSC.basics.fitting import (
    exp1_func,
    exp1B_func,
    fit_exp1_func,
    fit_exp1B_func,
)


@dataclass
class DecayResult(Results):
    """Data being described by a single decay, Ap^x."""

    # x and y data.
    x: Union[List[Number], np.ndarray]
    y: Union[List[Number], np.ndarray]
    # Fitting parameters.
    A: Optional[Number] = None
    Aerr: Optional[Number] = None
    p: Optional[Number] = None
    perr: Optional[Number] = None
    func: Iterable = field(default=exp1_func)

    def __post_init__(self):
        """Do some checks if the data given is correct. If no initial fitting parameters are given,
        choose a wise guess based on the given data.
        """
        if len(self.y) != len(self.x):
            raise ValueError(
                "Lenght of y and x must agree. len(x)={} != len(y)={}".format(
                    len(self.x), len(self.y)
                )
            )
        self.y_scatter = None
        if isinstance(self.y[0], Iterable):
            self.y_scatter = self.y
            self.y = [np.mean(y_row) for y_row in self.y]
        if self.A is None:
            self.A = np.max(self.y) - np.min(self.y)
        if self.p is None:
            self.p = 0.9

    @property
    def fitting_params(self):
        return (self.A, self.p)

    @fitting_params.setter
    def fitting_params(self, value):
        self.A, self.p, self.B = value

    @property
    def fitting_errors(self):
        return (self.Aerr, self.perr)

    @fitting_errors.setter
    def fitting_errors(self, value):
        self.Aerr, self.perr = value

    def fit_func(self, x, y, **kwargs):
        return fit_exp1_func(x, y, **kwargs)

    def fit(self, n_bootstrap=0, sample_size=1, **kwargs):
        """Fits the data, all parameters given through kwargs will be passed on to the optimization function."""

        def samples_to_p0(samples):
            ground = np.array([0] * len(samples[0]))
            return np.sum(np.product(samples == ground, axis=1)) / len(samples)

        # Perform fit on the initial data
        kwargs.setdefault("bounds", ((0, 0, 0), (1, 1, 1)))
        kwargs.setdefault("p0", (self.A, self.p, self.B))
        kwargs.setdefault("sigma", np.std(np.array(self.y_scatter), axis=1))
        popt, pcov = self.fit_func(self.x, self.y, **kwargs)
        self.fitting_params = popt

        if n_bootstrap < 1:
            self.fitting_errors = pcov
            error_y = np.std(self.y_scatter, axis=1)
            self.error_y = error_y  # np.vstack([error_y, error_y])
            return

        # Semi-parametric bootstrap resampling
        bootstrap_estimates = [popt]
        y_estimates = [self.y]
        for _ in range(n_bootstrap):
            fit_y = []
            bootstrap_y_scatter = []
            for y in self.y_scatter:
                # Non-parametric bootstrap: Resample sequences with replacement
                bootstrap_y = np.random.choice(y, size=len(y), replace=True)

                # Parametrically sample the number of "correct" shots with binomial distribution
                bootstrap_y_scatter.append([])
                for y_prob in bootstrap_y:
                    samples_corrected = np.random.binomial(
                        n=1, p=1 - y_prob, size=(sample_size, 1)
                    )
                    bootstrap_y_scatter[-1].append(samples_to_p0(samples_corrected))
                fit_y.append(np.mean(bootstrap_y_scatter[-1]))

            # Fit the resampled data to extract parameters
            kwargs["sigma"] = np.std(bootstrap_y_scatter, axis=1)
            params, _ = self.fit_func(self.x, fit_y, **kwargs)
            bootstrap_estimates.append(params)
            y_estimates.append(fit_y)

        # 95% confidence intervals for the error bars
        confidence = 0.95
        percentiles = [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)]
        self.error_y = np.abs(
            np.vstack([self.y, self.y])
            - np.percentile(self.y_scatter, percentiles, axis=1)
        )

        # 95% confidence intervals for the fitting parameters
        self.fitting_errors = np.abs(
            np.vstack([popt, popt])
            - np.percentile(bootstrap_estimates, percentiles, axis=0)
        ).T

    def plot(self):
        """Plots the histogram data for each point and the averges plus the fit."""

        if self.y_scatter is not None:
            self.fig = plot_hists_result(self)
        self.fig = plot_decay_result(self, self.fig)
        return self.fig

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters if there are any."""

        if self.perr is not None:
            return "Fit: y=Ap^x<br>A: {}<br>p: {}".format(
                ci_to_str(self.A, self.Aerr), ci_to_str(self.p, self.perr)
            )
        else:
            return "DecayResult: Ap^x"


@dataclass
class DecayWithOffsetResult(DecayResult):
    """Data being described by a single decay with offset, Ap^x + B."""

    B: Optional[Number] = None
    Berr: Optional[Number] = None
    func: Iterable = field(default=exp1B_func)

    def __post_init__(self):
        super().__post_init__()
        if self.B is None:
            self.B = np.mean(np.array(self.y))

    @property
    def fitting_params(self):
        return (*super().fitting_params, self.B)

    @fitting_params.setter
    def fitting_params(self, value):
        self.A, self.p, self.B = value

    @property
    def fitting_errors(self):
        return (*super().fitting_errors, self.Berr)

    @fitting_errors.setter
    def fitting_errors(self, value):
        self.Aerr, self.perr, self.Berr = value

    def fit_func(self, x, y, **kwargs):
        return fit_exp1B_func(x, y, **kwargs)

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters if there are any."""

        if self.perr is not None:
            return "Fit: y=Ap^x+B<br>A: {}<br>p: {}<br>B: {}".format(
                ci_to_str(self.A, self.Aerr),
                ci_to_str(self.p, self.perr),
                ci_to_str(self.B, self.Berr),
            )
        else:
            return "DecayResult: Ap^x+B"


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
    # If result.error_y is given, create a dictionary for the error bars
    error_y_dict = None
    if isinstance(result.error_y, numeric):
        error_y_dict = dict(
            type="constant",
            value=result.error_y,
        )
    elif isinstance(result.error_y[0], Iterable) is True:
        error_y_dict = dict(
            type="data",
            symmetric=False,
            array=result.error_y[1],
            arrayminus=result.error_y[0],
        )
    else:
        error_y_dict = dict(
            type="data",
            array=result.error_y,
        )
    # Plot the x and y data from the result, they are (normally) the averages.
    fig.add_trace(
        go.Scatter(
            x=result.x,
            y=result.y,
            error_y=error_y_dict,
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    # Build the fit and plot the fit.
    x_fit = np.linspace(min(result.x), max(result.x), 100)
    y_fit = result.func(x_fit, *result.fitting_params)
    fig.add_trace(
        go.Scatter(x=x_fit, y=y_fit, name=str(result), line=go.scatter.Line(dash="dot"))
    )
    return fig


def plot_hists_result(result: DecayResult) -> go.Figure:
    """Plots the distribution of data around each point.

    Args:
        result (DecayResult): Where the histogramm data comes from.

    Returns:
        go.Figure: A plotly figure with one single trace with the distribution of
    """
    counts_list, bins_list = get_hists_data(result.y_scatter)
    counts_list = sum(counts_list, [])
    fig_hist = go.Figure(
        go.Scatter(
            x=np.repeat(result.x, [len(bins) for bins in bins_list]),
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


def get_hists_data(signal: Union[List[Number], np.ndarray]) -> Tuple[list, list]:
    """From a dataframe extract for each point the histogram data.

    Args:
        data_agg (DataFrame): The raw data for the histogram.
        xlabel (str, optional): The label where the x data is stored in the data frame. Defaults to 'depth'.
        ylabel (str, optional): The label where the y data is stored in the data frame. Defaults to 'signal'.

    Returns:
        Tuple[list, list]: Counts and bin for each point.
    """

    # Get the exact number of occurences
    counters = [Counter(np.round(x, 3)) for x in signal]
    bins_list = [list(counter_x.keys()) for counter_x in counters]
    counts_list = [list(counter_x.values()) for counter_x in counters]

    return counts_list, bins_list
