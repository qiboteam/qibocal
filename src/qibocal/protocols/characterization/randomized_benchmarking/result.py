from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Number
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from qibocal.auto.operation import Results
from qibocal.config import raise_error
from qibocal.protocols.characterization.randomized_benchmarking.fitting import (
    exp1_func,
    exp1B_func,
    fit_exp1_func,
    fit_exp1B_func,
)
from qibocal.protocols.characterization.randomized_benchmarking.utils import (
    data_mean_errors,
    number_to_str,
)


@dataclass
class DecayResult(Results):
    """Data being described by a single decay, Ap^x."""

    # x and y data.
    x: Union[List[Number], np.ndarray]
    y: Union[List[Number], np.ndarray]
    # Fitting parameters and errors.
    A: Optional[Number] = None
    A_err: Optional[Number] = None
    p: Optional[Number] = None
    p_err: Optional[Number] = None
    # Model and the fitting function
    model: Iterable = field(default=exp1_func)
    fit_func: Iterable = field(default=fit_exp1_func)
    meta_data: Dict = field(default_factory=dict)
    """The result data should behave according to that to that model."""

    def __post_init__(self):
        """Do some checks if the data given is correct. If no initial fitting parameters are given,
        choose a wise guess based on the given data.
        """
        if len(self.y) != len(self.x):
            raise_error(
                ValueError(
                    f"Length of y and x must agree. len(x)={len(self.x)} != len(y)={len(self.y)}"
                )
            )
        self.y_scatter = None
        self.error_y = None
        self.fig = None
        if isinstance(self.y[0], Iterable):
            self.y_scatter = self.y
            self.y = [np.mean(y_row) for y_row in self.y]
        if self.A is None:
            self.A = np.max(self.y) - np.min(self.y)
        if self.p is None:
            self.p = 0.9
        self.fig = None
        self.resample_func = None

    @property
    def fitting_params(self):
        return (self.A, self.p)

    @fitting_params.setter
    def fitting_params(self, value):
        self.A, self.p, self.B = value

    @property
    def fitting_errors(self):
        return (self.A_err, self.p_err)

    @fitting_errors.setter
    def fitting_errors(self, value):
        self.A_err, self.p_err = value

    def fit(self, **kwargs):
        """Fits the data and performs bootstrap resampling, all parameters given through `kwargs`
        will be passed on to the optimization function.
        """

        # Update kwargs for the optimization function
        init_kwargs = deepcopy(kwargs)
        init_kwargs.setdefault("bounds", ((0, 0, 0), (1, 1, 1)))
        init_kwargs.setdefault("p0", (self.A, self.p, self.B))
        init_sigma = init_kwargs.pop("sigma", None)

        # Extract bootstrap parameters if given
        uncertainties = self.meta_data.get("uncertainties", None)
        n_bootstrap = self.meta_data.get("n_bootstrap", 0)

        # Perform bootstrap resampling
        y_estimates, popt_estimates = bootstrap(
            self.x,
            self.y_scatter,
            uncertainties,
            n_bootstrap,
            self.resample_func,
            self.fit_func,
        )

        # Fit the initial data
        sigma = (
            data_mean_errors(y_estimates, uncertainties, symmetric=True)
            if init_sigma is None
            else init_sigma
        )
        popt, pcov = self.fit_func(self.x, self.y, sigma=sigma, **init_kwargs)
        self.fitting_params = popt

        # Compute fitting errors
        if len(popt_estimates) == 0:
            self.fitting_errors = pcov
        else:
            fitting_errors = data_mean_errors(popt_estimates, uncertainties)
            self.fitting_errors = (
                fitting_errors.T
                if fitting_errors is not None
                else [0] * len(self.fitting_errors)
            )

        # Compute y data errors
        self.error_y = data_mean_errors(y_estimates, uncertainties, symmetric=False)

    def plot(self):
        """Plots the histogram data for each point and the averges plus the fit."""
        self.fig = plot_decay_result(self)
        return self.fig

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters
        if there are any.
        """

        if all(param is not None for param in self.fitting_params):
            return "Fit: y=Ap^x<br>A: {}<br>p: {}".format(
                number_to_str(self.A, self.A_err), number_to_str(self.p, self.p_err)
            )
        return "DecayResult: Ap^x"


@dataclass
class DecayWithOffsetResult(DecayResult):
    """Data being described by a single decay with offset, Ap^x + B."""

    # Fitting parameters and errors
    B: Optional[Number] = None
    B_err: Optional[Number] = None
    # Model and the fitting function
    model: Iterable = field(default=exp1B_func)
    fit_func: Iterable = field(default=fit_exp1B_func)
    """The result data should behave according to that to that model."""

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
        return (*super().fitting_errors, self.B_err)

    @fitting_errors.setter
    def fitting_errors(self, value):
        self.A_err, self.p_err, self.B_err = value

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters
        if there are any.
        """

        if all(param is not None for param in self.fitting_params):
            return "Fit: y=Ap^x+B<br>A: {}<br>p: {}<br>B: {}".format(
                number_to_str(self.A, self.A_err),
                number_to_str(self.p, self.p_err),
                number_to_str(self.B, self.B_err),
            )
        return "DecayResult: Ap^x+B"


def plot_decay_result(result: DecayResult) -> go.Figure:
    """Plots the average and the fitting data from a `DecayResult`.

    Args:
        result (DecayResult): Data to plot.
        fig (Optional[go.Figure], optional): If given, traces. Defaults to None.

    Returns:
        go.Figure: Figure with at least two traces, one for the data, one for the fit.
    """

    if result.y_scatter is not None:
        fig = plot_hists_result(result)
    else:
        fig = go.Figure()

    # Plot the x and y data from the result, they are (normally) the averages.
    fig.add_trace(
        go.Scatter(
            x=result.x,
            y=result.y,
            line={"color": "#aa6464"},
            mode="markers",
            name="average",
        )
    )
    # If result.error_y is given, create a dictionary for the error bars
    if hasattr(result, "error_y") and result.error_y is not None:
        if isinstance(result.error_y, Number):
            error_y_dict = {"type": "constant", "value": result.error_y}
        elif isinstance(result.error_y[0], Iterable) is False:
            error_y_dict = {"type": "data", "array": result.error_y}
        else:
            error_y_dict = {
                "type": "data",
                "symmetric": False,
                "array": result.error_y[1],
                "arrayminus": result.error_y[0],
            }
        fig.add_trace(
            go.Scatter(
                x=result.x,
                y=result.y,
                error_y=error_y_dict,
                line={"color": "#aa6464"},
                mode="markers",
                name="error bars",
            )
        )
    # Build the fit and plot the fit.
    x_fit = np.linspace(min(result.x), max(result.x), 100)
    y_fit = result.model(x_fit, *result.fitting_params)
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name=str(result),
            line=go.scatter.Line(dash="dot", color="#00cc96"),
        )
    )
    return fig


def bootstrap(
    x_data,
    y_data,
    uncertainties: Union[str, float] = None,
    n_bootstrap: int = 0,
    resample_func=None,
    fit_func=fit_exp1B_func,
    **kwargs,
):
    """Semiparametric bootstrap resampling.

    Args:
        x_data (list or np.ndarray): 1d array of x values.
        y_data (list or np.ndarray): 2d array with rows containing data points
            from which the mean values for y are computed.
        uncertainties (str or float, optional): method of computing ``sigma`` for the ``fit_func``.
            If ``std``, computes the standard deviation. If type ``float`` between 0 and 1,
            computes the maximum of low and high errors from the corresponding confidence interval.
            If ``None``, does not compute the uncertainties. Defaults to ``None``.
        n_bootstrap (int): number of bootstrap iterations. If `0`,
            returns `y_data` for y estimates and an empty list for fitting parameters estimates.
        resample_func (callable): function that preforms resampling of given a list of y values.
            (see :func:`qibocal.protocols.characterization.randomized_benchmarking.standard_rb.resample_p0`)
            If ``None``, only non-parametric resampling is performed. Defaults to ``None``.
        fit_func (callable): fitting function that returns parameters and errors, given
            `x`, `y`, `sigma` and `**kwargs`.
            Defaults to :func:`qibocal.protocols.characterization.randomized_benchmarking.fitting.fit_exp1B_func`.

    Returns:
        Tuple[list, list]: y data estimates and fitting parameters estimates.
    """

    if isinstance(n_bootstrap, int) is False:
        raise_error(
            TypeError,
            f"`n_bootstrap` must be of type int. Got {type(n_bootstrap)} instead.",
        )
    if n_bootstrap < 0:
        raise_error(ValueError, f"`n_bootstrap` cannot be negative. Got {n_bootstrap}.")
    if n_bootstrap == 0:
        return y_data, []

    if resample_func is not None and callable(resample_func) is False:
        raise_error(
            TypeError,
            f"`resample_func must be callable. Got {type(resample_func)} instead.",
        )
    if resample_func is None:
        resample_func = lambda data: data

    popt_estimates = []
    y_estimates = []

    for _ in range(n_bootstrap):
        bootstrap_y_scatter = []
        for y in y_data:
            # Non-parametric bootstrap: resample data points with replacement
            bootstrap_y = np.random.choice(y, size=len(y), replace=True)

            # Parametrically resample the new data
            bootstrap_y_scatter.append(resample_func(bootstrap_y))
        fit_y = np.mean(bootstrap_y_scatter, axis=1)

        # Fit the resampled data to get parameters estimates
        bootstrap_sigma = data_mean_errors(
            bootstrap_y_scatter, uncertainties, symmetric=True
        )
        popt, _ = fit_func(x_data, fit_y, sigma=bootstrap_sigma, **kwargs)
        popt_estimates.append(popt)
        y_estimates.append(fit_y)

    return np.array(y_estimates).T, np.array(popt_estimates).T


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
            text=counts_list,
            hovertemplate="<br>x:%{x}<br>y:%{y}<br>count:%{text}",
            name="iterations",
        )
    )

    return fig_hist


def get_hists_data(signal: Union[List[Number], np.ndarray]) -> Tuple[list, list]:
    """From a dataframe extract for each point the histogram data.

    Args:
        signal (list or np.ndarray): The raw data for the histogram.

    Returns:
        Tuple[list, list]: Counts and bins for each point.
    """

    # Get the exact number of occurences
    counters = [Counter(np.round(x, 3)) for x in signal]
    bins_list = [list(counter_x.keys()) for counter_x in counters]
    counts_list = [list(counter_x.values()) for counter_x in counters]

    return counts_list, bins_list
