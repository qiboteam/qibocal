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
    samples_to_p0,
)


@dataclass
class DecayResult(Results):
    """Data being described by a single decay, Ap^x."""

    # x and y data.
    x: Union[List[Number], np.ndarray]
    y: Union[List[Number], np.ndarray]
    # Fitting parameters and errors.
    A: Optional[Number] = None
    Aerr: Optional[Number] = None
    p: Optional[Number] = None
    perr: Optional[Number] = None
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

    def fit(
        self,
        uncertainties=None,
        n_bootstrap: int = 0,
        sample_size: int = 1,
        samples_to_y=None,
        **kwargs,
    ):
        """Fits the data and performs bootstrap resampling.

        Args:
            uncertainties: method of computing the uncertainties if ``sigma`` in ``kwargs``
                is not given. If ``std``, computes the standard deviation. If type ``float`` between
                0 and 1, computes the corresponding confidence interval. If ``None``, does not
                compute the uncertainties. Defaults to ``None``.
            n_bootstrap (int): number of bootstrap iterations.
            sample_size (int): number of "corrected" samples from the binomial distribution.
            samples_to_y (callable): function that transforms samples to y data. If ``None``,
                computes the probability of 0 (see
                :func:`qibocal.protocols.characterization.randomized_benchmarking.utils.samples_to_p0`).
                Defaults to ``None``.
            kwargs: parameters passed on to the optimization function.
        """

        # Update kwargs for the optimization function
        init_kwargs = deepcopy(kwargs)
        init_kwargs.setdefault("bounds", ((0, 0, 0), (1, 1, 1)))
        init_kwargs.setdefault("p0", (self.A, self.p, self.B))
        init_sigma = init_kwargs.pop("sigma", None)

        # Perform bootstrap resampling
        y_estimates, popt_estimates = self.semiparametric_bootstrap(
            uncertainties, n_bootstrap, sample_size, samples_to_y, **init_kwargs
        )
        if len(y_estimates) == 0:
            y_estimates = self.y_scatter

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

    def semiparametric_bootstrap(
        self,
        uncertainties=None,
        n_bootstrap: int = 0,
        sample_size: int = 1,
        samples_to_y=None,
        **kwargs,
    ) -> Tuple[list, list]:
        """Semiparametric bootstrap resampling.
        All parameters given through kwargs will be passed on to the optimization function.

        Args:
            uncertainties: method of computing the uncertainties if ``sigma`` in ``kwargs``
                is not given. If ``std``, computes the standard deviation. If type ``float`` between
                0 and 1, computes the corresponding confidence interval. If ``None``, does not
                compute the uncertainties. Defaults to ``None``.
            n_bootstrap (int): number of bootstrap iterations.
            sample_size (int): number of "corrected" samples from the binomial distribution.
            samples_to_y (callable): function that transforms samples to y data. If ``None``,
                computes the probability of 0 (see
                :func:`qibocal.protocols.characterization.randomized_benchmarking.utils.samples_to_p0`).
                Defaults to ``None``.

        Returns:
            Tuple[list, list]: y data estimates and fitting parameters estimates.
        """

        if isinstance(n_bootstrap, int) is False:
            raise_error(
                TypeError,
                f"`n_bootstrap` must be of type int. Got {type(n_bootstrap)} instead.",
            )
        if isinstance(sample_size, int) is False:
            raise_error(
                TypeError,
                f"`sample_size` must be of type int. Got {type(sample_size)} instead.",
            )
        if n_bootstrap < 0:
            raise_error(
                ValueError, f"`n_bootstrap` cannot be negative. Got {n_bootstrap}."
            )
        if sample_size < 0:
            raise_error(
                ValueError, f"`sample_size` cannot be negative. Got {sample_size}."
            )
        if samples_to_y is not None and callable(samples_to_y) is False:
            raise_error(
                TypeError,
                f"`samples_to_y must be callable. Got {type(samples_to_y)} instead.",
            )

        if samples_to_y is None:
            samples_to_y = samples_to_p0

        popt_estimates = []
        y_estimates = []

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
                    bootstrap_y_scatter[-1].append(samples_to_y([samples_corrected])[0])
                fit_y.append(np.mean(bootstrap_y_scatter[-1]))

            # Fit the resampled data to get parameters estimates
            bootstrap_sigma = data_mean_errors(
                bootstrap_y_scatter, uncertainties, symmetric=True
            )
            popt, _ = self.fit_func(self.x, fit_y, sigma=bootstrap_sigma, **kwargs)
            popt_estimates.append(popt)
            y_estimates.append(fit_y)

        return np.array(y_estimates).T, np.array(popt_estimates).T

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
                number_to_str(self.A, self.Aerr), number_to_str(self.p, self.perr)
            )
        return "DecayResult: Ap^x"


@dataclass
class DecayWithOffsetResult(DecayResult):
    """Data being described by a single decay with offset, Ap^x + B."""

    # Fitting parameters and errors
    B: Optional[Number] = None
    Berr: Optional[Number] = None
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
        return (*super().fitting_errors, self.Berr)

    @fitting_errors.setter
    def fitting_errors(self, value):
        self.Aerr, self.perr, self.Berr = value

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters
        if there are any.
        """

        if all(param is not None for param in self.fitting_params):
            return "Fit: y=Ap^x+B<br>A: {}<br>p: {}<br>B: {}".format(
                number_to_str(self.A, self.Aerr),
                number_to_str(self.p, self.perr),
                number_to_str(self.B, self.Berr),
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
