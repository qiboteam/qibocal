"""New magic-free protocol execution interface.

This module introduces a new API for defining and executing calibration protocols
without the "magic" of the legacy Executor. It provides cleaner type hints and
more explicit control over the execution flow.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from qibolab import Platform

__all__ = ["BoundProtocol", "Completed", "Protocol"]

_ParametersT = TypeVar("_ParametersT")
_FitParsT = TypeVar("_FitParsT")
_ReportParsT = TypeVar("_ReportParsT")
_DataT = TypeVar("_DataT")
_ResultsT = TypeVar("_ResultsT")


@dataclass
class Protocol(Generic[_ParametersT, _FitParsT, _ReportParsT, _DataT, _ResultsT]):
    """A calibration protocol with explicit type-safe interface.

    This protocol definition separates concerns into distinct phases:
    - acquisition: collects raw data from hardware
    - fit: processes data and produces results
    - report: generates visualizations and summaries
    - update: applies results back to the platform configuration

    All phases except acquisition are optional, allowing flexible workflow composition.

    Type Parameters:
        _ParametersT: Type of acquisition parameters
        _FitParsT: Type of fit function parameters
        _ReportParsT: Type of report function parameters
        _DataT: Type of data returned by acquisition
        _ResultsT: Type of results returned by fit
    """

    acquisition: Callable[[_ParametersT], _DataT]
    """Acquire data from hardware. Takes parameters, returns data."""

    fit: Callable[[_DataT, _FitParsT | None], _ResultsT] | None = None
    """Process data and produce results. Takes data and optional fit params."""

    report: Callable[[_DataT, _ResultsT, _ReportParsT | None], None] | None = None
    """Generate reports/visualizations. Takes data, results, and optional report params."""

    update: Callable[[_ResultsT, Platform], None] | None = None
    """Update platform with results."""

    @property
    def parameters_type(self) -> type:
        """Extract the type of acquisition parameters."""
        sig = inspect.signature(self.acquisition)
        param = next(iter(sig.parameters.values()))
        return param.annotation

    @property
    def data_type(self) -> type:
        """Extract the return type of acquisition."""
        return inspect.signature(self.acquisition).return_annotation

    @property
    def results_type(self) -> type:
        """Extract the return type of fit."""
        if self.fit is None:
            return None
        return inspect.signature(self.fit).return_annotation

    def __call__(
        self,
        pars: _ParametersT | None = None,
        fit: _FitParsT | None = None,
        report: _ReportParsT | None = None,
        **kwargs: Any,
    ) -> "BoundProtocol":
        """Bind parameters to this protocol.

        Returns a BoundProtocol that can be executed with all necessary information.

        Args:
            pars: Protocol acquisition parameters. If None, will be constructed from kwargs.
            fit: Optional parameters for the fit phase.
            report: Optional parameters for the report phase.
            **kwargs: Keyword arguments used to construct parameters if pars is None.

        Returns:
            A BoundProtocol ready for execution.
        """
        if pars is None:
            pars = self.parameters_type(**kwargs)
        return BoundProtocol(
            protocol=self, parameters=pars, fitpars=fit, reportpars=report
        )


@dataclass
class BoundProtocol(Generic[_ParametersT, _FitParsT, _ReportParsT, _DataT, _ResultsT]):
    """A protocol bound with specific parameters and execution configuration.

    This represents a specific protocol execution with all required parameters
    and optional phase parameters already determined.
    """

    protocol: Protocol[_ParametersT, _FitParsT, _ReportParsT, _DataT, _ResultsT]
    """The protocol being bound."""

    parameters: _ParametersT
    """Acquisition parameters."""

    fitpars: _FitParsT | None = None
    """Optional fit phase parameters."""

    reportpars: _ReportParsT | None = None
    """Optional report phase parameters."""


@dataclass
class Completed:
    """Result of a protocol execution.

    Stores the outcomes and metadata of a complete or partial protocol execution.
    """

    data: Any = None
    """Data acquired during execution."""

    results: Any = None
    """Results produced by fitting."""

    protocol_id: str | None = None
    """Identifier for the protocol that was executed."""

    error: Exception | None = None
    """Error that occurred during execution, if any."""

    success: bool = True
    """Whether execution completed successfully."""
