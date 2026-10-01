from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Any, Generic, NewType, TypeVar

import numpy as np
import numpy.typing as npt
from qibolab import Platform, Qubit

from qibocal.calibration.calibration import QubitId, QubitPairId
from qibocal.config import log

from .serialize import deserialize, load, serialize

__all__ = ["BoundProtocol", "Completed", "Protocol", "ProtocolsCollection"]

OperationId = NewType("OperationId", str)
"""Identifier for a calibration routine."""
Qubits = dict[QubitId, Qubit]
"""Convenient way of passing qubit pairs in the routines."""


DATAFILE = "data"
"""Name of the file where data are dumped."""
RESULTSFILE = "results"
"""Name of the file where results are dumped."""


def show_logs(func):
    """Decorator to add logs."""

    @wraps(func)
    # necessary to maintain the function signature
    def wrapper(*args, **kwds):
        start = time.perf_counter()
        out = func(*args, **kwds)
        end = time.perf_counter()
        if end - start < 1:
            message = " in less than 1 second."
        else:
            message = f" in {end - start:.2f} seconds"
        log.info(f"Finished {func.__name__[1:]}" + message)
        return out, end - start

    return wrapper


DEFAULT_PARENT_PARAMETERS = {
    "nshots": None,
    "relaxation_time": None,
}
"""Default values of the parameters of `Parameters`"""


class Parameters:
    """Generic action parameters.

    Implement parameters as Algebraic Data Types (similar to), by
    subclassing this marker in actual parameters specification for each
    calibration routine.

    The actual parameters structure is only used inside the routines
    themselves.
    """

    nshots: int
    """Number of executions on hardware."""
    relaxation_time: float
    """Wait time for the qubit to decohere back to the ground state."""

    @classmethod
    def load(cls, input_parameters):
        """Load parameters from runcard.

        Possibly looking into previous steps outputs.
        Parameters defined in Parameters class are removed from `parameters`
        before `cls` is created.
        Then `nshots` and `relaxation_time` are assigned to cls.

        .. todo::

            move the implementation to History, since it is required to resolve
            the linked outputs
        """
        default_parent_parameters = deepcopy(DEFAULT_PARENT_PARAMETERS)
        parameters = deepcopy(input_parameters)
        for parameter, value in default_parent_parameters.items():
            default_parent_parameters[parameter] = parameters.pop(parameter, value)
        instantiated_class = cls(**parameters)
        for parameter, value in default_parent_parameters.items():
            setattr(instantiated_class, parameter, value)
        return instantiated_class


class AbstractData:
    """Abstract data class."""

    def __init__(
        self, data: dict[tuple[QubitId, int] | QubitId, npt.NDArray] | None = None
    ):
        self.data = data if data is not None else {}

    def __getitem__(self, qubit: QubitId | tuple[QubitId, int]):
        """Access data attribute member."""
        return self.data[qubit]

    @property
    def params(self) -> dict:
        """Convert non-arrays attributes into dict."""
        global_dict = asdict(self)
        if hasattr(self, "data"):
            global_dict.pop("data")
        return global_dict

    def save(self, path: Path, filename: str):
        """Dump class to file."""
        self._to_json(path, filename)
        self._to_npz(path, filename)

    def _to_npz(self, path: Path, filename: str):
        """Helper function to use np.savez while converting keys into
        strings."""
        if hasattr(self, "data"):
            np.savez(
                path / f"{filename}.npz",
                **{json.dumps(i): self.data[i] for i in self.data},
            )

    def _to_json(self, path: Path, filename: str):
        """Helper function to dump to json."""
        if self.params:
            (path / f"{filename}.json").write_text(
                json.dumps(serialize(self.params), indent=4), encoding="utf-8"
            )

    @classmethod
    def load(cls, path: Path, filename: str):
        """Generic load method."""
        data_dict = cls.load_data(path, filename)
        params = cls.load_params(path, filename)
        if data_dict is not None:
            if params is not None:
                return cls(data=data_dict, **params)
            else:
                return cls(data=data_dict)
        elif params is not None:
            return cls(**params)

    @staticmethod
    def load_data(path: Path, filename: str):
        """Load data stored in a npz file."""
        file = path / f"{filename}.npz"
        if file.is_file():
            raw_data_dict = dict(np.load(file))
            data_dict = {}

            for data_key, array in raw_data_dict.items():
                data_dict[load(data_key)] = np.rec.array(array)

            return data_dict

    @staticmethod
    def load_params(path: Path, filename: str):
        """Load parameters stored in a json file."""
        file = path / f"{filename}.json"
        if file.is_file():
            params = json.loads(file.read_text())
            params = deserialize(params)
            return params


class Data(AbstractData):
    """Data resulting from acquisition routine."""

    @property
    def qubits(self) -> list[QubitId]:
        """Access qubits from data structure."""
        # TODO: In the two-qubit case, a set of the first elements of the tuples is
        # returned. This behaviour is not reflected in the name of the property so may
        # lead to confusion and should therefore be changed.
        if set(map(type, self.data)) == {tuple}:
            return list({q[0] for q in self.data})
        return [q for q in self.data]

    @property
    def pairs(self) -> list[QubitPairId]:
        """Access qubit pairs from data structure."""
        return list({tuple(q[:2]) for q in self.data})

    def register_qubit(self, dtype, data_keys, data_dict):
        """Store output for single qubit.

        Args:
            data_keys (tuple): Keys of Data.data.
            data_dict (dict): The keys are the fields of `dtype` and
            the values are the related arrays.
        """
        # to be able to handle the non-sweeper case
        ar = np.empty(np.shape(data_dict[next(iter(data_dict))]), dtype=dtype)
        for key, value in data_dict.items():
            ar[key] = value

        if data_keys in self.data:
            self.data[data_keys] = np.rec.array(
                np.concatenate((self.data[data_keys], ar))
            )
        else:
            self.data[data_keys] = np.rec.array(ar)

    def save(self, path: Path, filename: str = DATAFILE):
        """Store data to file."""
        super().save(path, filename)

    @classmethod
    def load(cls, path: Path, filename: str = DATAFILE):
        """Load data and parameters."""
        return super().load(path, filename)


@dataclass
class Results(AbstractData):
    """Generic runcard update."""

    def __contains__(self, key: QubitId | QubitPairId | tuple[QubitId, ...]) -> bool:
        """Checking if qubit is in Results.

        If key is not present means that fitting failed or was not
        performed.
        """
        return all(
            key in getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), dict)
        )

    @classmethod
    def load(cls, path: Path, filename: str = RESULTSFILE):
        """Load results."""
        return super().load(path, filename)

    def save(self, path: Path, filename: str = RESULTSFILE):
        """Store results to file."""
        super().save(path, filename)


# Type variables for generic Protocol
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

    acquisition: Callable[..., _DataT]
    """Acquire data with parameters and optional platform/targets."""

    fit: Callable[..., _ResultsT] | None = None
    """Process data and produce results. Takes data and optional fit params."""

    report: Callable[..., Any] | None = None
    """Generate reports/visualizations and return their output."""

    update: Callable[..., None] | None = None
    """Update platform with results."""

    two_qubit_gates: bool | None = False
    """Flag to determine whether to allocate list of Qubits or Pairs."""

    @property
    def parameters_type(self) -> type:
        """Extract the type of acquisition parameters."""
        sig = inspect.signature(self.acquisition, eval_str=True)
        param = next(iter(sig.parameters.values()))
        return param.annotation

    @property
    def data_type(self) -> type:
        """Extract the return type of acquisition."""
        return inspect.signature(self.acquisition, eval_str=True).return_annotation

    @property
    def results_type(self) -> type:
        """Extract the return type of fit."""
        if self.fit is None:
            return None
        return inspect.signature(self.fit, eval_str=True).return_annotation

    @property
    def platform_dependent(self) -> bool:
        """Check if acquisition involves platform."""
        return "platform" in inspect.signature(self.acquisition).parameters

    @property
    def targets_dependent(self) -> bool:
        """Check if acquisition involves qubits."""
        return "targets" in inspect.signature(self.acquisition).parameters

    def __call__(
        self,
        pars: _ParametersT | None = None,
        fit: _FitParsT | None = None,
        report: _ReportParsT | None = None,
        **kwargs: Any,
    ) -> BoundProtocol:
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
            parameters_type = self.parameters_type
            pars = (
                parameters_type.load(kwargs)
                if isinstance(parameters_type, type)
                and issubclass(parameters_type, Parameters)
                else parameters_type(**kwargs)
            )
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

    reports: Any = None
    """Report output, keyed by target for per-target callbacks."""


ProtocolsCollection = dict[str, Protocol]
"""Collection of protocols.

This collection is supposed to be a bundle, either built-in or provided by external
extensions.
"""


@dataclass
class DummyPars(Parameters):
    """Dummy parameters."""


@dataclass
class DummyData(Data):
    """Dummy data."""

    def save(self, path):
        """Dummy method for saving data."""


@dataclass
class DummyRes(Results):
    """Dummy results."""


def _dummy_acquisition(pars: DummyPars, platform: Platform) -> DummyData:
    """Dummy data acquisition."""
    return DummyData()


def _dummy_update(
    results: DummyRes, platform: Platform, qubit: QubitId | QubitPairId
) -> None:
    """Dummy update function."""


dummy_operation = Protocol(_dummy_acquisition)
"""Example of a dummy operation."""
