import inspect
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Callable, Generic, NewType, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId

from qibocal.config import log

from .serialize import deserialize, load, serialize

OperationId = NewType("OperationId", str)
"""Identifier for a calibration routine."""
ParameterValue = Union[float, int]
"""Valid value for a routine and runcard parameter."""
Qubits = dict[QubitId, Qubit]
"""Convenient way of passing qubit pairs in the routines."""
QubitsPairs = dict[tuple[QubitId, QubitId], QubitPair]


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
            message = f" in {end-start:.2f} seconds"
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
    """Wait time for the qubit to decohere back to the `gnd` state."""
    hardware_average: bool = False
    """By default hardware average will be performed."""
    classify: bool = False
    """By default qubit state classification will not be performed."""

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

    @property
    def execution_parameters(self):
        """Default execution parameters."""
        averaging_mode = (
            AveragingMode.CYCLIC if self.hardware_average else AveragingMode.SINGLESHOT
        )
        acquisition_type = (
            AcquisitionType.DISCRIMINATION
            if self.classify
            else AcquisitionType.INTEGRATION
        )
        return ExecutionParameters(
            nshots=self.nshots,
            relaxation_time=self.relaxation_time,
            acquisition_type=acquisition_type,
            averaging_mode=averaging_mode,
        )


class AbstractData:
    """Abstract data class."""

    def __init__(
        self, data: dict[Union[tuple[QubitId, int], QubitId], npt.NDArray] = None
    ):
        self.data = data if data is not None else {}

    def __getitem__(self, qubit: Union[QubitId, tuple[QubitId, int]]):
        """Access data attribute member."""
        if isinstance(qubit, list):
            qubit = tuple(qubit)
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
                json.dumps(serialize(self.params), indent=4)
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
    def qubits(self):
        """Access qubits from data structure."""
        if set(map(type, self.data)) == {tuple}:
            return list({q[0] for q in self.data})
        return [q for q in self.data]

    @property
    def pairs(self):
        """Access qubit pairs ordered alphanumerically from data structure."""
        return list({tuple(sorted(q[:2])) for q in self.data})

    def register_qubit(self, dtype, data_keys, data_dict):
        """Store output for single qubit.

        Args:
            data_keys (tuple): Keys of Data.data.
            data_dict (dict): The keys are the fields of `dtype` and
            the values are the related arrays.
        """
        # to be able to handle the non-sweeper case
        ar = np.empty(np.shape(data_dict[list(data_dict)[0]]), dtype=dtype)
        for key, value in data_dict.items():
            ar[key] = value

        if data_keys in self.data:
            self.data[data_keys] = np.rec.array(
                np.concatenate((self.data[data_keys], ar))
            )
        else:
            self.data[data_keys] = np.rec.array(ar)

    def save(self, path: Path):
        """Store data to file."""
        super()._to_json(path, DATAFILE)
        super()._to_npz(path, DATAFILE)

    @classmethod
    def load(cls, path: Path):
        """Load data and parameters."""
        return super().load(path, filename=DATAFILE)


@dataclass
class Results(AbstractData):
    """Generic runcard update."""

    def __contains__(self, key: Union[QubitId, QubitPairId, tuple[QubitId, ...]]):
        """Checking if qubit is in Results.

        If key is not present means that fitting failed or was not
        performed.
        """
        if isinstance(key, list):
            key = tuple(key)
        return all(
            key in getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), dict)
        )

    @classmethod
    def load(cls, path: Path):
        """Load results."""
        return super().load(path, filename=RESULTSFILE)

    def save(self, path: Path):
        """Store results to file."""
        super()._to_json(path, RESULTSFILE)
        super()._to_npz(path, RESULTSFILE)


# Internal types, in particular `_ParametersT` is used to address function
# contravariance on parameter type
_ParametersT = TypeVar("_ParametersT", bound=Parameters, contravariant=True)
_DataT = TypeVar("_DataT", bound=Data)
_ResultsT = TypeVar("_ResultsT", bound=Results)


@dataclass
class Routine(Generic[_ParametersT, _DataT, _ResultsT]):
    """A wrapped calibration routine."""

    acquisition: Callable[[_ParametersT], _DataT]
    """Data acquisition function."""
    fit: Callable[[_DataT], _ResultsT] = None
    """Post-processing function."""
    report: Callable[[_DataT, _ResultsT], None] = None
    """Plotting function."""
    update: Callable[[_ResultsT, Platform], None] = None
    """Update function platform."""
    two_qubit_gates: Optional[bool] = False
    """Flag to determine whether to allocate list of Qubits or Pairs."""

    def __post_init__(self):
        # add decorator to show logs
        self.acquisition = show_logs(self.acquisition)
        self.fit = show_logs(self.fit)
        if self.update is None:
            self.update = _dummy_update

    @property
    def parameters_type(self):
        """Input parameters type."""
        sig = inspect.signature(self.acquisition)
        param = next(iter(sig.parameters.values()))
        return param.annotation

    @property
    def data_type(self):
        """Data object type return by data acquisition."""
        return inspect.signature(self.acquisition).return_annotation

    @property
    def results_type(self):
        """Results object type returned by data acquisition."""
        return inspect.signature(self.fit).return_annotation

    # TODO: I don't like these properties but it seems to work
    @property
    def platform_dependent(self):
        """Check if acquisition involves platform."""
        return "platform" in inspect.signature(self.acquisition).parameters

    @property
    def targets_dependent(self):
        """Check if acquisition involves qubits."""
        return "targets" in inspect.signature(self.acquisition).parameters


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
    results: DummyRes, platform: Platform, qubit: Union[QubitId, QubitPairId]
) -> None:
    """Dummy update function."""


dummy_operation = Routine(_dummy_acquisition)
"""Example of a dummy operation."""
