import inspect
import json
import time
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Callable, Generic, NewType, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId, QubitPairId

from qibocal.config import log

from .serialize import deserialize, load, serialize

OperationId = NewType("OperationId", str)
"""Identifier for a calibration routine."""
ParameterValue = Union[float, int]
"""Valid value for a routine and runcard parameter."""
Qubits = dict[QubitId, Qubit]
"""Convenient way of passing qubit pairs in the routines."""
QubitsPairs = dict[tuple[QubitId, QubitId], Qubit]


DATAFILE = "data.npz"
"""Name of the file where data acquired (arrays) by calibration are dumped."""
JSONFILE = "conf.json"
"""Name of the file where data acquired (global configuration) by calibration are dumped."""
RESULTSFILE = "results.json"
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

    Implement parameters as Algebraic Data Types (similar to), by subclassing
    this marker in actual parameters specification for each calibration routine.

    The actual parameters structure is only used inside the routines themselves.

    """

    nshots: int
    """Number of executions on hardware"""
    relaxation_time: float
    """Wait time for the qubit to decohere back to the `gnd` state"""

    @classmethod
    def load(cls, parameters):
        """Load parameters from runcard.

        Possibly looking into previous steps outputs.
        Parameters defined in Parameters class are removed from `parameters`
        before `cls` is created.
        Then `nshots` and `relaxation_time` are assigned to cls.

        .. todo::

            move the implementation to History, since it is required to resolve
            the linked outputs

        """
        for parameter, value in DEFAULT_PARENT_PARAMETERS.items():
            DEFAULT_PARENT_PARAMETERS[parameter] = parameters.pop(parameter, value)
        instantiated_class = cls(**parameters)
        for parameter, value in DEFAULT_PARENT_PARAMETERS.items():
            setattr(instantiated_class, parameter, value)
        return instantiated_class


class Data:
    """Data resulting from acquisition routine."""

    data: dict[Union[tuple[QubitId, int], QubitId], npt.NDArray]
    """Data object to store arrays"""

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

    def __getitem__(self, qubit: Union[QubitId, tuple[QubitId, int]]):
        """Access data attribute member."""
        return self.data[qubit]

    @property
    def global_params(self) -> dict:
        """Convert non-arrays attributes into dict."""
        global_dict = asdict(self)
        global_dict.pop("data")
        return global_dict

    def save(self, path):
        """Store results."""
        self._to_json(path)
        self._to_npz(path)

    def _to_npz(self, path):
        """Helper function to use np.savez while converting keys into strings."""
        np.savez(path / DATAFILE, **{json.dumps(i): self.data[i] for i in self.data})

    def _to_json(self, path):
        """Helper function to dump to json in JSONFILE path."""
        if self.global_params:
            (path / JSONFILE).write_text(
                json.dumps(serialize(self.global_params), indent=4)
            )

    @classmethod
    def load(cls, path):
        with open(path / DATAFILE) as f:
            raw_data_dict = dict(np.load(path / DATAFILE))
            data_dict = {}

            for data_key, array in raw_data_dict.items():
                data_dict[load(data_key)] = np.rec.array(array)
        if (path / JSONFILE).is_file():
            params = json.loads((path / JSONFILE).read_text())

            params = deserialize(params)
            obj = cls(data=data_dict, **params)
        else:
            obj = cls(data=data_dict)

        return obj

    def register_qubit(self, dtype, data_keys, data_dict):
        """Store output for single qubit.

        Args:
            data_keys (tuple): Keys of Data.data.
            data_dict (dict): The keys are the fields of `dtype` and
            the values are the related arrays.
        """
        # to be able to handle the non-sweeper case
        ar = np.empty(np.shape(data_dict[list(data_dict.keys())[0]]), dtype=dtype)
        for key, value in data_dict.items():
            ar[key] = value
        if data_keys in self.data:
            self.data[data_keys] = np.rec.array(
                np.concatenate((self.data[data_keys], ar))
            )
        else:
            self.data[data_keys] = np.rec.array(ar)


@dataclass
class Results:
    """Generic runcard update.

    As for the case of :class:`Parameters` the explicit structure is only useful
    to fill the specific update, but in this case there should be a generic way

    Each field might be annotated with an ``update`` metadata field, in order
    to mark them for later use in the runcard::

        @dataclass
        class Cmd1Res(Results):
            res: str = field(metadata=dict(update="myres"))
            num: int

    .. todo::

        Implement them as ``source: dest``, where ``source`` will be the field
        name in the class, corresponding to the same field in ``Result``

    """

    def save(self, path):
        """Store results to json."""
        (path / RESULTSFILE).write_text(json.dumps(serialize(asdict(self))))

    @classmethod
    def load(cls, path):
        params = json.loads((path / RESULTSFILE).read_text())
        params = deserialize(params)
        return cls(**params)


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
        """ "Data object type return by data acquisition."""
        return inspect.signature(self.acquisition).return_annotation

    @property
    def results_type(self):
        """ "Results object type return by data acquisition."""
        return inspect.signature(self.fit).return_annotation

    # TODO: I don't like these properties but it seems to work
    @property
    def platform_dependent(self):
        """Check if acquisition involves platform."""
        return "platform" in inspect.signature(self.acquisition).parameters

    @property
    def qubits_dependent(self):
        """Check if acquisition involves qubits."""
        return "qubits" in inspect.signature(self.acquisition).parameters


@dataclass
class DummyPars(Parameters):
    """Dummy parameters."""


@dataclass
class DummyData(Data):
    """Dummy data."""

    def save(self, path):
        """Dummy method for saving data"""


@dataclass
class DummyRes(Results):
    """Dummy results."""


def _dummy_acquisition(pars: DummyPars, platform: Platform) -> DummyData:
    """Dummy data acquisition."""
    return DummyData()


def _dummy_update(
    results: DummyRes, platform: Platform, qubit: Union[QubitId, QubitPairId]
) -> None:
    """Dummy update function"""


dummy_operation = Routine(_dummy_acquisition)
"""Example of a dummy operation."""
