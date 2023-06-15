import inspect
import json
from dataclasses import dataclass, fields
from typing import Callable, Dict, Generic, NewType, TypeVar, Union

import numpy as np
from qibolab.platform import Platform
from qibolab.qubits import Qubit

OperationId = NewType("OperationId", str)
"""Identifier for a calibration routine."""
ParameterValue = Union[float, int]
"""Valid value for a routine and runcard parameter."""
Qubits = Dict[int, Qubit]
"""Convenient way of passing qubits in the routines."""

DATAFILE = "data.npz"
"""Name of the file where data acquired (arrays) by calibration are dumped."""
JSONFILE = "conf.json"
"""Name of the file where data acquired (global configuration) by calibration are dumped."""


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

        .. todo::

            move the implementation to History, since it is required to resolve
            the linked outputs

        """
        return cls(**parameters)


class Data:
    """Data resulting from acquisition routine."""

    @staticmethod
    def to_npz(path, data_dict: dict):
        """Helper function to use np.savez while converting keys into strings."""
        np.savez(path / DATAFILE, **{str(i): data_dict[i] for i in data_dict})

    @staticmethod
    def to_json(path, data_dict: dict):
        """Helper function to dump to json in JSONFILE path."""
        (path / JSONFILE).write_text(json.dumps(data_dict, indent=4))


@dataclass
class Results:
    """Generic runcard update.

    As for the case of :cls:`Parameters` the explicit structure is only useful
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

    @property
    def update(self) -> Dict[str, ParameterValue]:
        """Produce an update from a result object.

        This is later used to update the runcard.

        """
        up: Dict[str, ParameterValue] = {}
        for fld in fields(self):
            if "update" in fld.metadata:
                up[fld.metadata["update"]] = getattr(self, fld.name)

        return up


# Internal types, in particular `_ParametersT` is used to address function
# contravariance on parameter type
_ParametersT = TypeVar("_ParametersT", bound=Parameters, contravariant=True)
_DataT = TypeVar("_DataT", bound=Data)
_ResultsT = TypeVar("_ResultsT", bound=Results)


@dataclass
class Routine(Generic[_ParametersT, _DataT, _ResultsT]):
    """A wrapped calibration routine."""

    acquisition: Callable[[_ParametersT], _DataT]
    fit: Callable[[_DataT], _ResultsT] = None
    report: Callable[[_DataT, _ResultsT], None] = None

    def __post_init__(self):
        # TODO: this could be improved
        if self.fit is None:
            self.fit = _dummy_fit
        if self.report is None:
            self.report = _dummy_report

    @property
    def parameters_type(self):
        sig = inspect.signature(self.acquisition)
        param = next(iter(sig.parameters.values()))
        return param.annotation

    @property
    def data_type(self):
        return inspect.signature(self.acquisition).return_annotation

    @property
    def results_type(self):
        return inspect.signature(self.fit).return_annotation

    # TODO: I don't like these properties but it seems to work
    @property
    def platform_dependent(self):
        return "platform" in inspect.signature(self.acquisition).parameters

    @property
    def qubits_dependent(self):
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
    return DummyData()


def _dummy_fit(data: DummyData) -> DummyRes:
    return DummyRes()


def _dummy_report(data: DummyData, result: DummyRes):
    return [], ""


dummy_operation = Routine(_dummy_acquisition, _dummy_fit, _dummy_report)
