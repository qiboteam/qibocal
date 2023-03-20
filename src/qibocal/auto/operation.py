import inspect
from dataclasses import dataclass, field, fields
from typing import Callable, Dict, Generic, NewType, TypeVar, Union

from qibolab.platforms.abstract import AbstractPlatform, Qubit

OperationId = NewType("OperationId", str)
"""Identifier for a calibration routine."""
ParameterValue = Union[float, int]
"""Valid value for a routine and runcard parameter."""
Qubits = Dict[int, Qubit]


class Parameters:
    """Generic action parameters.

    Implement parameters as Algebraic Data Types (similar to), by subclassing
    this marker in actual parameters specification for each calibration routine.

    The actual parameters structure is only used inside the routines themselves.

    """

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
_ResultsT = TypeVar("_ResultsT", bound=Results, covariant=True)
_QubitsT = TypeVar("_QubitsT", bound=Qubits, contravariant=True)
_PlatformT = TypeVar("_PlatformT", bound=AbstractPlatform, contravariant=True)


@dataclass
class Routine(Generic[_PlatformT, _QubitsT, _ParametersT, _DataT, _ResultsT]):
    """A wrapped calibration routine."""

    acquisition: Callable[[_PlatformT, _QubitsT, _ParametersT], _DataT]
    fit: Callable[[_DataT], _ResultsT]

    @property
    def parameters_type(self):
        sig = inspect.signature(self.acquisition)
        param = list(sig.parameters.values())[2]
        return param.annotation

    @property
    def data_type(self):
        return inspect.signature(self.acquisition).return_annotation

    @property
    def results_type(self):
        return inspect.signature(self.fit).return_annotation


@dataclass
class DummyPars(Parameters):
    """Dummy parameters."""


@dataclass
class DummyData(Data):
    """Dummy data."""


@dataclass
class DummyRes(Results):
    """Dummy results."""


def _dummy_acquisition(pars: DummyPars) -> DummyData:
    return DummyData()


def _dummy_fit(data: DummyData) -> DummyRes:
    return DummyRes()


dummy_operation = Routine(_dummy_acquisition, _dummy_fit)

#  --- from here on start the examples ---


@dataclass
class Cmd1Pars(Parameters):
    a: int
    b: int


@dataclass
class Cmd1Data(Data):
    c: float


@dataclass
class Cmd1Res(Results):
    res: str = field(metadata=dict(update="myres"))
    num: int


def _cmd1_acq(args: Cmd1Pars) -> Cmd1Data:
    print("command_1")
    return Cmd1Data(3.4)


def _cmd1_fit(args: Cmd1Data) -> Cmd1Res:
    return Cmd1Res("command_1", 3)


command_1 = Routine(_cmd1_acq, _cmd1_fit)


@dataclass
class Cmd2Res(Results):
    res: str = field(metadata=dict(update="res2"))


def _cmd2_acq(*args: Cmd1Pars) -> Cmd1Data:
    print("command_2")
    return Cmd1Data(1.8)


def _cmd2_fit(*args: Cmd1Data) -> Cmd2Res:
    return Cmd2Res("command_2")


command_2 = Routine(_cmd1_acq, _cmd1_fit)


#  ---
#  the following enum should exist, with this name, so only its content is
#  supposed to be an example, but it should not be exported by this module, and
#  instead should be placed inside `qibocal.calibrations.__init__.py`
#  ---


# class Operation(Enum):
#     command_1 = command_1
#     command_2 = command_2
