from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Callable, Dict, Generic, NewType, TypeVar, Union

OperationId = NewType("OperationId", str)
"""Identifier for a calibration routine."""
ParameterValue = Union[float, int]
"""Valid value for a routine and runcard parameter."""


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
        return cls()


@dataclass
class Results:
    """Generic runcard update.

    As for the case of :cls:`Parameters` the explicit structure is only useful
    to fill the specific update, but in this case there should be a generic way

    Each field might be annotated with an ``updata`` metadata field, in order
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
_ResultsT = TypeVar("_ResultsT", bound=Results, covariant=True)


@dataclass
class Routine(Generic[_ParametersT, _ResultsT]):
    """A wrapped calibration routine."""

    routine: Callable[[_ParametersT], _ResultsT]


#  --- from here on start the examples ---


@dataclass
class Cmd1Pars(Parameters):
    a: int
    b: int


@dataclass
class Cmd1Res(Results):
    res: str = field(metadata=dict(update="myres"))
    num: int


def _command_1(args: Cmd1Pars) -> Cmd1Res:
    print("command_1")
    return Cmd1Res("command_1", 3)


command_1 = Routine(_command_1)


@dataclass
class Cmd2Res(Results):
    res: str = field(metadata=dict(update="res2"))


def _command_2(*args: Cmd1Pars) -> Cmd2Res:
    print("command_2")
    return Cmd2Res("command_2")


command_2 = Routine(_command_2)


#  ---
#  the following enum should exist, with this name, so only its content is
#  supposed to be an example, but it should not be exported by this module, and
#  instead should be placed inside `qibocal.calibrations.__init__.py`
#  ---


class Operation(Enum):
    command_1 = command_1
    command_2 = command_2
