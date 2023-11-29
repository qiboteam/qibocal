"""Describe the status of a completed task.

Simple and general statuses are defined here, but more of them can be defined
by individual calibrations routines, and user code as well::

    class PinkFirst(Status):
        '''Follow the pink arrow as the next one.'''

    @dataclass
    class ParametrizedException(Status):
        '''Trigger exceptional workflow, passing down a further parameter.

        Useful if the handler function is using some kind of threshold, or can
        make somehow use of the parameter to decide, but in a way that is not
        completely established, so it should not be hardcoded in the status
        type.

        '''
        myvalue: int

    @dataclass
    class ExceptionWithInput(Status):
        '''Pass to next routine as input.'''
        routine_x_input: float

In general, statuses can encode a predetermined decision about what to do next,
so the decision has been handled by the fitting function, or an open decision,
that is left up to the handler function.

"""


class Status:
    """The exit status of a calibration routine."""


class Normal(Status):
    """All green."""


class Failure(Status):
    """Unrecoverable."""
