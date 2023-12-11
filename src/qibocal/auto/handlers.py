"""Handlers available in Qibocal."""
from dataclasses import dataclass


@dataclass
class Handler:
    """Generic handler object.

    .. todo::

        Add call method with recipe for executing protocol
        varying parameters.

    """

    id: str
    """New Id where graph will jump to."""
