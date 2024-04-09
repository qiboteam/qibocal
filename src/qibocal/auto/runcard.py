"""Specify runcard layout, handles (de)serialization."""

import os
from typing import Optional

from pydantic.dataclasses import dataclass
from qibo.backends import Backend, GlobalBackend
from qibo.transpiler.pipeline import Passes
from qibolab.platform import Platform

from .experiment import Experiment, Targets

MAX_ITERATIONS = 5
"""Default max iterations."""


@dataclass(config=dict(smart_union=True))
class Runcard:
    """Structure of an execution runcard."""

    actions: list[Experiment]
    """List of action to be executed."""
    targets: Optional[Targets] = None
    """Qubits to be calibrated.
       If `None` the protocols will be executed on all qubits
       available in the platform."""
    backend: str = "qibolab"
    """Qibo backend."""
    platform: str = os.environ.get("QIBO_PLATFORM", "dummy")
    """Qibolab platform."""
    max_iterations: int = MAX_ITERATIONS
    """Maximum number of iterations."""

    def __post_init__(self):
        if self.targets is None and self.platform_obj is not None:
            self.targets = list(self.platform_obj.qubits)

    @property
    def backend_obj(self) -> Backend:
        """Allocate backend."""
        GlobalBackend.set_backend(self.backend, platform=self.platform)
        backend = GlobalBackend()
        if backend.platform is not None:
            backend.transpiler = Passes(connectivity=backend.platform.topology)
            backend.transpiler.passes = backend.transpiler.passes[-1:]
        return backend

    @property
    def platform_obj(self) -> Platform:
        """Allocate platform."""
        return self.backend_obj.platform

    @classmethod
    def load(cls, params: dict):
        """Load a runcard (dict)."""
        return cls(**params)

    @property
    def raw(self):
        runcard = {}
        runcard["targets"] = self.targets
        runcard["backend"] = self.backend
        runcard["platform"] = self.platform

        runcard["actions"] = [exp.raw for exp in self.actions]
        return runcard
