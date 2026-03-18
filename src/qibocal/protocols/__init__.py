from typing import Any, Optional

from qibolab import Platform

from qibocal.auto.mode import AUTOCALIBRATION, ExecutionMode
from qibocal.auto.operation import Data, OperationId, Parameters, Results, Routine
from qibocal.auto.task import Action, Completed, Id, Targets, Task

from . import (
    allxy,
    classification,
    coherence,
    dispersive_shift,
    drag,
    flipping,
    flux_dependence,
    qubit_spectroscopies,
    rabi,
    ramsey,
    randomized_benchmarking,
    readout,
    readout_optimization,
    resonator_spectroscopies,
    signal_experiments,
    tomographies,
    two_qubit_interaction,
    twpa,
)
from .allxy import *
from .classification import *
from .coherence import *
from .dispersive_shift import *
from .drag import *
from .flipping import *
from .flux_dependence import *
from .qubit_spectroscopies import *
from .rabi import *
from .ramsey import *
from .randomized_benchmarking import *
from .readout import *
from .readout_optimization import *
from .resonator_spectroscopies import *
from .signal_experiments import *
from .tomographies import *
from .two_qubit_interaction import *
from .twpa import *

__all__ = []
__all__ += ["allxy"]
__all__ += ["coherence"]
__all__ += ["flux_dependence"]
__all__ += ["classification"]
__all__ += ["rabi"]
__all__ += ["ramsey"]
__all__ += ["randomized_benchmarking"]
__all__ += ["readout_optimization"]
__all__ += ["signal_experiments"]
__all__ += ["dispersive_shift"]
__all__ += ["classification"]
__all__ += ["drag"]
__all__ += ["flipping"]
__all__ += ["readout"]
__all__ += ["tomographies"]
__all__ += ["resonator_spectroscopies"]
__all__ += ["qubit_spectroscopies"]
__all__ += ["two_qubit_interaction"]
__all__ += ["twpa"]


class CalibrationProtocol:
    """Protocol to calibrate a chip."""

    def __call__(
        self,
        *args: Any,
        parameters: Optional[dict] = None,
        id: str = "",
        mode: ExecutionMode = AUTOCALIBRATION,
        update: bool = True,
        targets: Optional[Targets] = None,
        **kwargs,
    ) -> Completed:
        """Invoke calibration experiment.

        The signature depend on the specific protocol. For the core set, the documentation
        can be found at <url>. Otherwise compare the documentation for the Qibocal extension
        providing the protocol invoked.
        """
        _ = self, args, parameters, id, mode, update, targets, kwargs

        def _acquisition(x: Parameters) -> Data:
            _ = x
            return Data()

        def _fit(x: Data) -> Results:
            _ = x
            return Results()

        def _report(x: Data, y: Results) -> None:
            _ = x, y

        def _update(x: Results, y: Platform) -> None:
            _ = x, y

        return Completed(
            task=Task(
                action=Action(id=Id(""), operation=OperationId("")),
                operation=Routine(_acquisition, _fit, _report, _update),
            )
        )


_ph = CalibrationProtocol()
"Placeholder."


class BaseSet:
    a: CalibrationProtocol = _ph
    b: CalibrationProtocol = _ph
    c: CalibrationProtocol = _ph


PROTOCOLS: dict[str, Routine] = {}
