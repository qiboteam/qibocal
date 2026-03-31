from typing import Any, Optional

from qibolab import Platform

from qibocal.auto.mode import AUTOCALIBRATION, ExecutionMode
from qibocal.auto.operation import (
    Data,
    OperationId,
    Parameters,
    ProtocolsCollection,
    Results,
    Routine,
)
from qibocal.auto.task import Action, Completed, Id, Targets, Task

from . import (
    allxy as allxy_,
)
from . import (
    classification,
    coherence,
    drag,
    flux_dependence,
    qubit_spectroscopies,
    rabi,
    randomized_benchmarking,
    readout,
    readout_optimization,
    resonator_spectroscopies,
    signal_experiments,
    tomographies,
    two_qubit_interaction,
    twpa,
)
from . import (
    dispersive_shift as dispersive_shift_,
)
from . import (
    flipping as flipping_,
)
from . import (
    ramsey as ramsey_,
)
from .allxy import *
from .classification import *
from .coherence import *
from .coherence import spin_echo, t1, t2
from .dispersive_shift import *
from .drag import *
from .flipping import *
from .flux_dependence import *
from .qubit_spectroscopies import *
from .qubit_spectroscopies import qubit_spectroscopy
from .rabi import *
from .rabi import rabi_amplitude, rabi_length
from .ramsey import *
from .ramsey import ramsey
from .randomized_benchmarking import *
from .readout import *
from .readout_optimization import *
from .resonator_spectroscopies import *
from .resonator_spectroscopies import resonator_spectroscopy
from .signal_experiments import *
from .tomographies import *
from .two_qubit_interaction import *
from .twpa import *

__all__ = []
__all__ += allxy_.__all__
__all__ += coherence.__all__
__all__ += flux_dependence.__all__
__all__ += classification.__all__
__all__ += rabi.__all__
__all__ += ramsey_.__all__
__all__ += randomized_benchmarking.__all__
__all__ += readout_optimization.__all__
__all__ += signal_experiments.__all__
__all__ += dispersive_shift_.__all__
__all__ += classification.__all__
__all__ += drag.__all__
__all__ += flipping_.__all__
__all__ += readout.__all__
__all__ += tomographies.__all__
__all__ += resonator_spectroscopies.__all__
__all__ += qubit_spectroscopies.__all__
__all__ += two_qubit_interaction.__all__
__all__ += twpa.__all__


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


# The following class is exposed only for documentation purpose, to statically annotate
# a base set of protocols as `Executor` attributes.
# They are supposed to be dynamically overwritten with the `PROTOCOLS` below anyhow,
# during the `Executor` initialization.
class BaseSet:
    resonator_spectroscopy: CalibrationProtocol = _ph
    qubit_spectroscopy: CalibrationProtocol = _ph
    rabi_amplitude: CalibrationProtocol = _ph
    rabi_length: CalibrationProtocol = _ph
    ramsey: CalibrationProtocol = _ph
    t1: CalibrationProtocol = _ph
    t2: CalibrationProtocol = _ph
    spin_echo: CalibrationProtocol = _ph


PROTOCOLS: ProtocolsCollection = {
    "resonator_spectroscopy": resonator_spectroscopy,
    "qubit_spectroscopy": qubit_spectroscopy,
    "rabi_amplitude": rabi_amplitude,
    "rabi_length": rabi_length,
    "ramsey": ramsey,
    "t1": t1,
    "t2": t2,
    "spin_echo": spin_echo,
}
