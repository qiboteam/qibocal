from qibocal.auto.operation import ProtocolsCollection

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

PROTOCOLS: ProtocolsCollection = {name: globals()[name] for name in __all__}
