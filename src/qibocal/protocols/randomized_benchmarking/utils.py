"""Deprecated re-export shim.

.. deprecated::
    The content of this module has been split into themed modules (see
    discussion #1660). Import directly from
    ``qibocal.protocols.randomized_benchmarking.types``,
    ``qibocal.protocols.randomized_benchmarking.circuit_generation``,
    ``qibocal.protocols.randomized_benchmarking.acquisition`` or
    ``qibocal.protocols.randomized_benchmarking.fitting`` instead.
"""

from .acquisition import rb_acquisition, twoq_rb_acquisition
from .circuit_generation import (
    RBGenerator,
    _execute_indexed_circuits,
    _generate_indexed_circuits,
    add_inverse_layer,
    add_measurement_layer,
    layer_circuit,
    random_2q_clifford,
    random_clifford,
    setup_data,
)
from .dict_utils import generate_inv_dict_cliffords_file, load_cliffords
from .fitting import data_uncertainties, fit, number_to_str
from .types import (
    GLOBAL_PHASES,
    NPULSES_PER_CLIFFORD,
    SINGLE_QUBIT_CLIFFORDS,
    CircuitDepth,
    CircuitIndex,
    IndexedCircuit,
    IndexedResult,
    RB2QData,
    RB2QInterData,
    RBData,
    RBType,
    StandardRBResult,
)

__all__ = [
    "generate_inv_dict_cliffords_file",
    "load_cliffords",
    "rb_acquisition",
    "twoq_rb_acquisition",
    "RBGenerator",
    "_execute_indexed_circuits",
    "_generate_indexed_circuits",
    "add_inverse_layer",
    "add_measurement_layer",
    "layer_circuit",
    "random_2q_clifford",
    "random_clifford",
    "setup_data",
    "data_uncertainties",
    "fit",
    "number_to_str",
    "GLOBAL_PHASES",
    "NPULSES_PER_CLIFFORD",
    "RB2QData",
    "RB2QInterData",
    "RBData",
    "RBType",
    "SINGLE_QUBIT_CLIFFORDS",
    "StandardRBResult",
    "CircuitDepth",
    "CircuitIndex",
    "IndexedCircuit",
    "IndexedResult",
]
