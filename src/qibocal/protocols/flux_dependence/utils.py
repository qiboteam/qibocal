"""Deprecated backward-compatibility shim.

The helpers previously defined in this module have been split into themed modules:

- :mod:`~qibocal.protocols.flux_dependence.parameters`
- :mod:`~qibocal.protocols.flux_dependence.acquisition`
- :mod:`~qibocal.protocols.flux_dependence.physics`
- :mod:`~qibocal.protocols.flux_dependence.fitting`
- :mod:`~qibocal.protocols.flux_dependence.plotting`

Import from those modules directly; everything re-exported here is kept only so that
existing imports of ``qibocal.protocols.flux_dependence.utils.<name>`` keep working.
"""

from .acquisition import create_data_array
from .fitting import (
    _continuity_score,
    _function_dof,
    filter_data,
    flux_extract_feature,
    ransac_fit,
    select_sweetspot,
)
from .parameters import FluxFrequencySweepParameters
from .physics import G_f_d, transmon_frequency, transmon_readout_frequency
from .plotting import flux_crosstalk_plot, flux_dependence_plot

__all__ = [
    "FluxFrequencySweepParameters",
    "create_data_array",
    "flux_dependence_plot",
    "flux_crosstalk_plot",
    "G_f_d",
    "transmon_frequency",
    "transmon_readout_frequency",
    "filter_data",
    "flux_extract_feature",
    "_function_dof",
    "select_sweetspot",
    "_continuity_score",
    "ransac_fit",
]
