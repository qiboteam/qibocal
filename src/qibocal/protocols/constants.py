"""Shared constants and base classes for qibocal protocols."""

from enum import Enum

from scipy import constants

GHZ_TO_HZ = 1e9
HZ_TO_GHZ = 1e-9
V_TO_UV = 1e6
S_TO_NS = 1e9
MESH_SIZE = 50
MARGIN = 0
SPACING = 0.1
COLUMNWIDTH = 600
LEGEND_FONT_SIZE = 20
TITLE_SIZE = 25
EXTREME_CHI = 1e4
"""Chi2 output when errors list contains zero elements"""
KB = constants.k
H = constants.h
COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"
DELAY_FIT_PERCENTAGE = 10
"""Percentage of the first and last points used to fit the cable delay."""
STRING_TYPE = "<U100"

# constants for signal detection
MAX_PIXELS = 2
"""How many pixels at most two clusters' endpoints should be far for merging them."""
DISTANCE_XY = 1.5 * MAX_PIXELS  # very heuristic
""" Minimum distance for separate clusters.
Clusters below this distance will be merged.
Since it is given in a 3D-space, with a compressed vertical dimension, and the horizontal plane measured in pixels,
this distance correspond to diagonally adjacent pixels, with some additional leeway for the extra dimension.
"""
DISTANCE_Z = 0.5
"""See :const:`DISTANCE_XY`."""


class FeatExtractionError(Exception):
    """Exception for feature extraction errors."""


class PowerLevel(str, Enum):
    """Power Regime for Resonator Spectroscopy"""

    high = "high"
    low = "low"
