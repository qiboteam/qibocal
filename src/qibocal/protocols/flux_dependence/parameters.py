"""Parameters for flux frequency sweep protocols."""

from dataclasses import dataclass

import numpy as np

from ...auto.operation import Parameters


@dataclass(kw_only=True)
class FluxFrequencySweepParameters(Parameters):
    """Parameters to define flux DC sweep."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [a.u.]."""
    bias_step: float
    """Bias step for sweep [a.u.]."""

    @property
    def frequency_range(self) -> np.ndarray:
        return np.arange(-self.freq_width / 2, self.freq_width / 2, self.freq_step)

    @property
    def bias_range(self) -> np.ndarray:
        return np.arange(-self.bias_width / 2, self.bias_width / 2, self.bias_step)
