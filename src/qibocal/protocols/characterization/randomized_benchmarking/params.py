from dataclasses import dataclass, field
from typing import TypedDict, Union

from qibocal.auto.operation import Parameters


class DepthsDict(TypedDict):
    start: int
    stop: int
    step: int


@dataclass
class RBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    nqubits: int
    """The amount of qubits on the chip."""
    qubits: list
    """A list of indices which qubit(s) should be benchmarked """
    depths: Union[list, DepthsDict]
    """A list of depths/sequence lengths. If a dictionary is given the list will be build."""
    niter: int
    """Sets how many iterations over the same depth value."""
    nshots: int
    """For each sequence how many shots for statistics should be performed."""
    n_bootstrap: int = 100
    """Number of bootstrap iterations for the fit uncertainties and error bars.
    If ``0``, gets the fit uncertainties from the fitting function and the error bars
    from the distribution of the measurements. Defaults to ``1``."""
    uncertainties: Union[str, float] = 0.95
    """Method of computing the error bars and uncertainties of the data. If ``None``, does not
    compute the errors. If ``"std"``, computes the standard deviation. If a value is of type ``float``
    between 0 and 1, computes the corresponding confidence interval. Defaults to ``0.95``."""
    noise_model: str = ""
    """For simulation purposes, string has to match what is in
    :mod:`qibocal.protocols.characterization.randomized_benchmarking.noisemodels`"""
    noise_params: list = field(default_factory=list)
    """With this the noise model will be initialized, if not given random values will be used."""

    def __post_init__(self):
        if isinstance(self.depths, dict):
            self.depths = list(
                range(self.depths["start"], self.depths["stop"], self.depths["step"])
            )
