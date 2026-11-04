"""Two-Gaussian mixture fit of the projected single-shot distributions.

Shots prepared in each state are projected on the axis connecting the two state
centroids. Each projected distribution is modeled as a mixture of the same two
Gaussians, one per qubit state, with independent weights: the shots prepared in the
ground state may contain a spurious excited population (e.g. thermal), and those
prepared in the excited state a spurious ground population (e.g. decay, or an
imperfect pi pulse). The two Gaussians, describing the readout of each state, are
shared by both preparations.
"""

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

__all__ = ["GaussianMixture", "fit_mixture", "project"]

MAX_ITERATIONS = 1000
TOLERANCE = 1e-10
"""Relative tolerance on the log-likelihood, to stop the iterations."""
INITIAL_SPURIOUS = 0.05
"""Initial guess of the spurious populations."""
DISTANCE_SCALE = 1.5
DISTANCE_FLOOR = 0.02
"""Acceptance of the fit, cf. :attr:`GaussianMixture.tolerance`."""


def project(points: npt.NDArray, angle: float) -> npt.NDArray:
    """Project IQ points on the states axis.

    The `angle` is the classifier rotation angle, rotating the states axis to the I
    axis, cf. :meth:`qibocal.fitting.classifier.qubit_fit.QubitFit.rotate`.
    """
    return points @ np.array([np.cos(angle), -np.sin(angle)])


def _robust_std(values: npt.NDArray) -> float:
    """Standard deviation estimated from the median absolute deviation."""
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


@dataclass
class GaussianMixture:
    """Two Gaussians shared by both preparations, with independent weights."""

    means: list[float]
    """Means of the ground and excited state Gaussians."""
    stds: list[float]
    """Standard deviations of the ground and excited state Gaussians."""
    spurious: list[float]
    """Spurious populations: excited in the ground preparation, and ground in the
    excited one."""
    distance: float | None = None
    """Largest Kolmogorov-Smirnov distance between the fitted and the measured
    distributions, among the two preparations."""
    tolerance: float | None = None
    """Largest acceptable :attr:`distance`.

    It is the statistical fluctuation expected for the number of shots, but not
    smaller than a fixed floor, tolerating small deviations from the Gaussian shape
    (e.g. due to decays during the readout) in measurements with many shots.
    """

    @property
    def consistent(self) -> bool:
        """Whether the two Gaussians describe the measured distributions.

        Further clusters (e.g. leakage to higher states) are absorbed by broadening
        the Gaussians, resulting in an inconsistent fit.
        """
        return (
            self.distance is not None
            and self.tolerance is not None
            and self.distance <= self.tolerance
        )

    def weights(self, prepared: int) -> npt.NDArray:
        """Weights of the ground and excited state Gaussians for a preparation."""
        spurious = self.spurious[prepared]
        return np.array(
            [1 - spurious, spurious] if prepared == 0 else [spurious, 1 - spurious]
        )

    def pdf(self, x: npt.ArrayLike, prepared: int) -> npt.NDArray:
        """Probability density of the shots prepared in a given state."""
        x = np.asarray(x)
        return sum(
            weight * norm.pdf(x, mean, std)
            for weight, mean, std in zip(self.weights(prepared), self.means, self.stds)
        )

    @property
    def threshold(self) -> float:
        """Threshold minimizing the readout errors of the two Gaussians.

        It is the point between the two means where their densities are equal, i.e.
        the optimal threshold with equal priors, unaffected by the spurious
        populations.
        """
        (m0, m1), (s0, s1) = self.means, self.stds
        midpoint = (m0 + m1) / 2
        if np.isclose(s0, s1, rtol=1e-9):
            return float(midpoint)
        roots = np.roots(
            [
                1 / s0**2 - 1 / s1**2,
                -2 * (m0 / s0**2 - m1 / s1**2),
                m0**2 / s0**2 - m1**2 / s1**2 + 2 * np.log(s0 / s1),
            ]
        )
        roots = roots[np.isreal(roots)].real
        between = roots[(roots - m0) * (roots - m1) <= 0]
        candidates = between if len(between) > 0 else roots
        if len(candidates) == 0:
            return float(midpoint)
        return float(candidates[np.argmin(np.abs(candidates - midpoint))])

    def errors(self, threshold: float | None = None) -> list[float]:
        """Readout errors of the two Gaussians, P(1|0) and P(0|1).

        The ground state Gaussian is assumed to lie below the `threshold`, which
        defaults to the optimal one.
        """
        threshold = self.threshold if threshold is None else threshold
        (m0, m1), (s0, s1) = self.means, self.stds
        return [
            float(norm.sf(threshold, m0, s0)),
            float(norm.cdf(threshold, m1, s1)),
        ]

    def cdf(self, x: npt.ArrayLike, prepared: int) -> npt.NDArray:
        """Cumulative distribution of the shots prepared in a given state."""
        x = np.asarray(x)
        return sum(
            weight * norm.cdf(x, mean, std)
            for weight, mean, std in zip(self.weights(prepared), self.means, self.stds)
        )

    def dump(self) -> dict:
        """Serializable representation."""
        return asdict(self)


def _distance(mixture: GaussianMixture, x: npt.NDArray, prepared: int) -> float:
    """Kolmogorov-Smirnov distance from the measured distribution."""
    x = np.sort(x)
    model = mixture.cdf(x, prepared)
    above = np.arange(1, len(x) + 1) / len(x) - model
    below = model - np.arange(len(x)) / len(x)
    return float(max(np.max(above), np.max(below)))


def _log_likelihoods(
    x: npt.NDArray, mixture: GaussianMixture, prepared: int
) -> npt.NDArray:
    """Weighted log-likelihoods of each point for each Gaussian, shape (2, N)."""
    with np.errstate(divide="ignore"):
        log_weights = np.log(mixture.weights(prepared))
    return np.stack(
        [
            log_weight + norm.logpdf(x, mean, std)
            for log_weight, mean, std in zip(log_weights, mixture.means, mixture.stds)
        ]
    )


def fit_mixture(ground: npt.NDArray, excited: npt.NDArray) -> GaussianMixture:
    """Fit the projected shots prepared in the ground and excited states.

    The fit uses the expectation-maximization algorithm, initialized with the median
    and spread of each preparation.

    Raises:
        ValueError: if the fit does not produce finite parameters.
    """
    ground, excited = np.asarray(ground, float), np.asarray(excited, float)
    floor = 1e-6 * (float(np.ptp(np.concatenate([ground, excited]))) or 1.0)
    mixture = GaussianMixture(
        means=[float(np.median(ground)), float(np.median(excited))],
        stds=[max(_robust_std(ground), floor), max(_robust_std(excited), floor)],
        spurious=[INITIAL_SPURIOUS, INITIAL_SPURIOUS],
    )

    previous = -np.inf
    for _ in range(MAX_ITERATIONS):
        # expectation: probability of each shot belonging to each Gaussian
        responsibilities, likelihood = [], 0.0
        for prepared, x in enumerate((ground, excited)):
            logs = _log_likelihoods(x, mixture, prepared)
            total = np.logaddexp(logs[0], logs[1])
            likelihood += float(np.sum(total))
            responsibilities.append(np.exp(logs - total))

        # maximization: independent weights, shared Gaussians
        mixture.spurious = [
            float(np.mean(responsibilities[0][1])),
            float(np.mean(responsibilities[1][0])),
        ]
        points = np.concatenate([ground, excited])
        for state in (0, 1):
            weights = np.concatenate([r[state] for r in responsibilities])
            total = np.sum(weights)
            mean = np.sum(weights * points) / total
            mixture.means[state] = float(mean)
            mixture.stds[state] = max(
                float(np.sqrt(np.sum(weights * (points - mean) ** 2) / total)), floor
            )

        if not np.isfinite(likelihood):
            raise ValueError("Gaussian mixture fit diverged")
        if abs(likelihood - previous) <= TOLERANCE * abs(likelihood):
            break
        previous = likelihood

    parameters = np.array([*mixture.means, *mixture.stds, *mixture.spurious])
    if not np.all(np.isfinite(parameters)):
        raise ValueError("Gaussian mixture fit diverged")
    mixture.distance = max(
        _distance(mixture, x, prepared) for prepared, x in enumerate((ground, excited))
    )
    shots = min(len(ground), len(excited))
    mixture.tolerance = max(DISTANCE_SCALE / np.sqrt(shots), DISTANCE_FLOOR)
    return mixture
