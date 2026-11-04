"""Exclusion of outliers from the single-shot Gaussian mixture fit.

Shots outside of the two state distributions (e.g. from leakage to higher states)
would otherwise be absorbed by broadening the fitted Gaussians. Each state core is
estimated robustly from the shots prepared in that state, and shots far from both
cores, and from the segment connecting them, are excluded. Shots along that segment
are retained, since they may result from decays during the readout.
"""

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["CoreRegion", "core_region", "rotate"]

MAX_DISTANCE = 5.0
"""Largest distance from the state cores, in units of their spread."""
MAX_EXCLUDED = 0.25
"""Largest excluded fraction of each preparation.

Beyond it, the cores are not considered representative of the data, and no shot is
excluded.
"""


def rotate(points: npt.NDArray, angle: float) -> npt.NDArray:
    """Coordinates along and across the states axis.

    The `angle` is the classifier rotation angle, rotating the states axis to the I
    axis, cf. :meth:`qibocal.fitting.classifier.qubit_fit.QubitFit.rotate`.
    """
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


def _robust_center_and_spread(points: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Median and spread from the median absolute deviation, per coordinate."""
    center = np.median(points, axis=0)
    spread = 1.4826 * np.median(np.abs(points - center), axis=0)
    return center, spread


@dataclass
class CoreRegion:
    """Region around the two state cores, in coordinates along and across the
    states axis."""

    centers: list[list[float]]
    """Cores of the ground and excited state distributions."""
    spreads: list[list[float]]
    """Spreads of the ground and excited state distributions."""
    outside: list[int]
    """Shots outside of the region, for each preparation."""
    excluded: bool
    """Whether the shots outside are excluded from the fit."""

    def distance(self, points: npt.NDArray) -> npt.NDArray:
        """Distance from the cores and the segment connecting them.

        Between the cores, the center and spread are linearly interpolated along the
        states axis.
        """
        centers, spreads = np.array(self.centers), np.array(self.spreads)
        position = (points[:, 0] - centers[0, 0]) / (centers[1, 0] - centers[0, 0])
        position = np.clip(position, 0, 1)[:, np.newaxis]
        center = centers[0] + position * (centers[1] - centers[0])
        spread = spreads[0] + position * (spreads[1] - spreads[0])
        return np.sqrt(np.sum(((points - center) / spread) ** 2, axis=1))

    def inside(self, points: npt.NDArray) -> npt.NDArray:
        """Whether the points are within the region."""
        return self.distance(points) <= MAX_DISTANCE

    def outline(self, samples: int = 50) -> npt.NDArray:
        """Closed boundary of the region.

        It consists of the two sides along the segment, and two half ellipses
        around the cores.
        """
        centers, spreads = np.array(self.centers), MAX_DISTANCE * np.array(self.spreads)
        position = np.linspace(0, 1, samples)[:, np.newaxis]
        center = centers[0] + position * (centers[1] - centers[0])
        spread = spreads[0] + position * (spreads[1] - spreads[0])
        angles = np.linspace(-np.pi / 2, np.pi / 2, samples)
        cap = np.column_stack([np.cos(angles), np.sin(angles)])
        return np.concatenate(
            [
                center - spread * [0, 1],
                centers[1] + spreads[1] * cap,
                (center + spread * [0, 1])[::-1],
                centers[0] - spreads[0] * cap,
                (center - spread * [0, 1])[:1],
            ]
        )

    def dump(self) -> dict:
        """Serializable representation."""
        return asdict(self)


def core_region(ground: npt.NDArray, excited: npt.NDArray) -> CoreRegion:
    """Region of the state cores, from the shots in each preparation.

    `ground` and `excited` are the shots in coordinates along and across the states
    axis, cf. :func:`rotate`. Spurious populations of each preparation are a minority,
    so the robust estimates of center and spread mostly describe its own state. If
    they do not (e.g. a large decay), the spread increases, and fewer shots are
    excluded.
    """
    cores = [_robust_center_and_spread(points) for points in (ground, excited)]
    region = CoreRegion(
        centers=[center.tolist() for center, _ in cores],
        spreads=[spread.tolist() for _, spread in cores],
        outside=[0, 0],
        excluded=False,
    )
    # degenerate cores: no meaningful region
    if (
        np.any(np.array(region.spreads) <= 0)
        or region.centers[0][0] >= region.centers[1][0]
    ):
        return region
    region.outside = [
        int(np.sum(~region.inside(points))) for points in (ground, excited)
    ]
    region.excluded = all(
        outside <= MAX_EXCLUDED * len(points)
        for outside, points in zip(region.outside, (ground, excited))
    )
    return region
