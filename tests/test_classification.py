import numpy as np
import plotly.graph_objects as go
import pytest
from scipy.stats import norm

from qibocal.protocols.classification.classification import (
    ClassificationType,
    SingleShotClassificationData,
    _fit,
    _plot,
    _update,
)
from qibocal.protocols.classification.mixture import fit_mixture
from qibocal.protocols.classification.outliers import (
    MAX_DISTANCE,
    MAX_EXCLUDED,
    core_region,
)

CENTERS = [(0.2, 0.5), (0.6, 0.1)]


def single_shot_data(nshots: int, classifiers=("qubit_fit",), seed=0):
    rng = np.random.default_rng(seed)
    data = SingleShotClassificationData(
        nshots=nshots,
        savedir=" ",
        qubit_frequencies={0: 5e9},
        classifiers_list=list(classifiers),
    )
    for state, center in enumerate(CENTERS):
        points = rng.normal(center, 0.08, size=(nshots, 2))
        data.register_qubit(
            ClassificationType,
            0,
            {"i": points[:, 0], "q": points[:, 1], "state": [state] * nshots},
        )
    return data


@pytest.mark.parametrize("classifiers", [["qubit_fit"], ["qubit_fit", "qblox_fit"]])
def test_plot_size_independent_of_shots(classifiers):
    sizes = []
    for nshots in (500, 20_000):
        data = single_shot_data(nshots, classifiers)
        figures, _ = _plot(data, 0, _fit(data))
        sizes.append(
            max(
                np.size(trace.z) if isinstance(trace, go.Heatmap) else np.size(trace.x)
                for trace in figures[0].data
            )
        )
        assert [type(trace) for trace in figures[0].data].count(go.Contour) == len(
            classifiers
        )
        assert len(figures) == (2 if len(classifiers) == 1 else 4)
    # densities are binned: the largest trace does not scale with the shots
    assert sizes[1] < 2 * sizes[0]
    assert sizes[1] < 20_000


def test_plot_without_fit():
    data = single_shot_data(500)
    figures, table = _plot(data, 0, None)

    assert len(figures) == 1
    assert table == ""


def mixture_shots(means, stds, spurious, nshots, seed=0):
    """Projected shots of the two preparations, from two Gaussians."""
    rng = np.random.default_rng(seed)

    def prepared(excited_weight):
        excited = rng.binomial(nshots, excited_weight)
        return np.concatenate(
            [
                rng.normal(means[0], stds[0], nshots - excited),
                rng.normal(means[1], stds[1], excited),
            ]
        )

    return prepared(spurious[0]), prepared(1 - spurious[1])


@pytest.mark.parametrize(
    "means, stds, spurious",
    [
        ([-1.0, 1.0], [0.3, 0.4], [0.03, 0.15]),
        ([-0.3, 0.3], [0.25, 0.25], [0.1, 0.2]),
        ([0.0, 5.0], [1.0, 1.0], [0.0, 0.0]),
    ],
)
def test_fit_mixture(means, stds, spurious):
    ground, excited = mixture_shots(means, stds, spurious, 20_000)
    mixture = fit_mixture(ground, excited)

    assert mixture.means == pytest.approx(means, abs=0.02)
    assert mixture.stds == pytest.approx(stds, abs=0.02)
    assert mixture.spurious == pytest.approx(spurious, abs=0.01)
    assert mixture.consistent
    # optimal threshold: equal densities of the two Gaussians
    threshold = mixture.threshold
    assert means[0] < threshold < means[1]
    assert norm.pdf(threshold, mixture.means[0], mixture.stds[0]) == pytest.approx(
        norm.pdf(threshold, mixture.means[1], mixture.stds[1])
    )
    # errors at the optimal threshold are minimal
    assert sum(mixture.errors()) <= sum(mixture.errors(threshold + 0.05))
    assert sum(mixture.errors()) <= sum(mixture.errors(threshold - 0.05))


def test_fit_mixture_inconsistent_with_further_cluster():
    ground, excited = mixture_shots([-1.0, 1.0], [0.3, 0.3], [0.02, 0.05], 5_000)
    # a third cluster in both preparations, as from leakage
    third = np.random.default_rng(1).normal(-3.0, 0.3, 1_000)
    mixture = fit_mixture(np.concatenate([ground, third]), excited)

    assert not mixture.consistent
    assert mixture.distance > mixture.tolerance


def test_mixture_results_and_update(platform):
    data = single_shot_data(5_000, seed=1)
    fit = _fit(data)
    mixture = fit.gaussian_mixture(0, consistent=True)

    assert mixture is not None
    assert 0 in fit
    assert fit.confusion[0] == pytest.approx([0.0, 0.0], abs=0.01)
    _, table = _plot(data, 0, fit)
    assert "naive / Gaussian" in table

    _update(fit, platform, 0)
    readout = platform.calibration.single_qubits[0].readout
    assert readout.fidelity == pytest.approx(1 - sum(mixture.errors()))
    assert readout.effective_temperature == fit.mixture_effective_temperature[0][0]


def test_results_without_mixture(platform):
    """Results produced before the mixture fit are still reported and applied."""
    data = single_shot_data(500)
    fit = _fit(data)
    fit.confusion, fit.mixture, fit.mixture_effective_temperature = {}, {}, {}

    assert 0 in fit
    figures, table = _plot(data, 0, fit)
    assert len(figures) == 2
    assert "naive / Gaussian" not in table
    _update(fit, platform, 0)
    assert platform.calibration.single_qubits[0].readout.fidelity == fit.fidelity[0]


def shots_2d(nshots, spurious=(0.02, 0.05), extra=None, bridge=0, seed=0):
    """Shots along and across the states axis, with an optional further cluster
    and shots between the states, as from decays during the readout."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-1.0, 0.0], [1.0, 0.0]])

    def prepared(state):
        wrong = rng.binomial(nshots, spurious[state])
        points = [
            rng.normal(centers[state], 0.3, (nshots - wrong, 2)),
            rng.normal(centers[1 - state], 0.3, (wrong, 2)),
        ]
        if extra is not None:
            points.append(rng.normal(extra[0], 0.3, (int(extra[1] * nshots), 2)))
        if bridge and state == 1:
            along = rng.uniform(-1, 1, bridge)
            points.append(np.column_stack([along, rng.normal(0, 0.3, bridge)]))
        return np.concatenate(points)

    return prepared(0), prepared(1)


def test_core_region_clean():
    ground, excited = shots_2d(20_000)
    region = core_region(ground, excited)

    assert region.excluded
    assert sum(region.outside) <= 2


def test_core_region_excludes_further_cluster():
    ground, excited = shots_2d(10_000, extra=([0.0, 4.0], 0.08))
    region = core_region(ground, excited)

    assert region.excluded
    assert region.outside == pytest.approx([800, 800], abs=20)
    ground, excited = ground[region.inside(ground)], excited[region.inside(excited)]
    mixture = fit_mixture(ground[:, 0], excited[:, 0])
    assert mixture.consistent
    assert mixture.spurious == pytest.approx([0.02, 0.05], abs=0.01)


def test_core_region_keeps_decays_along_the_axis():
    ground, excited = shots_2d(10_000, bridge=500)
    region = core_region(ground, excited)

    assert region.excluded
    assert region.outside[1] <= 2


def test_core_region_refrains_when_unsure():
    ground, excited = shots_2d(10_000, extra=([0.0, 4.0], 0.5))
    region = core_region(ground, excited)

    assert not region.excluded
    assert min(region.outside) > MAX_EXCLUDED * 10_000


def test_region_outline_on_boundary():
    ground, excited = shots_2d(5_000)
    region = core_region(ground, excited)

    assert region.distance(region.outline()) == pytest.approx(MAX_DISTANCE)
