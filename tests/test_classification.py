import json

import numpy as np
import plotly.graph_objects as go
import pytest
from scipy.stats import norm

from qibocal.protocols.classification.classification import (
    ClassificationType,
    SingleShotClassificationData,
    SingleShotClassificationParameters,
    SingleShotClassificationResults,
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
from qibocal.protocols.classification.plot import MAX_BINS, _density, _edges
from qibocal.protocols.utils import classify, linear_classifier

CENTERS = [(0.2, 0.5), (0.6, 0.1)]


def single_shot_data(nshots: int, seed=0):
    rng = np.random.default_rng(seed)
    data = SingleShotClassificationData(nshots=nshots, qubit_frequencies={0: 5e9})
    for state, center in enumerate(CENTERS):
        points = rng.normal(center, 0.08, size=(nshots, 2))
        data.register_qubit(
            ClassificationType,
            0,
            {"i": points[:, 0], "q": points[:, 1], "state": [state] * nshots},
        )
    return data


def boundaries(figure):
    return [trace for trace in figure.data if trace.name == "boundary"]


def test_plot_size_independent_of_shots():
    sizes = []
    for nshots in (20_000, 200_000):
        data = single_shot_data(nshots)
        figures, _ = _plot(data, 0, _fit(data))
        (density,) = [t for t in figures[0].data if isinstance(t, go.Image)]
        sizes.append(np.size(density.z))
        assert len(boundaries(figures[0])) == 1
        assert len(figures) == 2
        assert all(figure.layout.title.text is None for figure in figures)
    # densities are binned: beyond the finest resolution, the image only grows with the
    # data extent (slowly, through the tails), not with the shots, and it is bounded
    assert sizes[1] < 1.5 * sizes[0]
    assert sizes[1] <= 4 * MAX_BINS**2


def test_density_blends_states_symmetrically():
    data = single_shot_data(2_000)
    states = [data.state_zero(0), data.state_one(0)]
    edges = _edges(states)
    image = np.array(_density(states, edges).z)
    swapped = np.array(_density(states[::-1], edges).z)

    # the opacity only depends on the total count, the colour on the counts' ratio
    np.testing.assert_array_equal(image[..., 3], swapped[..., 3])
    assert image[..., 3].max() == 1
    assert np.all((image[..., 3] > 0) == (image[..., :3].sum(axis=-1) > 0))


def test_boundary_is_straight_and_exact():
    data = single_shot_data(5_000, seed=2)
    fit = _fit(data)
    figure = _plot(data, 0, fit)[0][0]
    (line,) = boundaries(figure)

    # a segment, drawn on top of the densities
    assert len(line.x) == 2
    assert figure.data[-1] is line
    # on the exact boundary of the linear classifier
    angle, threshold = fit.rotation_angle[0], fit.threshold[0]
    normal = np.array([np.cos(angle), -np.sin(angle)])
    np.testing.assert_allclose(np.column_stack([line.x, line.y]) @ normal, threshold)


def test_plot_without_fit():
    data = single_shot_data(500)
    figures, table = _plot(data, 0, None)

    assert len(figures) == 1
    assert table == ""


def test_bins_adapt_to_shots():
    rng = np.random.default_rng(0)

    def width(nshots):
        states = [
            rng.normal([0, 0], 1, (nshots, 2)),
            rng.normal([0, 3], 1, (nshots, 2)),
        ]
        edges = _edges(states)
        return edges[1, 0] - edges[0, 0]

    # wider bins with fewer shots, not narrower than a third of the spread
    assert width(200) > width(2_000) > width(20_000)
    assert width(200_000) == pytest.approx(1 / 3, rel=0.05)


def test_linear_classifier():
    rng = np.random.default_rng(3)
    ground = rng.normal([0, 0], 1, (50_000, 2))
    excited = rng.normal([1.2, 1.6], 1, (50_000, 2))
    angle, threshold = linear_classifier(ground, excited)

    # the states axis is rotated to the I axis, and the excited state lies beyond
    assert np.cos(angle) * 1.2 - np.sin(angle) * 1.6 == pytest.approx(2, abs=0.02)
    # equal overlapping Gaussians: the optimal threshold is the midpoint, and the
    # errors are the tail beyond one standard deviation
    assert threshold == pytest.approx(1, abs=0.1)
    assert classify(ground, angle, threshold).mean() == pytest.approx(0.159, abs=0.02)
    assert 1 - classify(excited, angle, threshold).mean() == pytest.approx(
        0.159, abs=0.02
    )


def test_fit_uses_all_shots():
    data = single_shot_data(2_000, seed=4)
    fit = _fit(data)
    ground, excited = data.state_zero(0), data.state_one(0)
    angle, threshold = fit.rotation_angle[0], fit.threshold[0]
    p10 = classify(ground, angle, threshold).mean()
    p01 = 1 - classify(excited, angle, threshold).mean()

    # fidelities evaluated on the same shots used to fit the classifier
    assert fit.fidelity[0] == pytest.approx(1 - p10 - p01)
    assert fit.assignment_fidelity[0] == pytest.approx(1 - (p10 + p01) / 2)
    np.testing.assert_allclose(fit.mean_gnd_states[0], ground.mean(axis=0))


LEGACY_PARAMETERS = {"classifiers_list": ["qubit_fit"], "savedir": " "}


def test_legacy_parameters(caplog):
    params = SingleShotClassificationParameters.load(
        {"nshots": 10, **LEGACY_PARAMETERS}
    )

    assert params.nshots == 10
    assert "classifiers have been removed" in caplog.text
    with pytest.raises(TypeError):
        SingleShotClassificationParameters.load({"unknown": 1})


def test_legacy_data_and_results(tmp_path):
    """Data and results saved with the removed classifiers are still loaded."""
    data = single_shot_data(500)
    fit = _fit(data)
    data.save(tmp_path)
    fit.save(tmp_path)
    for filename, extra in [
        ("data.json", LEGACY_PARAMETERS),
        ("results.json", {"names": ["qubit_fit"], "grid_preds": {}, "x_tests": {}}),
    ]:
        path = tmp_path / filename
        content = json.loads(path.read_text())
        content.update({json.dumps(key): value for key, value in extra.items()})
        path.write_text(json.dumps(content))

    loaded = SingleShotClassificationData.load(tmp_path)
    results = SingleShotClassificationResults.load(tmp_path)
    assert loaded.nshots == 500
    np.testing.assert_array_equal(loaded.data[0], data.data[0])
    assert results.threshold == fit.threshold
    assert 0 in results


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


def test_centers_at_gaussian_means():
    rng = np.random.default_rng(5)
    nshots = 20_000
    ground = rng.normal([0, 0], 0.1, (nshots, 2))
    # a large decay, pulling the plain average towards the ground state
    excited = np.vstack(
        [
            rng.normal([1, 0], 0.1, (int(0.7 * nshots), 2)),
            rng.normal([0, 0], 0.1, (int(0.3 * nshots), 2)),
        ]
    )
    data = SingleShotClassificationData(nshots=nshots, qubit_frequencies={0: 5e9})
    for state, points in enumerate((ground, excited)):
        data.register_qubit(
            ClassificationType,
            0,
            {"i": points[:, 0], "q": points[:, 1], "state": [state] * len(points)},
        )
    fit = _fit(data)
    assert fit.gaussian_mixture(0, consistent=True) is not None
    figure = _plot(data, 0, fit)[0][0]
    excited_marker = next(
        t for t in figure.data if t.name == "state 1" and t.mode == "markers"
    )

    assert fit.mean_exc_states[0][0] == pytest.approx(0.7, abs=0.02)
    assert excited_marker.x[0] == pytest.approx(1, abs=0.01)
    assert excited_marker.y[0] == pytest.approx(0, abs=0.01)
    assert "Gaussian mean" in excited_marker.hovertemplate


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
