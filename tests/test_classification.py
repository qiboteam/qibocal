import json

import numpy as np
import plotly.graph_objects as go
import pytest

from qibocal.protocols.classification.classification import (
    ClassificationType,
    SingleShotClassificationData,
    SingleShotClassificationParameters,
    SingleShotClassificationResults,
    _fit,
    _plot,
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
