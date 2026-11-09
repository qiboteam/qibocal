import numpy as np
import plotly.graph_objects as go
import pytest

from qibocal.protocols.classification.classification import (
    ClassificationType,
    SingleShotClassificationData,
    _fit,
    _plot,
)
from qibocal.protocols.classification.plot import _density, _edges

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


def boundaries(figure):
    return [trace for trace in figure.data if (trace.name or "").endswith("boundary")]


@pytest.mark.parametrize("classifiers", [["qubit_fit"], ["qubit_fit", "qblox_fit"]])
def test_plot_size_independent_of_shots(classifiers):
    sizes = []
    for nshots in (500, 20_000):
        data = single_shot_data(nshots, classifiers)
        figures, _ = _plot(data, 0, _fit(data))
        (density,) = [t for t in figures[0].data if isinstance(t, go.Image)]
        sizes.append(np.size(density.z))
        assert len(boundaries(figures[0])) == len(classifiers)
        assert len(figures) == (2 if len(classifiers) == 1 else 4)
        assert all(figure.layout.title.text is None for figure in figures)
    # densities are binned: the image does not scale with the shots
    assert sizes[1] < 2 * sizes[0]


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
    # on the exact boundary of the qubit_fit classifier
    angle, threshold = fit.rotation_angle[0], fit.threshold[0]
    normal = np.array([np.cos(angle), -np.sin(angle)])
    np.testing.assert_allclose(
        np.column_stack([line.x, line.y]) @ normal, threshold, atol=2e-3
    )


def test_plot_without_fit():
    data = single_shot_data(500)
    figures, table = _plot(data, 0, None)

    assert len(figures) == 1
    assert table == ""
