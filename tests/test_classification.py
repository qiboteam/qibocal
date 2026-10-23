import numpy as np
import plotly.graph_objects as go
import pytest

from qibocal.protocols.classification.classification import (
    ClassificationType,
    SingleShotClassificationData,
    _fit,
    _plot,
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
