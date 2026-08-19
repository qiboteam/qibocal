"""Data processing helpers (classification, whitening, clustering, signal processing)
for qibocal protocols."""

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import HDBSCAN

from .constants import FeatExtractionError
from .physics import euclidean_metric


def compute_qnd(
    ones_first_measure,
    zeros_first_measure,
    ones_second_measure,
    zeros_second_measure,
    pi=False,
) -> tuple[float, list, list]:
    """QND calculation.

    For the standard QND we follow https://arxiv.org/pdf/2106.06173
    for the pi variant we follow https://arxiv.org/pdf/2110.04285

    Returns the QND and the two measurement matrices."""

    p_m1 = np.mean([zeros_first_measure, ones_first_measure], axis=1)
    p_m2 = np.mean([zeros_second_measure, ones_second_measure], axis=1)

    lambda_m = np.stack([1 - p_m1, p_m1])
    lambda_m2 = np.stack([1 - p_m2, p_m2])

    # pinv to avoid tests failing due to singular matrix
    p_o = np.linalg.pinv(lambda_m) @ lambda_m2

    qnd = np.sum(np.diag(p_o)) / 2 if not pi else np.sum(np.diag(p_o[::-1])) / 2
    return qnd, lambda_m.tolist(), lambda_m2.tolist()


def marginalize_qubit_counts(
    counts: Counter[str], qubit_id: Sequence[int] | int
) -> Counter[str]:
    """
    Extract marginal distribution from measurement counts over selected qubit indices.

    Args:
        counts: Counter mapping big-endian bitstrings to counts (e.g. {'0101': 10, ...})
        qubit_id: Qubit ids to marginalize over.

    Returns:
        Counter of the marginal distribution.
    """
    out = Counter()
    indices_list = [qubit_id] if isinstance(qubit_id, int) else qubit_id
    # Indices are the qubit ids. Since results are returned in big-endian format this
    # means that the qubit with id 0 is the rightmost bit in the bitstring, so we need to
    # remap the indices to account for this.
    assert len(set(map(len, counts))) == 1, "All bitstrings must have the same length"
    nqubits = len(next(iter(counts)))
    state_indices = [nqubits - 1 - i for i in indices_list]
    for state, count in counts.items():
        reduced = "".join(state[i] for i in state_indices)
        out[reduced] += count
    return out


def compute_assignment_fidelity(
    one_samples: np.ndarray, zero_samples: np.ndarray
) -> float:
    """Computing assignment fidelity from shots.
    The first argument are the samples when preparing state 1 and the second argument are
    the samples when preparing state 0.
    """

    p_m1_i0 = np.mean(zero_samples)
    p_m1_i1 = np.mean(one_samples)
    p_m0_i1 = 1 - p_m1_i1

    # compute assignment fidelity
    fidelity = 1 - (p_m1_i0 + p_m0_i1) / 2
    return fidelity


def classify(arr: np.ndarray, angle: float, threshold: float) -> np.ndarray:
    """Mapping IQ array in 0s and 1s given angle and threshold."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    rotated = arr @ rot.T
    return (rotated[:, 0] > threshold).astype(int)


def norm(x_mags):
    return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))


def cumulative(input_data, points):
    r"""Evaluates in data the cumulative distribution
    function of `points`.
    """
    return np.searchsorted(np.sort(points), np.sort(input_data))


def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    https://en.wikipedia.org/wiki/Whitening_transformation
    This implementation is analoguous of calling :func:`np.linalg.svd` function and
    multiplying `U` and `Vh` matrices;
    Example for matrix `X`:

    ```python
    U, S, Vh = np.linalg.svd(V)
    ZCA_X = X @ U @ Vh
    ```
    The aforementioned method does not require any regularization term `EPS`, making it formally more correct;
    however the current method is preferred because it scales better with respect to `X` dimensions and
    the relative error scales linear with `EPS`.

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert X.ndim == 2
    EPS = 10e-5

    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1.0 / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)

    X_white = np.dot(X, W)

    return X_white


def minmax_scaling(sig: np.ndarray, axis: int | None) -> np.ndarray:
    """Min–max scaling over a specific axis of the np.ndarray."""
    sig_min = np.min(sig, axis=axis, keepdims=True)
    sig_max = np.max(sig, axis=axis, keepdims=True)
    return (sig - sig_min) / (sig_max - sig_min)


# not used - we can remove
def horizontal_diagonal(xs: np.ndarray, ys: np.ndarray) -> float:
    """Computing the lenght of the diagonal of a two dimensional grid."""
    sizes = np.empty(2)
    for i, values in enumerate([xs, ys]):
        sizes[i] = np.max(values) - np.min(values)
    return np.sqrt((sizes**2).sum())


def build_clustering_data(peaks_dict: dict, z: np.ndarray):
    """Preprocessing of the data to cluster."""
    x_ = peaks_dict["x"]["idx"]
    y_ = peaks_dict["y"]["idx"]
    z_ = z[y_, x_]

    return np.stack((x_, y_, minmax_scaling(z_, axis=None))).T


def peaks_finder(x, y, z) -> dict | None:
    """Function for finding the peaks over the whole signal.

    This function takes as input 3 features of the signal. It slices the dataset along a
    preferred direction (`y` dimension, corresponding to the flux bias) and for each
    slice it determines the biggest peaks by using `scipy.signal.find_peaks` routine.

    If peaks are found, it returns a dictionary `peaks_dict` containing all the features
    for the computed peaks. If no peaks are found returns None.
    """

    # filter data using find_peaks
    peaks = {"x": {"idx": [], "val": []}, "y": {"idx": [], "val": []}}
    for y_idx, y_val in enumerate(y):
        peak, info = find_peaks(z[y_idx], prominence=0.2)
        if len(peak) > 0:
            idx = np.argmax(info["prominences"])
            # if multiple peaks per bias are found, select the one with the highest prominence
            x_idx = peak[idx]
            peaks["x"]["idx"].append(x_idx)
            peaks["x"]["val"].append(x[x_idx])
            peaks["y"]["idx"].append(y_idx)
            peaks["y"]["val"].append(y_val)

    if len(peaks["x"]["idx"]) == 0:
        return None

    return {
        feat: {kind: np.array(vals) for kind, vals in smth.items()}
        for feat, smth in peaks.items()
    }


def merging(
    data: tuple,
    labels: list,
    min_points_per_cluster: int,
    distance_xy: float,
    distance_z: float,
):
    """Divides the processed signal into clusters for separating signal from noise.

    `data` is a 3D tuple of the data to cluster, while `labels` is the classification made by the clustering algorithm;
    `min_points_per_cluster` is the minimum size of points for a cluster to be considered relevant signal.
    It is also possible to set the parameter `distance`, which represents the Euclidean distance between neighboring points of two clusters.
    If this distance is smaller than `distance`, the two clusters are merged.
    It allows a `min_cluster_size=2` in order to decrease as much as possible misclassification of few points.
    The function returns a boolean list corresponding to the indices of the relevant signal.
    """

    # removing data classified as noise
    unique_labels = np.unique(labels[labels >= 0])
    if len(unique_labels) == 0:  # if all points are noise
        """
        Clustering Failed:
        no signal but random noise is found.
        """
        raise FeatExtractionError()

    indices_list = np.arange(len(labels)).astype(int)
    indexed_labels = np.stack((labels, indices_list)).T
    data = np.vstack((data.T, indices_list))

    clusters = [data[:, labels == lab] for lab in unique_labels if lab >= 0]
    noise_points = data[:, labels < 0]

    for i in range(noise_points.shape[1]):
        clusters.append(noise_points[:, i][:, np.newaxis])

    clusters = sorted(
        clusters,
        key=lambda c: np.min(c[1]),
    )

    first = clusters[0]
    first_leftmost = first[:, np.argmin(first[1, :])]
    first_rightmost = first[:, np.argmax(first[1, :])]
    first_label = indexed_labels[first_leftmost[3].astype(int), 0]
    # If the leftmost point is classified as noise (label = -1),
    # we still use it as the initial cluster for the merge step.
    # Its label is reassigned to a unique value;
    # This avoids edge cases where true signal is fused first with one
    # noise points and then since it takes -1 label gets fused with
    # all other unmerged noise points
    if first_label < 0:
        max_lab = np.max(indexed_labels[:, 0]) + 1
        first_label = max_lab
        unique_labels = np.append(unique_labels, max_lab)

    active_clusters = {
        first_label: {
            "cluster": first,
            "leftmost": first_leftmost,
            "rightmost": first_rightmost,
        }
    }

    if len(unique_labels) == 1:
        # only one cluster found
        return active_clusters

    for cluster in clusters[1:]:
        distances_list = []
        indices = []

        for idx in active_clusters:
            cluster_rightmost = cluster[:, np.argmax(cluster[1, :])]
            cluster_leftmost = cluster[:, np.argmin(cluster[1, :])]
            cluster_label = indexed_labels[cluster_leftmost[3].astype(int), 0]

            d_xy = euclidean_metric(
                active_clusters[idx]["rightmost"][:-2], cluster_leftmost[:-2]
            )
            d_z = euclidean_metric(
                active_clusters[idx]["rightmost"][-2], cluster_leftmost[-2]
            )
            if d_xy <= distance_xy and d_z <= distance_z:  # keep the list
                distances_list.append(np.sqrt(d_xy**2 + d_z**2))
                indices.append(idx)

        if len(distances_list) != 0:
            best_dist = np.argmin(distances_list)
            best_idx = indices[best_dist]
            old_cluster = active_clusters[best_idx]["cluster"]
            updated_cluster = np.hstack((old_cluster, cluster))
            active_clusters[best_idx]["cluster"] = updated_cluster
            active_clusters[best_idx]["rightmost"] = updated_cluster[
                :, np.argmax(updated_cluster[1, :])
            ]
        else:
            if cluster_label < 0:
                cluster_label = np.max(unique_labels) + 1
                unique_labels = np.append(unique_labels, cluster_label)

            active_clusters[cluster_label] = {
                "cluster": cluster,
                "leftmost": cluster_leftmost,
                "rightmost": cluster_rightmost,
            }

    valid_clusters = {
        lab: v_clust
        for lab, v_clust in active_clusters.items()
        if v_clust["cluster"].shape[1] >= min_points_per_cluster
    }

    # since we allowed for clustering even a group of 2 points, we filter the allowed eligible clusters
    # to be at least composed by a minimum number of points given by min_points_per_cluster parameter
    if len(valid_clusters.keys()) == 0:  # if no big enough clusters are found
        """
        Clustering Failed:
        not enough big clusters after merging routine.
        """
        raise FeatExtractionError()

    return valid_clusters


def clustering(peaks_dict, z_masked):
    """In this function Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm is used;
    HDBSCAN is a good algorithm for successfully capture clusters with different densities.
    """

    # normalizing peaks for clustering
    peaks = build_clustering_data(peaks_dict, z_masked)

    # clustering
    hdb = HDBSCAN(copy=True, min_cluster_size=2)
    hdb.fit(peaks)
    labels = hdb.labels_

    return peaks, labels


def reshaping_raw_signal(x, y, z):
    x_ = np.unique(x)
    y_ = np.unique(y)
    # background removed over y axis
    z_ = z.reshape(len(y_), len(x_))

    return x_, y_, z_
