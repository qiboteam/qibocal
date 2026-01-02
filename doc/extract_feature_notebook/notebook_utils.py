from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from sklearn.cluster import HDBSCAN

DISTANCE = 5


def euclidean_metric(point1: np.ndarray, point2: np.ndarray):
    """Euclidean distance between two arrays."""
    return np.linalg.norm(point1 - point2)


def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    https://en.wikipedia.org/wiki/Whitening_transformation

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


def filter_data(matrix_z: np.ndarray):
    """Filter data with a ZCA transformation and then a unit-variance Gaussian."""

    # adding zca filter for filtering out background noise gradient
    zca_z = zca_whiten(matrix_z)
    # adding gaussian fliter with unitary variance for blurring the signal and reducing noise
    return ndimage.gaussian_filter(zca_z, 1)


def scaling_global(sig: np.ndarray) -> np.ndarray:
    """Min–max scaling over the whole np.ndarray (global)."""
    return scaling_slice(sig, axis=None)


def scaling_slice(sig: np.ndarray, axis: Optional[int]) -> np.ndarray:
    """Min–max scaling over a specific axis of the np.ndarray."""

    def expand(a):
        return np.expand_dims(a, axis) if axis is not None else a

    sig_min = expand(np.min(sig, axis=axis))
    return (sig - sig_min) / (expand(np.max(sig, axis=axis)) - sig_min)


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

    rescaling_fact = np.sqrt(2)
    return np.stack((x_, y_, scaling_global(z_) * rescaling_fact)).T


def peaks_finder(x, y, z) -> dict:
    """Function for finding the peaks over the whole signal.

    This function takes as input 3 features of the signal.
    It slices the dataset along a preferred direction (`y` dimension, corresponding to the flux bias) and for each slice it determines the biggest peaks
    by using `scipy.signal.find_peaks` routine.
    It returns a dictionary `peaks_dict` containing all the features for the computed peaks.
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

    return {
        feat: {kind: np.array(vals) for kind, vals in smth.items()}
        for feat, smth in peaks.items()
    }


def merging(
    data: tuple, labels: list, min_points_per_cluster: int, distance: float = 5.0
) -> list[bool]:
    """Divides the processed signal into clusters for separating signal from noise.

    `data` is a 3D tuple of the data to cluster, while `labels` is the classification made by the clustering algorithm;
    `min_points_per_cluster` is the minimum size of points for a cluster to be considered relevant signal.
    It is also possible to set the parameter `distance`, which represents the Euclidean distance between neighboring points of two clusters.
    If this distance is smaller than `distance`, the two clusters are merged.
    It allows a `min_cluster_size=2` in order to decrease as much as possible misclassification of few points.
    The function returns a boolean list corresponding to the indices of the relevant signal.
    """

    unique_labels = np.unique(labels)

    indices_list = np.arange(len(labels)).astype(int)
    indexed_labels = np.stack((labels, indices_list)).T
    data = np.vstack((data.T, indices_list))

    clusters = sorted(
        [data[:, labels == lab] for lab in unique_labels],
        key=lambda c: np.min(c[1]),
    )

    first = clusters[0]
    first_leftmost = first[:, np.argmin(first[1, :])]
    first_rightmost = first[:, np.argmax(first[1, :])]
    first_label = indexed_labels[first_leftmost[3].astype(int), 0]

    active_clusters = {
        first_label: {
            "cluster": first,
            "leftmost": first_leftmost,
            "rightmost": first_rightmost,
        }
    }

    for cluster in clusters[1:]:
        threshold = distance
        distances_list = []
        indices = []

        for idx in active_clusters.keys():
            cluster_rightmost = cluster[:, np.argmax(cluster[1, :])]
            cluster_leftmost = cluster[:, np.argmin(cluster[1, :])]
            cluster_label = indexed_labels[cluster_leftmost[3].astype(int), 0]

            d = euclidean_metric(
                active_clusters[idx]["rightmost"][:-1], cluster_leftmost[:-1]
            )
            if d <= threshold:  # keep the list
                distances_list.append(d)
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

    medians = np.array(
        [[lab, np.median(cl["cluster"][2, :])] for lab, cl in valid_clusters.items()]
    )
    # we only take the first three values of each point in the cluster because they correspond to the 3 features (x,y,z)

    signal_labels = np.zeros(indices_list.size, dtype=bool)
    if len(medians) != 0:
        signal_idx = medians[np.argmax(medians[:, 1]), 0]
        signal_labels[valid_clusters[signal_idx]["cluster"][-1, :].astype(int)] = True

    return signal_labels


def extract_feature(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, find_min: bool, min_points: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features of the signal by filtering out background noise.

    It first applies a custom filter mask (see `custom_filter_mask`)
    and then finds the biggest peak for each DC bias value;
    the masked signal is then clustered (see `clustering`) in order to classify the relevant signal for the experiment.
    If `find_min` is set to `True` it finds minimum peaks of the input signal;
    `min_points` is the minimum number of points for a cluster to be considered relevant signal.
    Position of the relevant signal is returned.
    """

    x_ = np.unique(x)
    y_ = np.unique(y)
    # background removed over y axis
    z_ = z.reshape(len(y_), len(x_))

    z_ = -z_ if find_min else z_

    # masking
    z_masked = filter_data(z_)

    # renormalizing
    # z_masked_norm = scaling_signal(z_masked)
    z_masked_norm = scaling_slice(z_masked, axis=1)

    # filter data using find_peaks
    peaks_dict = peaks_finder(x_, y_, z_masked_norm)

    # normalizing peaks for clustering
    peaks = build_clustering_data(peaks_dict, z_masked)

    # clustering
    # In this function Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm is used;
    # HDBSCAN good for successfully capture clusters with different densities.
    hdb = HDBSCAN(copy=True, min_cluster_size=2)
    hdb.fit(peaks)
    labels = hdb.labels_

    # merging close clusters
    signal_classification = merging(peaks, labels, min_points, DISTANCE)

    return peaks_dict["x"]["val"][signal_classification], peaks_dict["y"]["val"][
        signal_classification
    ]
