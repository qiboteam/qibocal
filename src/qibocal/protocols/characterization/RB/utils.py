from math import log10
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
from pandas import DataFrame

from qibocal.config import raise_error


def ci_to_str(value, confidence=None, precision=None):
    """"""
    if confidence is None:
        precision = precision if precision is not None else 3
        return f"{value:.{precision}f}"

    if isinstance(confidence, float):
        if confidence < 0:
            raise_error(
                ValueError,
                f"`confidence` cannot be negative. Got {confidence} instead.",
            )
        if confidence == 0:
            precision = precision if precision is not None else 3
            return f"{value:.{precision}f}"
        if precision is None:
            precision = max(-int(log10(confidence)), 0) + 1
        return f"{value:.{precision}f} \u00B1 {confidence:.{precision}f}"

    if isinstance(confidence, Iterable) is False:
        raise_error(
            TypeError,
            f"`confidence` must be iterable or a number. Got {type(confidence)} instead.",
        )
    if len(confidence) != 2:
        raise_error(
            ValueError,
            f"`confidence` list must contain 2 elements. Got {len(confidence)} instead.",
        )
    if any(confidence < 0):
        raise_error(
            ValueError,
            f"`confidence` values cannot be negative. Got {confidence} instead.",
        )

    if precision is None:
        precision = max(-int(log10(confidence[0])), -int(log10(confidence[1])), 0) + 1
    if all(c == 0 for c in confidence):
        precision = precision if precision is not None else 3
        return f"{value:.{precision}f}"
    if abs(confidence[0] - confidence[1]) < 10 ** (-precision):
        return f"{value:.{precision}f} \u00B1 {confidence[0]:.{precision}f}"
    return f"{value:.{precision}f} +{confidence[1]:.{precision}f} / -{confidence[0]:.{precision}f}"


def extract_from_data(
    data: Union[List[dict], DataFrame],
    output_key: str,
    groupby_key: str = "",
    agg_type: Union[str, Callable] = "",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return wanted values from list of dictionaries via a dataframe and its properties.

    If ``groupby_key`` given, aggregate the dataframe, extract the data by which the frame was
    grouped, what was calculated given the ``agg_type`` parameter. Two arrays are returned then,
    the group values and the grouped (aggregated) data. If no ``agg_type`` given use a linear function.
    If ``groupby_key`` not given, only return the extracted data from given key.

    Args:
        output_key (str): Key name of the wanted output.
        groupby_key (str): If given, group with that key name.
        agg_type (str): If given, calcuted aggregation function on groups.

    Returns:
        Either one or two np.ndarrays. If no grouping wanted, just the data. If grouping
        wanted, the values after which where grouped and the grouped data.
    """
    if isinstance(data, list):
        data = DataFrame(data)
    # Check what parameters where given.
    if not groupby_key and not agg_type:
        # No grouping and no aggreagtion is wanted. Just return the wanted output key.
        return np.array(data[output_key].tolist())
    elif not groupby_key and agg_type:
        # No grouping wanted, just an aggregational task on all the data.
        return data[output_key].apply(agg_type)
    elif groupby_key and not agg_type:
        df = data.get([output_key, groupby_key])
        # Sort by the output key for making reshaping consistent.
        df.sort_values(by=output_key)
        # Grouping is wanted but no aggregation, use a linear function.
        grouped_df = df.groupby(groupby_key, group_keys=True).apply(lambda x: x)
        return grouped_df[groupby_key].to_list(), grouped_df[output_key].to_list()
    else:
        df = data.get([output_key, groupby_key])
        grouped_df = df.groupby(groupby_key, group_keys=True).agg(agg_type)
        return grouped_df.index.to_list(), grouped_df[output_key].to_list()
