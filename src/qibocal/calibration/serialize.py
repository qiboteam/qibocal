import base64
import io
from typing import Annotated, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import PlainSerializer, PlainValidator
from scipy.sparse import csr_matrix, lil_matrix


# TODO: add tests about this
def sparse_serialize(matrix: lil_matrix) -> str:
    """Serialize a lil_matrix to a base64 string."""
    csr_matrix = matrix.tocsr()
    buffer = io.BytesIO()
    np.save(buffer, csr_matrix.shape)
    np.save(buffer, csr_matrix.data)
    np.save(buffer, csr_matrix.indices)
    np.save(buffer, csr_matrix.indptr)
    buffer.seek(0)
    return base64.standard_b64encode(buffer.read()).decode()


def sparse_deserialize(data: str) -> Optional[lil_matrix]:
    """Deserialize a base64 string back into a lil_matrix."""
    buffer = io.BytesIO(base64.standard_b64decode(data))
    shape = np.load(buffer, allow_pickle=True)
    data_array = np.load(buffer, allow_pickle=True)
    indices_array = np.load(buffer, allow_pickle=True)
    indptr_array = np.load(buffer, allow_pickle=True)
    csr = csr_matrix((data_array, indices_array, indptr_array), shape=shape)
    return lil_matrix(csr)


SparseArray = Annotated[
    lil_matrix,
    PlainValidator(sparse_deserialize),
    PlainSerializer(sparse_serialize, return_type=str),
]


def ndarray_serialize(ar: npt.NDArray) -> str:
    """Serialize array to string."""
    buffer = io.BytesIO()
    np.save(buffer, ar)
    buffer.seek(0)
    return base64.standard_b64encode(buffer.read()).decode()


def ndarray_deserialize(x: Union[str, npt.NDArray]) -> npt.NDArray:
    """Deserialize array."""
    buffer = io.BytesIO()
    buffer.write(base64.standard_b64decode(x))
    buffer.seek(0)
    return np.load(buffer)


NdArray = Annotated[
    npt.NDArray,
    PlainValidator(ndarray_deserialize),
    PlainSerializer(ndarray_serialize, return_type=str),
]
"""Pydantic-compatible array representation."""
