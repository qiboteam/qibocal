from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data

# RBType = np.dtype(
#     [("depth", np.int64), ("signal", np.float64),]
# )
# """Custom dtype for RB."""

RBType = np.dtype(
    [
        ("signal", np.float64),
    ]
)
"""Custom dtype for RB."""


@dataclass
class RBData(Data):
    """A pandas DataFrame bastard child. The output of the acquisition function."""

    params: dict
    # depths: dict
    data: dict[QubitId, npt.NDArray[RBType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, dtype, data_keys, data_dict):
        """Store output for single qubit.

        Args:
            data_keys (tuple): Keys of Data.data.
            data_dict (dict): The keys are the fields of `dtype` and
            the values are the related arrays.
        """
        # to be able to handle the non-sweeper case
        ar = np.empty(np.shape(data_dict[list(data_dict.keys())[0]]), dtype=dtype)
        for key, value in data_dict.items():
            ar[key] = value

        if data_keys in self.data:
            # FIXME: Let's work directly with a list for now
            self.data[data_keys] = np.append(self.data[data_keys], data_dict["signal"])
        else:
            # print("here")
            self.data[data_keys] = data_dict["signal"]
