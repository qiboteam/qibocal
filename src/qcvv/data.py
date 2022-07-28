# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pint_pandas
import yaml

from qcvv.config import raise_error


class Dataset:
    """First prototype of dataset for calibration routines."""

    def __init__(self, points=100, quantities=None):
        self._data = {
            "MSR": pd.Series([], dtype="pint[V]"),
            "i": pd.Series([], dtype="pint[V]"),
            "q": pd.Series([], dtype="pint[V]"),
            "phase": pd.Series([], dtype="pint[deg]"),
        }

        if quantities is not None:
            if isinstance(quantities, tuple):
                self._data[quantities[0]] = pd.Series(
                    [], dtype=f"pint[{quantities[1]}]"
                )
            elif isinstance(quantities, list):
                for item in quantities:
                    self._data[item[0]] = pd.Series([], dtype=f"pint[{item[1]}]")
            else:
                raise_error(RuntimeError, f"Format of {quantities} is not valid.")
        self.df = pd.DataFrame(self._data)
        self.points = points

    def add(self, *args):
        self.df.loc[len(self), list(self._data.keys())] = args

    def __len__(self):
        return len(self.df)
