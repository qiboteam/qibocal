# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pint_pandas
import yaml

from qcvv.config import raise_error


class Dataset:
    """First prototype of dataset for calibration routines."""

    def __init__(self, points=100, quantities=None):

        self.points = points
        self.df = pd.DataFrame(
            {
                "MSR": pd.Series(dtype="pint[V]"),
                "i": pd.Series(dtype="pint[V]"),
                "q": pd.Series(dtype="pint[V]"),
                "phase": pd.Series(dtype="pint[deg]"),
            }
        )

        if quantities is not None:
            for name, unit in quantities.items():
                self.df.insert(0, name, pd.Series(dtype=f"pint[{unit}]"))

    def add(self, data):
        import re

        from pint import UnitRegistry

        ureg = UnitRegistry()
        l = len(self)
        for key, value in data.items():
            name = key.split("[")[0]
            unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
            # TODO: find a better way to do this
            self.df.loc[l + l // len(list(data.keys())), name] = value * ureg(unit)

    def __len__(self):
        return len(self.df)
