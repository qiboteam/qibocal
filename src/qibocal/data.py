# -*- coding: utf-8 -*-
"""Implementation of Dataset class to store measurements."""

import re
from abc import abstractmethod

import numpy as np
import pandas as pd
import pint_pandas

from qibocal.config import raise_error


class AbstractDataset:
    def __init__(self, name=None):

        if name is None:
            self.name = "data"
        else:
            self.name = name

        self.df = pd.DataFrame()
        self.quantities = None

    def __add__(self, data):
        self.df = pd.concat([self.df, data.df], ignore_index=True)
        return self

    @abstractmethod
    def add(self, data):
        raise_error(NotImplementedError)

    def __len__(self):
        """Computes the length of the dataset."""
        return len(self.df)

    @classmethod
    def load_data(cls, folder, routine, format, name):
        raise_error(NotImplementedError)

    @abstractmethod
    def to_csv(self, path):
        """Save data in csv file.

        Args:
            path (str): Path containing output folder."""
        if self.quantities == None:
            self.df.to_csv(f"{path}/{self.name}.csv")
        else:
            self.df.pint.dequantify().to_csv(f"{path}/{self.name}.csv")

    def to_pickle(self, path):
        """Save data in pickel file.

        Args:
            path (str): Path containing output folder."""
        self.df.to_pickle(f"{path}/{self.name}.pkl")


class Dataset(AbstractDataset):
    """Class to store the data measured during the calibration routines.
    It is a wrapper to a pandas DataFrame with units of measure from the Pint
    library.

    Args:
        quantities (dict): dictionary containing additional quantities that the user
                        may save other than the pulse sequence output. The keys are the name of the
                        quantities and the corresponding values are the units of measure.
        options (list): list containing additional values to be saved.
    """

    def __init__(self, name=None, quantities=None, options=None):

        super().__init__(name=name)

        self.df = pd.DataFrame(
            {
                "MSR": pd.Series(dtype="pint[V]"),
                "i": pd.Series(dtype="pint[V]"),
                "q": pd.Series(dtype="pint[V]"),
                "phase": pd.Series(dtype="pint[deg]"),
            }
        )
        self.quantities = {"MSR": "V", "i": "V", "q": "V", "phase": "rad"}
        self.options = []

        if quantities is not None:
            self.quantities.update(quantities)
            for name, unit in quantities.items():
                self.df.insert(0, name, pd.Series(dtype=f"pint[{unit}]"))

        if options is not None:
            self.options = options
            for option in options:
                self.df.insert(0, option, pd.Series(dtype=object))

        from pint import UnitRegistry

        self.ureg = UnitRegistry()

    def add(self, data):
        """Add a row to dataset.

        Args:
            data (dict): dictionary containing the data to be added.
                        Every key should have the following form:
                        ``<name>[<unit>]``.
        """
        l = len(self)

        for key, value in data.items():
            if "[" in key:
                name = key.split("[")[0]
                unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
                # TODO: find a better way to do this
                self.df.loc[l, name] = np.array(value) * self.ureg(unit)
            else:
                self.df.loc[l, key] = value

    def set(self, data):
        """Add a row to dataset.

        Args:
            data (dict): dictionary containing the data to be added.
                        Every key should have the following form:
                        ``<name>[<unit>]``.
        """
        processed_data = {}
        for key, values in data.items():
            if "[" in key:
                name = key.split("[")[0]
                unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
                processed_data[name] = pd.Series(data=(np.array(values) * self.ureg(unit)), dtype=f"pint[{unit}]")
            else:
                processed_data[name] = pd.Series(data=(values), dtype=object)
        self.df = pd.DataFrame(processed_data)

    def get_values(self, key, unit=None):
        """Get values of a quantity in specified units.

        Args:
            quantity (str): Quantity to get the values of.
            unit (str): Unit of the returned values.

        Returns:
            ``pd.Series`` with the quantity values in the given units.
        """
        if unit is None:
            return self.df[key]
        else:
            return self.df[key].pint.to(unit).pint.magnitude

    @classmethod
    def load_data(cls, folder, routine, format, name):
        """Load data from specific format.

        Args:
            folder (path): path to the output folder from which the data will be loaded
            routine (str): calibration routine data to be loaded
            format (str): data format. Possible choices are 'csv' and 'pickle'.

        Returns:
            dataset (``Dataset``): dataset object with the loaded data.
        """
        obj = cls()
        if format == "csv":
            file = f"{folder}/data/{routine}/{name}.csv"
            obj.df = pd.read_csv(file, header=[0, 1])
            obj.df.pop("Unnamed: 0_level_0")
            quantities_label = []
            obj.options = []
            for column in obj.df.columns:
                if "Unnamed" not in column[1]:
                    quantities_label.append(column[0])
                else:
                    obj.options.append(column[0])
            quantities_df = obj.df[quantities_label].pint.quantify()
            options_df = obj.df[obj.options]
            options_df.columns = options_df.columns.droplevel(1)
            obj.df = pd.concat([quantities_df, options_df], axis=1)
        elif format == "pickle":
            file = f"{folder}/data/{routine}/{name}.pkl"
            obj.df = pd.read_pickle(file)
        else:
            raise_error(ValueError, f"Cannot load data using {format} format.")

        return obj

    def to_csv(self, path):
        """Save data in csv file.

        Args:
            path (str): Path containing output folder."""
        df = self.df[list(self.quantities)].pint.dequantify()
        firsts = df.index.get_level_values(None)
        df[self.options] = self.df[self.options].loc[firsts].values
        df.to_csv(f"{path}/{self.name}.csv")


class Data(AbstractDataset):
    """Class to store the data obtained from calibration routines.
    It is a wrapper to a pandas DataFrame.

    Args:
        quantities (dict): dictionary quantities to be saved.
    """

    def __init__(self, name=None, quantities=None):

        super().__init__(name=name)

        if quantities is not None:
            self.quantities = quantities
            for name in quantities:
                self.df.insert(0, name, pd.Series(dtype=object))

    def add(self, data):
        """Add a row to dataset.

        Args:
            data (dict): dictionary containing the data to be added.
                        Every key should have the following form:
                        ``<name>[<unit>]``.
        """
        l = len(self)
        for key, value in data.items():
            self.df.loc[l, key] = value

    def get_values(self, quantity):
        """Get values of a quantity in specified units.

        Args:
            quantity (str): Quantity to get the values of.

        Returns:
            ``pd.Series`` with the quantity values in the given units.
        """
        return self.df[quantity].values

    @classmethod
    def load_data(cls, folder, routine, format, name):
        """Load data from specific format.

        Args:
            folder (path): path to the output folder from which the data will be loaded
            routine (str): calibration routine data to be loaded
            format (str): data format. Possible choices are 'csv' and 'pickle'.

        Returns:
            dataset (``Dataset``): dataset object with the loaded data.
        """
        obj = cls()
        if format == "csv":
            file = f"{folder}/data/{routine}/{name}.csv"
            obj.df = pd.read_csv(file)
            obj.df.pop("Unnamed: 0")
        elif format == "pickle":
            file = f"{folder}/data/{routine}/{name}.pkl"
            obj.df = pd.read_pickle(file)
        else:
            raise_error(ValueError, f"Cannot load data using {format} format.")

        return obj

    def to_csv(self, path):
        """Save data in csv file.

        Args:
            path (str): Path containing output folder."""
        self.df.to_csv(f"{path}/{self.name}.csv")

    def to_pickle(self, path):
        """Save data in pickel file.

        Args:
            path (str): Path containing output folder."""
        self.df.to_pickle(f"{path}/{self.name}.pkl")
