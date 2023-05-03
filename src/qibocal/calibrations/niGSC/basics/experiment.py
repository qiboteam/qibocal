from __future__ import annotations

import pickle
from collections.abc import Iterable
from os.path import isfile
from typing import Callable

import numpy as np
import pandas as pd
from qibo.models import Circuit
from qibo.noise import NoiseModel

from qibocal.cli.utils import generate_output_folder
from qibocal.config import raise_error


class Experiment:
    """Experiment objects which holds an iterable circuit factory along with
    a simple data structure associated to each circuit.

    Args:
        circuitfactory (Iterable): Gives a certain amount of circuits when
            iterated over.
        nshots (int): For execution of circuit, indicates how many shots.
        data (list): If filled, ``data`` can be used to specifying parameters
                     while executing a circuit or deciding how to process results.
                     It is used to store all relevant data.
    """

    def __init__(
        self,
        circuitfactory: Iterable | None,
        data: Iterable | None = None,
        nshots: int | None = 128,
        noise_model: NoiseModel | None = None,
    ) -> None:
        """ """

        if circuitfactory is not None and not isinstance(circuitfactory, Iterable):
            raise_error(
                TypeError,
                f"Invalid type {type(circuitfactory)} for CircuitFactory,  must be Iterable or None.",
            )
        self.circuitfactory = circuitfactory
        if data is not None and not isinstance(data, Iterable):
            raise_error(
                TypeError,
                f"Invalid data type  {type(data)}. Data must be Iterable or None.",
            )
        self.data = data
        if nshots is not None and not isinstance(nshots, int):
            raise_error(
                TypeError,
                f"Invalid nshots type {type(nshots)}. Nshots must be int or None.",
            )
        self.nshots = nshots
        if noise_model is not None and not isinstance(noise_model, NoiseModel):
            raise_error(
                TypeError,
                f"NoiseModel has wrong type {type(noise_model)}. NoiseModel must be qibo NoiseModel or None .",
            )
        self.__noise_model = noise_model
        self.name = "Abstract"

    @property
    def noise_model(self):
        return self.__noise_model

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    @classmethod
    def load(cls, path: str) -> Experiment:
        """Creates an experiment object with data and if possible with circuits.

        Args:
            path (str): The directory from where the object should be restored.

        Returns:
            Experiment: The object with data (and circuitfactory).
        """
        datapath = f"{path}/experiment_data.pkl"
        circuitspath = f"{path}/circuits.pkl"
        if isfile(datapath):
            with open(datapath, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame):  # pragma: no cover
                data = data.to_dict("records")
            nshots = len(data[0]["samples"])
        else:
            data, nshots = None, None
        if isfile(circuitspath):
            with open(circuitspath, "rb") as f:
                circuitfactory = pickle.load(f)
        else:
            circuitfactory = None
        # Initiate an instance of the experiment class.
        obj = cls(circuitfactory, data=data, nshots=nshots)
        return obj

    def save_circuits(self, path: str | None = None, force: bool = False) -> str:
        """Creates a path if None given and pickles ``self.circuitfactory``
        if it is a list.

        Returns:
            (str): The path of stored experiment.
        """

        # Check if path to store is given, if not create one.
        if path is None:
            self.path = generate_output_folder(path, force)
        else:
            self.path = path
        if isinstance(self.circuitfactory, list):
            with open(f"{self.path}/circuits.pkl", "wb") as f:
                pickle.dump(self.circuitfactory, f)
        return self.path

    def save(self, path: str | None = None, force: bool = False) -> str:
        """Creates a path if None given and pickles relevant data from ``self.data``.

        Returns:
            (str): The path of stored experiment.
        """

        # Check if path to store is given, if not create one.
        if path is None:
            self.path = generate_output_folder(path, force)
        else:
            self.path = path
        # And only if data is not None the data list (full of dicionaries) will be
        # stored.
        if self.data is not None:
            with open(f"{self.path}/experiment_data.pkl", "wb") as f:
                pickle.dump(self.data, f)
        # It is convenient to know the path after storing, so return it.
        return self.path

    def extract(
        self, output_key: str, groupby_key: str = "", agg_type: str | Callable = ""
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return wanted values from ``self.data`` via the dataframe property.

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

        # Check what parameters where given.
        if not groupby_key and not agg_type:
            # No grouping and no aggreagtion is wanted. Just return the wanted output key.
            return np.array(self.dataframe[output_key].tolist())
        elif not groupby_key and agg_type:
            # No grouping wanted, just an aggregational task on all the data.
            return self.dataframe[output_key].apply(agg_type)
        elif groupby_key and not agg_type:
            # Grouping is wanted but no aggregation, use a linear function.
            df = self.dataframe.get([output_key, groupby_key])
            grouped_df = df.groupby(groupby_key, group_keys=True).apply(lambda x: x)
            return grouped_df[groupby_key].to_numpy(), grouped_df[output_key].to_numpy()
        else:
            df = self.dataframe.get([output_key, groupby_key])
            grouped_df = df.groupby(groupby_key, group_keys=True).apply(agg_type)
            return grouped_df.index.to_numpy(), grouped_df[output_key].to_numpy()

    def prebuild(self) -> None:
        """Converts the attribute ``circuitfactory`` which is in general
        an iterable into a list.
        """
        self.circuitfactory = list(self.circuitfactory)

    def perform(self, sequential_task: Callable[[Circuit, dict], dict]) -> None:
        """Takes a given function, checks the status of attribute ``circuitfactory``
        and ``data`` and executes the sequential function row by row altering the
        ``self.data`` attribute.

        Either ``self.circuitfactory`` or ``self.data`` cannot be ``None`` and
        if not ``None`` they have to have the right length.

        Args:
            sequential_task (callable[[Circuit, dict], dict]): A function applied
                row by row alterting each datarow.
        """
        # Either the circuit factory or the data rows can be empty.
        # If ``self.data`` is not empty the actual list element is altered without
        # storing it after alternation.
        # Both ``circuit`` and ``datarow`` can be provided:
        if self.circuitfactory is not None and self.data is not None:
            for circuit, datarow in zip(self.circuitfactory, self.data):
                datarow = sequential_task(circuit.copy(deep=True), datarow)
        # Only``datarow`` can be provided:
        elif self.circuitfactory is None and self.data is not None:
            for datarow in self.data:
                datarow = sequential_task(None, datarow)
        # Only ``circuit`` can be provided:
        elif self.circuitfactory is not None and self.data is None:
            newdata = []
            for circuit in self.circuitfactory:
                newdata.append(sequential_task(circuit.copy(deep=True), {}))
            self.data = newdata
        else:
            raise_error(ValueError, "Both attributes circuitfactory and data are None.")

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        """Executes a circuit, returns the single shot results in a dict.

        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and immediate
                postprocessing information.
        """

        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        samples = circuit(nshots=self.nshots).samples()
        return {"samples": samples}
