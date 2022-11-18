from __future__ import annotations

import pickle
from collections.abc import Iterable
from copy import deepcopy
from itertools import product
from os.path import isfile

import numpy as np
import pandas as pd
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

from qibocal.calibrations.protocols.utils import (
    ONEQUBIT_CLIFFORD_PARAMS,
    experiment_directory,
)


class Circuitfactory:
    """TODO write documentation
    TODO make the embedding into lager qubit space possible"""

    def __init__(self, qubits: list, depths: list, runs: int) -> None:
        self.qubits = qubits
        self.depths = depths
        self.runs = runs
        self.embed_func = None
        # TODO implement a test for that function?

    def __iter__(self) -> None:
        self.n = 0
        return self

    def __next__(self) -> None:
        if self.n >= self.runs * len(self.depths):
            raise StopIteration
        else:
            circuit = self.build_circuit(self.depths[self.n % len(self.depths)])
            self.n += 1
            if self.embed_func is not None:
                return self.embed_func(circuit)
            else:
                return circuit

    def build_circuit(self, depth: int):
        pass


class SingleCliffordsFactory(Circuitfactory):
    def __init__(self, qubits: list, depths: list, runs: int) -> None:
        super().__init__(qubits, depths, runs)

    def circuit_generator(self):
        for _ in range(self.runs):
            for depth in self.depths:
                yield self.build_circuit(depth)

    def build_circuit(self, depth: int):
        circuit = Circuit(len(self.qubits))
        for _ in range(depth):
            circuit.add(self.gate())
        circuit.add(gates.M(*self.qubits))
        return circuit

    def clifford_unitary(
        self, theta: float = 0, nx: float = 0, ny: float = 0, nz: float = 0
    ) -> np.ndarray:
        """Four given parameters are used to build one Clifford unitary.

        Args:
            theta (float) : An angle
            nx (float) : prefactor
            ny (float) : prefactor
            nz (float) : prefactor

        Returns:
            ``qibo.gates.Unitary`` with the drawn matrix as unitary.
        """
        matrix = np.array(
            [
                [
                    np.cos(theta / 2) - 1.0j * nz * np.sin(theta / 2),
                    -ny * np.sin(theta / 2) - 1.0j * nx * np.sin(theta / 2),
                ],
                [
                    ny * np.sin(theta / 2) - 1.0j * nx * np.sin(theta / 2),
                    np.cos(theta / 2) + 1.0j * nz * np.sin(theta / 2),
                ],
            ]
        )
        return matrix

    def gate(self) -> gates.Unitary:
        """Draws the parameters and builds the gate.

        Args:
            seed (int): optional, set the starting seed for random
                        number generator.
        Returns:
            ``qibo.gates.Unitary``: the simulatanous Clifford gates
        """
        # There are this many different Clifford matrices.
        amount = len(ONEQUBIT_CLIFFORD_PARAMS)
        # Initiate the matrix to start the kronecker (tensor) product.
        unitary = self.clifford_unitary(
            *ONEQUBIT_CLIFFORD_PARAMS[np.random.randint(amount)]
        )
        # Choose as many random integers between 0 and 23 as there are used
        # qubits. Get the clifford parameters and build the unitary.
        for rint in np.random.randint(0, amount, size=len(self.qubits) - 1):
            # Build the random Clifford matrix and take the tensor product
            # with the matrix before.
            unitary = np.kron(
                self.clifford_unitary(*ONEQUBIT_CLIFFORD_PARAMS[rint]), unitary
            )
        # Make a unitary gate out of 'unitary' for the qubits.
        return gates.Unitary(unitary, *self.qubits)


class Experiment:
    """Experiment objects which holds an iterable circuit factory along with
    a simple data structure associated to each circuit.

    Args:
        circuitfactory (Iterable): Gives a certain amount of circuits when
            iterated over.
        data (list): If filled ``data`` can be used to specifying parameters
            while executing a circuit or deciding how to process results.
        nshots (int): For execution of circuit, indicates how many shots.
    """

    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
    ) -> None:
        """ """
        self.circuitfactory = circuitfactory
        self.nshots = nshots
        self.data = data
        self.__noise_model = noisemodel

    @classmethod
    def load(cls, path: str) -> Experiment:
        """Creates an object with data and if possible with circuits.

        Args:
            path (str): The directory from where the object should be restored.

        Returns:
            Experiment: The object with data (and circuitfactory).
        """
        datapath = f"{path}data.pkl"
        circuitspath = f"{path}circuits.pkl"
        if isfile(datapath):
            with open(datapath, "rb") as f:
                data = pickle.load(f)
            nshots = len(data[0]["samples"])
        else:
            data = None
        if isfile(circuitspath):
            with open(circuitspath, "rb") as f:
                circuitfactory = pickle.load(f)
        else:
            circuitfactory = None
        # Initiate an instance of the experiment class.
        obj = cls(circuitfactory, data=data, nshots=nshots)
        return obj

    def prebuild(self) -> None:
        """Converts the attribute ``circuitfactory`` which is in general
        an iterable into a list.
        """
        self.circuitfactory = list(self.circuitfactory)

    def execute(self) -> None:
        """Calls method ``single_task`` while iterating over attribute
        ``circuitfactory```.

        Collects data given the already set data and overwrites
        attribute ``data``.
        """
        if self.circuitfactory is None:
            raise NotImplementedError("There are no circuits to execute.")
        newdata = []
        for circuit in self.circuitfactory:
            try:
                datarow = next(self.data)
            except TypeError:
                datarow = {}
            newdata.append(self.single_task(deepcopy(circuit), datarow))
        self.data = newdata

    def single_task(self, circuit: Circuit, datarow: dict) -> None:
        """Executes a circuit, returns the single shot results.

        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """
        if self.__noise_model is not None:
            circuit = self.__noise_model.apply(circuit)
        samples = circuit(nshots=self.nshots).samples()
        return {"samples": samples}

    def save(self) -> None:
        """Creates a path and pickles relevent data from ``self.data`` and
        if ``self.circuitfactory`` is a list that one too.
        """
        self.path = experiment_directory("standardrb")
        if isinstance(self.circuitfactory, list):
            with open(f"{self.path}circuits.pkl", "wb") as f:
                pickle.dump(self.circuitfactory, f)
        with open(f"{self.path}data.pkl", "wb") as f:
            pickle.dump(self.data, f)

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def _append_data(self, name: str, datacolumn: list) -> None:
        """Adds data column to ``data`` attribute.

        Args:
            name (str): Name of data column.
            datacolumn (list): A list of the right shape
        """
        if len(datacolumn) != len(self.data):
            raise ValueError("Given data column doesn't have the right length.")
        df = self.dataframe
        df[name] = datacolumn
        self.data = df.to_dict("records")

    @property
    def samples(self) -> np.ndarray:
        """Returns the samples from ``self.data`` in a 2d array.

        Returns:
            np.ndarray: 2d array of samples.
        """

        try:
            return np.array(self.dataframe["samples"].tolist())
        except KeyError:
            print("No samples here. Execute experiment first.")
            return None

    @property
    def probabilities(self) -> np.ndarray:
        """Takes the stored samples and returns probabilities for each
        possible state to occure.

        Returns:
            np.ndarray: Probability array of 2 dimension.
        """

        allsamples = self.samples
        if allsamples is None:
            print("No probabilities either.")
            return None
        # Create all possible state vectors.
        allstates = list(product([0, 1], repeat=len(allsamples[0][0])))
        # Iterate over all the samples and count the different states.
        probs = [
            [np.sum(np.product(samples == state, axis=1)) for state in allstates]
            for samples in allsamples
        ]
        probs = np.array(probs) / (self.nshots)
        return probs

    def apply_task(self, gtask):
        self = gtask(self)


class Result:
    """Once initialized with the correct parameters an Result object can build
    reports to display results of an randomized benchmarking experiment.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        self.all_figures = []

    def extract(self, group_by: str, output: str, agg_type: str):
        """Aggregates the dataframe, extracts the data by which the frame was
        grouped, what was calculated given the ``agg_type`` parameters.

        Args:
            group_by (str): _description_
            output (str): _description_
            agg_type (str): _description_
        """
        grouped_df = self.df.groupby(group_by)[output].apply(agg_type)
        return np.array(grouped_df.index), np.array(grouped_df.values)

    def scatter_plot(self):
        pass

    def report(self):
        pass
