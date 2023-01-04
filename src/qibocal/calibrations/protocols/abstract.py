from __future__ import annotations

import abc
import pickle
from collections.abc import Iterable
from copy import deepcopy
from itertools import product
from os.path import isfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

from qibocal.calibrations.protocols.utils import (
    ONEQUBIT_CLIFFORD_PARAMS,
    experiment_directory,
)
from qibocal.config import raise_error

""" TODO
    - Make row by row execution nicer -> ask Ingo
    - Don't load the whole experiment into the results class
    - Write validation functions
    - Make function names in each module generic
    - qubits_active, qubits_passive ?
    - Noise model integration
"""


class Circuitfactory:
    """TODO write documentation"""

    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        self.nqubits = nqubits if nqubits is not None else len(qubits)
        self.qubits = qubits if qubits is not None else list(range(nqubits))
        self.depths = depths
        self.runs = runs

    def __len__(self):
        return self.runs * len(self.depths)

    def __iter__(self) -> Circuitfactory:
        self.n = 0
        return self

    def __next__(self) -> Circuit:
        if self.n >= self.runs * len(self.depths):
            raise StopIteration
        else:
            circuit = self.build_circuit(self.depths[self.n % len(self.depths)])
            self.n += 1
            # Distribute the circuit onto the given support.
            bigcircuit = Circuit(self.nqubits)
            bigcircuit.add(circuit.on_qubits(*self.qubits))
            return bigcircuit

    @abc.abstractmethod
    def build_circuit(self, depth: int):
        raise_error(NotImplementedError)


class SingleCliffordsFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits)

    def build_circuit(self, depth: int):
        circuit = Circuit(len(self.qubits))
        for _ in range(depth):
            circuit.add(self.gates())
        circuit.add(gates.M(*range(len(self.qubits))))
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

    def gates(self) -> list:
        """Draws the parameters and builds the unitary Clifford gates for
        a circuit layer.

        Returns:
            list filled with ``qibo.gates.Unitary``:
                the simulatanous Clifford gates.
        """
        # There are this many different Clifford matrices.
        amount = len(ONEQUBIT_CLIFFORD_PARAMS)
        gates_list = []
        # Choose as many random integers between 0 and 23 as there are used
        # qubits. Get the clifford parameters and build the unitares.
        for count, rint in enumerate(
            np.random.randint(0, amount, size=len(self.qubits))
        ):
            # Build the random Clifford matrices append them
            gates_list.append(
                gates.Unitary(
                    self.clifford_unitary(*ONEQUBIT_CLIFFORD_PARAMS[rint]), count
                )
            )
        # Make a unitary gate out of 'unitary' for the qubits.
        return gates_list


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

    @property
    def noise_model(self):
        return self.__noise_model

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    @property
    def samples(self) -> np.ndarray:
        """Returns the samples from ``self.data`` in a 2d array.

        Returns:
            np.ndarray: 2d array of samples.
        """

        try:
            return np.array(self.dataframe["samples"].tolist())
        except KeyError:
            raise_error(KeyError, "No samples here. Execute experiment first.")

    @property
    def probabilities(self) -> np.ndarray:
        """Takes the stored samples and returns probabilities for each
        possible state to occure.

        Returns:
            np.ndarray: Probability array of 2 dimension.
        """

        allsamples = self.samples
        # Create all possible state vectors.
        allstates = list(product([0, 1], repeat=len(allsamples[0][0])))
        # Iterate over all the samples and count the different states.
        probs = [
            [np.sum(np.product(samples == state, axis=1)) for state in allstates]
            for samples in allsamples
        ]
        probs = np.array(probs) / (self.nshots)
        return probs

    @classmethod
    def load(cls, path: str) -> Experiment:
        """Creates an experiment object with data and if possible with circuits.

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
                if isinstance(data, pd.DataFrame):
                    data = data.to_dict("records")
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

    def save(self) -> None:
        """Creates a path and pickles relevent data from ``self.data`` and
        if ``self.circuitfactory`` is a list that one too.
        """
        self.path = experiment_directory("rb")
        if isinstance(self.circuitfactory, list):
            with open(f"{self.path}circuits.pkl", "wb") as f:
                pickle.dump(self.circuitfactory, f)
        with open(f"{self.path}data.pkl", "wb") as f:
            pickle.dump(self.data, f)

    def prebuild(self) -> None:
        """Converts the attribute ``circuitfactory`` which is in general
        an iterable into a list.
        """
        self.circuitfactory = list(self.circuitfactory)

    def execute(self) -> None:
        """Calls method ``single_task`` while iterating over attribute
        ``circuitfactory`` and ``data``.

        Collects data given the already set data rows and overwrites
        attribute ``data``.
        """
        if self.circuitfactory is None:
            raise_error(NotImplementedError, "There are no circuits to execute.")
        newdata = []
        for circuit in self.circuitfactory:
            try:
                datarow = next(self.data)
            except TypeError:
                datarow = {}
            newdata.append(self.single_task(deepcopy(circuit), datarow))
        self.data = newdata

    def single_task(self, circuit: Circuit, datarow: dict) -> dict:
        """Executes a circuit, returns the single shot results.

        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)
        samples = circuit(nshots=self.nshots).samples()
        return {"samples": samples}

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

    def apply_task(self, gtask):
        self = gtask(self)


class Result:
    """Once initialized with the correct parameters an Result object can build
    reports to display results of an randomized benchmarking experiment.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        self.all_figures = []
        # The fitting function should return two tuples and two np.ndarrays.
        self.fitting_func = lambda x, y: ((0, 0, 0), (1, 1, 1), x, y)
        self.title = "Report"
        self.info_dict = {
            "nqubits": len(self.df["samples"].to_numpy()[0][0]),
            "nshots": len(self.df["samples"].to_numpy()[0]),
        }

    def extract(self, group_by: str, output: str, agg_type: str):
        """Aggregates the dataframe, extracts the data by which the frame was
        grouped, what was calculated given the ``agg_type`` parameters.

        Args:
            group_by (str): _description_
            output (str): _description_
            agg_type (str): _description_
        """
        grouped_df = self.df.groupby(group_by)[output].apply(agg_type)
        return np.array(grouped_df.index), np.array(grouped_df.values.tolist())

    def get_info(self):
        """Extract information from attribute ``self.info_dict`` and formats
        it as columns.
        """
        info_string = "<br>".join(
            [f"{key} : {value}\n" for key, value in self.info_dict.items()]
        )
        return info_string

    def scatter_fit_fig(self, xdata_scatter, ydata_scatter, xdata, ydata):
        myfigs = []
        popt, pcov, x_fit, y_fit = self.fitting_func(xdata, ydata)
        fig = go.Scatter(
            x=xdata_scatter,
            y=ydata_scatter,
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
        )
        myfigs.append(fig)
        fig = go.Scatter(
            x=xdata, y=ydata, line=dict(color="#aa6464"), mode="markers", name="average"
        )
        myfigs.append(fig)
        fig = go.Scatter(
            x=x_fit,
            y=y_fit,
            name="A: {:.3f}, p: {:.3f}, B: {:.3f}".format(popt[0], popt[1], popt[2]),
            line=go.scatter.Line(dash="dot"),
        )
        myfigs.append(fig)
        self.all_figures.append({"figs": myfigs})

    def report(self):
        from plotly.subplots import make_subplots

        l = len(self.all_figures)
        subplot_titles = [figdict.get("subplot_title") for figdict in self.all_figures]
        fig = make_subplots(
            rows=l, cols=1 if l == 1 else 2, subplot_titles=subplot_titles
        )
        for count, fig_dict in enumerate(self.all_figures):
            plot_list = fig_dict["figs"]
            for plot in plot_list:
                fig.add_trace(plot, row=count // 2 + 1, col=count % 2 + 1)
        fig.add_annotation(
            dict(
                font=dict(color="black", size=16),
                x=1.1,
                y=0.0,
                showarrow=False,
                text=self.get_info(),
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
        fig.update_xaxes(title_font_size=18, tickfont_size=16)
        fig.update_yaxes(title_font_size=18, tickfont_size=16)
        fig.update_layout(
            font_family="Averta",
            hoverlabel_font_family="Averta",
            title_text=self.title,
            title_font_size=24,
            legend_font_size=16,
            hoverlabel_font_size=16,
            showlegend=True,
            height=500 * int(l / 2) if l > 1 else 500,
            width=1000,
        )
        return fig
