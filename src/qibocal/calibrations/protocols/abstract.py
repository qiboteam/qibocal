from __future__ import annotations

import abc
import pickle
from collections.abc import Iterable
from copy import deepcopy
from os.path import isfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

import qibocal.fitting.rb_methods as fitting_methods
from qibocal.calibrations.protocols import noisemodels
from qibocal.calibrations.protocols.utils import (
    ONEQUBIT_CLIFFORD_PARAMS,
    experiment_directory,
)
from qibocal.config import raise_error
from qibocal.data import Data

""" TODO
    - Don't load the whole experiment into the results class -> Its just a copy
    of the pointer! So it's fine.
    - Write validation functions
    - qubits_active, qubits_passive ?
    - Noise model integration
    - Use Renatos functions from quantum_info module
      * Pauli basis
      * average_gate_fidelity
    - Make quantum hardware
    - Change result class structure
      * Add used noisemodel with parameters.
      * Add validation of simulation
      * Redo the fitting-
    - Crosstalk -> Correlated?
    - Wo bricht crosstalk RB wegen Groessenordnungen bei Irreps? -> Theorie frage,
    - Nor locales depolarizing, 2qubit irreps params should follow from 1 qubit irrpes

    - validation function:
        * Falls funktion maetichg (generell auf module anwendbar, zB fourier matrix )
        * Theoretical validation function if too specific into testing function
            of said module

"""
""" TODO with Liza
- Add initial state to abstract Experiment class or to data ??
   * XId in filter function result times rho
- Add possiblilty to choose different measurement basis -> watch out
- Change basic functions from validation in XId
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
        self.name = "Abstract"

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
        self.name = "SingleCliffords"

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
        """Creates a path and pickles relevant data from ``self.data`` and
        if ``self.circuitfactory`` is a list that one too.
        """
        self.path = experiment_directory("rb")
        if isinstance(self.circuitfactory, list):
            with open(f"{self.path}circuits.pkl", "wb") as f:
                pickle.dump(self.circuitfactory, f)
        with open(f"{self.path}data.pkl", "wb") as f:
            pickle.dump(self.data, f)

    def extract(self, group_by: str, output: str, agg_type: str | callable):
        """Aggregates the dataframe, extracts the data by which the frame was
        grouped, what was calculated given the ``agg_type`` parameters.

        Args:
            group_by (str): _description_
            output (str): _description_
            agg_type (str): _description_
        """
        grouped_df = self.dataframe.groupby(group_by)[output].apply(agg_type)
        return np.array(grouped_df.index), np.array(grouped_df.values.tolist())

    def prebuild(self) -> None:
        """Converts the attribute ``circuitfactory`` which is in general
        an iterable into a list.
        """
        self.circuitfactory = list(self.circuitfactory)

    def perform(self, sequential_task: callable[[Circuit, dict], dict]) -> None:
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
                datarow = sequential_task(deepcopy(circuit), datarow)
        # Only``datarow`` can be provided:
        elif self.circuitfactory is None and self.data is not None:
            for datarow in self.data:
                datarow = sequential_task(None, datarow)
        # Only ``circuit`` can be provided:
        elif self.circuitfactory is not None and self.data is None:
            newdata = []
            for circuit in self.circuitfactory:
                newdata.append(sequential_task(deepcopy(circuit), {}))
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


class Protocol:
    def __init__(self, module) -> None:
        self.module = module

    def execute(
        self,
        qubit: list,
        depths: list,
        runs: int,
        nshots: int,
        nqubit: int = None,
        noise_model: str = None,
        noise_params: list = None,
    ) -> None:
        # Check if noise should artificially be added.
        if noise_model is not None:
            # Get the wanted noise model class.
            noise_model = getattr(noisemodels, noise_model)(*noise_params)
        # Initiate the circuit factory and the Experiment object.
        factory = self.module.moduleFactory(nqubit, depths, runs, qubits=qubit)
        experiment = self.module.moduleExperiment(
            factory, nshots, noisemodel=noise_model
        )
        # Execute the experiment.
        experiment.perform(experiment.execute)
        data = Data()
        data.df = experiment.dataframe
        yield data


class Report:
    """Once initialized with the correct parameters an Report object can build
    reports to display results of an randomized benchmarking experiment.
    """

    def __init__(self) -> None:
        self.all_figures = []
        self.title = "Report"
        self.info_dict = {}

    def build(self):
        from plotly.subplots import make_subplots

        l = len(self.all_figures)
        subplot_titles = [figdict.get("subplot_title") for figdict in self.all_figures]
        fig = make_subplots(
            rows=int(l / 2) + l % 2 + 1,
            cols=1 if l == 1 else 2,
            subplot_titles=subplot_titles,
        )
        for count, fig_dict in enumerate(self.all_figures):
            plot_list = fig_dict["figs"]
            for plot in plot_list:
                fig.add_trace(plot, row=count // 2 + 1, col=count % 2 + 1)

        fig.add_annotation(
            dict(
                bordercolor="black",
                font=dict(color="black", size=16),
                x=0.0,
                y=1.0 / (int(l / 2) + l % 2 + 1) - len(self.info_dict) * 0.005,
                showarrow=False,
                text="<br>".join(
                    [f"{key} : {value}\n" for key, value in self.info_dict.items()]
                ),
                align="left",
                textangle=0,
                yanchor="top",
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
            height=500 * (int(l / 2) + l % 2) if l > 2 else 1000,
            width=1000,
        )

        return fig


def scatter_fit_fig(
    experiment: Experiment, df_aggr: pd.DataFrame, xlabel: str, index: str
):
    fig_traces = []
    dfrow = df_aggr.loc[index]
    fig_traces.append(
        go.Scatter(
            x=experiment.dataframe[xlabel],
            y=experiment.dataframe[index],
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
        )
    )
    fig_traces.append(
        go.Scatter(
            x=dfrow[xlabel],
            y=dfrow["data"],
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    x_fit = np.linspace(min(dfrow[xlabel]), max(dfrow[xlabel]), len(dfrow[xlabel]) * 20)
    y_fit = getattr(fitting_methods, dfrow["fit_func"])(x_fit, *dfrow["popt"].values())
    fig_traces.append(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name="".join(
                ["{}:{:.3f} ".format(key, dfrow["popt"][key]) for key in dfrow["popt"]]
            ),
            line=go.scatter.Line(dash="dot"),
        )
    )
    return {"figs": fig_traces, "xlabel": xlabel, "ylabel": index}
