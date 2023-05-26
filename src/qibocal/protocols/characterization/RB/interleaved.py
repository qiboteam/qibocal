from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel

from qibocal.auto.operation import Routine
from qibocal.calibrations.niGSC.basics.circuitfactory import SingleCliffordsFactory
from qibocal.protocols.characterization.RB.result import (
    DecayWithOffsetResult,
    get_hists_data,
)
from qibocal.protocols.characterization.RB.utils import extract_from_data

from .data import RBData
from .params import RBParameters

NoneType = type(None)


class Scan(SingleCliffordsFactory):
    def build_circuit(self, depth: int) -> Circuit:
        """Initiate a ``qibo.models.Circuit`` object and fill it with the wanted gates.

        Which gates are wanted is encoded in ``self.gates_layer()``.
        Add a measurement gate for every qubit.

        Args:
            depth (int): How many layers there are in the circuit.

        Returns:
            Circuit: the circuit with ``depth`` many layers.
        """
        # Initiate the ``Circuit`` object with the amount of active qubits.
        circuit = Circuit(len(self.qubits))
        # Go through the depth/layers of the circuit and add gate layers interleaved with CZ.
        cliffords_circuit = Circuit(len(self.qubits))
        for _ in range(depth - 1):
            cliffords = self.gate_layer()
            circuit.add(cliffords)
            cliffords_circuit.add(cliffords)
            circuit.add(gates.CZ(*range(len(self.qubits))))
            circuit.add(gates.CZ(*range(len(self.qubits))))
        # If there is at least one gate in the circuit, add an inverse.
        if depth > 0:
            cliffords = self.gate_layer()
            circuit.add(cliffords)
            cliffords_circuit.add(cliffords)
            # Build a gate out of the unitary of the whole circuit and
            # take the daggered version of that.
            # import pdb
            # pdb.set_trace()
            circuit_q0 = cliffords_circuit.light_cone(0)[0]
            circuit_q1 = cliffords_circuit.light_cone(1)[0]
            circuit.add(gates.Unitary(circuit_q0.unitary(), 0).dagger())
            circuit.add(gates.Unitary(circuit_q1.unitary(), 1).dagger())
            # circuit.add(
            #     gates.Unitary(circuit.unitary(), *range(len(self.qubits))).dagger()
            # )
        # Add a ``Measurement`` gate for every qubit.
        circuit.add(gates.M(*range(len(self.qubits))))
        # print(circuit.draw())
        return circuit


@dataclass
class InterleavedRBResult(DecayWithOffsetResult):
    """Inherits from `DecayWithOffsetResult`, a result class storing data and parameters
    of a single decay with statistics.

    Adds the method of calculating a fidelity out of the fitting parameters.
    TODO calculate SPAM errors with A and B
    TODO calculate the error of the fidelity

    """

    pass


def setup_scan(params: RBParameters) -> Iterable:
    """An iterator building random Clifford sequences with an inverse in the end.

    Args:
        params (RBParameters): The needed parameters.

    Returns:
        Iterable: The iterator of circuits.
    """

    return Scan(params.nqubits, params.depths * params.niter, params.qubits)


def execute(
    scan: Iterable,
    nshots: Union[int, NoneType] = None,
    noise_model: Union[NoiseModel, NoneType] = None,
) -> List[dict]:
    """Execute a given scan with the given number of shots and if its a simulation with the given
    noise model.

    Args:
        scan (Iterable): The ensemble of experiments (here circuits)
        nshots (Union[int, NoneType], optional): Number of shots per circuit. Defaults to None.
        noise_model (Union[NoiseModel, NoneType], optional): If its a simulation a noise model
            can be applied. Defaults to None.

    Returns:
        List[dict]: A list with one dictionary for each executed circuit where the data is stored.
    """

    data_list = []
    # Iterate through the scan and execute each circuit.
    for c in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (c.depth + 1) // 3 if c.depth > 0 else 0
        if noise_model is not None:
            c = noise_model.apply(c)
        samples = c.execute(nshots=nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append({"depth": depth, "samples": samples})
    return data_list


def aggregate(data: RBData) -> InterleavedRBResult:
    """Takes a data frame, processes it and aggregates data in order to create
    a routine result object.

    Args:
        data (RBData): Actually a data frame from where the data is processed.

    Returns:
        InterleavedRBResult: The aggregated data.
    """

    def p0s(samples_list):
        ground = np.array([0] * len(samples_list[0][0]))
        my_p0s = []
        for samples in samples_list:
            my_p0s.append(np.sum(np.product(samples == ground, axis=1)) / len(samples))
        return my_p0s

    # The signal is here the survival probability.
    data_agg = data.assign(signal=lambda x: p0s(x.samples.to_list()))
    # Histogram
    hists = get_hists_data(data_agg)
    # Build the result object
    return InterleavedRBResult(
        *extract_from_data(data_agg, "signal", "depth", "mean"),
        hists=hists,
        meta_data=data.attrs,
    )


def acquire(params: RBParameters) -> RBData:
    """The data acquisition stage of standard rb.

    1. Set up the scan
    2. Execute the scan
    3. Put the acquired data in a standard rb data object.

    Args:
        params (RBParameters): All parameters in one object.

    Returns:
        RBData: _description_
    """

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    scan = setup_scan(params)
    # For simulations, a noise model can be added.
    if params.noise_model:
        from qibocal.calibrations.niGSC.basics import noisemodels

        noise_model = getattr(noisemodels, params.noise_model)(*params.noise_params)
    else:
        noise_model = None
    # Execute the scan.
    data = execute(scan, params.nshots, noise_model)
    # Build the data object which will be returned and later saved.
    standardrb_data = RBData(data)
    standardrb_data.attrs = params.__dict__
    return standardrb_data


def extract(data: RBData) -> InterleavedRBResult:
    """Takes a data frame and extracts the depths,
    average values of the survival probability and histogram

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        InterleavedRBResult: Aggregated and processed data.
    """

    result = aggregate(data)
    result.fit()
    return result


def plot(
    data: RBData, result: InterleavedRBResult, qubit
) -> Tuple[List[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        result (InterleavedRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        Tuple[List[go.Figure], str]:
    """

    table_str = "".join(
        [
            f" | {key}: {value}<br>"
            for key, value in {**result.meta_data, **result.fidelity_dict}.items()
        ]
    )
    fig = result.plot()
    return [fig], table_str


standard_rb = Routine(acquire, extract, plot)
