from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel, PauliError
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.calibrations.protocols.abstract import (
    Experiment,
    Result,
    SingleCliffordsFactory,
    embed_unitary_circuit,
)
from qibocal.calibrations.protocols.fitting_methods import fit_exp1_func
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.plots.scatters import standardrb_plot



class CrossentropyRBExperiment(Experiment):
    def __init__(self, circuitfactory: Iterable, nshots: int = None, data: list = None, noisemodel: NoiseModel = None) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel)
    
    def single_task(self, circuit: Circuit, datarow: dict) -> None:
        """Executes a circuit, returns the single shot results
        Args:
            circuit (Circuit): Will be executed, has to return samples.
            datarow (dict): Dictionary with parameters for execution and
                immediate postprocessing information.
        """
        datadict = super().single_task(circuit, datarow)
        # FIXME and on the measurement branch of qibo the measurement is
        # counted as one gate on the master branch not.
        datadict["depth"] = int(
            (
                circuit.ngates - 1 if circuit.ngates > 1 else 0
            )/len(datadict['samples'][0])
        )
        datadict['circuit'] = circuit
        return datadict

    @property
    def depths(self) -> np.ndarray:
        """Extracts the used circuits depths.

        Returns:
            np.ndarray: Used depths for every data row.
        """
        try:
            return self.dataframe["depth"].to_numpy()
        except KeyError:
            print("No depths. Execute experiment first.")
            return None

def filter_function(experiment: Experiment):
    # Extract amount of used qubits.
    nqubits = len(experiment.data[0]['samples'][0])
    nshots = len(experiment.data[0]['samples'])
    d = 2
    biglist = []
    for datarow in experiment.data:
        samples = datarow['samples']
        # Extract the unitaries which acted on each qubit.
        idealoutcome_list = []
        for count in range(nqubits):
            unitary = np.eye(2, dtype = complex)
            # Don't take into account the measurement gate.
            # The queue is ordered, go through it to get each qubit.
            for gate in np.array(datarow['circuit'].queue[count:-1:nqubits]):
                unitary = gate.matrix@unitary
            idealoutcome_list.append(unitary@np.array([[1], [0]]))
        idealoutcomes = d*np.array(idealoutcome_list)

        f_list = []
        for l in np.array(list(product([False,True], repeat = 2))): 
            if not sum(l):
                bla = nshots # nshots/(d+1)
            else:
                supl = idealoutcomes[l][0]
                supsamples = samples[:,l]
                bla = 0
                for s in supsamples:
                    for b in np.array(list(product([False,True], repeat = sum(l)))):
                        bla += (-1)**sum(b)*np.prod(np.abs(supl[s[~b]]))
            f_list.append(bla)
        biglist.append(np.array(f_list)*(d+1)/(d**2*nshots))
    print(np.average(biglist, axis=0))


def analyze(experiment: Experiment, noisemodel: NoiseModel = None, **kwargs):
    pass

def perform(
    nqubits: int,
    depths: list,
    runs: int,
    nshots: int,
    qubits: list = None,
    noise_params: list = None,
):
    if noise_params is not None:
        # Define the noise model.
        paulinoise = PauliError(*noise_params)
        noise = NoiseModel()
        noise.add(paulinoise, gates.Unitary)
        depol = effective_depol(paulinoise)
    else:
        noise = None
    # Initiate the circuit factory and the faulty Experiment object.
    factory = SingleCliffordsFactory(nqubits, depths, runs, qubits=qubits)
    experiment = CrossentropyRBExperiment(factory, nshots, noisemodel=noise)
    # The circuits have to be stored, build the experiment first to make a list
    # (out of the factory) which will be stored when saving the experiment.
    experiment.build()
    # Execute the experiment.
    experiment.execute()
    analyze(experiment, noisemodel=noise).show()

