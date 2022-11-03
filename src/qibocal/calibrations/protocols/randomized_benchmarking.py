# -*- coding: utf-8 -*-

from datetime import datetime

from qibo.noise import NoiseModel, PauliError

from qibocal import plots
from qibocal.calibrations.protocols.experiments import Experiment
from qibocal.calibrations.protocols.generators import *
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.plots.scatters import rb_plot
from qibocal.fitting.methods import rb_exponential_fit
from datetime import datetime


@plot("Randomized benchmarking", plots.rb_plot)
def dummyrb(
    platform,
    qubit: list,
    circuit_generator_class: str,
    invert: bool,
    sequence_lengths: list,
    runs: int,
    nshots: int = 1024,
    inject_noise: list = None,
    active_qubit: int = None,
):
    print("start: ", datetime.now().strftime("%d.%b %y %H:%M:%S"))
    # Make the generator class out of the name.
    circuit_generator_class = eval(circuit_generator_class)
    # Make a generator object out of the generator class.
    circuit_generator = circuit_generator_class(
        qubit, invert=invert, act_on=active_qubit
    )
    # Initiate the Experiment object, not filled with circuits yet.
    experiment = Experiment(circuit_generator, sequence_lengths, qubit, runs, nshots)
    # Build the circuits.
    experiment.build()
    # Get the circuits object. To avoid the
    # TypeError: cannot pickle 'module' object,
    # initiate the data object now.
    data_circs = experiment.data_circuits
    yield data_circs
    # Execute the circuits.
    experiment.execute_experiment(paulierror_noiseparams=inject_noise)
    # Get the data objects.
    data_probs = experiment.data_probabilities
    data_samples = experiment.data_samples
    # Yield the circuits and outcome data objects.
    yield data_probs
    yield data_samples
    if not active_qubit:
        active_qubit = qubit
    yield rb_exponential_fit(experiment, active_qubit)
    # Store the effective depol parameter. If there is no noise to inject (
    # because it is run on hardware), make it zero.
    if not inject_noise:
        inject_noise = [0, 0, 0]
    pauli = PauliError(*inject_noise)
    noise = NoiseModel()
    noise.add(pauli, gates.Unitary)
    data_depol = Data("effectivedepol", quantities=["effective_depol"])
    data_depol.add({"effective_depol": effective_depol(pauli)})
    yield data_depol
    print("end: ", datetime.now().strftime("%d.%b %y %H:%M:%S"))
