
from qcvv.calibrations.protocols.experiments import Experiment
from qcvv.calibrations.protocols.generators import *
from qcvv.plots.scatters import rb_plot
from qibo.noise import PauliError, NoiseModel
from qcvv.data import Data
from qcvv.calibrations.protocols.utils import effective_depol
from qcvv.decorators import plot
from qcvv import plots


@plot("Randomized benchmarking", plots.rb_plot)
def dummyrb(
    platform,
    qubit : list,
    circuit_generator_class : str,
    invert : bool,
    sequence_lengths : list,
    runs : int,
    nshots: int=1024,
    inject_noise : list=None,
    active_qubit : int=None,
):
    # Make the generator class out of the name.
    circuit_generator_class = eval(circuit_generator_class)
    # Make a generator object out of the generator class.
    circuit_generator = circuit_generator_class(
        qubit, invert=invert, act_on=active_qubit)
    # Initiate the Experiment object, not filled with circuits yet. 
    experiment = Experiment(
        circuit_generator, sequence_lengths, qubit, runs, nshots)
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
    # Store the effective depol parameter.
    if not inject_noise or inject_noise is None or inject_noise == 'None':
        inject_noise = [0,0,0]
    pauli = PauliError(*inject_noise)
    noise = NoiseModel()
    noise.add(pauli, gates.Unitary)
    data_depol = Data("effectivedepol", quantities=["effective_depol"])
    data_depol.add({"effective_depol": effective_depol(pauli)})
    yield data_depol