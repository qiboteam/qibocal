
from qcvv.calibrations.protocols.experiments import Experiment
from qcvv.calibrations.protocols.generators import *
from qcvv.plots.scatters import rb_plot
from qibo.noise import PauliError, NoiseModel
from qcvv.data import Data
from qcvv.calibrations.protocols.utils import effective_depol
from qcvv.decorators import plot
from qcvv import plots


@plot("Test Standard RB", plots.rb_plot)
def dummyrb(
    platform,
    qubits : list,
    circuit_generator_class : str,
    invert : bool,
    sequence_lengths : list,
    runs : int,
    nshots: int,
    inject_noise : list,
):
    # Make the generator class out of the name.
    circuit_generator_class = eval(circuit_generator_class)
    # Make a generator object out of the generator class.
    circuit_generator = circuit_generator_class(qubits, invert=invert)
    # Initiate the Experiment object, not filled with circuits yet. 
    experiment = Experiment(
        circuit_generator, sequence_lengths, qubits, runs, nshots)
    # Build the circuits and store them.
    directory = experiment.build_a_save(yield_data=True)
    # Execute the circuits and store the outcome.
    experiment.execute_a_save(
        yield_data=True, paulierror_noisparams=inject_noise)
    # Store the effective depol parameter.
    pauli = PauliError(*inject_noise)
    noise = NoiseModel()
    noise.add(pauli, gates.Unitary)
    data_depol = Data("effectivedepol", quantities=["effective_depol"])
    data_depol.add({"effective_depol": effective_depol(pauli)})
    yield data_depol