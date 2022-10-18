from qcvv.data import Data
from qibo.noise import PauliError, NoiseModel
from qibo import gates
import numpy as np
import pdb
from  qcvv.calibrations.protocols.generators import *


class Experiment():
    """  The experiment class has methods to build, save, load and execute
    circuits with different depth with a given generator for random circuits. 
    After executing the experiment the outcomes are stored too.


    Attributes:

    circuits_list (list) : The list of lists of circuits. 
            axis 1: different runs, axis 2: different sequence lengths.

    """
    def __init__(self, circuit_generator:Generator=None,
            sequence_lengths:list=None, qubits:list=None, runs:int=None,
            nshots:int=1024, **kwargs) -> None:
        self.circuit_generator = circuit_generator
        self.sequence_lengths = sequence_lengths
        self.qubits = qubits
        self.runs = runs
        self.nshots = nshots
        if hasattr(circuit_generator, 'invert'):
            self.inverse = circuit_generator.invert

    def build_a_save(self, **kwargs):
        """ 
        """
        # Build the whole list of circuits.
        self.build(**kwargs)
        # Store it.
        return self.circuits_list, self.save_experiment(**kwargs)


    def build(self, **kwargs):
        """ Build a list out of the circuits required to run for the wanted
        experiment.
        """
        # Use the __call__ function of the circuit generator to retrieve a
        # random circuit 'runs' many times for each sequence length.
        circuits_list = [
            [next(self.circuit_generator(length))
            for length in self.sequence_lengths] for _ in range(self.runs)]
        # Create an attribute.
        # TODO should that be necessary if the experiment is stored in a 
        # pikle file?
        self.circuits_list = circuits_list
        return circuits_list
    
    def make_directory(self, **kwargs):
        """ Make the directory where the experiment will be stored.
        """
        import os
        from datetime import datetime
        overall_dir = 'experiments/'
        # Check if the overall directory exists. If not create it.
        if not os.path.isdir(overall_dir):
            os.mkdir(overall_dir)
        # Get the current date and time.
        dt_string = datetime.now().strftime("%y%b%d_%H%M%S")
        # Get the name of the generator.
        gname = self.circuit_generator.__class__.__name__
        # Every generator for the circuits gets its own directory.
        directory_generator = f'{overall_dir}{gname}/'
        if not os.path.isdir(directory_generator):
            os.mkdir(directory_generator)
        # Name the final directory for this experiment.
        directory = f'{directory_generator}experiment{dt_string}/'
        if not os.path.isdir(directory):
            os.mkdir(directory)
        # Store this as an attribute.
        self.directory = directory
        return directory

    def save_circuits(self, **kwargs) -> None:
        """ Save the given circuits list. 
        FIXME if the circuits were executed already this does not work!! 

        Args:
            kwargs (dict)
        
        Returns:
            None
        """
        # Check if there has been made a directory already for this experiment.
        if not hasattr(self, 'directory'):
            # Make and get the directory.
            self.make_directory()
        # Initiate the data structure from qibocal.
        data = Data(
            'circuits', quantities=list(self.sequence_lengths))
        # Store the data in a pandas dataframe. The columns are indexed by the
        # different sequence lengths. The rows are indexing the different runs.
        for count in range(self.runs):
            # The data object takes dictionaries.
            data.add({self.sequence_lengths[i]:self.circuits_list[count][i] \
                for i in range(len(self.sequence_lengths))})
        # Save the circuits in pickle format.
        data.to_pickle(self.directory)

    def save_metadata(self, **kwargs):
        """
        """
        from qcvv.calibrations.protocols.utils import dict_to_txt
        # Check if there has been made a directory already for this experiment.
        if not hasattr(self, 'directory'):
            # Make and get the directory.
            self.make_directory()
        # Store the metadata in a .txt file. For that create a dictionary.
        metadata_dict = {
            'qubits' : self.qubits,
            'nshots' : self.nshots,
            'runs' : self.runs,
            'inverse' : self.inverse,
            'circuit_generator' : self.circuit_generator.__class__.__name__
        }
        # One file in the directory stores the meta data.
        metadata_filename = f'{self.directory}metadata.txt'
        # Write the meta data as comments to the .txt file.
        dict_to_txt(metadata_filename, metadata_dict, openingstring='w')
        # Write the sequence lengths list.
        with open(metadata_filename, 'a') as f:
            np.savetxt(f, self.sequence_lengths)
        # The file is automatically closed.
    
    def save_experiment(self, **kwargs):
        """
        """
        self.save_metadata()
        self.save_circuits()
        return self.directory

    def retrive_from_file(self, filename:str, **kwargs):
        """
        """
        pass

    def build_onthefly(self, **kwargs):
        """
        """
        pass

    def execute_a_save(self, **kwargs):
        """
        Args:
            kwargs (dict):
                'paulierror_noiseparams' = [p1, p2, p3]
        """
        # Initiate the outcome lists, one for the single shot samples and
        # one for the probabilities.
        self.outcome_samples, self.outcome_probs = [], []
        # Also initiate the data structure where the outcomes will be stored.
        data_probabilities = Data(
            'experiment/probabilities', quantities=list(self.sequence_lengths))
        # If the circuits are simulated and not run on quantum hardware, the
        # noise has to be simulated, too.
        if kwargs.get('paulierror_noiseparams'):
            # Insert artificial noise, namely random Pauli flips.
            pauli = PauliError(*kwargs.get('paulierror_noiseparams'))
            noise = NoiseModel()
            # The noise should be applied with each unitary in the circuit.
            noise.add(pauli, gates.Unitary)
        # Loop 'runs' many times over the whole protocol.
        for count_runs in range(self.runs):
            # Go through every sequence in the protocol.
            for count_m in range(len(self.sequence_lengths)):
                # Get the circuit.
                circuit = self.circuits_list[count_runs][count_m]
                # For the simulation the noise has to be added to the circuit.
                if kwargs.get('paulierror_noiseparams'):
                    # Add the noise to the circuit (more like the other way
                    # around, the circuit to the noise).
                    noisy_circuit = noise.apply(circuit)
                    # Execute the noisy circuit.
                    executed = noisy_circuit(nshots=kwargs.get('nshots'))
                else:
                    # Execute the qibo circuit without artificial noise.
                    executed = circuit(nshots=kwargs.get('nshots'))
                # FIXME The samples (zeros and ones per shot) acquisition does 
                # not work for quantum hardware yet.
                try:
                    # Get the samples from the executed gate. It should be an
                    # object filled with as many integers as used shots.
                    # Append the samples.
                    self.outcome_samples.append(executed.samples())
                except:
                    print('Retrieving samples not possible.')
                # Either way store the probabilities. Since
                # 'executed.probabilities()' only contains an entry for qubit
                # if it is nonzero, the shape can vary, fix that FIXME.
                # Store them.
                self.outcome_probs.append(executed.probabilities())
            # Put the probabilities into the data object.
            data_probabilities.add({
                self.sequence_lengths[i]:self.outcome_probs[count_runs*i+i] \
                for i in range(len(self.sequence_lengths))})
        # Push data.
        return data_probabilities
    
    @classmethod
    def retreive_experiment(cls, path:str, **kwargs):
        """
        """
        from qcvv.calibrations.protocols.utils import pkl_to_list, dict_from_comments_txt
        import ast
        # Initiate an instance of the experiment class.
        obj = cls()
        # Get the metadata in form of a dictionary.
        metadata_dict = dict_from_comments_txt(f'{path}metadata.txt')
        # Get the (not commented) sequence lengths array.
        sequence_lenghts = np.loadtxt(f'{path}metadata.txt')
        # The circuit generator has to be restored, this will get the class.
        Generator = eval(metadata_dict['circuit_generator'])
        # Build the generator.
        circuit_generator = Generator(metadata_dict['qubits'])
        # Write it to the dictionary.
        metadata_dict['circuit_generator'] = circuit_generator
        # Give the objects the attributes as a dictionary. Every attribute
        # would be overwritten by that.
        obj.__dict__ = metadata_dict
        # Store the diven path.
        obj.directory = path
        # Get the circuits list and make it an attribute.
        sequences_frompkl, circuits_list = pkl_to_list(f'{path}circuits.pkl')
        # Make sure that the order is the same, right now the order is reversed.
        assert np.array_equal(
            np.array(sequences_frompkl)[::-1], sequence_lenghts), \
            'The order of the restored circuits is not the same as when build'
        # Every list in circuits_list has entries refering to the different
        # sequence lengths. Meaning that the order is crucial.
        obj.circuits_list = [x[::-1] for x in circuits_list]
        # Also store the sequence lenghts.
        obj.sequence_lengths = list(sequence_lenghts)
        return obj
        
