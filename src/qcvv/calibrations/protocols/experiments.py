from qcvv.data import Data


class Experiment():
    """
    """
    def __init__(self, circuit_generator, sequence_lengths:list, qubits:list,
            runs:int, nshots:int=1024, **kwargs) -> None:
        self.circuit_generator = circuit_generator
        self.sequence_lengths = sequence_lengths
        self.qubits = qubits
        self.runs = runs
        self.nshots = nshots
        self.inverse = circuit_generator.invert

    def prebuild_a_save(self, **kwargs):
        """ 
        """
        # Build the whole list of circuits.
        circuits_list = self.prebuild(**kwargs)
        # Store it.
        self.save(circuits_list, **kwargs)

    def prebuild(self, **kwargs):
        """ Build a list out of the circuits required to run for the wanted
        experiment.
        """
        # Use the __call__ function of the circuit generator to retrieve a
        # random circuit 'runs' many times for each sequence length.
        circuits_list = [
            [next(self.circuit_generator(length))
            for length in self.sequence_lengths] for _ in range(self.runs) ]
        return circuits_list

    def save(self, circuits_list, **kwargs):
        """ Save the given circuits list.
        Args:
            circuits_list (list) : The list of lists of circuits. 
            axis 1: different runs, axis 2: different sequence lengths.
        
        Returns:
            (data object): The data object to store the circuits and sequence
            lengths.
        """
        # Initiate the data structure from qibocal.
        data = Data("experiment", quantities=list(self.sequence_lengths))
        # Store the data in a pandas dataframe. The columns are indexed by the
        # different sequence lengths. The rows are indexing the different runs.
        for count in range(self.runs):
            # The data object takes dictionaries.
            data.add({self.sequence_lengths[i]:circuits_list[count][i] \
                for i in range(len(self.sequence_lengths))})
        yield data

    def retrive_from_file(self, filename:str, **kwargs):
        """
        """
        pass

    def build_onthefly(self, **kwargs):
        """
        """
        pass

    def execute_a_measure(self, **kwargs):
        """
        """
        pass
    
    def convert_to_object(generator_str):
        """
        """
        pass
