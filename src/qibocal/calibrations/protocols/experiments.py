# -*- coding: utf-8 -*-
from qibocal.calibrations.protocols import fitting_methods
from ast import literal_eval
from copy import deepcopy
from itertools import product
from os import mkdir
from os.path import isdir, isfile
from copy import deepcopy

import numpy as np
from qibo import gates
from qibo.noise import NoiseModel, PauliError

from qibocal.calibrations.protocols.generators import *
from qibocal.calibrations.protocols.utils import dict_to_txt, pkl_to_list
from qibocal.data import Data

# from typing import Union


class Experiment:
    """The experiment class has methods to build, save, load and execute
    circuits with different depth with a given generator for random circuits.
    After executing the experiment the outcomes are stored too.


    Attributes:
        TODO work in possible error models better

    circuits_list (list) : The list of lists of circuits.
            axis 1: different runs, axis 2: different sequence lengths.

    """

    def __init__(
        self,
        circuit_generator: Generator = None,
        sequence_lengths: list = None,
        qubits: list = None,
        runs: int = None,
        nshots: int = 1024,
        **kwargs,
    ) -> None:
        self.circuit_generator = circuit_generator
        self.sequence_lengths = sequence_lengths
        self.qubits = qubits
        self.runs = runs
        self.nshots = nshots
        if hasattr(circuit_generator, "invert"):
            self.inverse = circuit_generator.invert
        self.__number_simulations = 0

    ############################ PROPERTIES/SETTER ############################

    @property
    def number_simulations(self):
        return self.__number_simulations

    @property
    def data_circuits(self):
        """ """
        # Initiate the data structure from qibocal.
        data_circs = Data("circuits", quantities=list(self.sequence_lengths))
        # Store the data in a pandas dataframe. The columns are indexed by the
        # different sequence lengths. The rows are indexing the different runs.
        for count in range(self.runs):
            # The data object takes dictionaries.
            data_circs.add(
                {
                    self.sequence_lengths[i]: self.circuits_list[count][i]
                    for i in range(len(self.sequence_lengths))
                }
            )
        return data_circs

    @data_circuits.setter
    def data_circuits(self, gdata):
        """ """
        # Extract the data frame.
        dataframe = gdata.df
        # Put them in a list, first axis is the different runs, second axis
        # the sequence lengths.
        circuits_list = dataframe.values.tolist()
        # Check if the attribute does not exist yet.
        if not hasattr(self, "sequence_lengths") or self.sequence_lengths is None:
            # Get the sequence lengths.
            sequence_lenghts = dataframe.columns.tolist()
            # The pickeling process reverses the order, reoder it.
            self.sequence_lengths = np.array(sequence_lenghts[::-1])
        # Store the outcome as an attribute to further work with its.
        self.circuits_list = [x[::-1] for x in circuits_list]
        # Check how many gates there are in a single circuit. If it is one more
        # then the supposed sequence length, there is an inverse.
        # Since there is a measurement gate, actually its two more.
        first_sl = self.sequence_lengths[0]
        first_cl = len(self.circuits_list[0][0].queue)
        if first_sl + 2 == first_cl:
            self.inverse = True
        elif first_sl + 1 == first_cl:
            self.inverse = False
        else:
            raise ValueError("circuits don't match sequence length!")

    @property
    def data_samples(self):
        """ """
        if len(self.outcome_samples[0]) != 0:
            # Store the data in a pandas dataframe.
            # FIXME There are versions not supporting writing arrays to data frames.
            try:
                # raise ValueError
                # Initiate the data structure where the outcomes will be stored.
                data_samples = Data(
                    f"samples{self.__number_simulations}",
                    quantities=list(self.sequence_lengths),
                )
                # The columns are indexed by the different sequence lengths.
                # The rows are indexing the different runs.
                for count in range(self.runs):
                    # The data object takes dictionaries.
                    data_samples.add(
                        {
                            self.sequence_lengths[i]: self.outcome_samples[count][i]
                            for i in range(len(self.sequence_lengths))
                        }
                    )
            # This error: ValueError: Must have equal len keys and value when
            # setting with an iterable
            # is caught by this.
            except ValueError:
                # Initiate the data structure where the outcomes will be stored.
                # If the initialization of the data objext is not overwritten
                # here, the first row will be filled with Nan's (because it
                # tried to fill the data frame but failed) breaking the code.
                data_samples = Data(
                    f"samples{self.__number_simulations}",
                    quantities=list(self.sequence_lengths),
                )
                # FIXME Make the lists to strings.
                list_of_lists = [
                    [[list(x) for x in a] for a in b] for b in self.outcome_samples
                ]
                # FIXME Make the lists to strings.
                for count in range(self.runs):
                    # The data object takes dictionaries.
                    data_samples.add(
                        {
                            self.sequence_lengths[i]: str(list_of_lists[count][i])
                            for i in range(len(self.sequence_lengths))
                        }
                    )
        return data_samples

    @data_samples.setter
    def data_samples(self, gdata):
        """ """
        # Extract the data frame.
        dataframe = gdata.df
        # Put them in a list, first axis is the different runs, second axis
        # the sequence lengths.
        samples_list = dataframe.values.tolist()
        # FIXME
        if samples_list and type(samples_list[0][0]) == str:
            samples_list = [[literal_eval(x) for x in a] for a in samples_list]
        # Check if the attribute does not exist yet.
        if not hasattr(self, "sequence_lengths") or self.sequence_lengths is None:
            # Get the sequence lengths.
            sequence_lenghts = dataframe.columns.tolist()
            # The pickeling process reverses the order, reoder ot.
            self.sequence_lengths = np.array(sequence_lenghts[::-1])
        # Store the outcome as an attribute to further work with its.
        self.outcome_samples = [x[::-1] for x in samples_list]

    @property
    def data_probabilities(self):
        """ """
        # Store the data in a pandas dataframe.
        # FIXME There are versions not supporting writing arrays to data frames.
        try:
            # Initiate the data structure where the outcomes will be stored.
            data_probs = Data(
                f"probabilities{self.__number_simulations}",
                quantities=list(self.sequence_lengths),
            )
            # The columns are indexed by the different sequence lengths.
            # The rows are indexing the different runs.
            for count in range(self.runs):
                # The data object takes dictionaries.
                data_probs.add(
                    {
                        self.sequence_lengths[i]: self.outcome_probabilities[count][i]
                        for i in range(len(self.sequence_lengths))
                    }
                )
        # This error: ValueError: Must have equal len keys and value when
        # setting with an iterable
        # is caught by this.
        except ValueError:
            # Initiate the data structure where the outcomes will be stored.
            # If the initialization of the data objext is not overwritten here,
            # the first row will be filled with Nan's (because it tried to fill
            # the data frame but failed) breaking the code.
            data_probs = Data(
                f"probabilities{self.__number_simulations}",
                quantities=list(self.sequence_lengths),
            )
            # FIXME Make the lists to strings.
            for count in range(self.runs):
                # The data object takes dictionaries.
                data_probs.add(
                    {
                        self.sequence_lengths[i]: str(
                            self.outcome_probabilities[count][i]
                        )
                        for i in range(len(self.sequence_lengths))
                    }
                )
        return data_probs

    @data_probabilities.setter
    def data_probabilities(self, gdata):
        """ """
        # Extract the data frame.
        dataframe = gdata.df
        # Put them in a list, first axis is the different runs, second axis
        # the sequence lengths.
        probabilities_list = dataframe.values.tolist()
        # FIXME
        if type(probabilities_list[0][0]) == str:
            probabilities_list = [
                [literal_eval(x) for x in a] for a in probabilities_list
            ]
        # Check if the attribute does not exist yet.
        if not hasattr(self, "sequence_lengths") or self.sequence_lengths is None:
            # Get the sequence lengths.
            sequence_lenghts = dataframe.columns.tolist()
            # The pickeling process reverses the order, reoder ot.
            self.sequence_lengths = np.array(sequence_lenghts[::-1])
        # Store the outcome as an attribute to further work with its.
        self.outcome_probabilities = [x[::-1] for x in probabilities_list]

    ############################## CLASS METHODS ##############################

    @classmethod
    def retrieve_from_path(cls, path: str, **kwargs):
        """ """
        from qibocal.calibrations.protocols.utils import dict_from_comments_txt

        # Initiate an instance of the experiment class.
        obj = cls()
        # Get the metadata in form of a dictionary.
        metadata_dict = dict_from_comments_txt(f"{path}metadata.txt")
        # The circuit generator has to be restored, this will get the class.
        Generator = eval(metadata_dict["circuit_generator"])
        # Build the generator.
        circuit_generator = Generator(metadata_dict["qubits"])
        # Write it to the dictionary.
        metadata_dict["circuit_generator"] = circuit_generator
        # Give the objects the attributes as a dictionary. Every attribute
        # would be overwritten by that.
        obj.__dict__ = metadata_dict
        # Store the diven path.
        obj.directory = path
        # Get the circuits list and make it an attribute.
        obj.load_circuits(path)
        # Try to load the outcomes.
        try:
            obj.load_samples(path)
            obj.load_probabilities(path)
        except FileNotFoundError:
            # If there are no outcomes (yet), there will be no files.
            print("No outcomes to retrieve.")
        return obj

    @classmethod
    def retrieve_from_dataobjects(cls, data_circs, data_samples, data_probs, **kwargs):
        """ """
        # Initiate an instance of the experiment class.
        obj = cls()
        # Put the three different data objects.
        obj.data_circuits = data_circs
        # If there were no samples this data structure is empty but that
        # should be no problem.
        obj.data_samples = data_samples
        obj.data_probabilities = data_probs
        # Retrieve the meta data which can be extracted from the data frames.
        obj.runs = len(obj.circuits_list)
        if obj.outcome_samples:
            obj.nshots = len(obj.outcome_samples[0][0])
        else:
            obj.nshots = None
        if obj.outcome_probabilities:
            amount_qubits = int(np.log2(len(obj.outcome_probabilities[0][0])))
            obj.qubits = list(range(amount_qubits))
        return obj

    ################################# METHODS #################################

    ############################## Build ##############################

    def build(self, **kwargs):
        """Build a list out of the circuits required to run for the wanted
        experiment.
        """
        # Use the __call__ function of the circuit generator to retrieve a
        # random circuit 'runs' many times for each sequence length.
        circuits_list = [
            [next(self.circuit_generator(length)) for length in self.sequence_lengths]
            for _ in range(self.runs)
        ]
        # Create an attribute.
        # TODO should that be necessary if the experiment is stored in a
        # pikle file?
        self.circuits_list = circuits_list
        return circuits_list

    def build_onthefly(self, **kwargs):
        """ """
        pass

    def build_a_save(self, **kwargs):
        """ """
        # Build the whole list of circuits.
        self.build(**kwargs)
        # Store the list of circuits.
        self.save_experiment(**kwargs)

    def build_noise(self, **kwargs):
        """ """
        pass

    ################################ Execute ################################

    def execute_experiment(self, **kwargs):
        """FIXME the circuits have to be build already (or loaded),
        add something to check that and if they were not build yet build or
        load them.
        TODO execute run by run and yield the data inbetween?

        Args:
            kwargs (dict):
                'paulierror_noiseparams' = [p1, p2, p3]
        """
        # Initiate the outcome lists, one for the single shot samples and
        # one for the probabilities.
        self.outcome_samples, self.outcome_probabilities = [], []
        # If the circuits are simulated and not run on quantum hardware, the
        # noise has to be simulated, too.
        if kwargs.get("paulierror_noiseparams"):
            # Insert artificial noise, namely random Pauli flips.
            pauli = PauliError(*kwargs.get("paulierror_noiseparams"))
            noise = NoiseModel()
            # The noise should be applied with each unitary in the circuit.
            # noise.add(pauli, gates.Unitary)
            noise.add(pauli, gates.X)
        # Makes code easier to read.
        amount_m = len(self.sequence_lengths)
        # Loop 'runs' many times over the whole protocol.
        for count_runs in range(self.runs):
            # Initiate two lists to store the outcome for every sequence.
            probs_list, samples_list = [], []
            # Go through every sequence in the protocol.
            for count_m in range(amount_m):
                # Get the circuit. Deepcopy it to not alter the
                # the original circuit.
                circuit = deepcopy(self.circuits_list[count_runs][count_m])
                # For the simulation the noise has to be added to the circuit.
                if kwargs.get("paulierror_noiseparams"):
                    # Add the noise to the circuit (more like the other way
                    # around, the circuit to the noise).
                    noisy_circuit = noise.apply(circuit)
                    # Execute the noisy circuit.
                    executed = noisy_circuit(
                        kwargs.get('init_state'), nshots=self.nshots)
                else:
                    # Execute the qibo circuit without artificial noise.
                    executed = circuit(
                        kwargs.get('init_state'), nshots=self.nshots)
                # FIXME The samples (zeros and ones per shot) acquisition does
                # not work for quantum hardware yet.
                try:
                    # Get the samples from the executed gate. It should be an
                    # object filled with as many integers as used shots.
                    # Append the samples.
                    samples_list.append(executed.samples())
                except:
                    print("Retrieving samples not possible.")
                    # pass
                # Either way store the probabilities. Since
                # 'executed.probabilities()' only contains an entry for qubit
                # if it is nonzero, the shape can vary, fix that FIXME.
                # Store them.
                probs_list.append(list(executed.probabilities()))
            # For each run store the temporary lists in the attribute.
            # It could happend that the samples list is empty if the samples
            # cannot be retrieved.
            self.outcome_samples.append(samples_list)
            self.outcome_probabilities.append(probs_list)
        self.__number_simulations += 1

    def execute_a_save(self, **kwargs):
        """ """
        self.execute_experiment(**kwargs)
        self.save_outcome(**kwargs)

    ###################### Datastructures and save/load ######################

    def make_directory(self, **kwargs):
        """Make the directory where the experiment will be stored."""
        from datetime import datetime

        overall_dir = "experiments/"
        # Check if the overall directory exists. If not create it.
        if not isdir(overall_dir):
            mkdir(overall_dir)
        # Get the current date and time.
        dt_string = datetime.now().strftime("%y%b%d_%H%M%S")
        # Get the name of the generator.
        gname = self.circuit_generator.__class__.__name__
        # Every generator for the circuits gets its own directory.
        directory_generator = f"{overall_dir}{gname}/"
        if not isdir(directory_generator):
            mkdir(directory_generator)
        # Name the final directory for this experiment.
        directory = f"{directory_generator}experiment{dt_string}/"
        if not isdir(directory):
            mkdir(directory)
        # Store this as an attribute.
        self.directory = directory
        return directory

    def save_circuits(self, **kwargs) -> None:
        """Save the given circuits list.
        FIXME if the circuits were executed already this does not work!!

        Args:
            kwargs (dict)

        Returns:
            None
        """
        # Check if there has been made a directory already for this experiment.
        if not hasattr(self, "directory"):
            # Make and get the directory.
            self.make_directory()
        # Use the property.
        data_circs = self.data_circuits
        # Save the circuits in pickle format.
        data_circs.to_pickle(self.directory)

    def save_metadata(self, **kwargs):
        """ """
        # Check if there has been made a directory already for this experiment.
        if not hasattr(self, "directory"):
            # Make and get the directory.
            self.make_directory()
        # Store the metadata in a .txt file. For that create a dictionary.
        # Store any parameters given through kwargs.
        metadata_dict = {
            "qubits": self.qubits,
            "nshots": self.nshots,
            "runs": self.runs,
            "inverse": self.inverse,
            "circuit_generator": self.circuit_generator.__class__.__name__,
        }
        # One file in the directory stores the meta data.
        metadata_filename = f"{self.directory}metadata.txt"
        # Write the meta data as comments to the .txt file.
        dict_to_txt(metadata_filename, metadata_dict, openingstring="w")
        # The file is automatically closed.

    def save_experiment(self, **kwargs):
        """ """
        self.save_metadata(**kwargs)
        self.save_circuits(**kwargs)
        return self.directory

    def save_outcome(self, **kwargs):
        """ """
        # Check if there has been made a directory already for this experiment.
        if not hasattr(self, "directory"):
            # Make and get the directory.
            self.make_directory()
        if isfile(f"{self.directory}metadata.txt"):
            dict_to_txt(
                f"{self.directory}metadata.txt",
                kwargs,
                comments=True,
                openingstring="a",
            )
        # Use the properties.
        data_probs = self.data_probabilities
        data_samples = self.data_samples
        # Save the data structures.
        data_samples.to_pickle(self.directory)
        data_probs.to_pickle(self.directory)

    def load_circuits(self, path: str, **kwargs):
        """ """
        if isfile(f"{path}circuits.pkl"):
            # Get the pandas data frame from the pikle file.
            sequences_frompkl, circuits_list = pkl_to_list(f"{path}circuits.pkl")
            # Check if the attribute does not exist yet.
            if not hasattr(self, "sequence_lengths"):
                # The pickeling process reverses the order, reoder ot.
                self.sequence_lengths = list(np.array(sequences_frompkl)[::-1])
            # Store the outcome as an attribute to further work with its.
            self.circuits_list = [x[::-1] for x in circuits_list]
            return self.circuits_list
        else:
            raise FileNotFoundError("There is no file for circuits.")

    def load_samples(self, path: str, **kwargs):
        """ """
        if isfile(f"{path}samples.pkl"):
            # Get the pandas data frame from the pikle file.
            sequences_frompkl, samples_list = pkl_to_list(f"{path}samples.pkl")
            # Make sure that the order is the same.
            assert np.array_equal(
                np.array(sequences_frompkl)[::-1], self.sequence_lengths
            ), "The order of the restored outcome is not the same as when build"
            # Store the outcome as an attribute to further work with its.
            self.outcome_samples = np.array(
                [x[::-1] for x in samples_list]).reshape(
                    self.runs,
                    len(self.sequence_lengths),
                    self.nshots,
                    len(self.qubits)
                )
            return self.outcome_samples
        else:
            raise FileNotFoundError("There is no file for samples.")

    def load_probabilities(self, path: str, **kwargs):
        """ """
        if isfile(f"{path}probabilities.pkl"):
            # Get the pandas data frame from the pikle file.
            sequences_frompkl, probabilities_list = pkl_to_list(
                f"{path}probabilities.pkl"
            )
            # Make sure that the order is the same, right now the
            # order is reversed.
            assert np.array_equal(
                np.array(sequences_frompkl)[::-1], self.sequence_lengths
            ), "The order of the restored outcome is not the same as when build"
            # Store the outcome as an attribute to further work with its.
            self.outcome_probabilities = [x[::-1] for x in probabilities_list]
            return self.outcome_probabilities
        else:
            raise FileNotFoundError("There is no file for probabilities.")

    ########################### Outcome processing ###########################

    def probabilities(
        self,
        averaged: bool = True,
        run: int = None,
        from_samples: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """TODO this is exactly what resutl.frequencies() is doing!"""
        # Check if the samples attribute is not empty e.g. the first entry is
        # not just an empty list.
        if (
            len(self.outcome_samples[0]) != 0
            and from_samples
            and self.nshots is not None
        ):
            # Create all possible state vectors.
            allstates = np.array(list(product([0, 1], repeat=len(self.qubits))))
            # The attribute should be lists out of lists out of lists out
            # of lists, make it an array.
            samples = np.array(self.outcome_samples)
            if averaged:
                # Put the runs together, now the shape is
                # (amount sequences, runs*nshots, qubits).
                samples_conc = np.concatenate(samples, axis=1)
                # For each sequence length count the different state vectors and
                # divide by the total number of shots.
                probs = [
                    [
                        np.sum(np.product(samples_conc[countm] == state, axis=1))
                        for state in allstates
                    ]
                    for countm in range(len(self.sequence_lengths))
                ]
                probs = np.array(probs) / (self.runs * self.nshots)
            else:
                # If only a specific run (runs) is requested, choose that one.
                if run:
                    # Since the concatination in the next step only works when
                    # there are 4 dimensions, reshape it to 4 dimensions.
                    samples = samples[run].reshape(
                        -1, len(self.sequence_lengths), self.nshots, len(self.qubits)
                    )
                # Do the same thing as above just for every run.
                probs = [
                    [
                        [
                            np.sum(
                                np.product(samples[countrun, countm] == state, axis=1)
                            )
                            for state in allstates
                        ]
                        for countm in range(len(self.sequence_lengths))
                    ]
                    for countrun in range(len(samples))
                ]
                probs = np.array(probs) / (self.nshots)
        else:
            # The actual probabilites are used.
            probs = np.array(self.outcome_probabilities)
            if averaged:
                # If needed, average over the different runs for each sequence
                # length.
                probs = np.average(probs, axis=0)
            # Or pick a run. But if averaged is set to True this will not
            # happen.
            if run:
                probs = probs[run]
        return probs

    ############################ Filter functions ############################

    def filter_single_qubit(self, averaged: bool = True, **kwargs):
        """ """
        d = 2
        amount_sequences = len(self.sequence_lengths)
        # Initiate the list were the filter values will be stored.
        filterslist = []
        # If shots are available, use this way of calculating the filters.
        if len(self.outcome_samples[0]) != 0:
            # Go through all the runs one by one.
            for count in range(self.runs):
                # Go through each sequence length
                for m in range(amount_sequences):
                    # Get the circuit which was used to produce the data.
                    mycircuit = self.circuits_list[count][m]
                    # Execute it (of course without noise).
                    executed_circuit = mycircuit(nshots=self.nshots)
                    # Initiate the variable to average.
                    filterf = 0
                    # Go throught each shot outcome.
                    for count_shot in range(self.nshots):
                        # This is 0 or 1.
                        outcome = self.outcome_samples[count][m][count_shot][0]
                        # Take the probability that the ideal circuit has that
                        # outcome too.
                        prob = executed_circuit.execution_result[int(outcome)]
                        # Average over it with a renormalization.
                        filterf += (d + 1) * (np.abs(prob) - 1 / d)
                    # Divide by number of shots and append.
                    filterslist.append(filterf / self.nshots)
            # Reshape such that each run again is an array filled with filter
            # values corresponding to each sequence length.
            filtersarray = np.array(filterslist).reshape(self.runs, amount_sequences)
        # If not shots are available, use the probabilities.
        else:
            # Get the probabilities from the faulty circuits execution.
            probs = self.probabilities(averaged=False)
            # Go through every run and sequence length.
            for count in range(self.runs):
                for m in range(amount_sequences):
                    # Retrieve the ideal circuit.
                    mycircuit = self.circuits_list[count][m]
                    # Get the probability for ground state and excited state.
                    talpha, tbeta = mycircuit().probabilities()
                    alpha = np.sqrt(probs[count, m, 0])
                    beta = np.sqrt(probs[count, m, 1])
                    filterf = (d + 1) * (
                        np.abs(alpha * talpha + beta * tbeta) ** 2 - 1 / d
                    )
                    filterslist.append(filterf)
            filtersarray = np.array(filterslist).reshape(self.runs, amount_sequences)
        if averaged:
            filtersarray = np.average(filtersarray, axis=0)
        return filtersarray

    def plot_scatterruns(self, **kwargs):
        """ """
        import matplotlib.pyplot as plt

        colorfunc = plt.get_cmap("inferno")
        xdata = self.sequence_lengths
        xdata_scattered = np.tile(xdata, self.runs)
        if self.inverse:
            ydata_scattered = self.probabilities(averaged=False)[:, :, 0]
            fitting_func = fitting_methods.fit_exp1_func
        elif kwargs.get('sign'):
            ydata_scattered = self.filter_sign()
            fitting_func = fitting_methods.fit_exp2_func
        else:
            ydata_scattered = self.filter_single_qubit(averaged=False)
            fitting_func = fitting_methods.fit_exp2_func
        plt.scatter(
            xdata_scattered,
            ydata_scattered.flatten(),
            marker="_",
            linewidths=5,
            s=100,
            color=colorfunc(100),
            alpha=0.4,
            label="each run",
        )
        ydata = np.average(ydata_scattered, axis=0)
        # pdb.set_trace()
        plt.scatter(xdata, ydata, marker=5, label="averaged")
        xfitted, yfitted, popt = fitting_func(xdata, ydata)
        fitlegend =  ', '.join(format(f, '.3f') for f in popt)
        plt.plot(xfitted, yfitted, "--", color=colorfunc(50), label=fitlegend)
        plt.ylabel("survival probability")
        plt.xlabel("sequence length")
        plt.legend()
        plt.show()

    def crossvalidation(self, k: int, iterations: int, **kwargs):
        """Repeated random sub-sampling validation without the training,
        only testing.

        """
        import matplotlib.pyplot as plt

        colorfunc = plt.get_cmap("inferno")
        # Retrieve the single survival probabilities for each run.
        if self.inverse:
            ydata_scattered = self.probabilities(averaged=False)[:, :, 0]
        else:
            ydata_scattered = self.filter_single_qubit(averaged=False)
        # Store the sequence lengths for fitting purposes.
        xdata = self.sequence_lengths
        fittingparam = kwargs.get("fittingparam", 1)
        params_list = []
        # Loop over the amount of wanted iterations and draw k many samples
        # such that each time a random set of different runs is used to
        # calculate the decay parameters of the average of the given data.
        for _ in range(iterations):
            # Draw the random indices and get the belonging data.
            rand_data = ydata_scattered[np.random.randint(0, self.runs, size=k)]
            # Calculate the average and get the fitting parameters.
            xfit, yfit, popt = fitting_methods.fit_exponential(
                xdata, np.average(rand_data, axis=0))
            # In popt three fitting parameters are stored for A*f^x+B, in this
            # order. Make the tuple a list and store them.
            params_list.append(popt[fittingparam])
        # Plot the calculated fitting parameters.
        # Make two plots. A scatter plot and an histogram.
        plt.subplots(2, 1, figsize=(7, 7))
        plt.subplot(2, 1, 1)
        plt.scatter(
            params_list,
            np.zeros(iterations),
            marker="|",
            linewidths=5,
            s=150,
            color=colorfunc(50),
            alpha=0.4,
            label=f"{iterations} subsampling group of size {k}",
        )
        plt.plot(
            np.average(params_list),
        )
        plt.subplot(2, 1, 2)
        plt.hist(params_list)
        plt.legend()
        plt.show()

    def filter_sign(self):
        """
        """
        amount_sequences = len(self.sequence_lengths)
        # Initiate list for filter for each sequence length and all runs.
        avg_filterlist = []
        # Go through all the runs one by one.
        for count in range(self.runs):
            # Go through each sequence length
            for m in range(amount_sequences):
                # Get the circuit which was used to produce the data.
                mycircuit = self.circuits_list[count][m]
                # Count amount of X gates in the queue. TODO temporary!
                m_X = mycircuit.draw().count("X")
                filtersign = 0
                # Go throught each shot outcome.
                for count_shot in range(self.nshots):
                    # This is 0 or 1.
                    outcome = self.outcome_samples[count][m][count_shot][0]
                    filtersign += (-1)**(m_X%2+outcome)/2.
                avg_filterlist.append(filtersign/self.nshots)
        final = np.array(avg_filterlist).reshape(self.runs, amount_sequences)
        return final

                



