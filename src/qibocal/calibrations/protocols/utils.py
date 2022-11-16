from os import mkdir
from os.path import isdir, isfile

import numpy as np
from pandas import read_pickle

# To not define the parameters for one qubit Cliffords every time a
# new qubits is drawn define the parameters as global variable.
# This are parameters for all 24 one qubit clifford gates.
ONEQUBIT_CLIFFORD_PARAMS = [
    (0, 0, 0, 0),
    (np.pi, 1, 0, 0),
    (np.pi, 0, 1, 0),
    (np.pi, 0, 0, 1),
    (np.pi / 2, 1, 0, 0),
    (-np.pi / 2, 1, 0, 0),
    (np.pi / 2, 0, 1, 0),
    (-np.pi / 2, 0, 1, 0),
    (np.pi / 2, 0, 0, 1),
    (-np.pi / 2, 0, 0, 1),
    (np.pi, 1 / np.sqrt(2), 1 / np.sqrt(2), 0),
    (np.pi, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)),
    (np.pi, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)),
    (np.pi, -1 / np.sqrt(2), 1 / np.sqrt(2), 0),
    (np.pi, 1 / np.sqrt(2), 0, -1 / np.sqrt(2)),
    (np.pi, 0, -1 / np.sqrt(2), 1 / np.sqrt(2)),
    (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)),
    (2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
    (-2 * np.pi / 3, 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)),
]

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
pauli = [np.eye(2) / np.sqrt(2), X / np.sqrt(2), Y / np.sqrt(2), Z / np.sqrt(2)]


def experiment_directory(name: str):
    """Make the directory where the experiment will be stored."""
    from datetime import datetime

    overall_dir = "experiments/"
    # Check if the overall directory exists. If not create it.
    if not isdir(overall_dir):
        mkdir(overall_dir)
    # Get the current date and time.
    dt_string = datetime.now().strftime("%y%b%d_%H%M%S")
    # Every script name ``name`` gets its own directory.
    subdirectory = f"{overall_dir}{name}/"
    if not isdir(subdirectory):
        mkdir(subdirectory)
    # Name the final directory for this experiment.
    final_directory = f"{subdirectory}experiment{dt_string}/"
    if not isdir(final_directory):
        mkdir(final_directory)
    return final_directory


def liouville_representation_errorchannel(error_channel, **kwargs):
    """For single qubit error channels only."""
    # For single qubit the dimension is two.
    if error_channel.channel.__name__ == "PauliNoiseChannel":
        flipprobs = error_channel.options

        def acts(gmatrix):
            return (
                (1 - flipprobs[0] - flipprobs[1] - flipprobs[2]) * gmatrix
                + flipprobs[0] * X @ gmatrix @ X
                + flipprobs[1] * Y @ gmatrix @ Y
                + flipprobs[2] * Z @ gmatrix @ Z
            )

    return np.array(
        [[np.trace(p2.conj().T @ acts(p1)) for p1 in pauli] for p2 in pauli]
    )


def effective_depol(error_channel, **kwargs):
    """ """
    liouvillerep = liouville_representation_errorchannel(error_channel)
    d = int(np.sqrt(len(liouvillerep)))
    depolp = ((np.trace(liouvillerep) + d) / (d + 1) - 1) / (d - 1)
    return depolp


def dict_to_txt(
    filename: str, gdict: dict, comments: bool = True, openingstring: str = "a"
) -> None:
    """Writes a dictionary line by line as comments in a given file.

    Parameters
    ----------
    filename : str
        The whole name (with the right dictionary) for the
        file which will be writting in.
    gdict : dict
        The dictionary from which each key and value pair
        is written into a new line as comment.
    comments: bool (optional) default = True
        Set whether there is a # in the beginning of each line or not.
    openingstring : str
        Is given as an argument when opening the file,
        'a', 'w', 'x' are valid options.
    """
    # Open the file.
    with open(filename, openingstring) as f:
        # Go through the whole dictionary 'gdict', take each key
        # and store the key and the corresponding value, either
        # with a # in the beginning of the line or not.
        if comments:
            # # key1 : value1
            # # key2 : values 2
            # ...
            for key in gdict:
                f.write(f"# {key} : {gdict.get(key)} \n")
        else:
            # key1 : value1
            # key2 : values 2
            # ...
            for key in gdict:
                f.write(f"{key} : {gdict.get(key)} \n")
        # Closes automatically


def dict_from_comments_txt(filename: str):
    """Assumes that all comments are only in the beginning
    of the file.
    The comments should look like this:
    key1 : something
    key2 : something else
    """
    from ast import literal_eval

    # Initiate the dictionary in which the values are stored.
    comments_dict = {}
    with open(filename) as fh:
        for curline in fh:
            # Check if the current line starts with "#"
            if curline.startswith("#"):
                index = curline.find(":")
                key = curline[2 : index - 1]
                value = curline[index + 1 :]
                value = value.replace("\n", "").strip()
                # Try to convert 'value' which is a string to a list
                # or integer or tuple or whatever,  if it does not succeed
                # let the string be a string.
                try:
                    value = literal_eval(value)
                except:
                    value = value
                comments_dict[key] = value
            else:
                return comments_dict
        return comments_dict


def pkl_to_list(filename: str, **kwargs):
    """ """

    # Load the pickeled object.
    dataframe = read_pickle(filename)
    # Put them in a list, first axis is the different runs, second axis
    # the sequence lengths.
    object_list = dataframe.values.tolist()
    # Get the sequence lengths.
    sequence_lenghts = dataframe.columns.tolist()
    return sequence_lenghts, object_list
