from numpy import sqrt, pi
import pdb
# To not define the parameters for one qubit Cliffords every time a
# new qubits is drawn define the parameters as global variable.
# This are parameters for all 24 one qubit clifford gates.
onequbit_clifford_params = [
    (0, 0, 0, 0), (pi, 1, 0, 0), (pi,0, 1, 0),
    (pi, 0, 0, 1), (pi/2, 1, 0, 0), (-pi/2, 1, 0, 0),
    (pi/2, 0, 1, 0), (-pi/2, 0, 1, 0), (pi/2, 0, 0, 1),
    (-pi/2, 0, 0, 1),
    (pi, 1/sqrt(2), 1/sqrt(2), 0),
    (pi, 1/sqrt(2), 0, 1/sqrt(2)),
    (pi, 0, 1/sqrt(2), 1/sqrt(2)),
    (pi, -1/sqrt(2), 1/sqrt(2), 0),
    (pi, 1/sqrt(2), 0, -1/sqrt(2)),
    (pi, 0, -1/sqrt(2), 1/sqrt(2)),
    (2*pi/3, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    (-2*pi/3, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    (2*pi/3, -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    (-2*pi/3, -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    (2*pi/3, 1/sqrt(3), -1/sqrt(3), 1/sqrt(3)),
    (-2*pi/3, 1/sqrt(3), -1/sqrt(3), 1/sqrt(3)),
    (2*pi/3, 1/sqrt(3), 1/sqrt(3), -1/sqrt(3)),
    (-2*pi/3, 1/sqrt(3), 1/sqrt(3), -1/sqrt(3))]

def dict_to_txt(filename:str, gdict:dict, comments:bool=True,
        openingstring:str='a') -> None:
    """ Writes a dictionary line by line as comments in a given file.

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
                f.write(f'# {key} : {gdict.get(key)} \n')
        else:
            # key1 : value1
            # key2 : values 2
            # ...
            for key in gdict:
                f.write(f'{key} : {gdict.get(key)} \n')
        # Closes automatically

def dict_from_comments_txt(filename:str):
    """ Assumes that all comments are only in the beginning
    of the file.
    The comments should look like this:
    key1 : something
    key2 : something else
    """
    from ast import literal_eval
    
    # Initiate the dictionary in which the values are stored.
    comments_dict = {}
    with open(filename,'r') as fh:
        for curline in fh:
            # Check if the current line starts with "#"
            if curline.startswith("#"):
                index = curline.find(':')
                key = curline[2:index-1]
                value = curline[index+1:]
                value = value.replace('\n', '').strip()
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

def pkl_to_list(filename:str, **kwargs):
        """
        """
        from pandas import read_pickle
        # Load the pickeled object.
        dataframe = read_pickle(filename)
        # Put them in a list, first axis is the different runs, second axis
        # the sequence lengths.
        object_list = dataframe.values.tolist()
        # Get the sequence lengths.
        sequence_lenghts = dataframe.columns.tolist()
        return sequence_lenghts, object_list


