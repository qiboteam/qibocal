import numpy as np

def min_max_normalize(min_val, max_val, data):
    """Normalize each value in the data to the range -1 to 1"""  
    return [((x - min_val) / (max_val - min_val)) * 2 - 1 for x in data]