


def dummyrb(
    platform,
    qubit : list,
    experiment_name : str,
    inject_nois : list,
    nshots: int
):
    # There are several types of experiment objects
    #   1. the circuits are drawn randomly beforehand and stored (the numbers of it is a 
    # discrete set, don't forget the order/numbering!)
    #   2. They are drawn on the fly and the gate has to be known and stored somehow
    # experiment = convert_to_function(experiment_name)
    pass