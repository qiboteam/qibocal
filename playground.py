# from qibo import gates

# from qibocal.calibrations.protocols.noisemodels import (
#     PauliErrorOnUnitary,
#     PauliErrorOnX,
# )

# Define the necessary variables.
nqubits = 1  # Number of qubits in the quantum hardware.
depths = [0, 1, 4]  # How many random gates there are in each circuit.
runs = 2  # The amount of repetitions of the whole experiment.
nshots = 5  # When a circuit is executed how many shots are used.

# To not alter the iterator when using it, make deep copies.
from copy import deepcopy

from qibocal.calibrations.protocols import standardrb

factory = standardrb.moduleFactory(nqubits, depths, runs)
# ``factory`` is an iterator class object generating single clifford
# gates with the last gate always the inverse of the whole gate sequence.
# There are mainly three ways how to extract the circuits.
# 1. Make a list out of the iterator object.
circuits_list1 = list(deepcopy(factory))
# 2. Use a for loop.
circuits_list2 = []
for circuit in deepcopy(factory):
    circuits_list2.append(circuit)
# 3. Make an iterator and extract the circuits with the next method.
iter_factory = iter(deepcopy(factory))
circuits_list3, iterate = [], True
while iterate:
    try:
        circuits_list3.append(next(iter_factory))
    except StopIteration:
        iterate = False
# All the three lists have circuits constructed with
# single clifford gates according to the ``depths``list,
# repeated ``runs``many times.


def try_standardrb():
    import qibocal.calibrations.protocols.standardrb as rb

    nqubits = 1
    depths = [0, 3, 5, 10]
    runs = 100
    qubits = [0]
    nshots = 1000
    noisemodel = PauliErrorOnUnitary()
    factory = rb.moduleFactory(nqubits, depths, runs, qubits)
    experiment = rb.moduleExperiment(factory, nshots=nshots, noisemodel=noisemodel)
    experiment.perform(experiment.execute)

    rb.post_processing_sequential(experiment)
    df_aggr = rb.get_aggregational_data(experiment)
    fig = rb.build_report(experiment, df_aggr)
    fig.show()


def try_correlatedrb():
    import qibocal.calibrations.protocols.correlatedrb as rb

    nqubits = 2
    depths = [1, 3, 5, 10]
    runs = 10
    nshots = 1
    noisemodel = PauliErrorOnUnitary()
    factory = rb.moduleFactory(nqubits, depths, runs)
    experiment = rb.moduleExperiment(factory, nshots=nshots, noisemodel=noisemodel)
    experiment.perform(experiment.execute)

    rb.post_processing_sequential(experiment)
    print(experiment.dataframe)
    df_aggr = rb.get_aggregational_data(experiment)
    fig = rb.build_report(experiment, df_aggr)
    fig.show()


def try_XIdRB():
    import qibocal.calibrations.protocols.XIdrb as rb

    nqubits = 2
    depths = [1, 2, 3, 4, 5, 6, 7]
    runs = 100
    nshots = 10
    noisemodel = PauliErrorOnX()
    factory = rb.moduleFactory(nqubits, depths, runs)
    experiment = rb.moduleExperiment(factory, nshots=nshots, noisemodel=noisemodel)
    experiment.perform(experiment.execute)

    rb.post_processing_sequential(experiment)
    print(experiment.dataframe)
    df_aggr = rb.get_aggregational_data(experiment)
    fig = rb.build_report(experiment, df_aggr)
    fig.show()


# try_standardrb()
# try_correlatedrb()
# try_XIdRB()

# from qibocal.plots import gateset
# folder = '2023-01-23-000-jadwiga-wilkens'
# routine = 'standardrb'

# from qibocal.web import report
# report.create_report(folder)

# print("".join(["{}:{:.3f}{} ".format(key, *(dfrow["popt"][key],'') if not isinstance(dfrow["popt"][key], complex) else (np.real(dfrow["popt"][key]), "{}{:3f}j".format('-' if np.imag(dfrow["popt"][key])<0 else '+', np.imag(dfrow["popt"][key])))) for key in dfrow["popt"]]))
