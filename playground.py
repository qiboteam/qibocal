from qibo import gates

from qibocal.calibrations.protocols.noisemodels import (
    PauliErrorOnUnitary,
    PauliErrorOnX,
)


def try_standardrb():
    import qibocal.calibrations.protocols.standardrb as rb

    nqubits = 1
    depths = [0, 3, 5, 10]
    runs = 1
    qubits = [0]
    nshots = 1000
    noisemodel = PauliErrorOnUnitary()
    factory = rb.moduleFactory(nqubits, depths, runs, qubits)
    experiment = rb.moduleExperiment(factory, nshots=nshots, noisemodel=noisemodel)
    experiment.perform(experiment.execute)
    rb.analyze(experiment, noisemodel).show()


def try_crosstalkrb():
    import qibocal.calibrations.protocols.crosstalkrb as rb

    nqubits = 2
    depths = [1, 2, 3, 5, 10]
    runs = 100
    nshots = 1
    noisemodel = PauliErrorOnUnitary(0.03, 0.1, 0.07)
    factory = rb.moduleFactory(nqubits, depths, runs)
    experiment = rb.moduleExperiment(factory, nshots=nshots, noisemodel=noisemodel)
    experiment.perform(experiment.execute)
    rb.analyze(experiment, noisemodel).show()


def try_XIdRB():
    import qibocal.calibrations.protocols.XIdrb as rb

    nqubits = 1
    depths = [1, 2, 3, 5, 10]
    runs = 10
    nshots = 100
    noisemodel = PauliErrorOnX(0.03, 0.1, 0.07)
    factory = rb.moduleFactory(nqubits, depths, runs)
    experiment = rb.moduleExperiment(factory, nshots=nshots, noisemodel=noisemodel)
    experiment.perform(experiment.execute)
    rb.analyze(experiment, noisemodel).show()


try_crosstalkrb()
try_XIdRB()
try_standardrb()
