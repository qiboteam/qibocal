from qibo import gates

from qibocal.calibrations.protocols.noisemodels import (
    PauliErrorOnUnitary,
    PauliErrorOnX,
)


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
try_XIdRB()
