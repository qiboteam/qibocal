import importlib

from qibo.noise import NoiseModel


def execute_simulation(
    module_name: str,
    depths: list,
    nqubits: int = 1,
    nshots: int = 500,
    noise_model: NoiseModel = None,
    ndecays: int = None,
):
    """Execute simulation of XId Radomized Benchmarking experiment and generate an html report for a specific module.

    Args:
        module_name (str): name of the module in `qibocal.calibrations.niGSC` or path to the custom module.
        A module has to have `ModuleFactory`, `ModuleExperiment`, `post_processing_sequential`, `get_aggregational_data`, `build_report` and an optional function `add_validation`.
        depths (list): list of depths for circuits
        nqubits (int): number of qubits.
        nshots (int): number of shots per measurement
        noise_model (:class:`qibo.noise.NoiseModel`): noise model applied to the circuits in the simulation
        ndecays (int): number of decay parameters to fit. If None, gets the default number from the module.

    Example:
        .. testcode::
            import qibo
            from qibocal.calibrations.niGSC.basics.execute import execute_simulation
            from qibocal.calibrations.niGSC.basics.noisemodels import PauliErrorOnX

            qibo.set_backend("numpy")
            # Build the noise model.
            noise_params = [0.01, 0.02, 0.05]
            pauli_noise_model = PauliErrorOnX(*noise_params)
            # Generate the list of depths repeating 20 times
            runs = 20
            depths = list(range(1, 31)) * runs
            # Run the simulation
            nqubits = 1
            execute_simulation(
                module_name="XIdrb",
                depths=depths,
                nqubits=nqubits,
                nshots=500,
                noise_model=pauli_noise_model,
                ndecays=2,
            )
    """
    # Extract the module.
    try:
        module = importlib.import_module(f"qibocal.calibrations.niGSC.{module_name}")
    except:
        module = importlib.import_module(module_name)

    # Execute an experiment.
    factory = module.ModuleFactory(nqubits, depths)
    experiment = module.ModuleExperiment(
        factory, nshots=nshots, noise_model=noise_model
    )
    experiment.perform(experiment.execute)

    # Post-processing phase.
    module.post_processing_sequential(experiment)
    aggr_df = module.get_aggregational_data(experiment, ndecays=ndecays)

    # Add theoretical validation if possible.
    try:
        aggr_df = module.add_validation(experiment, aggr_df)
    except:
        pass

    # Build a report.
    report_figure = module.build_report(experiment, aggr_df)
    report_figure.show()
