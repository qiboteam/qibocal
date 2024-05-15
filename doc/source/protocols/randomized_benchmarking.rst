Standard RB
===========

Experiment Description
----------------------

Perform a standard RB protocol with a calibrated QPU.

Example Runcard
---------------

.. code-block::

    - id: standard rb
      operation: standard_rb
      targets: [1]
      parameters:
        depths: [1, 2, 3, 5]
        niter: 5
        nshots: 50
        noise_model: PauliErrorOnAll

Parameters
----------


- depths: a list of depths/sequence lengths. If a dictionary is given the list will be build
- niter: sets how many iterations over the same depth value
- uncertainties: method of computing the error bars of the signal and uncertainties of the fit. If ``None``, it computes the standard deviation. Otherwise it computes the corresponding confidence interval. Defaults `None`
- unrolling: iff ``True`` it uses sequence unrolling to deploy multiple circuits in a single instrument call.
- seed: a fixed seed to initialize ``np.random.Generator``. If ``None``, uses a random seed.
- noise_model: for simulation purposes, string has to match what is in
    :mod:`qibocal.protocols.characterization.randomized_benchmarking.noisemodels`"""
- noise_params: with this, the noise model will be initialized, if not given random values will be used
