
==================================
Qubit characterization via samples
==================================

How does extracting features via sampling work?
===============================================

The aim of this method is to reduce the number of calls to the chip as much as possible while still extracting
information on the features of different calibration routines.

The method is built for routines that sweep over parameters, and try to extract a feature that is changes smoothly
along the sweep. We choose a sample point gaussianly distributed around an initial guess with a given standard variation.
The new sampling point is executed in the machine, and compared with a background noise. If the extracted sample
deviates from the background noise, that is, if the absolute signal to noise ratio is high, we consider that we have
found a potential feature. Otherwise we draw a new sample. When a feature is detected, a small sweep around that value is
performed in order to fine-tune the value. This is then used as initial guess for the next iteration of the characterization
protocol, with changed parameters.

This algorithm closely follows a feature as the experimental parameters are slowly shifted.

We allow for input on the precision and range the samples are taken around the initial guess. As well as the precision on the
fast sweep used to fine tune the result (see `runcard <https://github.com/qiboteam/qibocal/blob/main/runcards/actions_sample.yml/>`_).

The "sample" methods will contain the following entries:

.. code-block:: yaml

    max_runs: 40
    thr: 5
    spans: [10000000, 5000000]
    small_spans: [1000000, 100000]
    resolution: 100000

**max_runs**: sets the maximum number of gaussianly distributed samples to extract before the algorithm decides the feature
is not within the given range.

**thr**: minimum threshold in signal to noise ratio where the algorithm will decide that the feature has been detected and
will move on to the next step.

**spans**: list with the spans where the new parameters will be sampled around. A list with a deacreasing span will find the
feature with more precision, at a cost of running the sampling algorothm more times. Knowledge of the scale of the desired
feature greatly helps in designing the span list. Example showcased in Hz, for a resonator punchout.

**small_spans**: list of spans for the small scans that are run when the feature is located. Scans of 10 equaly distributes
points will be executed. This will directly correlate to the final desired precision of the feature. Example showcased in Hz,
for a resonator punchout.

**resolution**: precision in the sampled space for the gaussianly distributed samples. In the initial part of the algorithm,
this value will be the precision used to detect the feature. This value needs to be small enough for the feature to be
detactable.


The sampling based characterization can be adapted to other routines, as long as they fulfill the desired criteria.
