How to add a new protocol
=========================

In this tutorial we show how to add new protocol to ``Qibocal``.

Protocol implementation in ``Qibocal``
--------------------------------------

Currently, characterization/calibration protocols in ``Qibocal`` are implemented
by performing a clear separation between *input parameters*, *data acquired* and
*results*.

These are then connected through the following functions:

* `acquisition` that receives as input `parameters` and outputs `data`
* `fit` that receives as input `data` and outputs `results`
* `plot` that receives as input `data` and `results` to visualize the protocol

We believe that this approach is useful in order to code protocols that
aims at achieving a specific task. However, this approach is flexible enough
to show some data acquired without performing a post-processing analysis.

Tutorial
--------

All protocols are located in `src/qibocal/protocols/characterization <https://github.com/qiboteam/qibocal/tree/main/src/qibocal/protocols/characterization>`_.
Suppose that we want to code a protocol to perform a RX rotation for different
angles.

We create a file ``rotate.py`` in ``src/qibocal/protocols/characterization``.

First, we define the input parameters.
