CPMG spectroscopy
==================

This protocol is a variation of the CPMG sequence (see :doc:`cpmg`) aimed at
extracting :math:`T_2` as a function of the CPMG filter-function frequency,
rather than as a function of the number of pulses at fixed total duration.

For a CPMG sequence with fixed inter-pulse delay :math:`\tau` and :math:`n` pulses,
the noise filter function is peaked around :math:`\omega_0 \sim \pi / \tau`,
independently of :math:`n`. Therefore, fixing :math:`\tau` and sweeping
:math:`n` (i.e. sweeping the total free evolution time :math:`t = n \tau`)
allows extracting a coherence decay, and thus :math:`T_2(\tau)`, filtered at a
single, well defined frequency.

The protocol consists of two nested loops:

- **Inner loop**: for a fixed :math:`\tau`, the number of pulses :math:`n` is
  swept from ``min_number_pulses`` (``1`` corresponding to a spin-echo) up to
  ``max_duration / tau``. The resulting decay of the excited state probability
  is fitted against the actual elapsed time of the sequence

  .. math::

      t = n (\tau + t_{RY}) + 2 t_{RX90}

  which includes the duration of the CPMG (:math:`RY`) and boundary
  (:math:`RX90`) pulses, not just the waits, with a dumped exponential

  .. math::

      p_e(t) = A + B  e^{ - t / T_2(\tau)}

  to extract :math:`T_2(\tau)`.

- **Outer loop**: :math:`\tau` is swept from ``delay_between_pulses_start`` to
  ``delay_between_pulses_end`` in steps of ``delay_between_pulses_step``,
  repeating the inner loop for every value, so as to obtain :math:`T_2` as a
  function of :math:`\tau` (or, equivalently, of the CPMG filter frequency
  :math:`\pi / \tau`).

The parameters defining the :math:`\tau` range are inherited from
:class:`qibocal.protocols.coherence.spin_echo.SpinEchoParameters`, the same
ones used by the standard CPMG/spin-echo protocols, where they represent the
total free-evolution time of a single pulse; here they instead set the fixed
inter-pulse delay :math:`\tau` swept across the outer loop. ``unrolling``
defaults to ``True``, since for each :math:`\tau` the inner loop deploys one
sequence per value of :math:`n`.

Parameters
^^^^^^^^^^

.. autoclass:: qibocal.protocols.coherence.cpmg_spectroscopy.CpmgSpectroscopyParameters
  :noindex:

Example
^^^^^^^

A possible runcard to launch a CPMG spectroscopy experiment could be the following:

.. code-block:: yaml

    - id: CPMG spectroscopy
      operation: cpmg_spectroscopy
      parameters:
        delay_between_pulses_start: 40
        delay_between_pulses_end: 4000
        delay_between_pulses_step: 200
        min_number_pulses: 1
        max_duration: 40000
        nshots: 1000

:math:`T_2(\tau)` is determined by fitting, independently for each :math:`\tau`,
the decay of the excited state probability vs the total elapsed time :math:`t`
using the formula presented above.

Requirements
^^^^^^^^^^^^

- :ref:`single-shot`
