.. _coupler_zz:

Coupler ZZ and JAZZ nulling
==========================

``coupler_zz`` and ``coupler_jazz`` measure the conditional frequency shift

.. math::

    \zeta(A) = f_{\mathrm{target},c=1}(A)
               - f_{\mathrm{target},c=0}(A)

in Hz versus a rectangular coupler flux-pulse amplitude. Both acquire the
target population with the control initially in each computational state,
fit both signals, and minimize :math:`|\zeta(A)|`. Targets are ordered
``[target, control]`` pairs. Only one pair can be acquired per action.

Unlike the minimum-variance heuristic in :ref:`coupler_ramsey_zz`, these
protocols estimate ZZ explicitly. They require calibrated single-qubit
rotations, target readout discrimination, and a pair-to-coupler assignment in
``platform.calibration.two_qubits`` with a configured coupler flux channel.
The control should remain coherent/excited over the measurement interval;
control relaxation and flux-induced leakage can bias the frequency estimates.

Conditional Ramsey
------------------

``coupler_zz`` applies a coupler pulse during the target Ramsey wait.
``delay_range`` sweeps the wait and pulse duration in ns. The target drive
is temporarily shifted by ``detuning`` in Hz, without changing the platform's
stored drive frequency. Each amplitude gives two fringe frequencies
:math:`F_0,F_1`. Provided both conditional target-drive detunings have the
same sign as the artificial detuning :math:`d`,

.. math::

    \zeta = -\mathrm{sign}(d)(F_1-F_0).

Population-only Ramsey cannot distinguish positive and negative frequencies.
Choose a nonzero ``detuning`` larger than the conditional shifts, while
keeping both fringe frequencies below the sweep's Nyquist frequency. A
conditional frequency crossing through zero invalidates this sign assignment.
``coupler_zz`` reports the best relative pulse amplitude but does not update
the DC bias automatically.

Echoed JAZZ
-----------

``coupler_jazz`` uses the following sequence for each initial control state:

.. code-block:: text

    target:                 RX90 -- tau -- RX -- tau -- R(phi, pi/2) -- MZ
    control: prepare 0 or 1 -------- tau -- RX -- tau ---------------- MZ
    coupler:                       pulse ----- pulse

Both qubits' central pi rotations start together. Each flux pulse is applied
only during its free-evolution half-wait; the coupler returns to its idle bias
during the echo rotations. Unequal echo-pulse durations are padded before the
second window. Readout follows the final target rotation at every swept delay.

Echoing both qubits refocuses unconditional target detuning while preserving
the ZZ phase. To retain measurable fringes at zero ZZ, the final target
rotation's axis is swept as :math:`\phi(\tau)=2\pi d\tau`. Here ``detuning``
specifies an analysis-phase ramp, not a drive-frequency update, and
``delay_range`` is **each half-wait**, not the total evolution time.

Using :math:`H/h=f_t n_t+f_c n_c+\zeta n_t n_c` and positive rotation-axis
phase, on the branch :math:`|d|>|\zeta|` the two unsigned fringe frequencies are
:math:`F_0=|d+\zeta|` and :math:`F_1=|d-\zeta|`, so

.. math::

    \zeta = -\frac{\mathrm{sign}(d)}{2}(F_1-F_0).

The factor of two follows from the opposite conditional echo phases.
The phase ramp is swept in lockstep with both flux-pulse durations.
It requires hardware support for simultaneous duration and relative-phase sweeps.

Fitting and minimum search
-------------------------

The fit uses both traces simultaneously with independent readout offsets,
contrasts and phases, and a shared target dephasing rate. Linear quadratures
are solved at each nonlinear step; multiple spectral starting points reduce
frequency-fit local minima. At least eight distinct increasing waits are
needed. Use enough delay span to resolve several carrier cycles while
retaining contrast. Nonuniform waits are supported, with a conservative
Nyquist limit based on the largest sampling interval.

Failed, flat, low-contrast or unresolved fits are logged and stored as NaN
for that amplitude. If neither control trace can be fitted at any amplitude,
the pair is omitted. The report shows both population heatmaps, ZZ estimates
with one-sigma residual-based uncertainties, and the local minimum fit.
The uncertainty assumes the shared-decay model describes the data; it does
not include leakage, control relaxation or other systematic errors.

A local quadratic is fitted to up to five contiguous valid amplitudes around
the smallest measured :math:`|\zeta|`. Its roots, stationary point and endpoints
are compared to minimize its absolute value. Thus a smooth nonzero minimum
can be located even without a zero crossing. No extrapolation or interpolation
across failed amplitudes is performed. With fewer than three contiguous
points, only a sampled minimum or a two-point zero crossing is reported.
Boundary minima are warned about and should be remeasured over a wider range.
A reported minimum is **not proof of zero coupling**: inspect the fitted
residual ZZ and refine the amplitude sweep.

For JAZZ, only an interior, polynomial-fitted minimum produces an updateable
``nulling_point``. Its absolute bias is the fitted pulse amplitude plus the DC
offset recorded during acquisition. Enabled updates replace the coupler flux
offset and its calibration ``sweetspot`` with that absolute value; they never
add the current offset a second time. Qubit biases are unchanged. Disable
updates while exploring an unfamiliar amplitude range.

Example
-------

Both protocols accept ``[start, stop, step]`` or the usual ``RangeLike``
specifications, including ``["linspace", start, stop, count]`` with both
endpoints. Amplitudes are relative to the acquisition DC coupler bias.

.. code-block:: yaml

    - id: coupler ZZ scan
      operation: coupler_zz
      targets: [[0, 1]]
      update: false
      parameters:
        delay_range: [16, 6016, 40]
        amplitude_range: ["linspace", -0.05, 0.05, 21]
        detuning: 2000000
        nshots: 2048
        relaxation_time: 20000

    - id: coupler echoed null
      operation: coupler_jazz
      targets: [[0, 1]]
      update: false
      parameters:
        delay_range: [16, 6016, 40]
        amplitude_range: ["linspace", -0.05, 0.05, 21]
        detuning: 2000000
        nshots: 2048
        relaxation_time: 20000

Parameters
----------

.. autoclass:: qibocal.protocols.couplers.zz.CouplerZZParameters
    :noindex:

.. autoclass:: qibocal.protocols.couplers.jazz.CouplerJAZZParameters
    :noindex:
