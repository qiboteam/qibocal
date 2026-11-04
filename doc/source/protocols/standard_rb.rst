Standard Randomized Benchmarking (RB)
=====================================

An approach to obtain the average gate fidelity is to perform randomized
benchmarking :cite:p:`Emerson_2005`.
The key idea is that if we average the error process over the uniform space of
unitaries the result is a depolarizing channel that maps any pure state to the
maximally mixed state.
Such uniform space of unitaries is known as *Haar measure*.
It can be shown :cite:p:`Emerson_2005` that the average induced error is proportional
to the depolarization probability.
However, this approach is inefficient because we sample randomly from the Haar measure.
A simplification was proposed in :cite:p:`Knill2008` by restricting the unitaries
to the Clifford group, which consists of unitary rotations mapping the group
of Pauli operators in itself.
Among the advantages of such group are the fact of the number of Clifford
gates is finite given the Hilbert space and being a group we can easily found
the inverse within the group.
The generic procedure to perform a randomized benchmarking is the following:

1. initialize the system in ground state
2. for each sequence length :math:`m` draw sequence of Clifford group elements
3. calculate inverse gate
4. measure sequence and inverse gate
5. repeat the process for multiple sequence of same length and varying the length

The previous approach works because it has been shown :cite:p:`Nielsen_2002` that
randomization with Clifford gates provides again a depolarized noise channel

.. math::
    :name: eq:1

    \rho \rightarrow \frac{d}{2} I + ( 1 - d) \rho

with depolarization probability :math:`d`.
If we follow the previous procedure and we measure the survival probability, i.e.
the probability of measuring the qubit in :math:`\ket{0}`, for
different sequence length :math:`m` we expect the following behavior

.. math::
    :name: eq:2

    F(m) = A p^m + B

where :math:`1-p` is the rate of depolarization while :math:`A` and :math:`B`
capture state preparation and measurement errors.
Finally, we can extract the average error per Clifford as in Eq. (5) of
:cite:p:`Magesan_2011`

.. math::
    :name: eq:3

    \epsilon_\text{Clifford} = 1 - F_\text{Clifford} = \frac{d - 1}{d} (1 - p)

where :math:`d = 2^n` and :math:`n` is the number of qubits, i.e.
:math:`F_\text{Clifford} = (1 + p) / 2` for a single qubit and :math:`F_\text{Clifford}
= (1 + 3p) / 4` for two qubits.

For single qubit RB, the average gate fidelity is estimated to first order by dividing
the Clifford error by the number :math:`N` of physical gates per Clifford, which usually
is 1.875

.. math::

    F_\text{avg,gate} \approx 1 - \frac{1 - F_\text{Clifford}}{N}

This is the first order expansion, in the small error :math:`1 - p`, of
:math:`p_\text{gate} = p^{1/N}`, which holds assuming all physical gates have the same
depolarizing error.

One of the main feature of RB is the possibility to estimate the gate fidelity
alone without taking into account both state preparation and measurement errors
which can be computed using the :math:`A` and :math:`B` terms in :ref:`Eq. 2 <eq:2>`.

Interleaved Randomized Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fidelity of a specific gate :math:`\mathcal{C}` can be estimated with interleaved RB
:cite:p:`Magesan_2012`: :math:`\mathcal{C}` is interleaved after each random Clifford,
and the decay :math:`p_\text{int}` of the resulting sequences is compared with the decay
:math:`p_\text{ref}` of a standard RB run. For two qubits, the fitted
:math:`F_\text{step} = (1 + 3 p_\text{int}) / 4` is the fidelity of a Clifford followed
by :math:`\mathcal{C}`, while the fidelity of :math:`\mathcal{C}` follows from Eq. (4)
of :cite:p:`Magesan_2012`

.. math::

    F_\mathcal{C} = \frac{1}{4} + \frac{3}{4} \frac{p_\text{int}}{p_\text{ref}}.

This is only an estimate: it is exact for depolarizing noise, while in general the
true value lies within the bounds of Eq. (5) of :cite:t:`Magesan_2012`, which are not
computed.

Parameters
^^^^^^^^^^


.. autoclass::
	qibocal.protocols.randomized_benchmarking.standard_rb.StandardRBParameters
	:noindex:


Example
^^^^^^^

It follows a runcard where we execute a standard RB.

.. code-block:: yaml

    - id: standard rb
      operation: standard_rb
      parameters:
        depths: [1,5,10,20,50,100]
        niter: 20
        nshots: 100

The expected output is the following:

.. image:: standard_rb.png
