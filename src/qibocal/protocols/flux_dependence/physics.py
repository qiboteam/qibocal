"""Transmon physics models used to fit flux dependent frequency sweeps."""

import numpy as np


def G_f_d(xi, xj, offset, d, crosstalk_element, normalization):
    """Auxiliary function to calculate qubit frequency as a function of bias.

    It also determines the flux dependence of :math:`E_J`,:math:`E_J(\\phi)=E_J(0)G_f_d`.
    For more details see: https://arxiv.org/pdf/cond-mat/0703002.pdf

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        offset (float): phase_offset [a.u.].
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        crosstalk_element(float): off-diagonal crosstalk matrix element
        normalization(float): diagonal crosstalk matrix element
    Returns:
        (float)
    """
    return (
        d**2
        + (1 - d**2)
        * np.cos(
            np.pi
            * (xi * normalization + normalization * xj * crosstalk_element + offset)
        )
        ** 2
    ) ** 0.25


def transmon_frequency(
    xi, xj, w_max, d, normalization, offset, crosstalk_element, charging_energy
):
    r"""Approximation to transmon frequency.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        normalization(float): diagonal crosstalk matrix element
        offset (float): phase_offset [a.u.].
        crosstalk_element(float): off-diagonal crosstalk matrix element
        charging_energy (float): Ec / h

     Returns:
         (float): qubit frequency as a function of bias.
    """
    return (w_max + charging_energy) * G_f_d(
        xi,
        xj,
        offset=offset,
        d=d,
        normalization=normalization,
        crosstalk_element=crosstalk_element,
    ) - charging_energy


def transmon_readout_frequency(
    xi,
    xj,
    w_max,
    d,
    normalization,
    crosstalk_element,
    offset,
    resonator_freq,
    g,
    charging_energy,
):
    r"""Approximation to flux dependent resonator frequency.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
         xi (float): bias of target qubit
         xj (float): bias of neighbor qubit
         w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
         d (float): asymmetry between the two junctions of the transmon.
                    Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
         normalization(float): diagonal crosstalk matrix element
         offset (float): phase_offset [a.u.].
         crosstalk_element(float): off-diagonal crosstalk matrix element
         resonator_freq (float): bare resonator frequency
         g (float): readout coupling.
         charging_energy (float): Ec / h

     Returns:
         (float): resonator frequency as a function of bias.
    """

    qubit_frequency = transmon_frequency(
        xi=xi,
        xj=xj,
        w_max=w_max,
        d=d,
        normalization=normalization,
        offset=offset,
        crosstalk_element=crosstalk_element,
        charging_energy=charging_energy,
    )
    return resonator_freq + g**2 * (
        1 / (resonator_freq - qubit_frequency)
        - 1 / (resonator_freq - qubit_frequency + charging_energy)
    )
