"""Value reporting helpers (rounding, formatting, HTML tables) for qibocal."""

import numpy as np
import pandas as pd

from qibocal.auto.operation import QubitId


def eval_magnitude(value: float | np.number) -> int:
    """number of non decimal digits in `value`"""
    if value == 0 or not np.isfinite(value):
        return 0
    return int(np.floor(np.log10(abs(value))))


def significant_digit(number: float) -> int:
    """Computes the position of the first significant digit of a given number.

    Args:
        number (Number): number for which the significant digit is computed. Can be complex.

    Returns:
        int: position of the first significant digit. Returns ``-1`` if the given number
            is ``>= 1``, ``= 0`` or ``inf``.
    """

    if (
        np.isinf(np.real(number))
        or np.real(number) >= 1
        or number == 0
        or np.isnan(number)
    ):
        return -1

    position = max(np.ceil(-np.log10(abs(np.real(number)))), -1)

    if np.imag(number) != 0:
        position = max(position, np.ceil(-np.log10(abs(np.imag(number)))))

    return int(position)


def round_report(
    measure: list,
) -> tuple[list, list]:
    """
    Rounds the measured values and their errors according to their significant digits.

    Args:
        measure (list): Variable-Errors couples.

    Returns:
        A tuple with the lists of values and errors in the correct string format.
    """
    rounded_values = []
    rounded_errors = []
    for value, error in measure:
        if value:
            magnitude = eval_magnitude(value)
        else:
            magnitude = 0

        ndigits = max(significant_digit(error * 10 ** (-1 * magnitude)), 0)
        if magnitude != 0:
            rounded_values.append(
                f"{round(value * 10 ** (-1 * magnitude), ndigits)}e{magnitude}"
            )
            rounded_errors.append(
                f"{np.format_float_positional(round(error * 10 ** (-1 * magnitude), ndigits), trim='-')}e{magnitude}"
            )
        else:
            rounded_values.append(f"{round(value * 10 ** (-1 * magnitude), ndigits)}")
            rounded_errors.append(
                f"{np.format_float_positional(round(error * 10 ** (-1 * magnitude), ndigits), trim='-')}"
            )

    return rounded_values, rounded_errors


def format_error_single_cell(measure: tuple) -> str:
    """Helper function to print mean value and error in one line."""
    # extract mean value and error
    mean = measure[0][0]
    error = measure[1][0]
    if all("e" in number for number in measure[0] + measure[1]):
        magn = mean.split("e")[1]
        return f"({mean.split('e')[0]} ± {error.split('e')[0]}) 10<sup>{magn}</sup>"
    return f"{mean} ± {error}"


def table_dict(
    qubit: list[QubitId] | QubitId,
    names: list[str],
    values: list,
    display_error=False,
) -> dict:
    """
    Build a dictionary to generate HTML table with `table_html`.

    Args:
        qubit (list[QubitId] | QubitId): If qubit is a scalar value,
        the "Qubit" entries will have only this value repeated.
        names (list[str]): List of the names of the parameters.
        values (list): List of the values of the parameters.
        display_errors (bool): if `True`, it means that `values` is a list of value-error couples,
        so an `Errors` key will be displayed in the dictionary. The function will round the couples according to their significant digits. Default False.

    Return:
        A dictionary with keys `Qubit`, `Parameters`, `Values` (`Errors`).
    """
    if not np.isscalar(values):
        if np.isscalar(qubit):
            qubit = [qubit] * len(names)

        if display_error:
            rounded_values, rounded_errors = round_report(values)

            return {
                "Qubit": qubit,
                "Parameters": names,
                "Values": rounded_values,
                "Errors": rounded_errors,
            }
    else:  # If `values` is scalar also `qubit` should be a scalar
        qubit = [
            qubit
        ]  # In this way when the Dataframe is generated, an index is not required.
    return {"Qubit": qubit, "Parameters": names, "Values": values}


def table_html(data: dict) -> str:
    """This function converts a dictionary into an HTML table.

    Args:
        data (dict): the keys will be converted into table entries and the
        values will be the columns of the table.
        Values must be valid HTML strings.

    Return:
        str
    """
    return pd.DataFrame(data).to_html(
        classes="fitting-table", index=False, border=0, escape=False
    )
