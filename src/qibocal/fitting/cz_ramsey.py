import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from qibocal.data import DataUnits


def fit_amplitude_balance_cz(data):
    """Fit phase of the target qubit detuning for both the ON and OFF sequence.

    After normalizing the data between -1 and 1, the phase, phi, of the target
    qubit is fitted using sin(2*pi*x/360 + phi) where x is the detuning of the
    target.

    Args:
        data (dict): The data to fit.

    Returns:
        dict: The fit results.
    """

    data_fit = DataUnits(
        name=f"fit",
        quantities={
            "flux_pulse_amplitude": "dimensionless",
            "flux_pulse_ratio": "dimensionless",
            "initial_phase_ON": "degree",
            "initial_phase_OFF": "degree",
            "phase_difference": "degree",
            "leakage": "dimensionless",
        },
        options=["controlqubit", "targetqubit"],
    )

    # Extracting unique values
    combinations = np.unique(
        np.vstack(
            (data.df["targetqubit"].to_numpy(), data.df["controlqubit"].to_numpy())
        ).transpose(),
        axis=0,
    )
    amplitude_unique = np.unique(data.df["flux_pulse_amplitude"].to_numpy())
    ratio_unique = np.unique(data.df["flux_pulse_ratio"].to_numpy())
    detuning_unique = np.unique(data.df["detuning"].to_numpy())

    # Fitting function
    def f(x, phi):
        return np.sin(2 * np.pi * x / 360 + phi)

    for i in combinations:
        q_target = i[0]
        q_control = i[1]

        # Extracting Data
        prob_dict = sort_data(data, q_control, q_target, ["prob", "dimensionless"])
        amplitude_dict = sort_data(
            data, q_control, q_target, ["flux_pulse_amplitude", "dimensionless"]
        )
        ratio_dict = sort_data(
            data, q_control, q_target, ["flux_pulse_ratio", "dimensionless"]
        )
        detuning_dict = sort_data(data, q_control, q_target, ["detuning", "degree"])

        for ratio in ratio_unique:
            for amp in amplitude_unique:
                # Normalizing between -1 and 1
                for ON_OFF in ["ON", "OFF"]:
                    prob_dict["target"][ON_OFF][
                        (amplitude_dict["target"][ON_OFF] == amp)
                        & (ratio_dict["target"][ON_OFF] == ratio)
                    ] = prob_dict["target"][ON_OFF][
                        (amplitude_dict["target"][ON_OFF] == amp)
                        & (ratio_dict["target"][ON_OFF] == ratio)
                    ] - np.min(
                        prob_dict["target"][ON_OFF][
                            (amplitude_dict["target"][ON_OFF] == amp)
                            & (ratio_dict["target"][ON_OFF] == ratio)
                        ]
                    )
                    prob_dict["target"][ON_OFF][
                        (amplitude_dict["target"][ON_OFF] == amp)
                        & (ratio_dict["target"][ON_OFF] == ratio)
                    ] = prob_dict["target"][ON_OFF][
                        (amplitude_dict["target"][ON_OFF] == amp)
                        & (ratio_dict["target"][ON_OFF] == ratio)
                    ] / np.max(
                        prob_dict["target"][ON_OFF][
                            (amplitude_dict["target"][ON_OFF] == amp)
                            & (ratio_dict["target"][ON_OFF] == ratio)
                        ]
                    )

                # Fitting the data
                try:
                    popt_ON, pcov_ON = curve_fit(
                        f,
                        detuning_dict["target"]["ON"][
                            (amplitude_dict["target"]["ON"] == amp)
                            & (ratio_dict["target"]["ON"] == ratio)
                        ],
                        prob_dict["target"]["ON"][
                            (amplitude_dict["target"]["ON"] == amp)
                            & (ratio_dict["target"]["ON"] == ratio)
                        ],
                        p0=[0],
                        maxfev=100000,
                    )
                    popt_OFF, pcov_OFF = curve_fit(
                        f,
                        detuning_dict["target"]["OFF"][
                            (amplitude_dict["target"]["OFF"] == amp)
                            & (ratio_dict["target"]["OFF"] == ratio)
                        ],
                        prob_dict["target"]["OFF"][
                            (amplitude_dict["target"]["OFF"] == amp)
                            & (ratio_dict["target"]["OFF"] == ratio)
                        ],
                        p0=[0],
                        maxfev=100000,
                    )

                    results = {
                        "flux_pulse_amplitude[dimensionless]": amp,
                        "flux_pulse_ratio[dimensionless]": ratio,
                        "initial_phase_ON[degree]": np.rad2deg(popt_ON[0]) % 360,
                        "initial_phase_OFF[degree]": np.rad2deg(popt_OFF[0]) % 360,
                        "phase_difference[degree]": np.rad2deg(popt_ON[0] - popt_OFF[0])
                        % 360,
                        "leakage[dimensionless]": np.mean(
                            prob_dict["control"]["OFF"][
                                (amplitude_dict["target"]["OFF"] == amp)
                                & (ratio_dict["target"]["OFF"] == ratio)
                            ]
                        )
                        / 2,
                        "controlqubit": q_control,
                        "targetqubit": q_target,
                    }
                except:
                    results = {
                        "flux_pulse_amplitude[dimensionless]": np.nan,
                        "flux_pulse_ratio[dimensionless]": np.nan,
                        "initial_phase_ON[degree]": np.nan,
                        "initial_phase_OFF[degree]": np.nan,
                        "phase_difference[degree]": np.nan,
                        "leakage[dimensionless]": np.nan,
                        "controlqubit": np.nan,
                        "targetqubit": np.nan,
                    }

                data_fit.add(results)

    return data_fit


# Function to index out data for given control and target qubits
def sort_data(data, q_control, q_target, options):
    data_dict = {"control": {}, "target": {}}
    for k in ["ON", "OFF"]:
        data_dict["control"][k] = data.get_values(options[0], options[1])[
            (data.df["ON_OFF"] == k)
            & (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_control)
        ].to_numpy()
        data_dict["target"][k] = data.get_values(options[0], options[1])[
            (data.df["ON_OFF"] == k)
            & (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_target)
        ].to_numpy()
    return data_dict
