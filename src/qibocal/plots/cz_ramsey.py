import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import scipy.signal as ss
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.signal import lfilter

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


def amplitude_balance_cz(folder, routine, qubit, format):
    r"""
    Plotting function for the amplitude balance of the CZ gate.
    Args:
        folder (str): The folder where the data is stored.
        routine (str): The routine used to generate the data.
        qubit (int): The qubit to plot.
        format (str): The format of the data.
    Returns:
        fig (plotly.graph_objects.Figure): The figure.
    """
    try:
        data = DataUnits.load_data(folder, routine, format, f"fit")
    except:
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

    combinations = np.unique(
        np.vstack(
            (data.df["targetqubit"].to_numpy(), data.df["controlqubit"].to_numpy())
        ).transpose(),
        axis=0,
    )

    fig = make_subplots(
        cols=1,
        rows=len(combinations),
    )

    for comb, i in enumerate(combinations):
        q_target = comb[0]
        q_control = comb[1]
        fig.add_trace(
            go.Heatmap(
                z=data.get_values("initial_phase_ON", "degree").df[
                    (data.df["controlqubit"] == q_control)
                    & (data.df["targetqubit"] == q_target)
                ],
                x=data.get_values("flux_pulse_amplitude", "dimensionless").df[
                    (data.df["controlqubit"] == q_control)
                    & (data.df["targetqubit"] == q_target)
                ],
                y=data.get_values("flux_pulse_ratio", "dimensionless").df[
                    (data.df["controlqubit"] == q_control)
                    & (data.df["targetqubit"] == q_target)
                ],
                name=f"Q{q_control} Q{q_target}",
            ),
            row=i,
            col=1,
        )
    return fig
