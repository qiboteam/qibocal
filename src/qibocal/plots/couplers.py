import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits

# Plot SWAP data when changing the coupler frequency


def coupler_swap(folder, routine, qubit, format):
    fig = go.Figure()

    # Load data
    try:
        data = DataUnits.load_data(folder, "data", routine, format, "data")
    except:
        data = DataUnits(
            name="data",
            quantities={"frequency": "Hz"},
            options=["qubit", "coupler", "state", "probability"],
        )

    # Plot data
    for state, q in zip(
        [0, 1], [qubit, 2]
    ):  # When multiplex works zip([0, 1], [qubit, 2])
        fig.add_trace(
            go.Scatter(
                x=data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == f"c{qubit}")
                ]["frequency"]
                .pint.to("Hz")
                .pint.magnitude,
                y=data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == f"c{qubit}")
                ]["probability"],
                mode="lines",
                name=f"Qubit {q} |{state}>",
            )
        )
    return [fig], "No fitting data."


def coupler_swap_amplitude(folder, routine, qubit, format):
    fig = go.Figure()

    # Load data
    try:
        data = DataUnits.load_data(folder, "data", routine, format, "data")
    except:
        data = DataUnits(
            name="data",
            quantities={"amplitude": "dimensionless"},
            options=["qubit", "coupler", "state", "probability"],
        )

    # Plot data
    for state, q in zip(
        [0, 1], [qubit, 2]
    ):  # When multiplex works zip([0, 1], [qubit, 2])
        fig.add_trace(
            go.Scatter(
                x=data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == f"c{qubit}")
                ]["amplitude"]
                .pint.to("dimensionless")
                .pint.magnitude,
                y=data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == f"c{qubit}")
                ]["probability"],
                mode="lines",
                name=f"Qubit {q} |{state}>",
            )
        )
    return [fig], "No fitting data."
