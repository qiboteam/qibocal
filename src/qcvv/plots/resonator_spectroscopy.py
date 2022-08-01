# -*- coding: utf-8 -*-
import plotly.graph_objects as go


def resonator_spectroscopy_attenuation(dataframe, **figure_settings):
    X = dataframe["frequency"].pint.to("GHz").pint.magnitude
    Y = dataframe["attenuation"].pint.to("dB").pint.magnitude
    Z = dataframe["MSR"].pint.to("V").pint.magnitude
    fig = go.Figure(data=go.Heatmap(x=X, y=Y, z=Z))
    figure_settings["xaxis_title"] = "Frequency (GHz)"
    figure_settings["yaxis_title"] = "Attenuation (dB)"
    fig.update_layout(**figure_settings)
    return fig
