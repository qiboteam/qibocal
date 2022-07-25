import os
import yaml
import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Output, Input

# number of trials that ``os.stat(path)`` should remain unchanged to
# stop the live plotting
NTRIALS = 5

app = dash.Dash(__name__)

app.layout = html.Div(children=[
        dcc.Input(id='last-modified', value=0, type='number', style={'display': 'none'}),
        dcc.Input(id='stopper-trials', value=NTRIALS, type='number', style={'display': 'none'}),
        dcc.Interval(id='stopper-interval', interval=2000, n_intervals=0, disabled=False),
        dcc.Graph(id='graph'),
        dcc.Interval(id='graph-interval', interval=2000, n_intervals=0, disabled=False),
])


@app.callback(Output('graph', 'figure'),
              Input('graph-interval', 'n_intervals'))
def get_graph(n):
    if not os.path.exists(app.path):
        return go.Figure()

    with open(app.path, "r") as file:
        data = yaml.safe_load(file)

    df = pd.DataFrame({f"{v.get('name')} ({v.get('unit')})": v.get('data')
                       for v in data.values()})

    X = df["frequency (Hz)"]
    Y = df["attenuation (dB)"]
    Z = df["MSR (V)"]

    fig = go.Figure(data=go.Heatmap(x=X,y=Y,z=Z))
    fig.update_layout(
        xaxis_title='Frequency (Hz)',
        yaxis_title='Attenuation (dB)',
        autosize=False,
        width=1200,
        height=800
    )
    return fig


@app.callback(Output('last-modified', 'value'),
              Output('stopper-trials', 'value'),
              Output('graph-interval', 'disabled'),
              Output('stopper-interval', 'disabled'),
              Input('stopper-interval', 'n_intervals'),
              Input('stopper-trials', 'value'),
              Input('last-modified', 'value'))
def stop_live_plotting(n, trials, last_modified):
    if not os.path.exists(app.path):
        return 0, NTRIALS, True, True

    # TODO: The stopper-interval can probably run forever
    new_modified = os.stat(app.path)[-1]
    if new_modified == last_modified:
        trials -= 1
    else:
        trials = NTRIALS
    return new_modified, trials, new_modified == last_modified, trials <= 0
