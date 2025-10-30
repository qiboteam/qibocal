import plotly.express as px


def plot(data: dict, color: str, label: str):
    fig = px.scatter(
        data,
        x="I",
        y="Q",
        marginal_x="histogram",
        marginal_y="histogram",
        color_discrete_sequence=[color],
    )

    for i in range(3):
        fig.data[i].legendgroup = label
    fig.data[-1].name = label
    fig.data[-1].showlegend = True

    return fig
