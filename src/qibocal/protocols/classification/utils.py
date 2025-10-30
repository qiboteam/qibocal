import plotly.express as px


def plot_distribution(data: dict, color: str, label: str):
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


def plot_confusion_matrix(confusion_matrix: list, labels: list):
    matrix = px.imshow(
        confusion_matrix,
        x=labels,
        y=labels,
        aspect="auto",
        text_auto=True,
        color_continuous_scale="Mint",
        title="Confusion matrix",
    )
    matrix.update_xaxes(title="Predicted State")
    matrix.update_yaxes(title="Actual State")
    return matrix
