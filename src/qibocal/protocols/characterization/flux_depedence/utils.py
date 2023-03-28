import plotly.graph_objects as go
from plotly.subplots import make_subplots


def flux_dependence_plot(data, fit, qubit):
    figures = []
    fitting_report = "No fitting data"

    report_n = 0

    data.df = data.df[data.df["qubit"] == qubit]

    fluxlines = data.df["fluxline"].unique()

    if len(fluxlines) > 1:
        fig = make_subplots(
            rows=1,
            cols=len(fluxlines),
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=tuple(
                [f"MSR [V] - fluxline {fluxline}" for fluxline in fluxlines]
            ),
        )

        for fluxline_n, fluxline in enumerate(fluxlines):
            fluxline_df = data.df[data.df["fluxline"] == fluxline]
            fluxline_df = fluxline_df.drop(
                columns=["i", "q", "qubit", "fluxline", "iteration"]
            )

            fluxline_df = fluxline_df.groupby(
                ["frequency", "bias"], as_index=False
            ).mean()

            fig.add_trace(
                go.Heatmap(
                    x=fluxline_df["frequency"].pint.to("GHz").pint.magnitude,
                    y=fluxline_df["bias"].pint.to("V").pint.magnitude,
                    z=fluxline_df["MSR"].pint.to("uV").pint.magnitude,
                    showscale=False,
                ),
                row=1 + report_n,
                col=1 + fluxline_n,
            )
            fig.update_xaxes(
                title_text=f"q{qubit}/r{report_n}: Frequency (GHz)",
                row=1 + report_n,
                col=1 + fluxline_n,
            )

    elif len(fluxlines) == 1:
        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=(
                f"MSR [V] - fluxline {fluxlines[0]}",
                f"Phase [rad] - fluxline {fluxlines[0]}",
            ),
        )

        fluxline_df = data.df[data.df["fluxline"] == fluxlines[0]]
        fluxline_df = fluxline_df.drop(
            columns=["i", "q", "qubit", "fluxline", "iteration"]
        )

        fluxline_df = fluxline_df.groupby(["frequency", "bias"], as_index=False).mean()

        fig.add_trace(
            go.Heatmap(
                x=fluxline_df["frequency"].pint.to("GHz").pint.magnitude,
                y=fluxline_df["bias"].pint.to("V").pint.magnitude,
                z=fluxline_df["MSR"].pint.to("uV").pint.magnitude,
                colorbar_x=0.46,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)",
            row=1 + report_n,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=fluxline_df["frequency"].pint.to("GHz").pint.magnitude,
                y=fluxline_df["bias"].pint.to("V").pint.magnitude,
                z=fluxline_df["phase"].pint.to("rad").pint.magnitude,
                colorbar_x=1.01,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)",
            row=1 + report_n,
            col=2,
        )
        fig.update_yaxes(title_text="Bias (V)", row=1 + report_n, col=2)

    fig.update_yaxes(title_text="Bias (V)", row=1 + report_n, col=1)
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report
