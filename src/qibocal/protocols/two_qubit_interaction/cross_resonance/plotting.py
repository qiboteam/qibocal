import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.auto.operation import QubitPairId
from qibocal.protocols.utils import (
    angle_wrap,
    table_dict,
    table_html,
)

from . import fitting
from .cr_parent_classes import (
    Basis,
    HamiltonianTerm,
    HamiltonianTomographyData,
    HamiltonianTomographyResults,
    SetControl,
)
from .cross_resonance_processing import compute_bloch_vector


def tomography_cr_plot(
    data: HamiltonianTomographyData,
    target: QubitPairId,
    fit: HamiltonianTomographyResults | None = None,
) -> tuple[list[go.Figure], str]:
    """Create cross-resonance Hamiltonian tomography plots given the acquired data and
    the fit results for to a pair of qubits."""

    fig = make_subplots(
        rows=4,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        shared_xaxes=False,
        shared_yaxes=False,
        column_titles=["Target Qubit Evolution", "Control Qubit Evolution"],
    )

    if type(data).__name__ == "HamiltonianTomographyCRAmplData":
        annotation = "CR gate amplitude [a.u.]"
    else:
        annotation = "CR gate duration [ns]"

    target = target if target in data.pairs else (target[1], target[0])
    for i, basis in enumerate(Basis):
        for setup in SetControl:
            pair_data = data.data[target[0], target[1], basis, setup]
            fig.add_trace(
                go.Scatter(
                    x=pair_data.x,
                    y=pair_data.prob_target,
                    name=f"Target when Control at {0 if setup is SetControl.Id else 1}",
                    showlegend=True if basis is Basis.Z else False,
                    legendgroup=f"Target when Control at {0 if setup is SetControl.Id else 1}",
                    mode="markers",
                    marker=dict(color="blue" if setup is SetControl.Id else "red"),
                    error_y=dict(
                        type="data",
                        array=pair_data.error_target,
                        visible=True,
                    ),
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pair_data.x,
                    y=pair_data.prob_control,
                    name=f"Control in {0 if setup is SetControl.Id else 1}",
                    showlegend=True if basis is Basis.Z else False,
                    legendgroup=f"Control in {0 if setup is SetControl.Id else 1}",
                    mode="markers",
                    marker=dict(color="blue" if setup is SetControl.Id else "red"),
                    error_y=dict(
                        type="data",
                        array=pair_data.error_control,
                        visible=True,
                    ),
                ),
                row=i + 1,
                col=2,
            )
            if fit is not None and (*target, setup) in fit.fitted_parameters:
                x = np.linspace(
                    pair_data.x.min(), pair_data.x.max(), 50 * len(pair_data.x)
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=getattr(fitting, f"pauli_{basis.name.lower()}_expectation")(
                            x,
                            wx=fit.fitted_parameters[target[0], target[1], setup][0],
                            wy=fit.fitted_parameters[target[0], target[1], setup][1],
                            wz=fit.fitted_parameters[target[0], target[1], setup][2],
                            gamma=fit.fitted_parameters[target[0], target[1], setup][3],
                        ),
                        name=f"Simultaneous Fit of target when control at {0 if setup is SetControl.Id else 1}",
                        showlegend=True if basis is Basis.Z else False,
                        legendgroup=f"Simultaneous Fit target when control at {0 if setup is SetControl.Id else 1}",
                        mode="lines",
                        line=dict(
                            color="green" if setup is SetControl.Id else "orange",
                        ),
                    ),
                    row=i + 1,
                    col=1,
                )

                if type(fit).__name__ == "HamiltonianTomographyCRAmplResults":
                    fit_dict = fit.cr_amplitudes
                else:
                    fit_dict = fit.cr_lengths

                if target in fit_dict:
                    fig.add_trace(
                        go.Scatter(
                            x=[fit_dict[target]] * 2,
                            y=[
                                -1.2,
                                1.2,
                            ],
                            mode="lines",
                            line=go.scatter.Line(color="orange", width=3, dash="dash"),
                            name=annotation,
                            showlegend=False,
                            legendgroup=annotation,
                        ),
                        row=i + 1,
                        col=1,
                    )

    if fit is not None and fit.fitted_parameters:
        bloch_vect_targ, bloch_fit_targ, bloch_vect_ctrl = compute_bloch_vector(
            data, target, fit.fitted_parameters
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=pair_data.x,
                    y=y,
                    name=f"{label} Bloch vector |R(t)|",
                    legendgroup=f"{label} Bloch vector |R(t)|",
                    showlegend=True,
                    mode="markers",
                    marker=dict(
                        color="green",
                    ),
                )
                for y, label in [
                    (bloch_vect_targ, "Target"),
                    (bloch_vect_ctrl, "Control"),
                ]
            ],
            rows=[4, 4],
            cols=[1, 2],
        )

        if bloch_fit_targ is not None:
            x = np.linspace(pair_data.x.min(), pair_data.x.max(), len(bloch_fit_targ))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=bloch_fit_targ,
                    name="Fitted Bloch vector |R(t)|",
                    showlegend=True,
                    legendgroup="Fitted Bloch vector |R(t)|",
                    mode="lines",
                ),
                row=4,
                col=1,
            )

        if type(fit).__name__ == "HamiltonianTomographyCRAmplResults":
            fit_dict = fit.cr_amplitudes
        else:
            fit_dict = fit.cr_lengths

        if target in fit_dict:
            fig.add_trace(
                go.Scatter(
                    x=[fit_dict[target]] * 2,
                    y=[
                        -1.2,
                        1.2,
                    ],
                    mode="lines",
                    line=go.scatter.Line(color="orange", width=3, dash="dash"),
                    name=annotation,
                    showlegend=True,
                    legendgroup=annotation,
                ),
                row=4,
                col=1,
            )

    fig.update_layout(height=1000)
    fig.update_xaxes(title_text=annotation, row=4, col=1)
    fig.update_xaxes(title_text=annotation, row=4, col=2)
    fig.update_yaxes(title_text="<X(t)>", range=[-1.2, 1.2], row=1)
    fig.update_yaxes(title_text="<Y(t)>", range=[-1.2, 1.2], row=2)
    fig.update_yaxes(title_text="<Z(t)>", range=[-1.2, 1.2], row=3)
    fig.update_yaxes(title_text="|R(t)|", range=[-0.2, 1.2], row=4)

    return [fig], ""


def cancellation_calibration_plot(
    data: HamiltonianTomographyData,
    target: QubitPairId,
    fit: HamiltonianTomographyResults | None = None,
) -> tuple[list[go.Figure], str]:
    """Plot calibration results for cross-resonance Hamiltonian tomography when
    tuning cancellation pulses.

    Generates plots for either amplitude or phase calibration data of cancellation pulses
    in cross-resonance interactions. Fits effective Hamiltonian terms and visualizes the
    results with fitted curves overlaid on experimental data.
    """

    fitting_report = ""

    if fit is None:
        logging.warning("Fit failed, plotting only data.")
    else:
        if type(data).__name__ == "HamiltonianTomographyCANCAmplData":
            fit_func = fitting.linear_func
            x_title = "amplitude [a.u.]"
            tunable_params = fit.cancellation_pulse_amplitudes[target]
            plotting_line = {
                HamiltonianTerm.IX: ["ampl_ix", "red"],
                HamiltonianTerm.IY: ["ampl_iy", "blue"],
            }
            plotting_terms = list(plotting_line.keys())
            fig_title = "Ham terms vs Cancellation pulse " + x_title

        if type(data).__name__ == "HamiltonianTomographyCANCPhaseData":
            fit_func = fitting.sin_func
            x_title = "phase [rad.]"
            tunable_params = {}
            tunable_params["phi0"] = fit.cancellation_pulse_phases[target]["control"]
            tunable_params["phi1"] = angle_wrap(
                fit.cancellation_pulse_phases[target]["control"]
                - fit.cancellation_pulse_phases[target]["target"]
            )
            plotting_line = {
                HamiltonianTerm.ZY: ["phi0", "red"],
                HamiltonianTerm.IY: ["phi1", "blue"],
            }
            plotting_terms = list(plotting_line.keys()) + [
                HamiltonianTerm.IX,
                HamiltonianTerm.ZX,
            ]
            fig_title = "Ham terms vs CR pulse " + x_title

        fig = make_subplots(
            rows=len(plotting_terms) // 2,
            cols=1,
            vertical_spacing=0.1,
            shared_xaxes=False,
            shared_yaxes=False,
            column_titles=[fig_title],
        )

        for t in plotting_terms:
            eff_ham_term = []
            exp_sweeper = []
            for f in fit.hamiltonian_terms[target]:
                eff_ham_term.append(f[1][t])
                exp_sweeper.append(f[0])

            plot_row = 1 if "I" in t.name else 2
            fig.append_trace(
                go.Scatter(
                    x=exp_sweeper,
                    y=eff_ham_term,
                    opacity=1,
                    name=f"{t.name}",
                    showlegend=True,
                    legendgroup="Probability",
                    mode="markers",
                ),
                row=plot_row,
                col=1,
            )

            if target in fit.fitted_parameters and t in fit.fitted_parameters[target]:
                sweep_range = np.linspace(
                    min(exp_sweeper),
                    max(exp_sweeper),
                    2 * len(exp_sweeper) if len(exp_sweeper) >= 100 else 200,
                )
                fit_y = fit_func(sweep_range, **fit.fitted_parameters[target][t])
                fig.add_trace(
                    go.Scatter(
                        x=sweep_range,
                        y=fit_y,
                        name=f"{t.name} Fit",
                        mode="lines",
                    ),
                    row=plot_row,
                    col=1,
                )

                if t in plotting_line:
                    params_name = plotting_line[t][0]
                    fig.add_trace(
                        go.Scatter(
                            x=[tunable_params[params_name]] * 2,
                            y=[
                                min(fit_y) - 0.1 * (max(fit_y) - min(fit_y)),
                                max(fit_y) + 0.1 * (max(fit_y) - min(fit_y)),
                            ],
                            mode="lines",
                            line=go.scatter.Line(
                                color=plotting_line[t][1], width=3, dash="dash"
                            ),
                            name=f"{params_name}",
                            showlegend=True,
                            legendgroup=f"{params_name}",
                        ),
                        row=plot_row,
                        col=1,
                    )

        height = 500 if plot_row == 1 else 800
        fig.update_layout(height=height, showlegend=True)
        for i in range(1, plot_row + 1):
            fig.update_yaxes(title_text="Interaction strength [MHz]", row=i)
            fig.update_xaxes(title_text=f"{x_title}", row=i)

        fitting_report = table_html(
            table_dict(
                len(plotting_line) * [target],
                ([k for k in tunable_params.keys()]),
                ([v for v in tunable_params.values()]),
            )
        )

    return [fig], fitting_report
