from pathlib import Path

import numpy as np
from qibo.backends import construct_backend

from qibocal.auto.execute import Executor
from qibocal.auto.history import History
from qibocal.auto.output import Metadata, Output
from qibocal.cli.report import report
from qibocal.protocols.flux_dependence.qubit_flux_dependence import QubitFluxResults
from qibocal.protocols.flux_dependence.utils import (
    transmon_frequency,
)

# NOTE: Run in Dummy
target = 0
backend = construct_backend(backend="qibolab", platform="dummy")
# NOTE: Run in QW11Q
# target = "D4"
# backend = construct_backend(backend="qibolab", platform="qw11q")

platform = backend.platform
if platform is None:
    raise ValueError("Qibocal requires a Qibolab platform to run.")

executor = Executor(
    name="myexec", history=History(), platform=platform, targets=[target]
)

from myexec import qubit_spectroscopy, rabi_amplitude_signal, t1_signal

# connect and initialize platform
platform.connect()


# NOTE: No need resonator should be fixed at these flux scales
# from myexec import resonator_flux
# #Do a 2D resonator flux dependacy first
# folder = Path(f"test_T1vsFlux/resonator_flux_dependancy")
# # generate output folder
# path = Output.mkdir(folder, force=True)

# # generate meta
# meta = Metadata.generate(path.name, backend)
# output = Output(History(), meta, platform)
# output.dump(path)

# # run
# meta.start()

# resonator_flux_output = resonator_flux(
#     freq_width=200_000_000,
#     freq_step=5_000_000,
#     bias_width=.2,
#     bias_step=0.01,
# )

# fit_function=transmon_readout_frequency
# fitted_parameters = resonator_flux_output.results.fitted_parameters
# params_resonator = fitted_parameters[target]

# history = executor.history
# # dump history, metadata, and updated platform
# output.history = history
# output.dump(path)

# report(path, history)

# executor.history = History()

# meta.end()

# NOTE: Optional qubit flux dependancy map
# from myexec import qubit_flux
# #Do a 2D flux dependacy first
# folder = Path(f"test_T1vsFlux/qubit_flux_dependancy")
# # generate output folder
# path = Output.mkdir(folder, force=True)

# # generate meta
# meta = Metadata.generate(path.name, backend)
# output = Output(History(), meta, platform)
# output.dump(path)

# # run
# meta.start()

# freq_width=400_000_000

# qubit_frequency = platform.qubits[target].drive_frequency
# qubit_frequency -= freq_width/3
# platform.qubits[target].drive_frequency = qubit_frequency # Does not change the qubit frequency on the pulse
# platform.qubits[target].native_gates.RX.frequency = qubit_frequency

# qubit_flux_output = qubit_flux(
#     freq_width=freq_width,
#     freq_step=5_000_000,
#     bias_width=.3,
#     bias_step=0.01,
#     drive_duration=500,
#     drive_amplitude=0.01,
#     relaxation_time=30000,
#     nshots = 1024,
# )

# fit_function=transmon_frequency
# fitted_parameters = qubit_flux_output.results.fitted_parameters
# params_qubit = fitted_parameters[target]


# history = executor.history
# # dump history, metadata, and updated platform
# output.history = history
# output.dump(path)

# report(path, history)

# executor.history = History()

# meta.end()

# NOTE: Load qubit flux dependancy map
folder_map = Path(f"test_T1vsFlux_map/qubit_flux_dependancy")
path = folder_map / Path("data/qubit_flux")

results = QubitFluxResults.load(path)
fitted_parameters = results.fitted_parameters
params_qubit = fitted_parameters[target]
fit_function = transmon_frequency

# biases = [0.1, 0.15]
biases = list(np.arange(0, 0.15, 0.01))
t1s = {}
qfs = {}
i = 0
for flux in biases:

    # check if this changes flux
    platform.qubits[target].flux.offset = flux

    # NOTE: No need resonator should be fixed at these flux scales
    # resonator_frequency=fit_function(flux, **params_resonator)
    # platform.qubits[target].resonator.frequency = resonator_frequency
    # print(resonator_frequency)

    qubit_frequency = fit_function(flux, **params_qubit)
    qubit_frequency *= 1e9
    platform.qubits[target].drive_frequency = qubit_frequency
    platform.qubits[target].native_gates.RX.frequency = qubit_frequency

    folder = Path(f"test_T1vsFlux/flux{i}")
    path = Output.mkdir(folder, force=True)

    # generate meta
    meta = Metadata.generate(path.name, backend)
    output = Output(History(), meta, platform)
    output.dump(path)

    # run
    meta.start()

    # NOTE: Look and correct from the 1st estimate qubit frequency
    qubit_spectroscopy_output = qubit_spectroscopy(
        freq_width=100_000_000,
        freq_step=5_000_000,
        drive_duration=500,
    )

    qubit_spectroscopy_output.update_platform(platform)

    platform.qubits[target].native_gates.RX.amplitude = 0.5
    platform.qubits[target].native_gates.RX.duration = 60
    if qubit_spectroscopy_output.results.frequency:
        platform.qubits[target].native_gates.RX.frequency = (
            qubit_spectroscopy_output.results.frequency[target]
        )
    else:
        platform.qubits[target].native_gates.RX.frequency = qubit_frequency

    qfs[target, flux] = platform.qubits[target].native_gates.RX.frequency
    rabi_output = rabi_amplitude_signal(
        min_amp_factor=0.1,
        max_amp_factor=1,
        step_amp_factor=0.01,
        pulse_length=platform.qubits[target].native_gates.RX.duration,
    )

    if rabi_output.results.amplitude[target] > 0.5:
        print(
            f"Rabi fit has pi pulse amplitude {rabi_output.results.amplitude[target]}, greater than 0.5 not possible for QM. Skipping to next bias point."
        )
        continue
    else:
        rabi_output.update_platform(platform)

        t1_output = t1_signal(
            delay_before_readout_start=16,
            delay_before_readout_end=100_000,
            delay_before_readout_step=4_000,
        )
        t1_output.update_platform(platform)
        t1s[target, flux] = t1_output.results.t1[target][0]

    history = executor.history
    # dump history, metadata, and updated platform
    output.history = history
    output.dump(path)

    report(path, history)

    executor.history = History()

    i += 1

    print(qfs)
    print(t1s)

    meta.end()

# stop and disconnect platform
platform.disconnect()


# #TODO: Plotting
# def plot(data_t1, data_qf, biases, target: QubitId):
#     """Plotting function for T1 experiment."""

#     figure = go.Figure()

#     for bias in biases:
#         figure.add_trace(
#             go.Scatter(
#                 x=data_qf,
#                 y=data_t1,
#                 opacity=1,
#                 name="Signal",
#                 showlegend=True,
#                 legendgroup="Signal",
#             )
#         )

#     # last part
#     figure.update_layout(
#         showlegend=True,
#         xaxis_title="Frequency [GHZ]",
#         yaxis_title="Signal [a.u.]",
#     )

#     return figure


# TODO: What to do with the plot here

# def report_script(path: Path, history: Optional[History] = None):
#     """Report generation.

#     Generates the report for protocol dumped in `path`.
#     Executor can be passed to generate report on the fly.
#     """

#     if (path / "index.html").exists():  # pragma: no cover
#         log.warning(f"Regenerating {path}/index.html")
#     # load meta
#     output = Output.load(path)

#     if history is None:
#         history = output.history

#     css_styles = f"<style>\n{Path(STYLES).read_text()}\n</style>"

#     env = Environment(loader=FileSystemLoader(TEMPLATES))
#     template = env.get_template("template.html")
#     html = template.render(
#         is_static=True,
#         css_styles=css_styles,
#         path=path,
#         title=path.name,
#         report=Report(
#             path=path,
#             history=history,
#             meta=output.meta.dump(),
#             plotter=plotter,
#         ),
#     )

#     (path / "index.html").write_text(html)

# def plotter(
#     data, target: Union[QubitId, QubitPairId, list[QubitId]]
# ) -> tuple[str, str]:
#     """Run plotly pipeline for generating html.

#     Performs conversions of plotly figures in html rendered code for
#     completed node on specific target.
#     """
#     figures, fitting_report = plot(data_t1, data_qf, biases, target)
#     buffer = io.StringIO()
#     html_list = []
#     for figure in figures:
#         figure.write_html(buffer, include_plotlyjs=False, full_html=False)
#         buffer.seek(0)
#         html_list.append(buffer.read())
#     buffer.close()
#     all_html = "".join(html_list)
#     return all_html, fitting_report
