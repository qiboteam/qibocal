import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from lmfit import Model
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, lfilter, savgol_filter

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey
from qibocal.plots.utils import get_color, get_data_subfolders

MX_tag = "MX"
MY_tag = "MY"


def _get_flux_pulse_durations_and_amplitudes(data: DataUnits):
    flux_pulse_durations = data.get_values("flux_pulse_duration", "ns")[
        data.df["component"] == MX_tag
    ].to_numpy()
    flux_pulse_amplitudes = data.get_values("flux_pulse_amplitude", "dimensionless")[
        data.df["component"] == MX_tag
    ].to_numpy()
    amplitudes = flux_pulse_amplitudes[flux_pulse_durations == flux_pulse_durations[0]]
    durations = flux_pulse_durations[flux_pulse_amplitudes == flux_pulse_amplitudes[0]]
    return durations, amplitudes


def _get_values(data, component=None, duration=None, amplitude=None):
    if component:
        values = data.get_values("prob", "dimensionless")[
            data.df["component"] == component
        ]
    else:
        values = data.get_values("prob", "dimensionless")

    if duration:
        values = values[data.df["flux_pulse_duration"] == duration]
    if amplitude:
        values = values[data.df["flux_pulse_amplitude"] == amplitude]
    return _normalise(values.to_numpy())


def _normalise(values: np.array):
    values = ((values - np.min(values)) / (np.max(values) - np.min(values))) * 2 - 1

    # TODO: consider moving this normalisation to data acquisition (work with magnitudes instead of 0-1 probs)
    return values


def cryoscope_raw(folder, routine, qubit, format):
    figs = {}
    figs["raw"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            # data.df = data.df[data.df["qubit"] == qubit]
            # extract qubit data and average over iterations
            data.df = (
                data.df[data.df["qubit"] == qubit]
                .drop(columns=["qubit", "iteration"])
                .groupby(
                    ["flux_pulse_duration", "flux_pulse_amplitude", "component"],
                    as_index=False,
                )
                .mean()
            )
        except:
            data = DataUnits(
                name=f"data",
                quantities={
                    "flux_pulse_duration": "ns",
                    "flux_pulse_amplitude": "dimensionless",
                    "prob": "dimensionless",
                },
                options=["component", "qubit", "iteration"],
            )

        durations = (
            data.df[data.df["component"] == MY_tag]["flux_pulse_duration"]
            .pint.to("ns")
            .pint.magnitude.unique()
        )
        amplitudes = (
            data.df[data.df["component"] == MY_tag]["flux_pulse_amplitude"]
            .pint.to("dimensionless")
            .pint.magnitude.unique()
        )

        for amp in amplitudes:
            figs["raw"].add_trace(
                go.Scatter(
                    x=durations,
                    y=_get_values(data, MX_tag, amplitude=amp),
                    name=f"q{qubit}/r{report_n}: <X> | A = {amp:.3f}",
                ),
                row=1,
                col=1,
            )

            figs["raw"].add_trace(
                go.Scatter(
                    x=durations,
                    y=_get_values(data, MY_tag, amplitude=amp),
                    name=f"q{qubit}/r{report_n}: <Y> | A = {amp:.3f}",
                ),
                row=1,
                col=1,
            )
    figs["raw"].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Magnitude (dimensionless)",
        title=f"Raw data",
    )
    return [figs["raw"]]


def flux_pulse_timing(folder, routine, qubit, format):
    figs = {}
    figs["flux_pulse_timing"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = (
                data.df[data.df["qubit"] == qubit]
                .drop(columns=["qubit", "iteration"])
                .groupby(
                    ["flux_pulse_amplitude", "flux_pulse_start"],
                    as_index=False,
                )
                .mean()
            )
        except:
            data = DataUnits(
                name=f"data",
                quantities={
                    "flux_pulse_amplitude": "dimensionless",
                    "flux_pulse_start": "ns",
                    "prob": "dimensionless",
                },
                options=["qubit", "iteration"],
            )

        amplitudes = (
            data.df["flux_pulse_amplitude"]
            .pint.to("dimensionless")
            .pint.magnitude.unique()
        )
        starts = data.df["flux_pulse_start"].pint.to("ns").pint.magnitude.unique()

    for amp in amplitudes:
        figs["flux_pulse_timing"].add_trace(
            go.Scatter(
                x=starts,
                y=_get_values(data, amplitude=amp),
                name=f"q{qubit}/r{report_n}: <X> | A = {amp:.3f}",
            ),
            row=1,
            col=1,
        )

    figs["flux_pulse_timing"].update_layout(
        xaxis_title="Flux start time (ns)",
        yaxis_title="Magnitude X component (dimensionless)",
    )
    return [figs["flux_pulse_timing"]]


def cryoscope_dephasing_heatmap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    signal = []
    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)
        signal.append(re + 1j * im)
    signal = np.array(signal, dtype=np.complex128)
    phase = np.arctan2(signal.imag, signal.real)
    global_title = "Dephasing"
    title_x = "Flux Pulse Duration"
    title_y = "Amplitudes"
    figs["norm"].add_trace(
        go.Heatmap(
            x=durations,
            y=amplitudes,
            z=np.abs(signal),
            colorbar=dict(len=0.46, y=0.25),
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_fft_peak_fitting(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    signal = []  # TODO: replace lists with np arrays
    fft = []
    freqs = []

    curve_amps = []
    curve_detunings = []

    num_points = durations.shape[0]
    interval = (durations[1] - durations[0]) * 1e-9
    amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
    mask = np.argsort(amp_freqs)
    amp_freqs = amp_freqs[mask]

    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        amp_signal = re + 1j * im
        signal.append(amp_signal)

        freqs.append(amp_freqs)

        amp_fft = np.fft.fft(amp_signal - np.mean(amp_signal))[mask]
        fft.append(amp_fft)

        figs["norm"].add_trace(
            go.Scatter(x=amp_freqs, y=np.abs(amp_fft), name=f"fft | A = {amp:.3f}"),
            row=1,
            col=1,
        )

        smoothed_y = smooth(np.absolute(amp_fft), window_len=21)
        percval = np.percentile(
            smoothed_y, 90
        )  # finds the top perc points in amplitude smoothed_y
        filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
        array_peaks = argrelextrema(filtered_y, np.greater)
        peaks_x = np.arange(num_points)[array_peaks]
        sort_mask = np.argsort(np.absolute(amp_fft)[array_peaks])[::-1]

        if len(peaks_x[sort_mask]) > 0 and amp_freqs[peaks_x[sort_mask][0]] <= 0:
            curve_amps.append(amp)
            curve_detunings.append(amp_freqs[peaks_x[sort_mask][0]])

        figs["norm"].add_trace(
            go.Scatter(x=amp_freqs, y=smoothed_y, name=f"smoothed_y | A = {amp:.3f}"),
            row=1,
            col=1,
        )

        figs["norm"].add_trace(
            go.Scatter(
                x=amp_freqs,
                y=filtered_y,
                name=f"filtered_y | A = {amp:.3f} | percentile = {percval:.3f}",
            ),
            row=1,
            col=1,
        )

    global_title = "FFT Peak Fitting"
    title_x = "Detunning (GHz)"
    title_y = "Spectral Density"

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_fft(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    signal = []  # TODO: replace lists with np arrays
    fft = []
    freqs = []

    curve_amps = []
    curve_detunings = []

    num_points = durations.shape[0]
    interval = (durations[1] - durations[0]) * 1e-9
    amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
    mask = np.argsort(amp_freqs)
    amp_freqs = amp_freqs[mask]

    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        amp_signal = re + 1j * im
        signal.append(amp_signal)

        freqs.append(amp_freqs)

        amp_fft = np.fft.fft(amp_signal - np.mean(amp_signal))[mask]
        fft.append(amp_fft)

        smoothed_y = smooth(np.absolute(amp_fft), window_len=21)
        percval = np.percentile(
            smoothed_y, 90
        )  # finds the top perc points in amplitude smoothed_y
        filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
        array_peaks = argrelextrema(filtered_y, np.greater)
        peaks_x = np.arange(num_points)[array_peaks]
        sort_mask = np.argsort(np.absolute(amp_fft)[array_peaks])[::-1]

        if len(peaks_x[sort_mask]) > 0 and amp_freqs[peaks_x[sort_mask][0]] <= 0:
            curve_amps.append(amp)
            curve_detunings.append(amp_freqs[peaks_x[sort_mask][0]])

    # print(curve_amps,curve_detunings)
    pfit = np.polyfit(curve_amps, curve_detunings, 2)
    # DEBUG:
    _amp = -0.109
    print(f"Curve fit {pfit} _ {pfit[0]*_amp**2 + pfit[1]*_amp + pfit[2]}")

    curve_amps = np.array(curve_amps)

    global_title = "FFT with curve fit"
    title_x = "Detunning (GHz)"
    title_y = "Amplitudes"
    figs["norm"].add_trace(
        go.Heatmap(
            x=amp_freqs,
            y=amplitudes,
            z=np.abs(fft),
        ),
        row=1,
        col=1,
    )
    figs["norm"].add_trace(
        go.Scatter(
            x=curve_detunings,
            y=curve_amps,
            name=f"Data points",
            mode="markers",
            marker=dict(size=5),
        ),
        row=1,
        col=1,
    )
    figs["norm"].add_trace(
        go.Scatter(
            x=pfit[0] * curve_amps**2 + pfit[1] * curve_amps + pfit[2],
            y=curve_amps,
            name=f"Curve fit",
        ),
        row=1,
        col=1,
    )
    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_phase(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        phase = np.arctan2(im, re)

        # Phase vs. Time
        global_title = "Phase vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Phase"
        figs["norm"].add_trace(
            go.Scatter(
                x=durations,
                y=phase,
                name=f"Phase | A = {amp:.3f}",
                mode="lines+markers",
                marker=dict(size=5),
            ),
            row=1,
            col=1,
        )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_phase_heatmap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    signal = []
    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)
        signal.append(re + 1j * im)
    signal = np.array(signal, dtype=np.complex128)
    phase = np.arctan2(signal.imag, signal.real)

    global_title = "Phase vs. Time"
    title_x = "Flux Pulse duration"
    title_y = "Amplitude"
    figs["norm"].add_trace(
        go.Heatmap(
            x=durations,
            y=amplitudes,
            z=phase,
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_phase_unwrapped(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        phase = np.arctan2(im, re)
        phase_unwrapped = np.unwrap(phase)

        # Phase vs. Time
        global_title = "Phase Unwrapped vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Phase"
        figs["norm"].add_trace(
            go.Scatter(x=durations, y=phase_unwrapped, name=f"Phase | A = {amp:.3f}"),
            row=1,
            col=1,
        )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_phase_unwrapped_heatmap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    signal = []
    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)
        signal.append(re + 1j * im)
    signal = np.array(signal, dtype=np.complex128)
    phase = np.arctan2(signal.imag, signal.real)
    phase_unwrapped = np.unwrap(phase, axis=1)

    global_title = "Phase unwrapped along duration vs. Time"
    title_x = "Flux Pulse duration"
    title_y = "Phase"
    figs["norm"].add_trace(
        go.Heatmap(
            x=durations,
            y=amplitudes,
            z=phase_unwrapped,
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_phase_amplitude_unwrapped_heatmap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    # start the unroll desde amp = 0 (no phase shift)
    # unroll in both directions

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    signal = []
    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)
        signal.append(re + 1j * im)
    signal = np.array(signal, dtype=np.complex128)
    phase = np.arctan2(signal.imag, signal.real)
    phase_unwrapped = np.unwrap(phase, axis=1)

    global_title = "Phase unwrapped along amplitude vs. Time"
    title_x = "Flux Pulse duration"
    title_y = "Phase"
    phase = np.arctan2(signal.imag, signal.real)
    phase_unwrapped = np.unwrap(phase, axis=0)
    figs["norm"].add_trace(
        go.Heatmap(
            x=durations,
            y=amplitudes,
            z=phase_unwrapped,
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_detuning_time(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    sampling_rate = 1 / (durations[1] - durations[0])
    derivative_window_length = 7 / sampling_rate
    derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
    derivative_window_size += (derivative_window_size + 1) % 2
    derivative_window_size = 9
    derivative_order = 2
    nyquist_order = 0

    # signal = []
    # detuning_mean = []
    # signal_fft = []

    for amp in amplitudes:  # unwrap along duration
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        # Unwrap phase
        phase = np.arctan2(im, re)
        phase_unwrapped = np.unwrap(phase)

        # -----------------      Unwrap along durations     ---------------------------
        # phase_unwrapped_matrix = np.array(amplitudes).reshape(len(amplitudes),1)
        # for dur in durations: # unwrap along amplitudes
        #     re = _get_values(data, MX_tag, duration=dur)
        #     im = _get_values(data, MY_tag, duration=dur)

        #     # Unwrap phase
        #     phase = np.arctan2(im, re)
        #     phase_unwrapped = np.unwrap(phase)

        #     phase_unwrapped_matrix = np.append(phase_unwrapped_matrix, phase_unwrapped, axis=1)

        # for n in range(1, len(amplitudes)):
        #     phase_unwrapped = phase_unwrapped_matrix[n]
        # -----------------------------------------------------------------------------

        # use a savitzky golay filter: it takes sliding window of length
        # `window_length`, fits a polynomial, returns derivative at
        # middle point

        # Derivate
        detuning = (
            savgol_filter(
                phase_unwrapped / (2 * np.pi),
                window_length=derivative_window_size,
                polyorder=derivative_order,
                deriv=1,
            )
            * sampling_rate
            * 1e9
        )

        # nyquist_order = 0
        # detuning = get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order)

        # Alternative derivation method
        # dt = np.diff(duration_unique) * 1e-9
        # dphi_dt_unwrap = np.diff(phase_unwrapped) / dt
        # detuning = dphi_dt_unwrap / (2 * np.pi)
        # detuning = smooth(detuning, window_len=21)

        # Detunning vs. Time
        global_title = "Detuning vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Detunning"
        figs["norm"].add_trace(
            go.Scatter(x=durations, y=detuning, name=f"A = {amp:.3f}"),
            row=1,
            col=1,
        )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_distorted_amplitude_time(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    signal = []  # TODO: replace lists with np arrays
    fft = []
    freqs = []

    curve_amps = []
    curve_detunings = []

    num_points = durations.shape[0]
    interval = (durations[1] - durations[0]) * 1e-9
    amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
    mask = np.argsort(amp_freqs)
    amp_freqs = amp_freqs[mask]

    sampling_rate = 1 / (durations[1] - durations[0])
    derivative_window_length = 7 / sampling_rate
    derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
    derivative_window_size += (derivative_window_size + 1) % 2
    derivative_window_size = 9
    derivative_order = 3
    nyquist_order = 0

    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        amp_signal = re + 1j * im
        signal.append(amp_signal)

        freqs.append(amp_freqs)

        amp_fft = np.fft.fft(amp_signal - np.mean(amp_signal))[mask]
        fft.append(amp_fft)

        smoothed_y = smooth(np.absolute(amp_fft), window_len=21)
        percval = np.percentile(
            smoothed_y, 90
        )  # finds the top perc points in amplitude smoothed_y
        filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
        array_peaks = argrelextrema(filtered_y, np.greater)
        peaks_x = np.arange(num_points)[array_peaks]
        sort_mask = np.argsort(np.absolute(amp_fft)[array_peaks])[::-1]

        if len(peaks_x[sort_mask]) > 0 and amp_freqs[peaks_x[sort_mask][0]] <= 0:
            curve_amps.append(amp)
            curve_detunings.append(amp_freqs[peaks_x[sort_mask][0]])

    pfit = np.polyfit(curve_amps, curve_detunings, 2)
    curve_amps = np.array(curve_amps)

    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        # Unwrap phase
        phase = np.arctan2(im, re)
        phase_unwrapped = np.unwrap(phase)

        # -----------------      Unwrap along durations     ---------------------------
        # phase_unwrapped_matrix = np.array(amplitudes).reshape(len(amplitudes),1)
        # for dur in durations: # unwrap along amplitudes
        #     re = _get_values(data, MX_tag, duration=dur)
        #     im = _get_values(data, MY_tag, duration=dur)

        #     # Unwrap phase
        #     phase = np.arctan2(im, re)
        #     phase_unwrapped = np.unwrap(phase)

        #     phase_unwrapped_matrix = np.append(phase_unwrapped_matrix, phase_unwrapped, axis=1)

        # for n in range(1, len(amplitudes)):
        #     phase_unwrapped = phase_unwrapped_matrix[n]
        # -----------------------------------------------------------------------------

        # Derivate
        detuning = (
            savgol_filter(
                phase_unwrapped / (2 * np.pi),
                window_length=derivative_window_size,
                polyorder=derivative_order,
                deriv=1,
            )
            * sampling_rate
            * 1e9
        )

        clipped_detuning = np.clip(
            detuning,
            a_max=-1 / 4 * (pfit[1] ** 2) / pfit[0] + pfit[2],
            a_min=np.min(detuning),
        )
        plt.plot(detuning)
        plt.plot(clipped_detuning)
        plt.savefig("/nfs/users/alvaro.orgaz/clipped.png")
        plt.close()

        distorted_pulse = (
            -pfit[1]
            - np.sqrt(pfit[1] ** 2 - (4 * pfit[0] * (pfit[2] - clipped_detuning)))
        ) / (2 * pfit[0])
        # distorted_pulse = (-pfit[1] + np.sqrt((pfit[1]**2 - (4* pfit[0] * (pfit[2] - clipped_detuning)))))/(2*pfit[0])

        global_title = "Distorted Pulse Amplitude vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Amplitude"
        figs["norm"].add_trace(
            go.Scatter(
                x=durations,
                y=distorted_pulse,
                name=f"A = {amp:.3f}",
                mode="lines+markers",
                marker=dict(size=5),
            ),
            row=1,
            col=1,
        )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


def cryoscope_reconstructed_amplitude_time(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, "data")
    durations, amplitudes = _get_flux_pulse_durations_and_amplitudes(data)

    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    signal = []  # TODO: replace lists with np arrays
    fft = []
    freqs = []

    curve_amps = []
    curve_detunings = []

    num_points = durations.shape[0]
    interval = (durations[1] - durations[0]) * 1e-9
    amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
    mask = np.argsort(amp_freqs)
    amp_freqs = amp_freqs[mask]

    sampling_rate = 1 / (durations[1] - durations[0])
    derivative_window_length = 7 / sampling_rate
    derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
    derivative_window_size += (derivative_window_size + 1) % 2
    derivative_window_size = 15
    derivative_order = 3
    nyquist_order = 0

    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        amp_signal = re + 1j * im
        signal.append(amp_signal)

        freqs.append(amp_freqs)

        amp_fft = np.fft.fft(amp_signal - np.mean(amp_signal))[mask]
        fft.append(amp_fft)

        smoothed_y = smooth(np.absolute(amp_fft), window_len=21)
        percval = np.percentile(
            smoothed_y, 90
        )  # finds the top perc points in amplitude smoothed_y
        filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
        array_peaks = argrelextrema(filtered_y, np.greater)
        peaks_x = np.arange(num_points)[array_peaks]
        sort_mask = np.argsort(np.absolute(amp_fft)[array_peaks])[::-1]

        if len(peaks_x[sort_mask]) > 0 and amp_freqs[peaks_x[sort_mask][0]] <= 0:
            curve_amps.append(amp)
            curve_detunings.append(amp_freqs[peaks_x[sort_mask][0]])

    pfit = np.polyfit(curve_amps, curve_detunings, 2)
    curve_amps = np.array(curve_amps)
    # x=pfit[0]*curve_amps**2 + pfit[1]*curve_amps + pfit[2]

    for amp in amplitudes:
        re = _get_values(data, MX_tag, amplitude=amp)
        im = _get_values(data, MY_tag, amplitude=amp)

        # Unwrap phase
        phase = np.arctan2(im, re)
        phase_unwrapped = np.unwrap(phase)

        # -----------------      Unwrap along durations     ---------------------------
        # phase_unwrapped_matrix = np.array(amplitudes).reshape(len(amplitudes),1)
        # for dur in durations: # unwrap along amplitudes
        #     re = _get_values(data, MX_tag, duration=dur)
        #     im = _get_values(data, MY_tag, duration=dur)

        #     # Unwrap phase
        #     phase = np.arctan2(im, re)
        #     phase_unwrapped = np.unwrap(phase)

        #     phase_unwrapped_matrix = np.append(phase_unwrapped_matrix, phase_unwrapped, axis=1)

        # for n in range(1, len(amplitudes)):
        #     phase_unwrapped = phase_unwrapped_matrix[n]
        # -----------------------------------------------------------------------------

        # Derivate
        detuning = (
            savgol_filter(
                phase_unwrapped / (2 * np.pi),
                window_length=derivative_window_size,
                polyorder=derivative_order,
                deriv=1,
            )
            * sampling_rate
            * 1e9
        )  # TODO: fix units e28!!??
        print(f"pfit: {pfit}")

        clipped_detuning = np.clip(
            detuning,
            a_max=-1 / 4 * (pfit[1] ** 2) / pfit[0] + pfit[2],
            a_min=np.min(detuning),
        )
        plt.plot(detuning)
        plt.plot(clipped_detuning)
        plt.savefig("/nfs/users/alvaro.orgaz/clipped.png")
        plt.close()

        distorted_pulse = (
            -pfit[1]
            - np.sqrt(pfit[1] ** 2 - (4 * pfit[0] * (pfit[2] - clipped_detuning)))
        ) / (2 * pfit[0])
        # distorted_pulse = (-pfit[1] + np.sqrt((pfit[1]**2 - (4* pfit[0] * (pfit[2] - clipped_detuning)))))/(2*pfit[0])

        global_title = "Reconstructed Pulse Amplitude vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Amplitude"
        figs["norm"].add_trace(
            go.Scatter(
                x=durations,
                y=distorted_pulse,
                name=f"Distorted A = {amp:.3f}",
                mode="lines+markers",
                marker=dict(size=5),
            ),
            row=1,
            col=1,
        )

        # fit over amplitudes vs time
        ideal_pulse = np.ones(len(durations))
        start_fit = 40 - 4
        end_fit = 165 - 4  # len(durations)
        amp_guess = -0.78
        tau_guess = 37
        pguess = [amp_guess, tau_guess]
        b_fit, a_fit, popt = cryoscope_fit_exp(
            durations=durations[start_fit:end_fit],
            ideal_pulse=ideal_pulse[start_fit:end_fit],
            distorted_pulse=distorted_pulse[start_fit:end_fit],
            pguess=pguess,
        )  # FIXME
        # print(b_fit, a_fit)
        reconstructed_pulse = cryoscope_step_response_exp(
            t=durations[start_fit:end_fit], p=popt, pulse=ideal_pulse[start_fit:end_fit]
        )

        # simulated_step_extrap = cryoscope_step_response_exp(t=durations,
        #                                             p=popt,
        #                                             awg_reconstructed_pulse=ideal_pulse)

        # initial_guess = cryoscope_step_response_exp(t=durations,
        #                                             p=[0.78,37],
        #                                             awg_reconstructed_pulse=ideal_pulse)

        figs["norm"].add_trace(
            go.Scatter(
                x=durations[start_fit:end_fit],
                y=ideal_pulse,
                name=f"Ideal A = {amp:.3f}",
                mode="lines+markers",
                marker=dict(size=5),  # , color="lightcoral"),
            ),
            row=1,
            col=1,
        )

        figs["norm"].add_trace(
            go.Scatter(
                x=durations[start_fit:end_fit],
                y=reconstructed_pulse,
                name=f"Reconstructed A = {amp:.3f}",
                mode="lines+markers",
                marker=dict(size=5),  # , color="lightcoral"),
            ),
            row=1,
            col=1,
        )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return [figs["norm"]]


# Helper functions


def smooth(x, window_len=11, window="hanning"):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the
        signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are
        minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an
            odd integer
        window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman'
        flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve
    scipy.signal.lfilter
    """
    if int(window_len) & 0x1 == 0:
        window_len += 1

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]

    if window == "flat":  # moving average
        w = np.ones(window_len, "d")

    else:
        w = eval("np." + window + "(window_len)")
    y = np.convolve(w / w.sum(), s, mode="valid")

    # Cut edges of y since a mirror image is used
    edge = (window_len - 1) / 2
    edge = int(edge)
    return y[edge:-edge]


def coefs_from_exp(amp, tau):
    alpha = 1 - np.exp(-1 / (tau * (1 + amp)))
    # the coefficient k (and the filter transfer function)
    # depend on the sign of amp

    if amp >= 0.0:
        k = amp / (1 + amp - alpha)
        # compensating overshoot by a low-pass filter
        # correction filter y[n] = (1-k)*x[n] + k*u[n] = x[n] + k*(u[n]-x[n])
        # where u[n] = u[n-1] + alpha*(x[n] - u[n-1])
    else:
        k = amp / (1 + amp) / (1 - alpha)
        # compensating low-pass by an overshoot
        # correction filter y[n] = (1+k)*x[n] - k*u[n] = x[n] - k*(u[n]-x[n])
        # where u[n] = u[n-1] + alpha*(x[n] - u[n-1])

    b = [(1 - k + k * alpha), -(1 - k) * (1 - alpha)]

    # while the denominator stays the same
    a = [1, -(1 - alpha)]
    return b, a


def cryoscope_step_response_exp(t, p, pulse):
    # This function returns the convolution of a filer h(t,params=p)
    #   applied to meassured distorted pulse -> should return rectangular
    #   applied to a rectangular pulse -> returns a predistorted pulse
    # It is done via lfilter function, and using an exponential model for h(t,p)

    # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
    # p = [A, tau_iir]
    # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
    # p = [b0, b1, a0, a1]
    from scipy.signal import lfilter

    amp = p[0]  # ration between the shooting point and the settling amplitude
    tau_iir = p[1]
    b_fit, a_fit = coefs_from_exp(amp, tau_iir)
    return lfilter(b_fit, a_fit, pulse)


def cryoscope_fit_exp(durations, ideal_pulse, distorted_pulse, pguess):
    # print('calling curve_fit')
    # print(pguess)
    amp_guess = pguess[0]
    tau_guess = pguess[1]

    def wrapper(t, amp, tau, gain):
        return gain * cryoscope_step_response_exp(t, [amp, tau], distorted_pulse)

    fit_mod = Model(wrapper)
    fit_mod.set_param_hint("amp", value=amp_guess, min=-2, max=2, vary=True)
    fit_mod.set_param_hint("tau", value=tau_guess, min=0, max=1000, vary=True)
    fit_mod.set_param_hint("gain", value=1, min=0, max=2, vary=True)
    params = fit_mod.make_params()
    fit_result = fit_mod.fit(data=ideal_pulse, t=durations, params=params)
    print(fit_result.fit_report())
    # popt, pcov = curve_fit(, x_data, y_data, p0=pguess)
    print(fit_result.best_values)
    fit_result.plot_fit()
    amp = fit_result.best_values[
        "amp"
    ]  # ration between the shooting point and the settling amplitude
    tau_iir = fit_result.best_values["tau"]
    b_fit, a_fit = coefs_from_exp(amp, tau_iir)
    return b_fit, a_fit, [amp, tau_iir]


# def cryoscope_fit(x_data, y_data, awg_reconstructed_pulse):
#     pguess = [
#         1, #b0
#         0, #b1
#         1, #a0
#         0  #a1
#     ]
#     popt, pcov = curve_fit(lambda t,*p:cryoscope_step_response(t, p, awg_reconstructed_pulse), x_data, y_data, p0=pguess)
#     b_fit = popt[[0,1]]
#     a_fit = popt[[2,3]]
#     return b_fit, a_fit

# def cryoscope_step_response(t, p, awg_reconstructed_pulse):

#     # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
#     # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
#     # p = [b0, b1, a0, a1]
#     from scipy.signal import lfilter
#     b_fit = p[:2]
#     a_fit = p[2:]
#     return lfilter(b_fit, a_fit, awg_reconstructed_pulse)

# def cryoscope_step_response_exp(t, p, awg_reconstructed_pulse):

#     # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
#     # p = [A, tau_iir]
#     # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
#     # p = [b0, b1, a0, a1]
#     from scipy.signal import lfilter
#     amp = p[0] # ration between the shooting point and the settling amplitude
#     tau_iir = p[1]
#     alpha = 1-np.exp((1+amp)/tau_iir)
#     if amp<0:
#         k = amp/((1+amp)*(1-alpha))
#     else:
#         k = 1/(1+amp-alpha)
#     b_fit = [1-k+k*alpha, -(1-k)*(1-alpha)]
#     a_fit = [1,-(1-alpha)]
#     return lfilter(b_fit, a_fit, awg_reconstructed_pulse)

# def cryoscope_fit_exp(x_data, y_data, awg_reconstructed_pulse):
#     amp_guess = -0.78
#     tau_guess = 37
#     pguess = [amp_guess, tau_guess]
#     popt, pcov = curve_fit(lambda t,*p:cryoscope_step_response_exp(t, p, awg_reconstructed_pulse), x_data, y_data, p0=pguess)
#     amp = popt[0] # ration between the shooting point and the settling amplitude
#     tau_iir = popt[1]
#     alpha = 1-np.exp((1+amp)/tau_iir)
#     if amp<0:
#         k = amp/((1+amp)*(1-alpha))
#     else:
#         k = 1/(1+amp-alpha)
#     b_fit = [1-k+k*alpha, -(1-k)*(1-alpha)]
#     a_fit = [1,-(1-alpha)]
#     return b_fit, a_fit, popt

# ?
# def cryoscope_detuning_amplitude(folder, routine, qubit, format):
#     data = DataUnits.load_data(folder, "data", routine, format, "data")
#     import numpy as np
#     import scipy.signal as ss

#     MX_tag = "MX"
#     MY_tag = "MY"

#     amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
#     duration = data.get_values("flux_pulse_duration", "ns")
#     flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
#     flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
#     amplitude_unique = flux_pulse_amplitude[
#         flux_pulse_duration == flux_pulse_duration[0]
#     ]
#     duration_unique = flux_pulse_duration[
#         flux_pulse_amplitude == flux_pulse_amplitude[0]
#     ]

#     # Making figure
#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     mean_detuning = []

#     for amp in amplitude_unique:
#         x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#         y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

#         complex_data = x_data + 1j * y_data
#         #print (len(complex_data))
#         ws = 10 # Needs to be adjusted for data analysis
#         norm_data = normalize_sincos(complex_data, window_size=ws)

#         sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
#         derivative_window_length = 7 / sampling_rate
#         derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#         derivative_window_size += (derivative_window_size + 1) % 2
#         derivative_order = 2
#         nyquist_order=0

#         freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
#         demod_freq = - \
#                 freq_guess * sampling_rate

#         #y = amp*exp(2pi i f t + ph) + off.
#         #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
#         demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
#         # Di Carlo has smooth here!!!


#         # Unwrap phase
#         phase = np.arctan2(demod_data.imag, demod_data.real)
#         # phase_unwrapped = np.unwrap(phase)
#         phase_unwrapped = np.unwrap(np.angle(demod_data))

#         # use a savitzky golay filter: it take sliding window of length
#         # `window_length`, fits a polynomial, returns derivative at
#         # middle point
#         # phase = phase_unwrapped_data

#         # Di Carlo method
#         detuning = savgol_filter(
#             phase_unwrapped / (2 * np.pi),
#             window_length = derivative_window_size,
#             polyorder = derivative_order,
#             deriv=1) * sampling_rate * 1e9

#         nyquist_order = 0
#         detuning = get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order)

#         # Maxime method
#         # dt = np.diff(duration_unique) * 1e-9
#         # dphi_dt_unwrap = np.abs(np.diff(phase_unwrapped) / dt)
#         # detuning = dphi_dt_unwrap / (2 * np.pi)


#         mean_detuning.append(np.mean(detuning))

#     # Mean detunning vs. amplitude
#     global_title = "Mean Detuning vs. Amplitude"
#     title_x = "Amplitude (dimensionless)"
#     title_y = "Detunning mean (Hz)"
#     figs["norm"].add_trace(
#         go.Scatter(
#             x=amplitude_unique,
#             y=mean_detuning,
#             name=f"Detuning"
#         ),
#         row=1,
#         col=1,
#     )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     return [figs["norm"]]


# def peak_finder_v2(x, y, perc=90, window_len=11):
#     """
#     Peak finder based on argrelextrema function from scipy
#     only finds maximums, this can be changed to minimum by using -y instead of y
#     """
#     smoothed_y = smooth(y, window_len=window_len)
#     percval = np.percentile(smoothed_y, perc) # finds the top perc points in amplitude smoothed_y
#     filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
#     array_peaks = argrelextrema(filtered_y, np.greater)
#     peaks_x = x[array_peaks]
#     sort_mask = np.argsort(y[array_peaks])[::-1]
#     return peaks_x[sort_mask]


# def normalize_sincos(
#         data,
#         window_size_frac=500,
#         window_size=None,
#         do_envelope=True):

#     if window_size is None:
#         window_size = len(data) // window_size_frac

#         # window size for savgol filter must be odd
#         window_size -= (window_size + 1) % 2

#     mean_data_r = savgol_filter(data.real, window_size, 0, 0)
#     mean_data_i = savgol_filter(data.imag, window_size, 0, 0)

#     mean_data = mean_data_r + 1j * mean_data_i

#     if do_envelope:
#         envelope = np.sqrt(
#             savgol_filter(
#                 (np.abs(
#                     data -
#                     mean_data))**2,
#                 window_size,
#                 0,
#                 0))
#     else:
#         envelope = 1
#     norm_data = ((data - mean_data) / envelope)
#     return norm_data

# def fft_based_freq_guess_complex(y):
#     """
#     guess the shape of a sinusoidal complex signal y (in multiples of
#         sampling rate), by selecting the peak in the fft.
#     return guess (f, ph, off, amp) for the model
#         y = amp*exp(2pi i f t + ph) + off.
#     """
#     fft = np.fft.fft(y)[1:len(y)]
#     freq_guess_idx = np.argmax(np.abs(fft))
#     if freq_guess_idx >= len(y) // 2:
#         freq_guess_idx -= len(y)

#     freq_guess = 1 / len(y) * (freq_guess_idx + 1)
#     phase_guess = np.angle(fft[freq_guess_idx]) + np.pi / 2
#     amp_guess = np.absolute(fft[freq_guess_idx]) / len(y)
#     offset_guess = np.mean(y)

#     return freq_guess, phase_guess, offset_guess, amp_guess

# def get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order=0):
#     real_detuning = detuning - demod_freq + sampling_rate * nyquist_order
#     return real_detuning

# def normalize_data(x, y, amplitude_unique, duration_unique, flux_pulse_amplitude, flux_pulse_duration, amplitude):
#     x_norm = np.ones((len(amplitude_unique), len(duration_unique))) * np.nan
#     y_norm = np.ones((len(amplitude_unique), len(duration_unique))) * np.nan

#     for i, amp in enumerate(amplitude_unique):
#         idx = np.where(flux_pulse_amplitude == amp)
#         if amp == flux_pulse_amplitude[-1]:
#             n = int(np.where(flux_pulse_duration[-1] == duration_unique)[0][0]) + 1
#         else:
#             n = len(duration_unique)

#         x_norm[i, :n] = x[idx]
#         x_norm[i, :n] = x_norm[i, :n] - 0.5
#         x_norm[i, :n] = x_norm[i, :n] / max(abs(x_norm[i, :n]))
#         y_norm[i, :n] = y[idx]
#         y_norm[i, :n] = y_norm[i, :n] - 0.5
#         y_norm[i, :n] = y_norm[i, :n] / max(abs(y_norm[i, :n]))

#     return x_norm, y_norm

# def normalize_sincos(x, y, window_size_frac=61, window_size=None, do_envelope=True):

#     if window_size is None:
#         window_size = len(x) // window_size_frac

#         # window size for savgol filter must be odd
#         window_size -= (window_size + 1) % 2

#     x = savgol_filter(x, window_size, 0, 0)  ## why poly order 0
#     y = savgol_filter(y, window_size, 0, 0)

#     # if do_envelope:
#     #     envelope = np.sqrt(savgol_filter((np.abs(data - mean_data))**2, window_size, 0, 0))
#     # else:
#     #     envelope = 1

#     # norm_data = ((data - mean_data) / envelope)

#     return x, y

# def cryoscope_demod_circle(folder, routine, qubit, format):
#     data = DataUnits.load_data(folder, "data", routine, format, "data")
#     import numpy as np
#     import scipy.signal as ss

#     MX_tag = "MX"
#     MY_tag = "MY"

#     amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
#     duration = data.get_values("flux_pulse_duration", "ns")
#     flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
#     flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
#     amplitude_unique = flux_pulse_amplitude[
#         flux_pulse_duration == flux_pulse_duration[0]
#     ]
#     duration_unique = flux_pulse_duration[
#         flux_pulse_amplitude == flux_pulse_amplitude[0]
#     ]

#     # Making figure
#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     for amp in amplitude_unique:
#         x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#         y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

#         complex_data = x_data + 1j * y_data
#         #print (len(complex_data))
#         ws = 10 # Needs to be adjusted for data analysis
#         norm_data = normalize_sincos(complex_data, window_size=ws)

#         sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
#         derivative_window_length = 7 / sampling_rate
#         derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#         derivative_window_size += (derivative_window_size + 1) % 2
#         derivative_order = 2
#         nyquist_order=0

#         freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
#         demod_freq = - \
#                 freq_guess * sampling_rate

#         #y = amp*exp(2pi i f t + ph) + off.
#         #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
#         demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
#         # Di Carlo has smooth here!!!

#         # demod_data circle
#         global_title = "FFT demod data circle"
#         title_x = "<X>"
#         title_y = "<Y>"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=demod_data.real,
#                 y=demod_data.imag,
#                 name=f"A = {amp:.3f}"
#             ),
#             row=1,
#             col=1,
#         )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     return [figs["norm"]]

# def cryoscope_demod_fft(folder, routine, qubit, format):
#     data = DataUnits.load_data(folder, "data", routine, format, "data")
#     import numpy as np
#     import scipy.signal as ss

#     MX_tag = "MX"
#     MY_tag = "MY"

#     amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
#     duration = data.get_values("flux_pulse_duration", "ns")
#     flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
#     flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
#     amplitude_unique = flux_pulse_amplitude[
#         flux_pulse_duration == flux_pulse_duration[0]
#     ]
#     duration_unique = flux_pulse_duration[
#         flux_pulse_amplitude == flux_pulse_amplitude[0]
#     ]

#     # Making figure
#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     for amp in amplitude_unique:
#         x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#         y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

#         complex_data = x_data + 1j * y_data
#         #print (len(complex_data))
#         ws = 10 # Needs to be adjusted for data analysis
#         norm_data = normalize_sincos(complex_data, window_size=ws)

#         sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
#         derivative_window_length = 7 / sampling_rate
#         derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#         derivative_window_size += (derivative_window_size + 1) % 2
#         derivative_order = 2
#         nyquist_order=0

#         freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
#         demod_freq = - \
#                 freq_guess * sampling_rate

#         #y = amp*exp(2pi i f t + ph) + off.
#         #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
#         demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
#         # Di Carlo has smooth here!!!

#         # <X> sin <Y> cos demod data
#         global_title = "<X> , <Y> after FFT demodulation"
#         title_x = "Flux Pulse duration"
#         title_y = "<X> , <Y>"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=duration_unique,
#                 y=demod_data.real,
#                 name=f"<X> fft demod | A = {amp:.3f}"
#             ),
#             row=1,
#             col=1,
#         )

#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=duration_unique,
#                 y=demod_data.imag,
#                 name=f"<Y> fft demod | A = {amp:.3f}"
#             ),
#             row=1,
#             col=1,
#         )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     return [figs["norm"]]
