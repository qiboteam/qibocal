"""ZZ-interaction strength versus coupler flux-pulse amplitude.

Conditional Ramsey experiment on the target qubit, repeated with the control
qubit in :math:`|0\\rangle` and :math:`|1\\rangle`, while applying a rectangular
coupler flux pulse during the Ramsey wait. The pulsed ZZ coupling is extracted as

.. math::

    \\zeta(A) = f_{\\text{target}}^{(c=1)}(A)
               - f_{\\text{target}}^{(c=0)}(A),

where :math:`f_{\\text{target}}` is the physical conditional target frequency,
not the unsigned detuned fringe frequency. The
minimum-coupling pulse amplitude is located with a local quadratic fit to
:math:`\\zeta(A)`. Population-only Ramsey measures unsigned fringe frequencies:
both conditional detunings must stay on the branch selected by ``detuning``.

References:
    - Collodo et al., "Implementation of Conditional-Phase Gates based on
      tunable ZZ-Interactions", PRL 125, 240502 (2020).
    - Sung et al., "Realization of High-Fidelity CZ and ZZ-Free iSWAP Gates
      with a Tunable Coupler", PRX 11, 021058 (2021).
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.constants as const
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseLike,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibocal.auto.operation import Protocol, QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.couplers.jazz import (
    CouplerJAZZData,
    CouplerJAZZParameters,
    CouplerJAZZResults,
    CouplerZZType,
)
from qibocal.protocols.ramsey.processing import fitting as ramsey_fitting
from qibocal.protocols.ramsey.processing import ramsey_fit as ramsey_func

__all__ = ["coupler_zz"]

SETUPS = ["I", "X"]
"""Control qubit preparations: identity (ground state) and X (excited state)."""


@dataclass
class CouplerZZParameters(CouplerJAZZParameters):
    """CouplerZZ runcard inputs."""

    detuning: float | None = None
    """Artificial detuning of the target drive frequency [Hz], to produce
    Ramsey fringes."""


@dataclass
class CouplerZZResults(CouplerJAZZResults):
    """CouplerZZ outputs."""

    zz: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """ZZ coupling [Hz] for each pulse amplitude (NaN where fitting failed)."""
    fitting_params: dict[QubitPairId, dict[float, dict[str, list[float]]]] = field(
        default_factory=dict
    )
    """Fitting parameters for each qubit pair."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return pair in self.zz


@dataclass
class CouplerZZData(CouplerJAZZData):
    """CouplerZZ acquisition outputs."""

    detuning: float | None = None
    """Artificial detuning of the target drive frequency [Hz]."""
    data: dict[tuple[QubitId, QubitId, str], npt.NDArray[CouplerZZType]] = field(
        default_factory=dict
    )
    """Acquired data for each pulse amplitude."""

    def amplitudes(self, pair: QubitPairId) -> npt.NDArray:
        key = pair + ("X",)
        return np.unique(self.data[key]["amplitude"])

    def return_data_per_amplitude(
        self, pair: QubitPairId, amp: float, setup: str
    ) -> npt.NDArray[CouplerZZType]:
        return self.data[pair + (setup,)][
            np.array(self.data[pair + (setup,)])["amplitude"] == amp
        ]


def _acquisition(
    params: CouplerZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerZZData:
    """Conditional Ramsey on the target qubit versus coupler pulse amplitude.

    For each control state the sequence is::

        control: [RX if state == 1]
        target:               RX90 -- wait -- RX90 -- MZ
        coupler:                     -- flux pulse --

    The inner sweeper varies both the Ramsey wait and the flux-pulse duration;
    the outer sweeper varies the pulse amplitude, leaving the DC bias unchanged.
    """

    coupler_pulse = Pulse(
        duration=params.delay_range[0],
        amplitude=params._coupler_range[0],
        envelope=Rectangular(),
    )

    delay = Delay(duration=params.delay_range[0])

    updates = []
    coupler_offsets: dict[QubitPairId, float] = {}
    sequences: dict[str, PulseSequence] = defaultdict(PulseSequence)
    target_ro_pulses: dict[str, dict[QubitPairId, PulseLike]] = defaultdict(dict)
    spectator_ro_pulses: dict[str, dict[QubitPairId, PulseLike]] = defaultdict(dict)
    for s in SETUPS:
        for pair in targets:
            target_qubit, spectator_qubit = pair

            target_natives = platform.natives.single_qubit[target_qubit]
            target_rx90_channel, target_rx90_pulse = target_natives.R(theta=np.pi / 2)[
                0
            ]
            target_ro_channel, target_ro_pulse = target_natives.MZ()[0]

            spectator_natives = platform.natives.single_qubit[spectator_qubit]
            spectator_rx_channel, spectator_rx_pulse = spectator_natives.RX()[0]
            spectator_ro_channel, spectator_ro_pulse = spectator_natives.MZ()[0]

            target_ro_pulses[s][pair] = target_ro_pulse
            spectator_ro_pulses[s][pair] = spectator_ro_pulse

            if params.detuning is not None:
                f0 = platform.config(target_rx90_channel).frequency
                updates.append(
                    {target_rx90_channel: {"frequency": f0 + params.detuning}}
                )

            coupler_id = platform.calibration.two_qubits[pair].coupler
            if coupler_id is None:
                raise ValueError("Coupler ID should not be None")

            coupler_channel = platform.couplers[coupler_id].flux
            assert coupler_channel is not None, "Coupler channel should not be None"

            if pair not in coupler_offsets:
                coupler_offsets[pair] = platform.config(coupler_channel).offset

            seq = PulseSequence([(target_rx90_channel, target_rx90_pulse)])
            if s == "X":
                seq |= PulseSequence([(spectator_rx_channel, spectator_rx_pulse)])

            waiting_seq = PulseSequence(
                [
                    (target_ro_channel, delay),
                    (spectator_ro_channel, delay),
                    (target_rx90_channel, delay),
                    (coupler_channel, coupler_pulse),
                ]
            )

            ro_seq = target_natives.R(theta=np.pi / 2) | PulseSequence(
                [
                    (target_ro_channel, target_ro_pulse),
                    (spectator_ro_channel, spectator_ro_pulse),
                ]
            )

            sequences[s] += seq | waiting_seq | ro_seq

    wait_sweeper = Sweeper(
        parameter=Parameter.duration,
        range=params.delay_range,
        pulses=[delay, coupler_pulse],
    )
    amplitude_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=params._coupler_range,
        pulses=[coupler_pulse],
    )

    results = platform.execute(
        list(sequences.values()),
        [[amplitude_sweeper], [wait_sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        updates=updates,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = CouplerZZData(
        detuning=params.detuning,
        couplers_offsets=coupler_offsets,
    )

    amplitudes = np.asarray(amplitude_sweeper.values)
    delays = np.asarray(wait_sweeper.values)

    for s in SETUPS:
        for pair in targets:
            t_ro_pulse = target_ro_pulses[s][pair]
            s_ro_pulse = spectator_ro_pulses[s][pair]

            t_probs = np.asarray(results[t_ro_pulse.id])
            t_error = np.sqrt(t_probs * (1 - t_probs) / params.nshots)

            s_probs = np.asarray(results[s_ro_pulse.id])
            s_error = np.sqrt(s_probs * (1 - s_probs) / params.nshots)

            # Store absolute bias here; fitting and updates must not add DC again.
            data.register_qubit(
                CouplerZZType,
                (*pair, s),
                {
                    "delay": np.tile(delays, len(amplitudes)),
                    "amplitude": np.repeat(
                        amplitudes + coupler_offsets[pair], len(delays)
                    ),
                    "prob": t_probs.ravel(),
                    "error": t_error.ravel(),
                    "prob_control": s_probs.ravel(),
                    "error_control": s_error.ravel(),
                },
            )

    return data


def _fit(data: CouplerZZData) -> CouplerZZResults:
    """Extract physical ZZ from both conditional fringes and minimize its magnitude."""

    delays = data.delays
    zz: dict[QubitPairId, list[float]] = defaultdict(list)
    fitting_params: dict[QubitPairId, dict[float, dict[str, list[float]]]] = (
        defaultdict(dict)
    )
    delta_fitting_measure: dict[tuple[QubitId, QubitId, str], list[float]] = {}
    for pair in data.pairs:
        target, spectator = pair

        amplitudes = data.amplitudes(pair)
        for amp in amplitudes:
            try:  # try outside of the for loop because we need both signals to fit (both control in 0 or 1)
                for setup in SETUPS:
                    setup_data = data.return_data_per_amplitude(pair, amp, setup)
                    popt, perr = ramsey_fitting(
                        delays, setup_data["prob"], setup_data["error"]
                    )
                    delta_fitting_measure[target, spectator, setup] = [
                        -popt[2] / (2 * np.pi) * const.giga,
                        perr[2] * const.giga / (2 * np.pi),
                    ]

                    fitting_params[pair].setdefault(amp, {})[setup] = popt

                zz_amp = float(
                    delta_fitting_measure[target, spectator, "X"][0]
                    - delta_fitting_measure[target, spectator, "I"][0]
                )
                zz[pair].append(zz_amp)

            # here we are catching two possible errors: fitting error in curve_fit
            # call or error in estimating coupling strenght because anharmonicities
            # are unknown
            except (RuntimeError, ValueError) as e:
                log.warning(f"Ramsey fitting failed for qubit pair {pair} due to {e}.")

    return CouplerZZResults(zz=zz, fitting_params=fitting_params)


def _plot(
    data: CouplerZZData,
    target: QubitPairId,
    fit: CouplerZZResults | None = None,
):
    """Heatmaps of the conditional Ramsey fringes and ZZ versus pulse amplitude."""

    figures = []
    waits = data.delays
    amplitudes = data.amplitudes(target)

    heatmap_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Control in |0⟩",
            "Control in |1⟩",
        ),
        shared_yaxes=True,
    )

    for col, setup in enumerate(SETUPS, start=1):
        heatmap_fig.add_trace(
            _probability_heatmap(
                data.data.get((target + (setup,)), []),
                waits,
                amplitudes,
                showscale=(col == 2),
            ),
            row=1,
            col=col,
        )
        heatmap_fig.update_xaxes(
            title_text="Coupler pulse amplitude [a.u.]", row=1, col=col
        )

    heatmap_fig.update_yaxes(title_text="Wait time [ns]", row=1, col=1)
    figures.append(heatmap_fig)

    fitting_report = ""
    if fit is not None and target in fit.zz:
        zz_fig, fitting_report = _plot_zz(fit, target, amplitudes)
        figures.append(zz_fig)

    if fit is not None and target in fit.fitting_params:
        for amp in amplitudes:
            fig = go.Figure()
            for s in SETUPS:
                data_amp = data.return_data_per_amplitude(target, amp, s)
                plot_t = np.linspace(data_amp["delay"][0], data_amp["delay"][-1], 200)
                fig.add_traces(
                    [
                        go.Scatter(
                            x=data_amp["delay"],
                            y=data_amp["prob"],
                            mode="markers",
                            name=f"{amp} ({s})",
                        ),
                        go.Scatter(
                            x=plot_t,
                            y=ramsey_func(plot_t, *fit.fitting_params[target][amp][s]),
                            mode="lines",
                            name=f"{np.round(amp, 3)} ({s}) fit",
                        ),
                    ]
                )
            fig.update_layout(
                xaxis_title="Wait time [ns]",
                yaxis_title="Fitting parameter value",
                showlegend=True,
            )
            figures.append(fig)

    return figures, fitting_report


def _probability_heatmap(records, waits, amplitudes, **kwargs) -> go.Heatmap:
    """Excited-state probability over (wait, amplitude), robust to row order."""
    z = records["prob"].reshape((len(amplitudes), len(waits)))
    return go.Heatmap(
        x=amplitudes,
        y=waits,
        z=z.T,
        colorscale="Viridis",
        zmin=0,
        zmax=1,
        colorbar={"title": {"text": "Excited State Probability", "side": "right"}},
        **kwargs,
    )


def _plot_zz(
    fit: CouplerZZResults, target: QubitPairId, amp: np.ndarray
) -> tuple[go.Figure, str]:
    """Plot ZZ values versus coupler pulse amplitude for a qubit pair."""
    zz_fig = go.Figure(
        go.Scatter(
            x=amp,
            y=np.array(fit.zz[target]) * const.milli,
            mode="lines+markers",
            name="ZZ",
        )
    )
    zz_fig.add_hline(y=0, line={"color": "gray", "dash": "dot"})

    zz_fig.update_layout(
        xaxis_title="Coupler pulse amplitude [a.u.]",
        yaxis_title="ZZ coupling [kHz]",
        showlegend=True,
    )
    return zz_fig, ""


coupler_zz = Protocol(_acquisition, _fit, _plot, two_qubit_gates=True)
"""CouplerZZ Protocol object."""
