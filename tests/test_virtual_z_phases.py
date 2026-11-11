from inspect import Parameter, signature

import numpy as np
import pytest
from qibo import gates
from qibolab import Pulse, VirtualZ

from qibocal.protocols.two_qubit_interaction.utils import order_pair, sinusoid
from qibocal.protocols.two_qubit_interaction.virtual_z_phases import (
    VirtualZPhasesData,
    VirtualZPhasesParameters,
    _acquisition,
    _fit,
    _plot,
    _update,
    create_sequence,
)


def test_create_sequence_keyword_phase(platform):
    assert (
        signature(create_sequence).parameters["vzphase"].kind == Parameter.KEYWORD_ONLY
    )
    pair = order_pair((0, 1), platform)
    _, flux_pulse, _ = create_sequence(platform, "I", *pair, pair, 16)
    sequence, flux, vz = create_sequence(
        platform, "I", *pair, pair, 16, 80, flux_pulse, vzphase=-0.7
    )

    assert flux.duration == 80
    assert vz.phase == pytest.approx(-0.7)
    fluxes = [
        pulse
        for pulse in sequence.channel(platform.qubits[pair[1]].flux)
        if isinstance(pulse, Pulse)
    ]
    assert fluxes == [flux]
    with pytest.raises(TypeError):
        create_sequence(platform, "I", *pair, pair, 16, 80, flux_pulse, -0.7)


def test_phase_correction_sign(platform):
    phase_error = 2.1
    thetas = np.linspace(0, 2 * np.pi, 50)
    pair = order_pair((0, 1), platform)
    probabilities = []
    for theta in thetas:
        sequence, _, _ = create_sequence(platform, "I", *pair, pair, 16, vzphase=-theta)
        pulses = [
            pulse
            for pulse in sequence.to_relative_phases().channel(
                platform.qubits[pair[0]].drive
            )
            if isinstance(pulse, Pulse)
        ]
        state = (
            gates.GPI2(0, pulses[1].relative_phase).matrix()
            @ gates.RZ(0, phase_error).matrix()
            @ gates.GPI2(0, pulses[0].relative_phase).matrix()
            @ np.array([1, 0])
        )
        probabilities.append(abs(state[1]) ** 2)

    np.testing.assert_allclose(
        probabilities, sinusoid(thetas, 0.5, 0.5, phase_error), atol=1e-14
    )
    correction = _fit(
        VirtualZPhasesData(
            data={
                (*pair, setup): np.stack([probabilities, np.zeros_like(thetas)])
                for setup in ("I", "X")
            },
            thetas=thetas.tolist(),
        )
    ).virtual_phase[tuple(pair)]
    assert correction == pytest.approx(phase_error)
    # A frame shift of +correction implements RZ(-correction).
    corrected = gates.RZ(0, -correction).matrix() @ gates.RZ(0, phase_error).matrix()
    np.testing.assert_allclose(corrected, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("theta_end", [0.4, 6.2])
def test_acquisition_phase_range(platform, mocker, theta_end):
    params = VirtualZPhasesParameters.load(
        {"theta_start": 0.2, "theta_end": theta_end, "theta_step": 0.2}
    )
    assert "sweep" not in signature(VirtualZPhasesParameters).parameters
    thetas = np.arange(params.theta_start, params.theta_end, params.theta_step)

    def execute(sequences, sweepers, **kwargs):
        assert len(sequences) == 4
        assert len(sweepers) == len(sweepers[0]) == 1
        np.testing.assert_allclose(sweepers[0][0].values, -thetas)
        results = {}
        vzs = []
        for sequence in sequences:
            vz = next(pulse for _, pulse in sequence if isinstance(pulse, VirtualZ))
            assert vz.phase == 0
            vzs.append(vz)
            phases = sweepers[0][0].values
            probabilities = np.asarray(0.5 + 0.4 * np.cos(-phases + 2))
            for qubit in platform.qubits.values():
                readout = list(sequence.channel(qubit.acquisition))[-1]
                results[readout.id] = probabilities
        assert sweepers[0][0].pulses == vzs
        return results

    mocker.patch.object(platform, "execute", side_effect=execute)
    data = _acquisition(params, platform, [(0, 1)])

    np.testing.assert_allclose(data.thetas, thetas)
    assert "gate_repetition" not in data.params
    assert len(data.data) == 4
    for probabilities in data.data.values():
        assert probabilities.shape == (2, len(thetas))
        np.testing.assert_allclose(probabilities[0], 0.5 + 0.4 * np.cos(thetas + 2))
        np.testing.assert_allclose(probabilities[1], probabilities[0])


def test_fit_plot_update_single_cz(platform):
    thetas = np.linspace(0, 2 * np.pi, 50)
    phases = {(0, 1): 2.1, (1, 0): 2.7}
    data = VirtualZPhasesData(
        data={
            (target, control, setup): np.stack(
                [
                    sinusoid(thetas, 0.4, 0.5, phase + shift),
                    np.full_like(thetas, leakage),
                ]
            )
            for (target, control), phase in phases.items()
            for setup, shift, leakage in [("I", 0, 0.1), ("X", -np.pi, 0.3)]
        },
        thetas=thetas.tolist(),
    )
    fit = _fit(data)

    assert "gate_repetition" not in fit.params
    for pair, phase in phases.items():
        assert fit.virtual_phase[pair] == pytest.approx(phase)
        assert fit.angle[pair] == pytest.approx(np.pi)
        assert fit.leakage[pair] == pytest.approx(0.1)
    figures, _ = _plot(data, fit, (0, 1))
    for figure, phase in zip(figures, phases.values()):
        traces = [trace for trace in figure.data if trace.name == "Fit"]
        assert len(traces) == 2
        for trace, shift in zip(traces, [0, -np.pi]):
            np.testing.assert_allclose(
                trace.y, sinusoid(np.asarray(trace.x), 0.4, 0.5, phase + shift)
            )

    _update(fit, platform, (1, 0))
    sequence = platform.natives.two_qubit[0, 1].CZ
    for pair, phase in phases.items():
        vz = list(sequence.channel(platform.qubits[pair[0]].drive))[-1]
        assert isinstance(vz, VirtualZ)
        assert vz.phase == pytest.approx(phase)


def _drive_rotation(theta, phi):
    """Rotation of angle ``theta`` around the axis ``cos(phi) X + sin(phi) Y``."""
    return np.array(
        [
            [np.cos(theta / 2), -1j * np.exp(-1j * phi) * np.sin(theta / 2)],
            [-1j * np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )


def _noisy_iswap(phase_low, phase_high, phase_11):
    """iSWAP with single-qubit phases acquired after the exchange and a residual
    conditional phase on |11>, in the (low, high) qubit basis."""
    gate = np.zeros((4, 4), dtype=complex)
    gate[0, 0] = 1
    gate[2, 1] = 1j * np.exp(1j * phase_low)  # |01> -> |10>: low qubit excited
    gate[1, 2] = 1j * np.exp(1j * phase_high)  # |10> -> |01>: high qubit excited
    gate[3, 3] = np.exp(1j * phase_11)
    return gate


def _simulate(platform, sequence, pair, gate, target):
    """Excited state probability of ``target`` for ``sequence`` with ``gate`` as
    the two-qubit interaction."""
    sequence = sequence.align_to_delays()
    flux_start = 0.0
    for pulse in sequence.channel(platform.qubits[pair[1]].flux):
        if isinstance(pulse, Pulse):
            break
        flux_start += pulse.duration
    before, after = np.eye(4, dtype=complex), np.eye(4, dtype=complex)
    for i, qubit in enumerate(pair):
        rx_amplitude = platform.natives.single_qubit[qubit].RX()[0][1].amplitude
        start = 0.0
        for pulse in sequence.to_relative_phases().channel(
            platform.qubits[qubit].drive
        ):
            if isinstance(pulse, Pulse):
                rotation = _drive_rotation(
                    np.pi * pulse.amplitude / rx_amplitude, pulse.relative_phase
                )
                ops = [np.eye(2), np.eye(2)]
                ops[i] = rotation
                op = np.kron(*ops)
                if start < flux_start:
                    before = op @ before
                else:
                    after = op @ after
            start += pulse.duration
    state = after @ gate @ before @ np.array([1, 0, 0, 0])
    probabilities = np.abs(state.reshape(2, 2)) ** 2
    return probabilities.sum(axis=1 - pair.index(target))[1]


def test_iswap_sequence(platform):
    pair = order_pair((0, 1), platform)
    low, high = (platform.natives.single_qubit[q] for q in pair)
    sequence, flux, _ = create_sequence(platform, "X", *pair, pair, 16, native="iSWAP")
    assert flux == next(
        iter(
            platform.natives.two_qubit[pair].iSWAP.channel(
                platform.qubits[pair[1]].flux
            )
        )
    )
    low_drive = [
        p
        for p in sequence.channel(platform.qubits[pair[0]].drive)
        if isinstance(p, Pulse)
    ]
    high_drive = [
        p
        for p in sequence.channel(platform.qubits[pair[1]].drive)
        if isinstance(p, Pulse)
    ]
    # X on the target before the gate, X90 after it
    assert [p.amplitude for p in low_drive] == pytest.approx(
        [low.RX()[0][1].amplitude, low.R(theta=np.pi / 2)[0][1].amplitude]
    )
    # X90 on the control before the gate, X after it for leakage
    assert [p.amplitude for p in high_drive] == pytest.approx(
        [high.R(theta=np.pi / 2)[0][1].amplitude, high.RX()[0][1].amplitude]
    )


@pytest.mark.parametrize("phases", [(0.0, 0.0, 0.0), (2.1, 0.7, 0.3), (-1.3, 4.0, 5.9)])
def test_iswap_phase_correction(platform, phases):
    phase_low, phase_high, phase_11 = phases
    gate = _noisy_iswap(phase_low, phase_high, phase_11)
    pair = order_pair((0, 1), platform)
    thetas = np.linspace(0, 2 * np.pi, 60)
    data = {}
    for target, control in [pair, pair[::-1]]:
        for setup in ("I", "X"):
            target_prob, control_prob = [], []
            for theta in thetas:
                sequence, _, _ = create_sequence(
                    platform,
                    setup,
                    target,
                    control,
                    pair,
                    16,
                    vzphase=-theta,
                    native="iSWAP",
                )
                target_prob.append(_simulate(platform, sequence, pair, gate, target))
                control_prob.append(_simulate(platform, sequence, pair, gate, control))
            data[target, control, setup] = np.stack([target_prob, control_prob])

    fit = _fit(VirtualZPhasesData(data=data, thetas=thetas.tolist(), native="iSWAP"))
    conditional = np.mod(phase_11 - phase_low - phase_high, 2 * np.pi)
    for qubit, phase in zip(pair, (phase_low, phase_high)):
        key = (qubit, pair[1 - pair.index(qubit)])
        assert np.exp(1j * fit.virtual_phase[key]) == pytest.approx(np.exp(1j * phase))
        assert np.exp(1j * fit.angle[key]) == pytest.approx(np.exp(1j * conditional))
        assert fit.leakage[key] == pytest.approx(0, abs=1e-10)

    # A frame shift of +phase implements RZ(-phase) after the gate.
    corrected = (
        np.kron(
            gates.RZ(0, -fit.virtual_phase[pair]).matrix(),
            gates.RZ(0, -fit.virtual_phase[pair[::-1]]).matrix(),
        )
        @ gate
    )
    expected = gates.iSWAP(0, 1).matrix() @ np.diag([1, 1, 1, np.exp(1j * conditional)])
    # equal up to a global phase
    global_phase = corrected[0, 0] / expected[0, 0]
    np.testing.assert_allclose(corrected, global_phase * expected, atol=1e-6)

    _update(fit, platform, pair)
    sequence = platform.natives.two_qubit[pair].iSWAP
    for qubit in pair:
        vz = list(sequence.channel(platform.qubits[qubit].drive))[-1]
        assert isinstance(vz, VirtualZ)
        key = (qubit, pair[1 - pair.index(qubit)])
        assert vz.phase == pytest.approx(fit.virtual_phase[key])
