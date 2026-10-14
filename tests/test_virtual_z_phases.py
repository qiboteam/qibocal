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


@pytest.mark.parametrize("sweep", [True, False])
def test_acquisition_phase_range(platform, mocker, sweep):
    params = VirtualZPhasesParameters.load(
        {"theta_start": 0.2, "theta_end": 6.2, "theta_step": 0.2, "sweep": sweep}
    )
    thetas = np.arange(params.theta_start, params.theta_end, params.theta_step)

    def execute(sequences, sweepers, **kwargs):
        assert len(sequences) == 4 * (1 if sweep else len(thetas))
        if sweep:
            np.testing.assert_allclose(sweepers[0][0].values, -thetas)
        else:
            assert sweepers == []
        results = {}
        for sequence in sequences:
            vz = next(pulse for _, pulse in sequence if isinstance(pulse, VirtualZ))
            phases = sweepers[0][0].values if sweep else vz.phase
            probabilities = np.asarray(0.5 + 0.4 * np.cos(-phases + 2))
            for qubit in platform.qubits.values():
                readout = list(sequence.channel(qubit.acquisition))[-1]
                results[readout.id] = probabilities
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
