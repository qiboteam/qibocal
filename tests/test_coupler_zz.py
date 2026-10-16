from copy import deepcopy

import numpy as np
import pytest
from click.testing import CliRunner
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse

from qibocal import Executor, protocols, update
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.task import Action, Completed, Task
from qibocal.calibration.calibration import QubitCalibration, TwoQubitCalibration
from qibocal.cli._base import command
from qibocal.protocols.couplers import jazz, zz

PAIR = (0, 1)
COUPLER = 1
CHANNEL = "coupler_01/flux"
PARAMETERS = {
    "delay_range": ["linspace", 16, 6016, 81],
    "amplitude_range": ["linspace", -0.15, 0.15, 7],
    "detuning": 2e6,
    "nshots": 2048,
    "relaxation_time": 20000,
}


@pytest.fixture
def coupler_platform(platform):
    platform.calibration.two_qubits[PAIR] = TwoQubitCalibration(coupler=COUPLER)
    platform.calibration.couplers[COUPLER] = QubitCalibration()
    update.flux_offset(0.25, platform, COUPLER, coupler=True)
    return platform


def synthetic_data(module=zz, *, quadratic=False, detuning=2e6, noise=0.0):
    rng = np.random.default_rng(82)
    waits = np.linspace(16, 6016, 81)
    amplitudes = np.linspace(-0.15, 0.15, 7)
    strength = (
        8e6 * (amplitudes - 0.023) ** 2 + 4000
        if quadratic
        else 1.2e6 * (amplitudes - 0.023)
    )
    echoed = module is jazz
    data = (
        jazz.CouplerJAZZData(
            detuning=detuning,
            flux_offsets={PAIR: 0.25},
        )
        if echoed
        else zz.CouplerZZData(detuning=detuning)
    )
    for setup in zz.SETUPS:
        if echoed:
            frequency = detuning + (-strength if setup == "X" else strength)
        else:
            common_shift = 0.4e6 + 0.5e6 * amplitudes
            frequency = detuning - common_shift - (strength if setup == "X" else 0)
        angle = 2 * np.pi * np.abs(frequency[:, None]) * waits * 1e-9
        probs = (0.49 if setup == "I" else 0.52) + (
            0.4 if setup == "I" else 0.29
        ) * np.exp(-(waits - waits[0]) / 18000) * np.cos(
            angle + (0.2 if setup == "I" else -0.4)
        )
        probs += rng.normal(0, noise, probs.shape)
        data.register_qubit(
            zz.CouplerZZType,
            (PAIR, setup),
            {
                "wait": np.tile(waits, len(amplitudes)),
                "amplitude": np.repeat(amplitudes, len(waits)),
                "prob": probs.ravel(),
            },
        )
    return data, strength


@pytest.mark.parametrize("module", [zz, jazz])
@pytest.mark.parametrize("detuning", [-2e6, 2e6])
def test_signed_strength_and_subgrid_minimum(module, detuning):
    data, strength = synthetic_data(module, detuning=detuning, noise=0.002)
    results = module._fit(data)
    np.testing.assert_allclose(results.zz[PAIR], strength, atol=800)
    assert np.all(np.asarray(results.zz_error[PAIR]) > 0)
    assert results.zz_free_amplitude[PAIR] == pytest.approx(0.023, abs=0.001)
    assert abs(results.minimum_zz[PAIR]) < 1
    if module is jazz:
        assert results.nulling_point[PAIR] == pytest.approx(0.273, abs=0.001)


@pytest.mark.parametrize("module", [zz, jazz])
def test_minimum_without_zero_crossing(module):
    data, strength = synthetic_data(module, quadratic=True, noise=0.001)
    results = module._fit(data)
    np.testing.assert_allclose(results.zz[PAIR], strength, atol=500)
    assert results.zz_free_amplitude[PAIR] == pytest.approx(0.023, abs=0.001)
    assert results.minimum_zz[PAIR] == pytest.approx(4000, abs=400)


def test_joint_fit_handles_nonuniform_waits_and_independent_readout():
    rng = np.random.default_rng(2)
    waits = np.linspace(100, 6100, 101) + rng.uniform(-5, 5, 101)
    probs = np.array(
        [
            offset
            + contrast
            * np.exp(-(waits - waits[0]) / 12000)
            * np.cos(2 * np.pi * frequency * waits * 1e-9 + phase)
            for offset, contrast, frequency, phase in (
                (0.42, 0.3, 1.4e6, -2.1),
                (0.58, 0.22, 1.415e6, 0.7),
            )
        ]
    )
    difference, error, parameters = zz._conditional_fit(waits, probs, 2e6)
    assert difference == pytest.approx(15000, abs=1)
    assert error < 1
    assert parameters[0][0] == pytest.approx(0.42)
    assert parameters[1][0] == pytest.approx(0.58)


@pytest.mark.parametrize(
    "invalid", ["flat", "nan", "probability", "mismatch", "missing"]
)
def test_failed_traces_are_omitted(invalid, caplog):
    data, _ = synthetic_data()
    if invalid == "flat":
        data[PAIR, "I"].prob[:] = 0.5
    elif invalid == "nan":
        data[PAIR, "I"].prob[:] = np.nan
    elif invalid == "probability":
        data[PAIR, "I"].prob[:] = 1.5
    elif invalid == "mismatch":
        data[PAIR, "I"].wait[:] += 1
    else:
        del data.data[PAIR, "X"]
    results = zz._fit(data)
    assert PAIR not in results
    assert not results.zz_free_amplitude
    assert "failed" in caplog.text.lower()


def test_failed_amplitude_does_not_bridge_a_null(caplog):
    data, _ = synthetic_data(jazz)
    for setup in zz.SETUPS:
        data[PAIR, setup].prob[data[PAIR, setup].amplitude == 0] = np.nan
    result = jazz._fit(data)
    assert np.isnan(result.zz[PAIR][3])
    assert PAIR not in result.nulling_point
    assert "no update" in caplog.text


def test_minimum_does_not_extrapolate():
    amplitudes = np.linspace(-0.1, 0.1, 5)
    point, residual, _, _ = zz._minimum(
        amplitudes, 1e6 * (amplitudes - 0.2), np.ones(5)
    )
    assert point == pytest.approx(0.1)
    assert residual == pytest.approx(-1e5)
    assert zz._zero_crossing(np.arange(3), np.array([-1, np.nan, 1])) is None


def test_boundary_and_missing_bias_are_not_updated(caplog):
    data, _ = synthetic_data(jazz)
    data.flux_offsets.clear()
    results = jazz._fit(data)
    assert results.zz
    assert PAIR not in results
    assert "DC offset" in caplog.text
    for setup in zz.SETUPS:
        record = data[PAIR, setup]
        data.data[PAIR, setup] = record[record.amplitude < 0]
    data.flux_offsets[PAIR] = 0.25
    results = jazz._fit(data)
    assert PAIR not in results
    assert "not bracketed" in caplog.text


@pytest.mark.parametrize("module", [zz, jazz])
def test_serialization_plot_and_registry(module, tmp_path):
    name = "coupler_jazz" if module is jazz else "coupler_zz"
    protocol = protocols.PROTOCOLS[name]
    assert protocol.two_qubit_gates
    data, _ = synthetic_data(module)
    data.save(tmp_path)
    loaded = protocol.data_type.load(tmp_path)
    results = module._fit(loaded)
    results.save(tmp_path)
    fitted = protocol.results_type.load(tmp_path)
    assert fitted.zz_free_amplitude[PAIR] == pytest.approx(0.023, abs=1e-5)
    # Stored rows need not arrive in sweep order.
    for records in loaded.data.values():
        np.random.default_rng(2).shuffle(records)
    figures, report = module._plot(loaded, PAIR, fitted)
    assert len(figures) == 2
    assert len(figures[1].data) == 2
    assert "Fitted minimum ZZ" in report
    assert len(module._plot(loaded, PAIR, None)[0]) == 1


def pulse_times(sequence, channel):
    time = 0
    pulses = []
    for pulse in sequence.channel(channel):
        pulses.append((time, pulse))
        time += pulse.duration
    return pulses


@pytest.mark.parametrize("control_duration", [20, 100])
def test_jazz_sweep_timing_and_acquisition(
    coupler_platform, monkeypatch, control_duration
):
    platform = coupler_platform
    platform.update({"native_gates.single_qubit.1.RX.0.1.duration": control_duration})
    calls = []
    probs = np.arange(24).reshape(3, 8) / 24

    def execute(sequences, sweepers, **options):
        calls.append((sequences[0], sweepers, options))
        return {
            list(sequences[0].channel(platform.qubits[q].acquisition))[-1].id: probs
            for q in PAIR
        }

    monkeypatch.setattr(platform, "execute", execute)
    params = jazz.CouplerJAZZParameters.load(
        PARAMETERS
        | {
            "delay_range": ["linspace", 16, 128, 8],
            "amplitude_range": ["linspace", -0.1, 0.1, 3],
        }
    )
    data = jazz._acquisition(params, platform, [PAIR])
    assert len(calls) == 2
    for setup, (sequence, sweepers, options) in zip(zz.SETUPS, calls):
        (amplitude,) = sweepers[0]
        duration, phase = sweepers[1]
        assert amplitude.parameter is Parameter.amplitude
        assert duration.parameter is Parameter.duration
        assert phase.parameter is Parameter.relative_phase
        assert len(amplitude.pulses) == 2
        assert amplitude.pulses[0].id != amplitude.pulses[1].id
        np.testing.assert_allclose(
            phase.values, 2 * np.pi * params.detuning * duration.values * 1e-9
        )
        assert options["acquisition_type"] is AcquisitionType.DISCRIMINATION
        assert options["averaging_mode"] is AveragingMode.CYCLIC
        assert not options.get("updates")
        assert data.flux_offsets[PAIR] == pytest.approx(0.25)
        assert data.frequency_scale == -0.5
        np.testing.assert_allclose(data[PAIR, setup].prob, probs.ravel())
        for wait in duration.values:
            swept = deepcopy(sequence.align_to_delays())
            ids = {pulse.id for pulse in duration.pulses}
            for _, pulse in swept:
                if pulse.id in ids:
                    object.__setattr__(pulse, "duration", wait)
            compiled = swept.align_to_delays()
            target_pulses = [
                (start, pulse)
                for start, pulse in pulse_times(compiled, platform.qubits[0].drive)
                if isinstance(pulse, Pulse)
            ]
            first, middle, final = target_pulses
            flux_pulses = [
                (start, pulse)
                for start, pulse in pulse_times(compiled, CHANNEL)
                if isinstance(pulse, Pulse)
            ]
            control_pulses = [
                (start, pulse)
                for start, pulse in pulse_times(compiled, platform.qubits[1].drive)
                if isinstance(pulse, Pulse)
            ]
            echo = control_pulses[-1]
            assert first[0] == control_duration
            assert flux_pulses[0][0] == first[0] + first[1].duration
            assert middle[0] == echo[0] == flux_pulses[0][0] + wait
            echo_end = middle[0] + max(middle[1].duration, echo[1].duration)
            assert flux_pulses[1][0] == echo_end
            assert final[0] == echo_end + wait
            for qubit in PAIR:
                readout = pulse_times(compiled, platform.qubits[qubit].acquisition)[-1]
                assert readout[0] == final[0] + final[1].duration
    assert platform.config(CHANNEL).offset == pytest.approx(0.25)


def test_jazz_absolute_update_is_idempotent(coupler_platform):
    results = jazz._fit(synthetic_data(jazz)[0])
    for _ in range(2):
        jazz._update(results, coupler_platform, PAIR)
        assert coupler_platform.config(CHANNEL).offset == pytest.approx(0.273)
        assert coupler_platform.calibration.couplers[
            COUPLER
        ].qubit.sweetspot == pytest.approx(0.273)


@pytest.mark.parametrize("module", [zz, jazz])
def test_saved_task_pipeline(module, coupler_platform, tmp_path, monkeypatch):
    data, _ = synthetic_data(module)
    name = "coupler_jazz" if module is jazz else "coupler_zz"
    protocol = protocols.PROTOCOLS[name]
    setups = iter(zz.SETUPS)

    def execute(sequences, sweepers, **options):
        setup = next(setups)
        return {
            list(sequences[0].channel(coupler_platform.qubits[q].acquisition))[
                -1
            ].id: data[PAIR, setup].prob.reshape(7, 81)
            if q == 0
            else np.full((7, 81), 0.9)
            for q in PAIR
            if coupler_platform.qubits[q].acquisition in sequences[0].channels
        }

    monkeypatch.setattr(coupler_platform, "execute", execute)
    task = Task(
        Action(id=name, operation=name, targets=[PAIR], parameters=PARAMETERS),
        protocol,
    )
    task.run(ExecutionMode.ACQUIRE, tmp_path, coupler_platform)
    reloaded = Completed.load(tmp_path)
    completed = reloaded.task.run(ExecutionMode.FIT, tmp_path, coupler_platform)
    assert isinstance(completed.results, protocol.results_type)
    completed.update_platform(coupler_platform)
    figures, report = protocol.report(completed.data, PAIR, completed.results)
    assert len(figures) == 2
    assert report


@pytest.mark.parametrize("module", [zz, jazz])
def test_nonzero_carrier_required(module, coupler_platform):
    parameters = (
        jazz.CouplerJAZZParameters if module is jazz else zz.CouplerZZParameters
    )
    params = parameters.load(PARAMETERS | {"detuning": 0})
    with pytest.raises(ValueError, match="nonzero"):
        module._acquisition(params, coupler_platform, [PAIR])


@pytest.mark.parametrize("detuning", [-2e6, 2e6])
def test_jazz_hamiltonian_sign_echo_and_factor_two(
    coupler_platform, monkeypatch, detuning
):
    """Simulate generated pulses with H/h = f_t n_t + zeta n_t n_c."""
    platform = coupler_platform
    identity = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    native_amplitudes = {
        platform.qubits[q].drive: platform.natives.single_qubit[q].RX()[0][1].amplitude
        for q in PAIR
    }

    def execute(sequences, sweepers, **options):
        (amplitude_sweep,) = sweepers[0]
        duration_sweep, phase_sweep = sweepers[1]
        sequence = sequences[0].align_to_delays()
        probabilities = np.empty(
            (len(amplitude_sweep.values), len(duration_sweep.values))
        )
        for i, amplitude in enumerate(amplitude_sweep.values):
            for j, wait in enumerate(duration_sweep.values):
                swept = deepcopy(sequence)
                for _, pulse in swept:
                    for sweeper in (amplitude_sweep, duration_sweep, phase_sweep):
                        if pulse.id in {item.id for item in sweeper.pulses}:
                            value = (
                                amplitude
                                if sweeper is amplitude_sweep
                                else wait
                                if sweeper is duration_sweep
                                else phase_sweep.values[j]
                            )
                            object.__setattr__(pulse, sweeper.parameter.name, value)
                events = []
                for qubit in PAIR:
                    drive = platform.qubits[qubit].drive
                    for start, pulse in pulse_times(swept, drive):
                        if isinstance(pulse, Pulse):
                            events.append(
                                (start + pulse.duration, "rotation", (qubit, pulse))
                            )
                for start, pulse in pulse_times(swept, CHANNEL):
                    if isinstance(pulse, Pulse):
                        events.extend(
                            [
                                (start, "flux_on", None),
                                (start + pulse.duration, "flux_off", None),
                            ]
                        )
                state = np.array([1, 0, 0, 0], dtype=complex)
                time, biased = 0, False
                strength = 1.2e6 * (amplitude - 0.023)
                for end, kind, payload in sorted(events, key=lambda event: event[0]):
                    # An amplitude-dependent unconditional shift must echo out.
                    common = 0.9e6 + amplitude * 3e6
                    energies = np.array(
                        [0, 0, common, common + (strength if biased else 0)]
                    )
                    state *= np.exp(-2j * np.pi * energies * (end - time) * 1e-9)
                    time = end
                    if kind == "flux_on":
                        biased = True
                    elif kind == "flux_off":
                        biased = False
                    else:
                        qubit, pulse = payload
                        theta = (
                            np.pi
                            * pulse.amplitude
                            / native_amplitudes[platform.qubits[qubit].drive]
                        )
                        axis = (
                            np.cos(pulse.relative_phase) * sigma_x
                            + np.sin(pulse.relative_phase) * sigma_y
                        )
                        rotation = (
                            np.cos(theta / 2) * identity - 1j * np.sin(theta / 2) * axis
                        )
                        state = (
                            np.kron(rotation, identity)
                            if qubit == 0
                            else np.kron(identity, rotation)
                        ) @ state
                probabilities[i, j] = np.clip(np.sum(np.abs(state[2:]) ** 2), 0, 1)
        return {
            list(sequence.channel(platform.qubits[q].acquisition))[-1].id: probabilities
            for q in PAIR
        }

    monkeypatch.setattr(platform, "execute", execute)
    params = jazz.CouplerJAZZParameters.load(PARAMETERS | {"detuning": detuning})
    data = jazz._acquisition(params, platform, [PAIR])
    results = jazz._fit(data)
    amplitudes = np.array(results.amplitudes[PAIR])
    np.testing.assert_allclose(results.zz[PAIR], 1.2e6 * (amplitudes - 0.023), atol=2)
    assert results.nulling_point[PAIR] == pytest.approx(0.273, abs=1e-5)


@pytest.mark.parametrize("module", [zz, jazz])
def test_offline_mock_fit_and_html_report(module, coupler_platform, tmp_path):
    name = "coupler_jazz" if module is jazz else "coupler_zz"
    output = tmp_path / name
    with Executor.open(output, targets=[PAIR], platform=coupler_platform) as executor:
        getattr(executor, name)(
            mode=ExecutionMode.ACQUIRE,
            **(
                PARAMETERS
                | {
                    "delay_range": [16, 416, 40],
                    "amplitude_range": ["linspace", -0.1, 0.1, 3],
                }
            ),
        )
    runner = CliRunner()
    for args in (["report"], ["fit"], ["report"]):
        result = runner.invoke(
            command,
            [*args, str(output), *(["--no-update"] if args == ["fit"] else [])],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
    assert (output / "index.html").is_file()


def test_noise_only_is_not_a_coupling_measurement():
    rng = np.random.default_rng(0)
    waits = np.linspace(16, 6016, 81)
    with pytest.raises(ValueError, match="noise floor"):
        zz._conditional_fit(waits, 0.5 + rng.normal(0, 0.01, (2, 81)), 2e6)


@pytest.mark.parametrize("module", [zz, jazz])
@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"delay_range": [16, 112, 16]}, "eight"),
        ({"delay_range": [-16, 112, 16]}, "positive"),
        ({"delay_range": [16, 1616, 160], "detuning": 4e6}, "Nyquist"),
        ({"amplitude_range": [0, 0.1, 0]}, "positive step"),
    ],
)
def test_invalid_sweeps_fail_before_execution(
    module, overrides, match, coupler_platform, monkeypatch
):
    def execute(*args, **kwargs):
        pytest.fail("Invalid sweep must be rejected before acquisition.")

    monkeypatch.setattr(coupler_platform, "execute", execute)
    parameter_type = (
        jazz.CouplerJAZZParameters if module is jazz else zz.CouplerZZParameters
    )
    with pytest.raises(ValueError, match=match):
        module._acquisition(
            parameter_type.load(PARAMETERS | overrides), coupler_platform, [PAIR]
        )
