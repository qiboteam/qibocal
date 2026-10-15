from copy import deepcopy

import numpy as np
import pytest
from click.testing import CliRunner
from qibolab import AcquisitionType, AveragingMode, Delay, Parameter, Qubit

from qibocal import Executor, protocols, update
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.task import Action, Completed, Task
from qibocal.calibration.calibration import QubitCalibration, TwoQubitCalibration
from qibocal.calibration.platform import CalibrationPlatform
from qibocal.cli._base import command
from qibocal.protocols.couplers.coupler_ramsey import (
    CouplerRamseyData,
    CouplerRamseyParameters,
    CouplerRamseyResults,
    CouplerRamseyType,
    _acquisition,
    _fit,
    _plot,
    _update,
    coupler_ramsey,
)

PAIR = (0, 1)
COUPLER = 1
CHANNEL = "coupler_01/flux"
PARAMETERS = {
    "delay_range": [16, 64, 16],
    "amplitude_coupler_range": [-0.125, 0.25, 0.125],
    "nshots": 100,
    "relaxation_time": 1000,
}


@pytest.fixture
def coupler_platform(platform):
    platform.calibration.two_qubits[PAIR] = TwoQubitCalibration(coupler=COUPLER)
    platform.calibration.couplers[COUPLER] = QubitCalibration()
    update.flux_offset(0.25, platform, COUPLER)
    return platform


def synthetic_data():
    data = CouplerRamseyData()
    data.register_qubit(
        CouplerRamseyType,
        PAIR,
        {
            "tau": np.tile([16, 32, 48], 3),
            "amplitude": np.repeat([-0.1, 0.0, 0.1], 3),
            "prob": [0.9, 0.1, 0.9, 0.8, 0.3, 0.8, 0.95, 0.95, 0.95],
            "error": np.full(9, 0.01),
            "prob_control": np.full(9, 0.98),
            "error_control": np.full(9, 0.01),
        },
    )
    return data


def test_protocol_metadata():
    """Keep protocol registration and serialized data/result types consistent."""
    assert protocols.PROTOCOLS["coupler_ramsey"] is coupler_ramsey
    assert coupler_ramsey.data_type is CouplerRamseyData
    assert coupler_ramsey.parameters_type is CouplerRamseyParameters
    assert coupler_ramsey.results_type is CouplerRamseyResults
    assert coupler_ramsey.two_qubit_gates


def pulse_times(sequence, channel):
    time = 0
    pulses = []
    for pulse in sequence.channel(channel):
        pulses.append((time, pulse))
        time += pulse.duration
    return pulses


@pytest.mark.parametrize("control_duration", [20, 100])
@pytest.mark.parametrize("bias", [-0.25, 0.0, 0.25])
def test_acquisition_timing_and_grid(
    coupler_platform, monkeypatch, control_duration, bias, tmp_path
):
    """Preserve timing and save the signed DC bias exactly once end to end."""
    platform = coupler_platform
    update.flux_offset(bias, platform, COUPLER)
    control_drive = platform.qubits[1].drive
    platform.update({"native_gates.single_qubit.1.RX.0.1.duration": control_duration})
    captured = {}
    probabilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.8, 0.8, 0.8]])

    def execute(*, sequences, sweepers, **options):
        captured.update(sequences=sequences, sweepers=sweepers, options=options)
        sequence = sequences[0]
        return {
            list(sequence.channel(platform.qubits[0].acquisition))[
                -1
            ].id: probabilities,
            list(sequence.channel(platform.qubits[1].acquisition))[-1].id: 1
            - probabilities,
        }

    monkeypatch.setattr(platform, "execute", execute)
    data = _acquisition(CouplerRamseyParameters.load(PARAMETERS), platform, [PAIR])
    sequence = captured["sequences"][0]
    amplitude_sweep = captured["sweepers"][0][0]
    duration_sweep = captured["sweepers"][1][0]
    assert amplitude_sweep.parameter is Parameter.amplitude
    assert duration_sweep.parameter is Parameter.duration
    options = captured["options"]
    assert options["acquisition_type"] is AcquisitionType.DISCRIMINATION
    assert options["averaging_mode"] is AveragingMode.CYCLIC
    assert options["nshots"] == PARAMETERS["nshots"]
    assert options["relaxation_time"] == PARAMETERS["relaxation_time"]
    assert not options.get("updates")

    for wait in duration_sweep.values:
        swept = deepcopy(sequence.align_to_delays())
        swept_ids = {pulse.id for pulse in duration_sweep.pulses}
        for _, pulse in swept:
            if pulse.id in swept_ids:
                object.__setattr__(pulse, "duration", wait)
        compiled = swept.align_to_delays()
        drive = platform.qubits[0].drive
        target_pulses = [
            (start, pulse)
            for start, pulse in pulse_times(compiled, drive)
            if not isinstance(pulse, Delay)
        ]
        first_start, first = target_pulses[0]
        second_start, second = target_pulses[1]
        control_pulses = [
            (start, pulse)
            for start, pulse in pulse_times(compiled, control_drive)
            if not isinstance(pulse, Delay)
        ]
        assert first_start == control_pulses[0][0] == 0
        preparation_end = max(first.duration, control_duration)
        flux_start, flux = next(
            (start, pulse)
            for start, pulse in pulse_times(compiled, CHANNEL)
            if not isinstance(pulse, Delay)
        )
        assert flux_start == preparation_end
        assert flux.duration == wait
        assert second_start == preparation_end + wait
        for qubit in PAIR:
            acquisition = platform.qubits[qubit].acquisition
            readout_start = pulse_times(compiled, acquisition)[-1][0]
            assert readout_start == second_start + second.duration

    np.testing.assert_allclose(data[PAIR].tau, np.tile(duration_sweep.values, 3))
    np.testing.assert_allclose(
        data[PAIR].amplitude, np.repeat(amplitude_sweep.values + bias, 3)
    )
    np.testing.assert_allclose(data[PAIR].prob, probabilities.ravel())
    np.testing.assert_allclose(data[PAIR].prob_control, (1 - probabilities).ravel())
    np.testing.assert_allclose(
        data[PAIR].error,
        np.sqrt(probabilities.ravel() * (1 - probabilities.ravel()) / 100),
    )
    np.testing.assert_allclose(data[PAIR].error_control, data[PAIR].error)
    assert "flux_channels" not in data.params
    assert "flux_offsets" not in data.params
    assert platform.config(CHANNEL).offset == pytest.approx(bias)
    data.save(tmp_path)
    fitted = _fit(CouplerRamseyData.load(tmp_path))
    fitted.save(tmp_path)
    reloaded = CouplerRamseyResults.load(tmp_path)
    assert reloaded.offset_adjustment[PAIR] == pytest.approx(bias + 0.125)
    _update(reloaded, platform, PAIR)
    _update(reloaded, platform, PAIR)
    assert platform.config(CHANNEL).offset == pytest.approx(bias + 0.125)


@pytest.mark.parametrize("biases", [(0.25, 0.5), (-0.25, 0.125)])
def test_saved_amplitudes_include_each_coupler_bias(
    coupler_platform, monkeypatch, biases
):
    """Shared pulse sweeps must retain each pair's own DC bias and readouts."""
    platform = coupler_platform
    update.flux_offset(biases[0], platform, COUPLER)
    other_pair = (2, 3)
    other_channel = "other/flux"
    platform.couplers["other"] = Qubit(flux=other_channel)
    platform.parameters.configs[other_channel] = platform.config(CHANNEL).model_copy(
        update={"offset": biases[1]}
    )
    platform.calibration.two_qubits[other_pair] = TwoQubitCalibration(coupler="other")
    for source, target in zip(PAIR, other_pair):
        platform.qubits[target] = Qubit.default(target)
        natives = platform.natives.single_qubit[source]
        platform.natives.single_qubit[target] = natives.model_copy(
            update={
                name: type(native)(
                    [
                        (channel.replace(f"{source}/", f"{target}/"), pulse)
                        for channel, pulse in native
                    ]
                )
                for name in ("RX", "RX90", "MZ")
                if (native := getattr(natives, name)) is not None
            }
        )
    captured = {}

    def execute(*, sequences, sweepers, **options):
        captured.update(sweepers=sweepers, options=options)
        return {
            list(sequences[0].channel(platform.qubits[qubit].acquisition))[
                -1
            ].id: np.full((4, 3), 0.1 * (qubit + 1))
            for qubit in (*PAIR, *other_pair)
        }

    monkeypatch.setattr(platform, "execute", execute)
    params = CouplerRamseyParameters.load(
        PARAMETERS | {"amplitude_coupler_range": ["window", 0.0, 0.25, 0.0625]}
    )
    data = _acquisition(params, platform, [PAIR, other_pair])
    amplitude_sweepers, wait_sweepers = captured["sweepers"]
    assert len(amplitude_sweepers) == len(wait_sweepers) == 1
    amplitude_sweeper = amplitude_sweepers[0]
    wait_sweeper = wait_sweepers[0]
    assert len(amplitude_sweeper.pulses) == 2
    assert len(wait_sweeper.pulses) == 4
    assert amplitude_sweeper.pulses[0].id != amplitude_sweeper.pulses[1].id
    expected = np.arange(-0.125, 0.125, 0.0625)
    np.testing.assert_allclose(amplitude_sweeper.values, expected)
    for pair, bias in zip([PAIR, other_pair], biases):
        np.testing.assert_allclose(data[pair].amplitude, np.repeat(expected + bias, 3))
        np.testing.assert_allclose(data[pair].tau, np.tile(wait_sweeper.values, 4))
        np.testing.assert_allclose(data[pair].prob, 0.1 * (pair[0] + 1))
        np.testing.assert_allclose(data[pair].prob_control, 0.1 * (pair[1] + 1))
    assert not captured["options"].get("updates")
    assert platform.config(CHANNEL).offset == pytest.approx(biases[0])
    assert platform.config(other_channel).offset == pytest.approx(biases[1])


def test_missing_coupler(platform):
    """Reject acquisition when the target pair has no assigned coupler."""
    platform.calibration.two_qubits[PAIR] = TwoQubitCalibration()
    with pytest.raises(ValueError, match="Coupler ID"):
        _acquisition(CouplerRamseyParameters.load(PARAMETERS), platform, [PAIR])


def test_fit_plot_and_serialization(tmp_path):
    """Round-trip the fit and show its bias as a line without plot annotations."""
    data = synthetic_data()
    data.save(tmp_path)
    loaded = CouplerRamseyData.load(tmp_path)
    np.testing.assert_array_equal(loaded[PAIR], data[PAIR])
    assert "flux_channels" not in loaded.params
    assert "flux_offsets" not in loaded.params
    results = _fit(loaded)
    assert results.offset_adjustment[PAIR] == pytest.approx(0.1)
    assert "flux_offset" not in results.params
    results.save(tmp_path)
    results = CouplerRamseyResults.load(tmp_path)
    assert "flux_channels" not in results.params
    figures, report = _plot(loaded, PAIR, results)
    assert len(figures) == 2
    np.testing.assert_allclose(figures[0].data[0].z, data[PAIR].prob.reshape(3, 3).T)
    np.testing.assert_allclose(
        figures[1].data[0].z, data[PAIR].prob_control.reshape(3, 3).T
    )
    assert figures[0].layout.shapes[0].x0 == pytest.approx(0.1)
    assert figures[0].layout.shapes[0].x1 == pytest.approx(0.1)
    assert not figures[0].layout.annotations
    assert not figures[1].layout.shapes
    assert "Offset adjustment" in report
    assert "Errors" not in report
    for fit in (None, CouplerRamseyResults({})):
        unfitted_figures, unfitted_report = _plot(loaded, PAIR, fit)
        assert unfitted_report == ""
        assert not unfitted_figures[0].layout.shapes


@pytest.mark.parametrize(
    "invalid",
    [
        "nan",
        "amplitude_nan",
        "tau_nan",
        "probability",
        "single_delay",
        "single_amplitude_delay",
        "empty",
    ],
)
def test_invalid_fit_is_not_updated(invalid, caplog):
    """Invalid probabilities or insufficient delays must not produce updates."""
    data = synthetic_data()
    if invalid == "nan":
        data[PAIR].prob[:] = np.nan
    elif invalid == "amplitude_nan":
        data[PAIR].amplitude[0] = np.nan
    elif invalid == "tau_nan":
        data[PAIR].tau[0] = np.nan
    elif invalid == "probability":
        data[PAIR].prob[:] = 2
    elif invalid == "single_delay":
        data[PAIR].tau[:] = 16
    elif invalid == "single_amplitude_delay":
        data[PAIR].tau[data[PAIR].amplitude == 0] = 16
    else:
        data.data[PAIR] = data[PAIR][:0]
    results = _fit(data)
    assert PAIR not in results
    assert not results.offset_adjustment
    assert "failed" in caplog.text.lower()


def test_invalid_pair_does_not_discard_valid_fit(caplog):
    """A failed pair must not prevent another pair from being calibrated."""
    data = synthetic_data()
    invalid_pair = (2, 3)
    data.data[invalid_pair] = data[PAIR].copy()
    data[invalid_pair].prob[:] = np.nan
    results = _fit(data)
    assert PAIR in results
    assert invalid_pair not in results
    assert results.offset_adjustment == {PAIR: pytest.approx(0.1)}
    assert str(invalid_pair) in caplog.text


def test_update_sets_absolute_nulling_bias(coupler_platform, tmp_path):
    """Repeated updates must not add the DC bias to the fitted value again."""
    platform = coupler_platform
    platform.calibration.single_qubits[0].qubit.sweetspot = 0.42
    results = _fit(synthetic_data())
    _update(results, platform, PAIR)
    assert platform.config(CHANNEL).offset == pytest.approx(0.1)
    _update(results, platform, PAIR)
    assert platform.config(CHANNEL).offset == pytest.approx(0.1)
    assert platform.calibration.single_qubits[0].qubit.sweetspot == pytest.approx(0.42)
    assert platform.calibration.couplers[COUPLER].qubit.sweetspot == pytest.approx(0.1)
    platform.dump(tmp_path)
    offline = CalibrationPlatform.from_datafolder(tmp_path, platform.name)
    assert not offline.couplers
    offline.couplers = platform.couplers.copy()
    _update(results, offline, PAIR)
    assert offline.config(CHANNEL).offset == pytest.approx(0.1)


def test_update_uses_platform_coupler_mapping(coupler_platform):
    """Resolve the update channel from the platform, not acquisition metadata."""
    platform = coupler_platform
    remapped_channel = "remapped/flux"
    platform.parameters.configs[remapped_channel] = platform.config(
        CHANNEL
    ).model_copy()
    platform.couplers[COUPLER] = platform.couplers[COUPLER].model_copy(
        update={"flux": remapped_channel}
    )
    _update(_fit(synthetic_data()), platform, PAIR)
    assert platform.config(remapped_channel).offset == pytest.approx(0.1)
    assert platform.config(CHANNEL).offset == pytest.approx(0.25)


def test_update_requires_coupler_id(coupler_platform):
    """Reject updates without a coupler assignment for the target pair."""
    coupler_platform.calibration.two_qubits[PAIR].coupler = None
    with pytest.raises(ValueError, match="Coupler ID"):
        _update(_fit(synthetic_data()), coupler_platform, PAIR)


def test_saved_task_pipeline(coupler_platform, tmp_path):
    """Reload acquired data before fitting, updating, and generating figures."""
    task = Task(
        Action(
            id="coupler_ramsey",
            operation="coupler_ramsey",
            targets=[PAIR],
            parameters=PARAMETERS,
        ),
        coupler_ramsey,
    )
    platform = coupler_platform
    platform.connect()
    try:
        task.run(ExecutionMode.ACQUIRE, tmp_path, platform)
    finally:
        platform.disconnect()
    reloaded = Completed.load(tmp_path)
    assert isinstance(reloaded.data, CouplerRamseyData)
    completed = reloaded.task.run(ExecutionMode.FIT, tmp_path, platform)
    completed.update_platform(platform)
    assert isinstance(Completed.load(tmp_path).results, CouplerRamseyResults)
    figures, _ = coupler_ramsey.report(completed.data, PAIR, completed.results)
    assert len(figures) == 2


def test_offline_fit_and_report(coupler_platform, tmp_path):
    """Fit and report saved acquisitions without requiring coupler hardware."""
    output = tmp_path / "coupler"
    with Executor.open(output, targets=[PAIR], platform=coupler_platform) as executor:
        executor.coupler_ramsey(mode=ExecutionMode.ACQUIRE, **PARAMETERS)
    runner = CliRunner()
    result = runner.invoke(command, ["report", str(output)], catch_exceptions=False)
    assert result.exit_code == 0
    assert (output / "index.html").is_file()
    result = runner.invoke(
        command, ["fit", str(output), "--no-update"], catch_exceptions=False
    )
    assert result.exit_code == 0
    result = runner.invoke(command, ["report", str(output)], catch_exceptions=False)
    assert result.exit_code == 0
    assert (output / "index.html").is_file()
    completed = Completed.load(output / "data" / "coupler_ramsey-0")
    assert isinstance(completed.results, CouplerRamseyResults)


def test_flux_updates_keep_qubit_behavior(platform):
    """Updating a qubit must leave the coupler's flux configuration unchanged."""
    qubit = 0
    update.flux_offset(0.15, platform, qubit)
    update.sweetspot(0.15, platform, qubit)
    assert platform.config(platform.qubits[qubit].flux).offset == pytest.approx(0.15)
    assert platform.config(CHANNEL).offset == 0
    assert platform.calibration.single_qubits[qubit].qubit.sweetspot == pytest.approx(
        0.15
    )
