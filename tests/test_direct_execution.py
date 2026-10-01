from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qibocal import Executor, Protocol
from qibocal.protocols import rabi_amplitude
from qibocal.protocols.rabi.amplitude import (
    RabiAmplitudeData,
    RabiAmplitudeParameters,
    RabiAmplitudeResults,
    RabiAmpType,
)
from qibocal.protocols.rabi.utils import rabi_amplitude_function


@dataclass
class PlainParameters:
    samples: int
    offset: int = 1


def plain_acquisition(parameters: PlainParameters) -> int:
    return parameters.samples + parameters.offset


def test_bind_plain_parameters():
    protocol = Protocol(plain_acquisition)
    bound = protocol(samples=3)

    assert bound.protocol is protocol
    assert bound.parameters == PlainParameters(3)
    assert Executor(None)(bound).data == 4


@pytest.mark.parametrize("from_keywords", [True, False])
def test_bind_builtin_parameters(from_keywords):
    kwargs = {
        "min_amp": 0.0,
        "max_amp": 1.0,
        "step_amp": 0.05,
        "nshots": 4096,
        "relaxation_time": 0,
    }
    parameters = RabiAmplitudeParameters.load(kwargs)
    bound = (
        rabi_amplitude(**kwargs) if from_keywords else rabi_amplitude(pars=parameters)
    )

    assert bound.protocol is rabi_amplitude
    assert bound.parameters == parameters
    if not from_keywords:
        assert bound.parameters is parameters
    assert bound.parameters.nshots == 4096
    assert bound.parameters.relaxation_time == 0
    assert bound.parameters.rx90 is False
    assert bound.parameters.pulse_length is None
    assert bound.fitpars is None
    assert bound.reportpars is None


def test_builtin_acquisition(platform, mocker):
    bound = rabi_amplitude(
        min_amp=0, max_amp=1, step_amp=0.05, nshots=4096, relaxation_time=0
    )
    execute = mocker.spy(platform, "execute")
    connect = mocker.spy(platform, "connect")
    disconnect = mocker.spy(platform, "disconnect")
    executor = Executor(platform, targets=[0, 1], update=False)

    data = executor.acquire(bound, targets=[1])

    assert data.qubits == [1]
    assert len(data[1]) == 20
    assert execute.call_args.kwargs["nshots"] == 4096
    assert execute.call_args.kwargs["relaxation_time"] == 0
    assert executor.targets == [0, 1]
    connect.assert_not_called()
    disconnect.assert_not_called()
    assert executor.path is None
    assert list(executor.history) == []


@pytest.mark.parametrize("from_keywords", [True, False])
def test_builtin_acquisition_settings_defaults(platform, mocker, from_keywords):
    kwargs = {"min_amp": 0, "max_amp": 1, "step_amp": 0.05}
    bound = (
        rabi_amplitude(**kwargs)
        if from_keywords
        else rabi_amplitude(pars=RabiAmplitudeParameters(**kwargs))
    )
    execute = mocker.spy(platform, "execute")
    Executor(platform, targets=[0]).acquire(bound)

    assert execute.call_args.kwargs["nshots"] == platform.settings.nshots
    assert (
        execute.call_args.kwargs["relaxation_time"] == platform.settings.relaxation_time
    )
    assert getattr(bound.parameters, "nshots", None) is None
    assert getattr(bound.parameters, "relaxation_time", None) is None


@pytest.fixture
def rabi_data():
    data = RabiAmplitudeData(rx90=False, durations={0: 40, 1: 40})
    amplitudes = np.linspace(0, 1, 101)
    probabilities = rabi_amplitude_function(amplitudes, 0.5, 0.4, 0.8, 0.0)
    for target in [0, 1]:
        data.register_qubit(
            RabiAmpType,
            target,
            {"amp": amplitudes, "prob": probabilities, "error": np.full(101, 0.01)},
        )
    return data


def test_builtin_independent_phases(platform, rabi_data):
    bound = rabi_amplitude(min_amp=0, max_amp=1, step_amp=0.01, fit={}, report={})
    executor = Executor(platform, targets=[0, 1], update=False)
    results = executor.fit(rabi_data, bound)

    assert isinstance(results, RabiAmplitudeResults)
    assert 0 in results and 1 in results
    reports = executor.report(rabi_data, results, bound, targets=[1])
    assert list(reports) == [1]
    figures, table = reports[1]
    assert figures
    assert isinstance(table, str) and table
    assert executor.report(rabi_data, None, bound, targets=[0])[0][0]

    amplitude_before = platform.natives.single_qubit[0].RX[0][1].amplitude
    executor.update(results, bound, targets=[1])
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == amplitude_before
    assert platform.natives.single_qubit[1].RX[0][1].amplitude == pytest.approx(
        results.amplitude[1][0]
    )
    assert executor.targets == [0, 1]
    assert executor.path is None
    assert list(executor.history) == []


@pytest.mark.parametrize("update", [True, False])
@pytest.mark.parametrize("from_keywords", [True, False])
def test_builtin_complete_workflow(platform, rabi_data, mocker, update, from_keywords):
    acquisition = mocker.spy(rabi_amplitude, "acquisition")
    mocker.patch(
        "qibocal.protocols.rabi.amplitude.probability",
        return_value=rabi_data[1].prob,
    )
    parameters = {
        "min_amp": 0,
        "max_amp": 1.01,
        "step_amp": 0.01,
        "nshots": 4096,
    }
    bound = (
        rabi_amplitude(**parameters)
        if from_keywords
        else rabi_amplitude(pars=RabiAmplitudeParameters.load(parameters))
    )
    updater = mocker.spy(rabi_amplitude, "update")
    executor = Executor(platform, targets=[0, 1], update=update)
    completed = executor(bound, targets=[1])

    assert acquisition.call_args.kwargs["platform"] is platform
    assert acquisition.call_args.kwargs["targets"] == [1]
    assert completed.success
    assert completed.data.qubits == [1]
    assert 1 in completed.results
    assert list(completed.reports) == [1]
    if update:
        updater.assert_called_once_with(completed.results, platform=platform, target=1)
    else:
        updater.assert_not_called()


def test_all_phase_context_overrides(platform, mocker):
    other_platform = mocker.Mock()

    def acquire(pars, *, platform, targets):
        return pars

    def fit(data, *, platform, targets, fitpars):
        return data + 1

    def report(data, *, fit, platform, targets, reportpars):
        return (data, fit)

    def update(results, *, platform, targets):
        pass

    callbacks = [
        mocker.create_autospec(callback) for callback in [acquire, fit, report, update]
    ]
    callbacks[0].return_value = 3
    callbacks[1].return_value = 4
    callbacks[2].return_value = (3, 4)
    bound = Protocol(*callbacks)(pars=3, fit={"method": "new"}, report={"html": True})
    executor = Executor(platform, targets=[0])
    completed = executor(bound, platform=other_platform, targets=[1])

    assert completed.reports == (3, 4)
    for callback in callbacks:
        assert callback.call_args.kwargs["platform"] is other_platform
        assert callback.call_args.kwargs["targets"] == [1]
    assert callbacks[1].call_args.kwargs["fitpars"] == {"method": "new"}
    assert callbacks[2].call_args.kwargs["fit"] == 4
    assert callbacks[2].call_args.kwargs["reportpars"] == {"html": True}
    assert executor.platform is platform
    assert executor.targets == [0]

    executor.acquire(bound, platform=other_platform, targets=[1])
    executor.fit(3, bound, platform=other_platform, targets=[1])
    executor.report(3, 4, bound, platform=other_platform, targets=[1])
    executor.update(4, bound, platform=other_platform, targets=[1])
    for callback in callbacks:
        assert callback.call_count == 2
        assert callback.call_args.kwargs["platform"] is other_platform
        assert callback.call_args.kwargs["targets"] == [1]


def test_optional_parameter_aliases():
    protocol = Protocol(
        acquisition=lambda pars: pars,
        fit=lambda data, fit_params: data + fit_params,
        report=lambda data, results, report_params: (results, report_params),
    )
    completed = Executor(None)(protocol(pars=3, fit=2, report="html"))
    assert completed.results == 5
    assert completed.reports == (5, "html")


@pytest.mark.parametrize("target", [1, (0, 1), (0, 1, 2)])
@pytest.mark.parametrize("name", ["target", "qubit"])
def test_per_target_update(platform, target, name):
    calls = []

    def update_target(results, platform, target):
        calls.append((results, platform, target))

    def update_qubit(results, platform, qubit):
        calls.append((results, platform, qubit))

    bound = Protocol(
        lambda pars: pars, update=update_target if name == "target" else update_qubit
    )(pars=3)
    Executor(platform, targets=[target]).update(4, bound)
    assert calls == [(4, platform, target)]


def test_positional_only_context(platform):
    bound = Protocol(
        acquisition=lambda pars, platform, targets, /: (pars, platform, targets),
        fit=lambda data, fitpars, /: fitpars,
        report=lambda data, target, fit, /: fit,
        update=lambda results, platform, target, /: None,
    )(pars=3, fit=4)
    completed = Executor(platform, targets=[1])(bound)
    assert completed.data == (3, platform, [1])
    assert completed.results == 4
    assert completed.reports == {1: 4}


@pytest.mark.parametrize("phase", ["acquire", "fit", "report", "update"])
def test_override_targets_validation(platform, phase):
    bound = Protocol(
        lambda pars: pars,
        lambda data: data,
        lambda data, fit: fit,
        lambda results, platform: None,
    )(pars=3)
    args = {
        "acquire": (bound,),
        "fit": (3, bound),
        "report": (3, 3, bound),
        "update": (3, bound),
    }
    with pytest.raises(ValueError, match="target qubits were repeated"):
        getattr(Executor(platform), phase)(*args[phase], targets=[1, 1])


def test_callback_type_error_is_not_retried():
    calls = []

    def fit(data):
        calls.append(data)
        raise TypeError("inside fitting")

    bound = Protocol(lambda pars: pars, fit)(pars=3, fit={})
    with pytest.raises(TypeError, match="inside fitting"):
        Executor(None)(bound)
    assert calls == [3]


def test_update_platform_override_without_default(mocker):
    platform = mocker.Mock()
    update = mocker.create_autospec(lambda results, platform: None)
    bound = Protocol(lambda pars: pars, update=update)(pars=3)
    Executor(None).update(4, bound, platform=platform)
    update.assert_called_once_with(4, platform=platform)
