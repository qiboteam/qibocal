from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qibocal import Executor, Protocol
from qibocal.auto.execute import _resolve
from qibocal.auto.operation import Data
from qibocal.auto.task import Completed
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


@dataclass
class PlatformParameters:
    platform: str


def plain_acquisition(parameters: PlainParameters) -> int:
    return parameters.samples + parameters.offset


def test_bind_plain_parameters():
    protocol = Protocol(plain_acquisition)
    bound = protocol(samples=3)

    assert bound.protocol is protocol
    assert bound.parameters == PlainParameters(3)
    assert Executor(None, targets=[])(bound).data == 4


def test_resolve_protocol():
    protocol = Protocol(plain_acquisition)
    kwargs = {"samples": 3, "fit": {"method": "new"}, "report": {"html": True}}
    bound = protocol(**kwargs)

    assert bound.protocol is protocol
    assert bound.parameters == PlainParameters(3)
    assert bound.fitpars == {"method": "new"}
    assert bound.reportpars == {"html": True}
    assert _resolve(bound, {}) is bound
    assert kwargs == {"samples": 3, "fit": {"method": "new"}, "report": {"html": True}}


@pytest.mark.parametrize("phase", ["__call__", "acquire"])
@pytest.mark.parametrize("explicit_parameters", [False, True])
def test_unbound_protocol_phases(phase, explicit_parameters):
    protocol = Protocol(plain_acquisition)
    executor = Executor(None, targets=[])
    kwargs = {"pars": PlainParameters(3)} if explicit_parameters else {"samples": 3}
    with pytest.raises(TypeError, match="requires a BoundProtocol"):
        getattr(executor, phase)(protocol, **kwargs)


@pytest.mark.parametrize("phase", ["__call__", "acquire"])
def test_bound_protocol_rejects_binding_arguments(phase):
    bound = Protocol(plain_acquisition)(samples=3)
    args = {
        "__call__": (bound,),
        "acquire": (bound,),
    }

    with pytest.raises(TypeError, match="binding arguments with a BoundProtocol"):
        getattr(Executor(None, targets=[]), phase)(*args[phase], samples=5)


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
    kwargs = {
        "min_amp": 0,
        "max_amp": 1,
        "step_amp": 0.05,
        "nshots": 4096,
        "relaxation_time": 0,
    }
    protocol = rabi_amplitude(**kwargs)
    execute = mocker.spy(platform, "execute")
    connect = mocker.spy(platform, "connect")
    disconnect = mocker.spy(platform, "disconnect")
    executor = Executor(platform, targets=[0, 1], update=False)

    completed = executor.acquire(protocol, targets=[1])
    data = completed.data

    assert data.qubits == [1]
    assert len(data[1]) == 20
    assert completed.bound is protocol
    assert completed.targets == [1]
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
    acquired = Completed(bound=bound, _data=rabi_data)
    fitted = executor.fit(acquired)
    results = fitted.results

    assert isinstance(results, RabiAmplitudeResults)
    assert 0 in results and 1 in results
    reported = executor.report(fitted, targets=[1])
    reports = reported.reports
    assert list(reports) == [1]
    figures, table = reports[1]
    assert figures
    assert isinstance(table, str) and table
    assert executor.report(acquired, targets=[0]).reports[0][0]
    assert acquired.results is None
    assert fitted is not acquired
    assert reported is not fitted
    assert fitted.reports is None
    assert reported.bound is bound
    assert reported.data is rabi_data

    amplitude_before = platform.natives.single_qubit[0].RX[0][1].amplitude
    updated = executor.update(reported, targets=[1])
    assert updated is not reported
    assert updated.results is results
    assert updated.reports is reports
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


@pytest.mark.parametrize("default_targets", [None, [], [0]])
def test_all_phase_targets_overrides(platform, mocker, default_targets):
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
    protocol = Protocol(*callbacks)
    kwargs = {"pars": 3, "fit": {"method": "new"}, "report": {"html": True}}
    bound = protocol(**kwargs)
    executor = Executor(platform, targets=default_targets)
    completed = executor(bound, targets=[1])

    assert completed.reports == (3, 4)
    for callback in callbacks:
        assert callback.call_args.kwargs["platform"] is platform
        assert callback.call_args.kwargs["targets"] == [1]
    assert callbacks[1].call_args.kwargs["fitpars"] == {"method": "new"}
    assert callbacks[2].call_args.kwargs["fit"] == 4
    assert callbacks[2].call_args.kwargs["reportpars"] == {"html": True}
    assert executor.platform is platform
    assert executor.targets == default_targets

    executor.acquire(bound, targets=[1])
    executor.fit(completed, targets=[1])
    executor.report(completed, targets=[1])
    executor.update(completed, targets=[1])
    for callback in callbacks:
        assert callback.call_count == 2
        assert callback.call_args.kwargs["platform"] is platform
        assert callback.call_args.kwargs["targets"] == [1]


@pytest.mark.parametrize("phase", ["__call__", "acquire"])
@pytest.mark.parametrize("factory", ["constructor", "create"])
def test_invocation_requires_targets(platform, tmp_path, phase, factory):
    executor = (
        Executor(platform)
        if factory == "constructor"
        else Executor.create(tmp_path, platform=platform)
    )
    protocol = Protocol(plain_acquisition)
    bound = protocol(samples=3)
    args = {
        "__call__": (bound,),
        "acquire": (bound,),
    }

    assert executor.targets is None
    with pytest.raises(ValueError, match="Targets must be supplied"):
        getattr(executor, phase)(*args[phase])
    with pytest.raises(ValueError, match="Targets must be supplied"):
        getattr(executor, phase)(*args[phase], targets=None)
    assert executor.targets is None


@pytest.mark.parametrize("factory", ["constructor", "create"])
@pytest.mark.parametrize("targets", [[], [0]])
def test_invocation_targets_without_defaults(platform, tmp_path, factory, targets):
    executor = (
        Executor(platform)
        if factory == "constructor"
        else Executor.create(tmp_path, platform=platform)
    )
    protocol = Protocol(plain_acquisition)

    assert executor(protocol(samples=3), targets=targets).data == 4
    assert executor.targets is None
    with pytest.raises(ValueError, match="Targets must be supplied"):
        executor(protocol(samples=3))


def test_optional_parameter_aliases():
    protocol = Protocol(
        acquisition=lambda pars: pars,
        fit=lambda data, fit_params: data + fit_params,
        report=lambda data, results, report_params: (results, report_params),
    )
    completed = Executor(None, targets=[])(protocol(pars=3, fit=2, report="html"))
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
    acquired = Executor(None, targets=[target]).acquire(bound)
    acquired.results = 4
    Executor(platform).update(acquired)
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
        "fit": (Completed(bound=bound, _data=3, _results=3, _targets=[]),),
        "report": (Completed(bound=bound, _data=3, _results=3, _targets=[]),),
        "update": (Completed(bound=bound, _data=3, _results=3, _targets=[]),),
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
        Executor(None, targets=[])(bound)
    assert calls == [3]


@pytest.mark.parametrize("phase", ["__call__", "acquire", "fit", "report", "update"])
def test_platform_keyword_is_binding_argument(platform, phase):
    protocol = Protocol(plain_acquisition)
    bound = protocol(samples=3)
    completed = Completed(bound=bound, _data=4, _results=5, _targets=[])
    args = {
        "__call__": (bound,),
        "acquire": (bound,),
        "fit": (completed,),
        "report": (completed,),
        "update": (completed,),
    }

    message = (
        "binding arguments with a BoundProtocol"
        if phase in ("__call__", "acquire")
        else "unexpected keyword argument 'platform'"
    )
    with pytest.raises(TypeError, match=message):
        getattr(Executor(platform, targets=[]), phase)(*args[phase], platform=platform)


@pytest.mark.parametrize("phase", ["__call__", "acquire", "fit", "report", "update"])
def test_platform_keyword_can_bind_parameters(platform, phase, mocker):
    acquisitions = []

    def acquire(parameters: PlatformParameters, platform):
        acquisitions.append((parameters, platform))
        return parameters.platform

    callbacks = [acquire] + [
        mocker.create_autospec(callback)
        for callback in (
            lambda data, platform: data,
            lambda data, results, platform: results,
            lambda results, platform: None,
        )
    ]
    callbacks[1].return_value = "parameter"
    protocol = Protocol(*callbacks)
    bound = protocol(platform="parameter")
    completed = Completed(bound=bound, _data="data", _results="results", _targets=[])
    args = {
        "__call__": (bound,),
        "acquire": (bound,),
        "fit": (completed,),
        "report": (completed,),
        "update": (completed,),
    }
    executor = Executor(platform, targets=[])
    getattr(executor, phase)(*args[phase])

    for name, callback in zip(["fit", "report", "update"], callbacks[1:]):
        if phase in ("__call__", name):
            assert callback.call_args.kwargs["platform"] is platform
    if phase in ("__call__", "acquire"):
        assert acquisitions == [(PlatformParameters("parameter"), platform)]
    assert executor.platform is platform


@pytest.mark.parametrize("phase", ["__call__", "acquire"])
def test_execution_settings_precedence(platform, mocker, phase):
    parameters = RabiAmplitudeParameters.load(
        {
            "min_amp": 0,
            "max_amp": 1,
            "step_amp": 0.05,
            "nshots": 64,
            "relaxation_time": 100,
        }
    )
    bound = rabi_amplitude(pars=parameters)
    kwargs = {
        "nshots": 128,
        "relaxation_time": 20,
        "targets": [1],
    }
    if phase == "__call__":
        kwargs["skip_fit"] = True
    execute = mocker.spy(platform, "execute")
    executor = Executor(platform, targets=[0])

    output = getattr(executor, phase)(bound, **kwargs)

    assert output.data.qubits == [1]
    assert output.bound.parameters.nshots == 128
    assert output.bound.parameters.relaxation_time == 20
    assert execute.call_args.kwargs["nshots"] == 128
    assert execute.call_args.kwargs["relaxation_time"] == 20
    assert parameters.nshots == 64
    assert parameters.relaxation_time == 100
    assert bound.parameters is parameters
    assert executor.targets == [0]


def test_resolve_execution_settings_are_extensible(monkeypatch):
    from qibocal.auto.operation import DEFAULT_PARENT_PARAMETERS, DummyPars

    monkeypatch.setitem(DEFAULT_PARENT_PARAMETERS, "future_setting", None)
    protocol = Protocol(lambda pars: pars)
    parameters = DummyPars.load({"future_setting": 1})
    bound = protocol(pars=parameters)

    overridden = _resolve(bound, {"future_setting": 2})
    rebound = _resolve(protocol(pars=parameters), {"future_setting": 3})

    assert overridden.parameters.future_setting == 2
    assert rebound.parameters.future_setting == 3
    assert parameters.future_setting == 1
    assert overridden.protocol is protocol


def test_execution_settings_plain_parameters():
    @dataclass
    class ExecutionParameters:
        nshots: int

    protocol = Protocol(lambda pars: pars)
    parameters = ExecutionParameters(64)
    bound = protocol(pars=parameters)

    acquired = Executor(None, targets=[]).acquire(bound, nshots=128)

    assert acquired.data.nshots == 128
    assert parameters.nshots == 64


@pytest.mark.parametrize(
    "keys, two_qubit, expected",
    [
        ([1], False, [1]),
        ([(1, 0), (1, 1)], False, [1]),
        ([(0, 1), (0, 1, "extra")], True, [(0, 1)]),
        ([], False, []),
        ([], True, []),
    ],
)
def test_downstream_targets_from_data(platform, mocker, keys, two_qubit, expected):
    data = Data({key: np.array([1]) for key in keys})
    fit = mocker.create_autospec(lambda data, targets: None, return_value=5)
    report = mocker.create_autospec(
        lambda data, targets, results: None, return_value="html"
    )
    update = mocker.create_autospec(lambda results, platform, targets: None)
    bound = Protocol(lambda pars: data, fit, report, update, two_qubit_gates=two_qubit)(
        pars=3
    )
    acquired = Executor(platform, targets=[2]).acquire(bound)
    executor = Executor(platform, targets=[3])

    fitted = executor.fit(acquired)
    reported = executor.report(fitted)
    updated = executor.update(reported)

    fit.assert_called_once_with(data, targets=expected)
    report.assert_called_once_with(data, targets=expected, results=5)
    update.assert_called_once_with(5, platform=platform, targets=expected)
    assert acquired.targets == expected
    assert acquired.results is None
    assert fitted.reports is None
    assert updated.reports == "html"
    assert len({id(node) for node in [acquired, fitted, reported, updated]}) == 4
    assert all(
        node.bound is bound and node.data is data
        for node in [acquired, fitted, reported, updated]
    )


def test_refit_preserves_input_and_clears_reports():
    bound = Protocol(
        lambda pars: pars,
        lambda data, fitpars: data + fitpars,
        lambda data, results: str(results),
    )(pars=3, fit=2)
    executor = Executor(None, targets=[])
    original = executor(bound)
    refitted = Executor(None).fit(original)

    assert refitted is not original
    assert refitted.bound is bound
    assert refitted.results == original.results == 5
    assert original.reports == "5"
    assert refitted.reports is None
    assert refitted.data_time == original.data_time


@pytest.mark.parametrize("phase", ["fit", "report", "update"])
def test_downstream_requires_completed(phase):
    executor = Executor(None)
    with pytest.raises(TypeError, match="requires a Completed instance"):
        getattr(executor, phase)(3)
    with pytest.raises(ValueError, match="has no bound protocol"):
        getattr(executor, phase)(Completed(_data=Data()))


def test_completed_is_exported_from_task():
    import qibocal
    from qibocal.auto import operation

    assert qibocal.Completed is Completed
    assert not hasattr(operation, "Completed")
