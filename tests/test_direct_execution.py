from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qibocal import Executor, Protocol
from qibocal.auto.execute import resolve
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


def test_resolve_protocol():
    protocol = Protocol(plain_acquisition)
    kwargs = {"samples": 3, "fit": {"method": "new"}, "report": {"html": True}}
    bound = resolve(protocol, kwargs)

    assert bound.protocol is protocol
    assert bound.parameters == PlainParameters(3)
    assert bound.fitpars == {"method": "new"}
    assert bound.reportpars == {"html": True}
    assert resolve(bound, {}) is bound
    assert kwargs == {"samples": 3, "fit": {"method": "new"}, "report": {"html": True}}


@pytest.mark.parametrize("phase", ["__call__", "acquire", "fit", "report", "update"])
@pytest.mark.parametrize("explicit_parameters", [False, True])
def test_unbound_protocol_phases(mocker, phase, explicit_parameters):
    fit = mocker.create_autospec(lambda data, fitpars: None, return_value=5)
    report = mocker.create_autospec(
        lambda data, results, reportpars: None, return_value="html"
    )
    update = mocker.create_autospec(lambda results, platform: None)
    protocol = Protocol(plain_acquisition, fit, report, update)
    platform = mocker.Mock()
    executor = Executor(platform)
    kwargs = {"pars": PlainParameters(3)} if explicit_parameters else {"samples": 3}
    kwargs |= {"fit": {"method": "new"}, "report": {"html": True}}
    binding = mocker.spy(Protocol, "__call__")
    args = {
        "__call__": (protocol,),
        "acquire": (protocol,),
        "fit": (4, protocol),
        "report": (4, 5, protocol),
        "update": (5, protocol),
    }

    output = getattr(executor, phase)(*args[phase], **kwargs)

    binding.assert_called_once_with(protocol, **kwargs)
    if phase in ("__call__", "acquire"):
        assert (output.data if phase == "__call__" else output) == 4
    if phase in ("__call__", "fit"):
        fit.assert_called_once_with(4, fitpars={"method": "new"})
        assert (output.results if phase == "__call__" else output) == 5
    if phase in ("__call__", "report"):
        report.assert_called_once_with(4, results=5, reportpars={"html": True})
        assert (output.reports if phase == "__call__" else output) == "html"
    if phase in ("__call__", "update"):
        update.assert_called_once_with(5, platform=platform)


@pytest.mark.parametrize("phase", ["__call__", "acquire", "fit", "report", "update"])
def test_bound_protocol_rejects_binding_arguments(phase):
    bound = Protocol(plain_acquisition)(samples=3)
    args = {
        "__call__": (bound,),
        "acquire": (bound,),
        "fit": (4, bound),
        "report": (4, 5, bound),
        "update": (5, bound),
    }

    with pytest.raises(TypeError, match="binding arguments with a BoundProtocol"):
        getattr(Executor(None), phase)(*args[phase], samples=5)


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


@pytest.mark.parametrize("unbound", [False, True])
def test_builtin_acquisition(platform, mocker, unbound):
    kwargs = {
        "min_amp": 0,
        "max_amp": 1,
        "step_amp": 0.05,
        "nshots": 4096,
        "relaxation_time": 0,
    }
    protocol = rabi_amplitude if unbound else rabi_amplitude(**kwargs)
    execute = mocker.spy(platform, "execute")
    connect = mocker.spy(platform, "connect")
    disconnect = mocker.spy(platform, "disconnect")
    executor = Executor(platform, targets=[0, 1], update=False)

    data = executor.acquire(protocol, targets=[1], **(kwargs if unbound else {}))

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


@pytest.mark.parametrize("unbound", [False, True])
def test_all_phase_targets_overrides(platform, mocker, unbound):
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
    bound = protocol if unbound else protocol(**kwargs)
    binding = kwargs if unbound else {}
    executor = Executor(platform, targets=[0])
    completed = executor(bound, targets=[1], **binding)

    assert completed.reports == (3, 4)
    for callback in callbacks:
        assert callback.call_args.kwargs["platform"] is platform
        assert callback.call_args.kwargs["targets"] == [1]
    assert callbacks[1].call_args.kwargs["fitpars"] == {"method": "new"}
    assert callbacks[2].call_args.kwargs["fit"] == 4
    assert callbacks[2].call_args.kwargs["reportpars"] == {"html": True}
    assert executor.platform is platform
    assert executor.targets == [0]

    executor.acquire(bound, targets=[1], **binding)
    executor.fit(3, bound, targets=[1], **binding)
    executor.report(3, 4, bound, targets=[1], **binding)
    executor.update(4, bound, targets=[1], **binding)
    for callback in callbacks:
        assert callback.call_count == 2
        assert callback.call_args.kwargs["platform"] is platform
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


@pytest.mark.parametrize("phase", ["__call__", "acquire", "fit", "report", "update"])
@pytest.mark.parametrize("unbound", [False, True])
def test_platform_override_rejected(platform, phase, unbound):
    protocol = Protocol(plain_acquisition)
    bound = protocol if unbound else protocol(samples=3)
    args = {
        "__call__": (bound,),
        "acquire": (bound,),
        "fit": (4, bound),
        "report": (4, 5, bound),
        "update": (5, bound),
    }
    kwargs = {"samples": 3} if unbound else {}

    with pytest.raises(TypeError, match="platform cannot be overridden"):
        getattr(Executor(platform), phase)(*args[phase], platform=platform, **kwargs)


@pytest.mark.parametrize("phase", ["__call__", "acquire"])
@pytest.mark.parametrize("binding", ["bound", "pars", "keywords"])
def test_execution_settings_precedence(platform, mocker, phase, binding):
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
    protocol = bound if binding == "bound" else rabi_amplitude
    if binding == "pars":
        kwargs["pars"] = parameters
    elif binding == "keywords":
        kwargs |= {"min_amp": 0, "max_amp": 1, "step_amp": 0.05}
    if phase == "__call__":
        kwargs["skip_fit"] = True
    execute = mocker.spy(platform, "execute")
    executor = Executor(platform, targets=[0])

    output = getattr(executor, phase)(protocol, **kwargs)

    assert (output.data if phase == "__call__" else output).qubits == [1]
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

    overridden = resolve(bound, {"future_setting": 2})
    rebound = resolve(protocol, {"pars": parameters, "future_setting": 3})

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

    acquired = Executor(None).acquire(bound, nshots=128)

    assert acquired.nshots == 128
    assert parameters.nshots == 64
