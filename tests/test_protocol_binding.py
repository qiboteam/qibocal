"""Signature and docstring of the binding of each protocol."""

import copy
import dataclasses
import inspect
import pickle
import pydoc

import pytest

from qibocal.auto.operation import BoundProtocol, Protocol
from qibocal.protocols import rabi_amplitude
from qibocal.protocols.rabi.amplitude import RabiAmplitudeParameters

FIELDS = ["min_amp", "max_amp", "step_amp", "pulse_length", "rx90"]


def test_call_binds():
    bound = rabi_amplitude(min_amp=0, max_amp=1, step_amp=0.1, nshots=10)

    assert isinstance(bound, BoundProtocol)
    assert bound.protocol is rabi_amplitude
    assert bound.parameters.nshots == 10
    assert rabi_amplitude.__call__(min_amp=0, max_amp=1, step_amp=0.1).protocol is (
        rabi_amplitude
    )


def test_signature():
    signature = inspect.signature(rabi_amplitude)
    parameters = signature.parameters

    assert list(parameters) == ["pars", "fit", "report", *FIELDS, "nshots"] + [
        "relaxation_time"
    ]
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in [*FIELDS, "nshots", "relaxation_time"]
    )
    assert parameters["min_amp"].annotation is float
    assert parameters["min_amp"].default is inspect.Parameter.empty
    assert parameters["rx90"].default is False
    assert parameters["nshots"].default is None
    # the same for the explicit `__call__`, as read by IPython
    assert inspect.signature(rabi_amplitude.__call__) == signature


def test_docstring():
    doc = inspect.getdoc(rabi_amplitude)

    assert doc.startswith("Bind RabiAmplitudeParameters to this protocol.")
    assert inspect.getdoc(RabiAmplitudeParameters) in doc
    assert inspect.getdoc(rabi_amplitude.__call__) == doc
    assert "Bind RabiAmplitudeParameters" in pydoc.render_doc(
        rabi_amplitude, renderer=pydoc.plaintext
    )


def test_class_unaffected():
    assert Protocol.__doc__.startswith("A calibration protocol")
    assert inspect.getdoc(Protocol).startswith("A calibration protocol")
    assert next(iter(inspect.signature(Protocol).parameters)) == "acquisition"
    assert Protocol.__call__ is Protocol._bind
    assert "__call__" not in vars(rabi_amplitude)
    assert "__doc__" not in vars(rabi_amplitude)


def test_copies():
    copied = copy.deepcopy(rabi_amplitude)
    replaced = dataclasses.replace(rabi_amplitude, two_qubit_gates=True)
    pickled = pickle.loads(pickle.dumps(rabi_amplitude))

    for protocol in (copied, replaced, pickled):
        assert inspect.signature(protocol) == inspect.signature(rabi_amplitude)
        bound = protocol.__call__(min_amp=0, max_amp=1, step_amp=0.1)
        assert bound.protocol is protocol
    assert copied == rabi_amplitude


@pytest.mark.parametrize(
    "acquisition",
    [lambda samples: samples, lambda: None, lambda pars: pars],
    ids=["unannotated", "no-parameters", "plain"],
)
def test_generic_fallback(acquisition):
    protocol = Protocol(acquisition)

    assert inspect.signature(protocol) == inspect.signature(protocol._bind)
    assert inspect.getdoc(protocol) == inspect.getdoc(Protocol._bind)


def test_non_parameters_dataclass():
    @dataclasses.dataclass
    class Plain:
        samples: int
        offset: int = 1

    def acquisition(pars: Plain) -> int:
        return pars.samples + pars.offset

    protocol = Protocol(acquisition)
    parameters = inspect.signature(protocol).parameters

    assert list(parameters) == ["pars", "fit", "report", "samples", "offset"]
    assert protocol(samples=3).parameters == Plain(3)
