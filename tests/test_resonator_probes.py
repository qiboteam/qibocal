"""Rectangular readout probes of the resonator spectroscopies with flux and states."""

import pytest
from qibolab import Readout, Rectangular

from qibocal.protocols.dispersive_shift.dispersive_shift import (
    DispersiveShiftParameters,
)
from qibocal.protocols.dispersive_shift.dispersive_shift import (
    _acquisition as dispersive_shift,
)
from qibocal.protocols.dispersive_shift.dispersive_shift_qutrit import (
    _acquisition as dispersive_shift_qutrit,
)
from qibocal.protocols.flux_dependence.resonator_flux_dependence import (
    ResonatorFluxParameters,
)
from qibocal.protocols.flux_dependence.resonator_flux_dependence import (
    _acquisition as resonator_flux,
)

PROTOCOLS = [
    (dispersive_shift, DispersiveShiftParameters, {}, 2),
    (dispersive_shift_qutrit, DispersiveShiftParameters, {}, 3),
    (resonator_flux, ResonatorFluxParameters, {"bias": ["center", 0.2, 0.1]}, 1),
]


@pytest.mark.parametrize("acquisition, parameters, extra, nsequences", PROTOCOLS)
def test_rectangular_probe(
    platform, mocker, acquisition, parameters, extra, nsequences
):
    execute = mocker.spy(platform, "execute")
    params = parameters.load(
        {
            "frequency": ["center", 2e6, 1e6],
            "amplitude": 0.05,
            "duration": 1000,
            "nshots": 10,
            **extra,
        }
    )
    acquisition(params, platform, [0])

    sequences = execute.call_args.args[0]
    assert len(sequences) == nsequences
    for sequence in sequences:
        [readout] = [
            pulse
            for pulse in sequence.channel(platform.qubits[0].acquisition)
            if isinstance(pulse, Readout)
        ]
        assert isinstance(readout.probe.envelope, Rectangular)
        assert readout.probe.amplitude == 0.05
        assert readout.probe.duration == readout.acquisition.duration == 1000


@pytest.mark.parametrize("parameters, extra", [(p, e) for _, p, e, _ in PROTOCOLS])
def test_amplitude_required(parameters, extra):
    loaded = parameters.load(
        {"frequency": ["center", 2e6, 1e6], "amplitude": 0.1, **extra}
    )
    assert loaded.duration == 2000
    with pytest.raises(TypeError, match="amplitude"):
        parameters.load({"frequency": ["center", 2e6, 1e6], **extra})
