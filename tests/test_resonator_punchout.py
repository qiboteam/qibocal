import numpy as np
import pytest
from qibolab import Rectangular

from qibocal import update
from qibocal.protocols.resonator_spectroscopies.resonator_punchout import (
    ResonatorPunchoutParameters,
    ResonatorPunchoutResults,
)
from qibocal.protocols.resonator_spectroscopies.resonator_punchout import (
    _acquisition as punchout_acquisition,
)
from qibocal.protocols.resonator_spectroscopies.resonator_punchout import (
    _update as punchout_update,
)
from qibocal.protocols.resonator_spectroscopies.resonator_punchout_attenuation import (
    ResonatorPunchoutAttenuationParameters,
)
from qibocal.protocols.resonator_spectroscopies.resonator_punchout_attenuation import (
    _acquisition as attenuation_acquisition,
)

AMPLITUDES = {"min_amp": 0.1, "max_amp": 0.3, "step_amp": 0.1}
ATTENUATIONS = {"min_attenuation": 0, "max_attenuation": 20, "step_attenuation": 10}


def readouts(execute, platform, qubits):
    [sequence] = execute.call_args.args[0]
    return [
        next(iter(sequence.channel(platform.qubits[q].acquisition))) for q in qubits
    ]


@pytest.mark.parametrize(
    "acquisition, parameters, sweeps",
    [
        (punchout_acquisition, ResonatorPunchoutParameters, AMPLITUDES),
        (attenuation_acquisition, ResonatorPunchoutAttenuationParameters, ATTENUATIONS),
    ],
)
def test_rectangular_probe(platform, mocker, acquisition, parameters, sweeps):
    execute = mocker.spy(platform, "execute")
    params = parameters.load(
        {"frequency": ["center", 2e6, 1e6], "duration": 1000, "nshots": 10, **sweeps}
    )
    acquisition(params, platform, [0, 1])

    for readout in readouts(execute, platform, [0, 1]):
        assert isinstance(readout.probe.envelope, Rectangular)
        assert readout.probe.duration == readout.acquisition.duration == 1000
    if parameters is ResonatorPunchoutAttenuationParameters:
        assert params.amplitude == 1.0
        assert all(r.probe.amplitude == 1.0 for r in readouts(execute, platform, [0]))


def test_legacy_frequency_ranges():
    """The deprecated width and step define the frequency ranges."""
    center, width, step = 7e9, 10e6, 1e6
    punchout = ResonatorPunchoutParameters.load(
        {"freq_width": width, "freq_step": step, **AMPLITUDES}
    )
    attenuation = ResonatorPunchoutAttenuationParameters.load(
        {"freq_width": width, "freq_step": step, **ATTENUATIONS}
    )

    np.testing.assert_allclose(
        np.arange(*punchout.frequency_range(center)),
        center + np.arange(-width / 2, width / 2, step),
    )
    np.testing.assert_allclose(
        np.arange(*attenuation.frequency_range(center)),
        center + np.arange(-width / 2, width / 2 + step, step),
    )
    with pytest.raises(ValueError, match="Provide 'frequency'"):
        ResonatorPunchoutParameters.load(AMPLITUDES).frequency_range(center)


@pytest.mark.parametrize(
    "parameters, sweeps, legacy, method",
    [
        (
            ResonatorPunchoutParameters,
            {"amplitude": ["linspace", 0.1, 0.3, 3]},
            AMPLITUDES,
            "amplitude_range",
        ),
        (
            ResonatorPunchoutAttenuationParameters,
            {"attenuation": ["linspace", 0, 20, 3]},
            ATTENUATIONS,
            "attenuation_sweep",
        ),
    ],
)
def test_range_like_sweeps(parameters, sweeps, legacy, method):
    """Amplitude and attenuation accept the range syntax, and the legacy fields."""
    new = parameters.load({"frequency": ["center", 2e6, 1e6], **sweeps})
    old = parameters.load({"frequency": ["center", 2e6, 1e6], **legacy})

    assert pytest.approx(getattr(new, method)()) == getattr(old, method)()
    assert getattr(old, method)() == tuple(legacy.values())
    with pytest.raises(ValueError, match=f"Provide '{next(iter(sweeps))}'"):
        getattr(parameters.load({}), method)()
    with pytest.raises(ValueError, match="Center must be provided"):
        getattr(parameters.load({next(iter(sweeps)): ["center", 1, 0.1]}), method)()


def test_range_like_acquisition(platform, mocker):
    """The range syntax defines the swept amplitudes and attenuations."""
    frequency = {"frequency": ["center", 2e6, 1e6], "nshots": 10}
    data = punchout_acquisition(
        ResonatorPunchoutParameters.load({"amplitude": [0.1, 0.35, 0.1], **frequency}),
        platform,
        [0],
    )
    np.testing.assert_allclose(data.amplitudes, [0.1, 0.2, 0.3])

    execute = mocker.spy(platform, "execute")
    data = attenuation_acquisition(
        ResonatorPunchoutAttenuationParameters.load(
            {"attenuation": ["window", 10, 20, 10], **frequency}
        ),
        platform,
        [0],
    )
    np.testing.assert_allclose(data.attenuations, [0, 10])
    assert execute.call_count == 2


def test_punchout_update_records_amplitude(platform):
    results = ResonatorPunchoutResults(
        readout_frequency={0: 7.1e9},
        bare_frequency={0: 7.2e9},
        readout_amplitude={0: 0.25},
        successful_fit={0: True},
    )
    punchout_update(results, platform, 0)

    assert platform.natives.single_qubit[0].MZ()[0][1].probe.amplitude == 0.25
    assert platform.calibration.single_qubits[0].resonator.punchout_amplitude == 0.25


def test_readout_amplitude_creates_measurement(platform):
    platform.update({"native_gates.single_qubit.0.MZ": None})
    update.readout_amplitude(0.3, platform, 0)

    [(channel, readout)] = platform.natives.single_qubit[0].MZ()
    assert channel == platform.qubits[0].acquisition
    assert isinstance(readout.probe.envelope, Rectangular)
    assert readout.probe.amplitude == 0.3
    assert readout.probe.duration == readout.acquisition.duration
    assert readout.probe.duration == update.DEFAULT_READOUT_DURATION
    assert readout.time_of_flight == 0
