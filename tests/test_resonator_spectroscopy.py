import pytest
import yaml
from conftest import PATH_TESTING_DATA, approx_for_regression
from qibolab import Rectangular

from qibocal.protocols.resonator_spectroscopies.resonator_spectroscopy import (
    ResonatorSpectroscopyData,
    ResonatorSpectroscopyParameters,
    ResonatorSpectroscopyResults,
)
from qibocal.protocols.resonator_spectroscopies.resonator_spectroscopy import (
    _acquisition as resonator_spectroscopy_acquisition,
)
from qibocal.protocols.resonator_spectroscopies.resonator_spectroscopy import (
    _fit as resonator_spectroscopy_fit,
)
from qibocal.protocols.resonator_spectroscopies.resonator_spectroscopy import (
    _update as resonator_spectroscopy_update,
)


def test_resonator_spectroscopy_fit():
    results_folder = PATH_TESTING_DATA / "resonator_spectroscopy-0"
    data = ResonatorSpectroscopyData.load(results_folder)
    expected = ResonatorSpectroscopyResults.load(results_folder)
    assert data is not None and expected is not None

    action = yaml.safe_load((results_folder / "action.yml").read_text(encoding="utf-8"))
    [qubit] = action["targets"]

    fitted = resonator_spectroscopy_fit(data)
    assert fitted.frequency[qubit] == approx_for_regression(expected.frequency[qubit])
    assert fitted.fitted_parameters[qubit] == approx_for_regression(
        expected.fitted_parameters[qubit]
    )
    assert fitted.amplitude is not None and expected.amplitude is not None
    assert fitted.amplitude[qubit] == approx_for_regression(expected.amplitude[qubit])


def resonator_parameters(**kwargs):
    return ResonatorSpectroscopyParameters.load(
        {"frequency": ["center", 2e6, 1e6], "nshots": 10, **kwargs}
    )


def test_rectangular_probe(platform, mocker):
    execute = mocker.spy(platform, "execute")
    params = resonator_parameters(amplitude=0.05, duration=1000)
    data = resonator_spectroscopy_acquisition(params, platform, [0, 1])

    [sequence] = execute.call_args.args[0]
    for qubit in (0, 1):
        [readout] = sequence.channel(platform.qubits[qubit].acquisition)
        assert isinstance(readout.probe.envelope, Rectangular)
        assert readout.probe.amplitude == 0.05
        assert readout.probe.duration == readout.acquisition.duration == 1000
        assert data.amplitudes[qubit] == 0.05
    assert resonator_parameters(amplitude=0.05).duration == 2000


def test_amplitude_required():
    with pytest.raises(TypeError, match="amplitude"):
        resonator_parameters()


def test_update_keeps_readout_amplitude(platform):
    natives = platform.natives.single_qubit[0]
    amplitude = natives.MZ()[0][1].probe.amplitude
    results = ResonatorSpectroscopyResults(
        frequency={0: 7.1e9}, fitted_parameters={0: []}, amplitude={0: 0.05}
    )
    resonator_spectroscopy_update(results, platform, 0)

    assert platform.natives.single_qubit[0].MZ()[0][1].probe.amplitude == amplitude
