from conftest import TEST_FILE_DIR, approx_for_regression

from qibocal.protocols.rabi.amplitude import (
    RabiAmplitudeData,
    RabiAmplitudeResults,
)
from qibocal.protocols.rabi.amplitude import (
    _fit as rabi_amplitude_classification_fitting,
)
from qibocal.protocols.rabi.amplitude_signal import (
    RabiAmplitudeSignalData,
    RabiAmplitudeSignalResults,
)
from qibocal.protocols.rabi.amplitude_signal import (
    _fit as rabi_amplitude_signal_fitting,
)
from qibocal.protocols.rabi.length import (
    RabiLengthData,
    RabiLengthResults,
)
from qibocal.protocols.rabi.length import (
    _fit as rabi_length_classification_fitting,
)
from qibocal.protocols.rabi.length_signal import (
    RabiLengthSignalData,
    RabiLengthSignalResults,
)
from qibocal.protocols.rabi.length_signal import (
    _fit as rabi_length_signal_fitting,
)
from qibocal.protocols.ramsey.acquisition import RamseyResults
from qibocal.protocols.ramsey.classification import (
    RamseyData,
)
from qibocal.protocols.ramsey.classification import (
    _fit as ramsey_classification_fitting,
)
from qibocal.protocols.ramsey.signal import (
    RamseySignalData,
)
from qibocal.protocols.ramsey.signal import (
    _fit as ramsey_signal_fitting,
)

RABI_TEST_DIR = TEST_FILE_DIR / "rabi_fit_data"
RAMSEY_TEST_DIR = TEST_FILE_DIR / "ramsey_fit_data"


def test_ramsey_fit():
    results_folders = [p for p in RAMSEY_TEST_DIR.iterdir() if p.is_dir()]

    for ramsey_res in results_folders:
        if "signal" in ramsey_res.name:
            ramsey_fitting = ramsey_signal_fitting
            data = RamseySignalData.load(ramsey_res)
        else:
            ramsey_fitting = ramsey_classification_fitting
            data = RamseyData.load(ramsey_res)
        expected = RamseyResults.load(ramsey_res)

        assert data is not None and expected is not None

        fitted = ramsey_fitting(data)

        for qubit in data.qubits:
            assert fitted.frequency[qubit][0] == approx_for_regression(
                expected.frequency[qubit][0]
            )
            assert fitted.t2[qubit][0] == approx_for_regression(expected.t2[qubit][0])
            assert fitted.delta_phys[qubit][0] == approx_for_regression(
                expected.delta_phys[qubit][0]
            )
            assert fitted.delta_fitting[qubit][0] == approx_for_regression(
                expected.delta_fitting[qubit][0]
            )


def test_rabi_fit():
    results_folders = [p for p in RABI_TEST_DIR.iterdir() if p.is_dir()]

    for rabi_res in results_folders:
        if all(x in rabi_res.name for x in ["signal", "amplitude"]):
            rabi_fitting = rabi_amplitude_signal_fitting
            data = RabiAmplitudeSignalData.load(rabi_res)
            expected = RabiAmplitudeSignalResults.load(rabi_res)
            parameter = "amplitude"
        elif "signal" not in rabi_res.name and "amplitude" in rabi_res.name:
            parameter = "amplitude"
            rabi_fitting = rabi_amplitude_classification_fitting
            data = RabiAmplitudeData.load(rabi_res)
            expected = RabiAmplitudeResults.load(rabi_res)
        elif all(x in rabi_res.name for x in ["signal", "length"]):
            parameter = "length"
            rabi_fitting = rabi_length_signal_fitting
            data = RabiLengthSignalData.load(rabi_res)
            expected = RabiLengthSignalResults.load(rabi_res)
        else:  # "signal" not in rabi_res.name and "length" in rabi_res.name
            parameter = "length"
            rabi_fitting = rabi_length_classification_fitting
            data = RabiLengthData.load(rabi_res)
            expected = RabiLengthResults.load(rabi_res)

        assert data is not None and expected is not None

        fitted = rabi_fitting(data)

        for qubit in data.qubits:
            assert getattr(fitted, parameter)[qubit] == approx_for_regression(
                getattr(expected, parameter)[qubit]
            )

            if "signal" not in rabi_res.name:
                assert fitted.chi2[qubit] == approx_for_regression(expected.chi2[qubit])
