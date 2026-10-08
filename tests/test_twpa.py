import numpy as np
import pytest
from qibolab import (
    Acquisition,
    OscillatorConfig,
    Parameter,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    Sweeper,
)

from qibocal.protocols.twpa.twpa import (
    TwpaCalibrationParameters,
    _fit,
    _plot,
    _twpa_scan,
)


@pytest.mark.parametrize("pumps", [("a",), ("a", "a"), ("a", "b", "a")])
@pytest.mark.parametrize("legacy", [True, False])
def test_twpa_scan_pump_mapping(mocker, pumps, legacy):
    qubits = list(reversed(range(len(pumps))))
    configs = {
        "a": OscillatorConfig(frequency=6e9, power=0.1),
        "b": OscillatorConfig(frequency=7e9, power=0.2),
    }
    params = TwpaCalibrationParameters.load(
        {
            "nshots": 10,
            "relaxation_time": 0,
            "freq_width": 10e6,
            "freq_step": 1e6,
            "twpa_freq_width": 10e6,
            "twpa_freq_step": 2e6,
            "twpa_pow_width": 0.8,
            "twpa_pow_step": 0.1,
        }
    )
    if not legacy:
        params.probe_frequency = ("asym", (5e6, 5e6), 1e6)
        params.frequency = ("asym", (5e6, 5e6), 2e6)
        params.power = ("asym", (0.4, 0.4), 0.1)

    sequence = PulseSequence()
    sweepers = []
    channels = []
    results = {}
    for qubit in qubits:
        channel = f"{qubit}/acquisition"
        channels.append(channel)
        readout = Readout(
            acquisition=Acquisition(duration=100),
            probe=Pulse(duration=100, amplitude=0.1, envelope=Rectangular()),
        )
        sequence.append((channel, readout))
        sweeper = Sweeper(
            parameter=Parameter.frequency,
            range=params.probe_frequency_range(5e9),
            channels=[f"{qubit}/probe"],
        )
        sweepers.append(sweeper)
        results[readout.id] = np.ones((len(sweeper.values), 2))

    platform = mocker.Mock()
    platform.config.side_effect = configs.__getitem__
    platform.execute.return_value = results
    data = _twpa_scan(platform, sequence, sweepers, qubits, pumps, channels, params)

    assert (
        set(data.data)
        == set(data.twpa_frequency)
        == set(data.twpa_power)
        == set(qubits)
    )
    for qubit, pump, sweeper in zip(qubits, pumps, sweepers):
        expected_power = np.arange(*params.power_range(configs[pump].power))
        expected_frequency = np.arange(*params.frequency_range(configs[pump].frequency))
        np.testing.assert_array_equal(data.twpa_power[qubit], expected_power)
        np.testing.assert_array_equal(data.twpa_frequency[qubit], expected_frequency)
        assert data[qubit].shape == (
            len(expected_power),
            len(expected_frequency),
            len(sweeper.values),
            2,
        )
        data.reference_value[qubit] = results[
            list(sequence.channel(f"{qubit}/acquisition"))[-1].id
        ].tolist()

    unique_pumps = list(dict.fromkeys(pumps))
    expected_updates = [
        [
            {pump: {"power": power, "frequency": frequency}}
            for pump, power, frequency in zip(unique_pumps, powers, frequencies)
        ]
        for powers in zip(
            *(
                np.arange(*params.power_range(configs[pump].power))
                for pump in unique_pumps
            )
        )
        for frequencies in zip(
            *(
                np.arange(*params.frequency_range(configs[pump].frequency))
                for pump in unique_pumps
            )
        )
    ]
    assert [
        call.kwargs["updates"] for call in platform.execute.call_args_list
    ] == expected_updates

    fit = _fit(data)
    for qubit in qubits:
        assert fit.twpa_power[qubit] == data.twpa_power[qubit][0]
        assert fit.twpa_frequency[qubit] == data.twpa_frequency[qubit][0]
        assert len(_plot(data, None, qubit)[0]) == 1
        assert len(_plot(data, fit, qubit)[0]) == 1
