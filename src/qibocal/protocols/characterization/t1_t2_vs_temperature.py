import os
import signal
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from shutil import copy2, copytree
from time import sleep
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots
from qibo.config import log
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.instruments.bluefors import TemperatureController
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.result import SampleResults

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

K_to_mK = 1e3
GHz_to_MHz = 1e3
ms_to_ns = 1e6
uW_to_W = 1e-6

experiment_record_type = np.dtype(
    [
        ("heater_power", np.float64),
        ("sensor_temperature", np.float64),
        ("timestamp", np.float64),
    ]
)

qubits_record_type = np.dtype(
    [
        ("effective_temperature", np.float64),
        ("frequency", np.float64),
        ("T1", np.float64),
        ("T2", np.float64),
    ]
)

"""Custom dtypes to store experiment data."""

folder = Path("t1_t2_vs_temperature_data") / Path("data")
folder.mkdir(exist_ok=True)


@dataclass
class T1T2vsTemperatureParameters(Parameters):
    """T1T2vsTemperature runcard inputs."""

    temperature_controller_ip: Optional[str] = "192.168.0.197"
    """IP address of BF Temperature Controller."""
    min_heater_power: Optional[int] = 2  # uW
    """Minimum power to be applied to fridge heater."""
    max_heater_power: Optional[int] = 100  # uW
    """Maximum power to be applied to fridge heater."""
    intermediate_measurements: bool = False
    """Flag to request taking measurements also while fridge temperature stabilises."""
    num_datapoints: Optional[int] = 1
    """Number of measurements to be taken."""
    stabilization_time: Optional[int] = 60  # s
    """Delay between setting a heater power and taking a measurement."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns) between shots."""


@dataclass
class T1T2vsTemperatureData(Data):
    """A class to store the data acquired by the routine."""

    nshots: int
    """Number of shots."""
    data: dict = field(default_factory=dict)
    """The attribute where data will be stored. Its tyep is a dictionary that contains qubit specific data under QubitId keys,
    and non qubit specific data (experiment data like fridge temperature of applied power) under the key 'experiment_data'"""

    _experiment_data: npt.NDArray[experiment_record_type] = field(
        default_factory=lambda: np.array([]), init=False, repr=False
    )
    _qubits_data: dict[QubitId, npt.NDArray[qubits_record_type]] = field(
        default_factory=dict, init=False, repr=False
    )

    def add_data(
        self,
        experiment_record: npt.NDArray[experiment_record_type],
        qubits_record: dict[QubitId, npt.NDArray[qubits_record_type]],
    ):
        """Store experiment datapoint data."""
        if self._experiment_data.size == 0:
            self._experiment_data = np.rec.array(experiment_record)
        else:
            self._experiment_data = np.rec.array(
                np.append(self._experiment_data, experiment_record)
            )

        self.data.update({"experiment_data": self._experiment_data})

        for qubit in qubits_record:
            if not qubit in self._qubits_data:
                self._qubits_data[qubit] = np.rec.array(qubits_record[qubit])
            else:
                self._qubits_data[qubit] = np.rec.array(
                    np.append(self._qubits_data[qubit], qubits_record[qubit])
                )
        self.data.update(self._qubits_data)

    def save(self, destination_folder: Path):
        for item in os.listdir(folder):
            source_item: Path = folder / item
            destination_item: Path = destination_folder.parent / item

            if source_item.is_dir():
                # If the item is a directory, use copytree
                if not destination_item.exists():
                    copytree(source_item, destination_item)
            else:
                # If the item is a file, use copy2 (preserves file metadata)
                copy2(source_item, destination_item)
        super().save(destination_folder)

    @property
    def global_params_dict(self):
        """Convert non-arrays attributes into dict."""
        global_dict = asdict(self)
        global_dict.pop("data")
        # remove pribvate attributes
        global_dict.pop("_experiment_data")
        global_dict.pop("_qubits_data")

        return global_dict


@dataclass
class T1T2vsTemperatureResults(Results):
    """T1T2vsTemperature outputs."""

    pass
    # TODO: define what we aim to extract
    # effective_temperature: dict[QubitId, list[float]]
    # """Effective Temperature"""
    # frequency: dict[QubitId, list[float]]
    # """Effective Temperature"""
    # T1: dict[QubitId, list[int]]
    # """Effective Temperature"""
    # T2: dict[QubitId, list[int]]
    # """Effective Temperature"""
    # timestamp: list[float]
    # """Timestamp"""


def _acquisition(
    params: T1T2vsTemperatureParameters,
    platform: Platform,
    qubits: Qubits,
) -> T1T2vsTemperatureData:
    """
    This method acquires the data necessary to determine how the qubits T1 and T2 vary with Temperature.
    The routine controls the temperature of the fridge and triggers the execution of several other routines
    once the temperature of the fridge is stable.

    TODO
    The routine changes the temperature of the fridge to the values passed.
    The effective temperature, defined by Teff = -h_bar*w01/Kb/ln(P1/P0)
    where
        P0 = statistical frequency of measuring 0:  n0 / nshots
        P1 = statistical frequency of measuring 1:  n1 / nshots

    We prepare the qubit in a state |0> and mesure. A small proportion of the results will be
    measured as |1> due to thermal excitations.

    Args:
        nshots (int): number of times the pulse sequence will be repeated.
        relaxation_time (float): Relaxation time.

        Example:
        .. code-block:: yaml

            - id: T1 and T2 vs Temperature
              priority: 10
              operation: t1_t2_vs_temperature
              parameters:
                temperature_controller_ip: '192.168.0.197'
                max_heater_power: 3000    # uW
                min_heater_power: 0       # uW
                stabilization_time: 3600  # s
                num_datapoints: 24
                nshots: 50_000
                relaxation_time: 50_000   # ns

    """

    def _run_effective_temperature(
        nshots,
        relaxation_time,
        platform: Platform,
        qubits: Qubits,
    ) -> dict[QubitId, float]:
        sequence = PulseSequence()
        ro_pulses = {}
        frequencies = {}
        for qubit in qubits:
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
            sequence.add(ro_pulses[qubit])
            frequencies[qubit] = qubits[qubit].drive_frequency

        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=nshots,
                relaxation_time=relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
            ),
        )
        h_bar = 6.62607015e-34
        Kb = 1.380649e-23
        probability = {}
        effective_temperature = {}
        for qubit in qubits:
            result: SampleResults = results[ro_pulses[qubit].serial]
            P1 = probability[qubit] = float(result.probability(1))
            P0 = 1 - P1
            w01 = frequencies[qubit]
            T = effective_temperature[qubit] = -h_bar * w01 / Kb / np.log(P1 / P0)
        return effective_temperature

    def _run_qubit_spectroscopy(
        timestamp,
        params: T1T2vsTemperatureParameters,
        platform: Platform,
        qubits: Qubits,
    ) -> dict[QubitId, float]:
        from qibocal.protocols.characterization.qubit_spectroscopy import (
            QubitSpectroscopyData,
            QubitSpectroscopyParameters,
            QubitSpectroscopyResults,
            qubit_spectroscopy,
        )

        operation: Routine = qubit_spectroscopy
        qubit_spectroscopy_params: QubitSpectroscopyParameters = (
            QubitSpectroscopyParameters.load(
                {
                    "freq_width": 10_000_000,
                    "freq_step": 100_000,
                    "drive_duration": 5_000,
                    "drive_amplitude": 0.05,
                    "nshots": 2000,  # params.nshots,
                    "relaxation_time": 1000,
                }
            )
        )
        qubit_spectroscopy_data: QubitSpectroscopyData
        qubit_spectroscopy_results: QubitSpectroscopyResults

        qubit_spectroscopy_data, time = operation.acquisition(
            qubit_spectroscopy_params, platform=platform, qubits=qubits
        )
        qubit_spectroscopy_results, time = operation.fit(qubit_spectroscopy_data)
        time = datetime.now().strftime("%Y%m%d_%H%M")
        (folder / f"{time}_qubit_spectroscopy").mkdir(exist_ok=True)
        qubit_spectroscopy_data.save(folder / f"{time}_qubit_spectroscopy")
        qubit_spectroscopy_results.save(folder / f"{time}_qubit_spectroscopy")

        return qubit_spectroscopy_results.frequency

    def _run_t1(
        timestamp,
        params: T1T2vsTemperatureParameters,
        platform: Platform,
        qubits: Qubits,
    ) -> dict[QubitId, float]:
        from qibocal.protocols.characterization.coherence.t1_msr import (
            T1MSRData,
            T1MSRParameters,
            T1MSRResults,
            t1_msr,
        )

        operation: Routine = t1_msr
        t1_params: T1MSRParameters = T1MSRParameters.load(
            {
                "delay_before_readout_start": 4,
                "delay_before_readout_end": 8000,
                "delay_before_readout_step": 32,
                "nshots": 2000,  # params.nshots,
                "relaxation_time": params.relaxation_time,
            }
        )
        t1_data: T1MSRData
        t1_results: T1MSRResults

        t1_data, time = operation.acquisition(
            t1_params, platform=platform, qubits=qubits
        )
        t1_results, time = operation.fit(t1_data)
        time = datetime.now().strftime("%Y%m%d_%H%M")
        (folder / f"{time}_t1").mkdir(exist_ok=True)
        t1_data.save(folder / f"{time}_t1")
        t1_results.save(folder / f"{time}_t1")

        return t1_results.t1

    def _run_ramsey(
        timestamp,
        params: T1T2vsTemperatureParameters,
        platform: Platform,
        qubits: Qubits,
    ) -> dict[QubitId, float]:
        from qibocal.protocols.characterization.ramsey_msr import (
            RamseyMSRData,
            RamseyMSRParameters,
            RamseyMSRResults,
            ramsey_msr,
        )

        operation: Routine = ramsey_msr
        ramsey_params: RamseyMSRParameters = RamseyMSRParameters.load(
            {
                "delay_between_pulses_start": 4,
                "delay_between_pulses_end": 2000,
                "delay_between_pulses_step": 8,
                "n_osc": 10,
                "nshots": 2000,  # params.nshots,
                "relaxation_time": params.relaxation_time,
            }
        )
        ramsey_data: RamseyMSRData
        ramsey_results: RamseyMSRResults

        ramsey_data, time = operation.acquisition(
            ramsey_params, platform=platform, qubits=qubits
        )
        ramsey_results, time = operation.fit(ramsey_data)
        time = datetime.now().strftime("%Y%m%d_%H%M")
        (folder / f"{time}_ramsey").mkdir(exist_ok=True)
        ramsey_data.save(folder / f"{time}_ramsey")
        ramsey_results.save(folder / f"{time}_ramsey")

        return ramsey_results.t2

    def _get_datapoint():
        timestamp = datetime.now().timestamp()

        effective_temperature = _run_effective_temperature(
            params.nshots, params.relaxation_time, platform, qubits
        )
        qubit_frequency = _run_qubit_spectroscopy(
            timestamp, params, platform, qubits
        )  # ~5s
        t1s = _run_t1(timestamp, params, platform, qubits)  # ~30s
        t2s = _run_ramsey(timestamp, params, platform, qubits)  # ~30s

        sensor_temperature = tc.get_data("MXC-flange")["measurements"]["temperature"]
        experiment_record = np.array(
            [
                (
                    heater_power,
                    sensor_temperature,
                    timestamp,
                )
            ],
            dtype=experiment_record_type,
        )

        qubits_record = {}
        for qubit in qubits:
            t1 = t1s[qubit] if t1s[qubit] > 0 and t1s[qubit] < ms_to_ns else -1
            t2 = t2s[qubit] if t2s[qubit] > 0 and t2s[qubit] < ms_to_ns else -1
            qubits_record[qubit] = np.array(
                [(effective_temperature[qubit], qubit_frequency[qubit], t1, t2)],
                dtype=qubits_record_type,
            )

        data.add_data(experiment_record, qubits_record)
        time = datetime.now().strftime("%Y%m%d_%H%M")
        np.savez(
            folder / f"{time}_results.npz", **{str(i): data.data[i] for i in data.data}
        )
        log.info(
            f"heater_power: {heater_power}, sensor_temperature: {sensor_temperature * 1000:.2f} mk"
        )

    def _termination_handler(signum, frame):
        """Calls all modules to stop if the program receives a termination signal."""

        log.warning("Termination signal received, stopping heater.")
        tc.set_heater(channel="MXC-heater", active=False, power=heater_power)
        log.warning("Temperature Controller MXC heater stopped.")
        exit(0)

    # instantiate the driver of BF Temperature Controller
    tc = TemperatureController(params.temperature_controller_ip)
    # if the routine is cancelled, ensure heaters are deactivated
    signal.signal(signal.SIGTERM, _termination_handler)

    # create an T1T2vsTemperatureData object to store the data
    data = T1T2vsTemperatureData(nshots=params.nshots)

    # TODO: to be replaced with an array of temperatures, then powers will be interpolated
    heater_powers = (
        np.linspace(
            params.min_heater_power, params.max_heater_power, params.num_datapoints
        )
        * uW_to_W
    )
    for heater_power in heater_powers:
        tc.set_heater(channel="MXC-heater", active=True, power=heater_power)
        try:
            time_track = datetime.now().timestamp()
            while (datetime.now().timestamp() - time_track) < params.stabilization_time:
                if params.intermediate_measurements:
                    _get_datapoint()
                else:
                    sleep(1)
            _get_datapoint()

        except Exception as e:
            # if there is any error, ensure heaters are deactivated
            tc.set_heater(channel="MXC-heater", active=False, power=heater_power)
            raise e
    # when the experiment is finished, deactivate heaters
    tc.set_heater(channel="MXC-heater", active=False, power=heater_power)
    return data


def _fit(data: T1T2vsTemperatureData) -> T1T2vsTemperatureResults:
    # qubits = data.qubits[1:]
    # experiment_data = data["experiment_data"]
    # effective_temperature = {}
    # for qubit in qubits:
    #     effective_temperature[qubit] = data[qubit].effective_temperature.tolist()
    # timestamp = experiment_data.timestamp.tolist()
    # return T1T2vsTemperatureResults(
    #     effective_temperature,
    #     timestamp
    # )

    # TODO: define what we aim to extract
    return T1T2vsTemperatureResults()


def _plot(data: T1T2vsTemperatureData, fit: T1T2vsTemperatureResults, qubit):
    fitting_report = ""
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
    )
    qubit_data: npt.NDArray[qubits_record_type] = data[qubit]
    times = [datetime.fromtimestamp(ts) for ts in data["experiment_data"].timestamp]

    fig.add_trace(
        go.Scatter(
            x=data["experiment_data"].sensor_temperature * K_to_mK,
            y=qubit_data.T1,
            opacity=1,
            name="T1",
            showlegend=True,
            legendgroup="T1",
            yaxis="y",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["experiment_data"].sensor_temperature * K_to_mK,
            y=qubit_data.T2,
            opacity=1,
            name="T2",
            showlegend=True,
            legendgroup="T2",
            yaxis="y",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=data["experiment_data"].sensor_temperature * K_to_mK,
            y=qubit_data.frequency,
            opacity=1,
            name="qubit frequency",
            showlegend=True,
            legendgroup="qubit frequency",
            yaxis="y2",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=data["experiment_data"].sensor_temperature * K_to_mK,
            opacity=1,
            name="Sensor Temperature",
            showlegend=True,
            legendgroup="Sensor Temperature",
            yaxis="y3",
        ),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=qubit_data.effective_temperature * K_to_mK,
            opacity=1,
            name="Effective Temperature",
            showlegend=True,
            legendgroup="Effective Temperature",
            yaxis="y3",
        ),
        row=1,
        col=2,
        secondary_y=False,
    )

    # fig.update_yaxes(title_text="Coherance (ns)", row=1, col=1)
    # fig.update_yaxes(title_text="Temperature (mK)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (mK)", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        yaxis=dict(
            title="Coherence (ns)",
            autoshift=True,
            side="left",
        ),
        yaxis2=dict(
            title="Frequency (GHz)",
            autoshift=True,
            anchor="x",
            overlaying="y",
            side="right",
            titlefont=dict(color=DEFAULT_PLOTLY_COLORS[2]),
            tickfont=dict(color=DEFAULT_PLOTLY_COLORS[2]),
        ),
        yaxis3=dict(
            title="Temperature (mK)",
            autoshift=True,
            side="right",
        ),
    )
    figures.append(fig)

    fitting_report = ""
    # fitting_report += (
    # f"{qubit} | Effective Temperature: {float(fit.T_eff[qubit])*1000:.3f} mK<br>"
    # )
    fitting_report = "No fitting data" if fitting_report == "" else fitting_report

    return figures, fitting_report


t1_t2_vs_temperature = Routine(_acquisition, _fit, _plot)
