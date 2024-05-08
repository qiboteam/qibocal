from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log


@dataclass
class MixerCalibrationParameters(Parameters):
    """MixerCalibration runcard inputs."""

    n_points: float
    """Offset step for sweep [a.u.]."""


@dataclass
class MixerCalibrationResults(Results):
    """MixerCalibration outputs."""

    offset_i: dict[QubitId, float]
    """I waveform offset to minimise LO leakage."""
    offset_q: dict[QubitId, float]
    """Q waveform offset to minimise LO leakage."""


@dataclass
class MixerCalibrationData(Data):
    """Resonator type."""

    resonator_type: str

    offset_i_range: dict[QubitId, list] = field(default_factory=dict)
    offset_q_range: dict[QubitId, list] = field(default_factory=dict)
    gain_ratio_range: dict[QubitId, list] = field(default_factory=dict)
    phase_offset_range: dict[QubitId, list] = field(default_factory=dict)
    data: dict[QubitId, tuple[npt.NDArray, npt.NDArray]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(
        self,
        qubit_id: QubitId,
        offset_i_range: npt.NDArray,
        offset_q_range: npt.NDArray,
        gain_ratio_range: npt.NDArray,
        phase_offset_range: npt.NDArray,
        leakage_data: npt.NDArray,
        image_data: npt.NDArray,
    ):
        """Store output for single qubit."""
        self.offset_i_range[qubit_id] = offset_i_range.tolist()
        self.offset_q_range[qubit_id] = offset_q_range.tolist()
        self.gain_ratio_range[qubit_id] = gain_ratio_range.tolist()
        self.phase_offset_range[qubit_id] = phase_offset_range.tolist()
        self.data[qubit_id] = (leakage_data, image_data)


def _acquisition(
    params: MixerCalibrationParameters, platform: Platform, qubits: Qubits
) -> MixerCalibrationData:
    """Data acquisition for MixerCalibration experiment."""

    # auxiliary functions
    from time import sleep, time

    def wait_for_sequencer_stop(module, sequencer_number, time_out=30):
        t = time()
        while True:
            try:
                state = module.get_sequencer_state(sequencer_number)
            except:
                pass
            else:
                if state.status == "STOPPED":
                    module.stop_sequencer(sequencer_number)
                    break
                elif time() - t > time_out:
                    log.info(
                        f"Timeout - {module.sequencers[sequencer_number].name} state: {state}"
                    )
                    module.stop_sequencer(sequencer_number)
                    break
            sleep(1)

    leakage_data = {}
    image_data = {}

    for qubit_id in qubits:
        qubit: Qubit = qubits[qubit_id]

        flux_module = qubit.flux.port.module.device
        flux_port_number = qubit.flux.port.port_number
        # flux module
        flux_module.set(f"out{flux_port_number}_offset", value=qubit.sweetspot)

    for qubit_id in qubits:
        qubit: Qubit = qubits[qubit_id]

        # get qubit drive and readout frequencies, modules and sequencers
        drive_lo_frequency = qubit.drive.lo_frequency
        drive_frequency = qubit.drive_frequency
        drive_intermediate_frequency = drive_frequency - drive_lo_frequency
        drive_module = qubit.drive.port.module.device
        drive_port_number = qubit.drive.port.port_number
        drive_sequencer_number = qubit.drive.port.sequencer_number
        drive_sequencer = drive_module.sequencers[drive_sequencer_number]
        readout_lo_frequency = qubit.readout.lo_frequency
        readout_frequency = qubit.readout_frequency
        readout_intermediate_frequency = readout_frequency - readout_lo_frequency
        readout_module = qubit.readout.port.module.device
        readout_port_number = qubit.readout.port.port_number
        readout_sequencer_number = qubit.readout.port.sequencer_number
        readout_sequencer = readout_module.sequencers[readout_port_number]

        # save current values
        # current_offset_i = drive_module.get(f"out{drive_port_number}_offset_path0")
        # current_offset_q = drive_module.get(f"out{drive_port_number}_offset_path1")
        # current_gain_ratio = drive_sequencer.get(f"mixer_corr_gain_ratio")
        # current_phase_offset = drive_sequencer.get(f"mixer_corr_phase_offset_degree")

        # get parameter bounds
        offset_i_bounds = drive_module.__getattr__(
            f"out{drive_port_number}_offset_path0"
        ).vals.valid_values
        offset_q_bounds = drive_module.__getattr__(
            f"out{drive_port_number}_offset_path1"
        ).vals.valid_values
        gain_ratio_bounds = drive_sequencer.__getattr__(
            f"mixer_corr_gain_ratio"
        ).vals.valid_values
        phase_offset_bounds = drive_sequencer.__getattr__(
            f"mixer_corr_phase_offset_degree"
        ).vals.valid_values

        # calculate parameter ranges
        n_points = params.n_points
        offset_i_range = np.linspace(offset_i_bounds[0], offset_i_bounds[1], n_points)
        offset_q_range = np.linspace(offset_q_bounds[0], offset_q_bounds[1], n_points)
        # gain_ratio_range = np.linspace(gain_ratio_bounds[0], gain_ratio_bounds[1], n_points)
        gain_ratio_range = 2 ** (np.linspace(-1, 1, n_points))
        phase_offset_range = np.linspace(
            phase_offset_bounds[0], phase_offset_bounds[1], n_points
        )

        # configure modules
        # readout module
        ro_pulse = platform.create_qubit_readout_pulse(qubit_id, start=0)
        waveforms = {
            f"{ro_pulse.envelope_waveform_i.serial}": {
                "data": ro_pulse.envelope_waveform_i.data.tolist(),
                "index": 0,
            },
            f"{ro_pulse.envelope_waveform_q.serial}": {
                "data": ro_pulse.envelope_waveform_q.data.tolist(),
                "index": 1,
            },
        }
        acquisitions = {f"{ro_pulse.serial}": {"num_bins": 1, "index": 0}}
        navgs = params.nshots
        seq_prog = f"""
            # setup
            move 0, R3                      # navgs loop
            nop
            loop_R3:
                wait_sync 4
                reset_ph
                play  0,1,{qubit.feedback.port.acquisition_hold_off} # play readout pulse
                acquire 0,0,{ro_pulse.duration}
                wait {qubit.feedback.port.acquisition_hold_off+params.relaxation_time}
            add R3, 1, R3
            nop
            jlt R3, {navgs}, @loop_R3       # navgs loop
            # cleaup
            stop
        """
        sequence = {
            "waveforms": waveforms,
            "weights": {},
            "acquisitions": acquisitions,
            "program": seq_prog,
        }
        readout_sequencer.sequence(sequence)
        readout_sequencer.nco_freq(readout_intermediate_frequency)

        # drive module
        drive_pulse = platform.create_RX_pulse(qubit_id, start=0)
        drive_sequencer.offset_awg_path0(1.0)  # 0
        drive_sequencer.offset_awg_path1(1.0)  # 0
        drive_sequencer.nco_freq(drive_intermediate_frequency)  # 0

        for module, sequencer_number in [
            (drive_module, drive_sequencer_number),
            (readout_module, readout_sequencer_number),
        ]:
            for s in range(6):
                module.sequencers[s].set("sync_en", value=False)
                if s == sequencer_number:
                    module.sequencers[s].set("marker_ovr_en", value=True)
                    module.sequencers[s].set("marker_ovr_value", value=15)
                    module.sequencers[s].set("mod_en_awg", value=True)

                else:
                    module.sequencers[s].set("marker_ovr_en", value=False)
                    module.sequencers[s].set("marker_ovr_value", value=0)
                    module.sequencers[s].set("mod_en_awg", value=False)

        ####################################################################################
        # leakage
        ####################################################################################

        leakage_data[qubit_id] = np.zeros((len(offset_q_range), len(offset_i_range)))
        qubit.drive.lo_frequency = (
            drive_lo_frequency + drive_intermediate_frequency
        )  # drive_frequency

        # control_module.print_readable_snapshot(update=True)
        # readout_module.print_readable_snapshot(update=True)

        for i, offset_i in enumerate(offset_i_range):
            drive_module.set(f"out{drive_port_number}_offset_path0", offset_i)
            log.info(
                f"qubit {qubit_id} leakage mitigation progress: {int(i/n_points*100)}%"
            )

            for q, offset_q in enumerate(offset_q_range):
                drive_module.set(f"out{drive_port_number}_offset_path1", offset_q)

                drive_module.arm_sequencer(drive_sequencer_number)
                readout_sequencer.delete_acquisition_data(all=True)
                readout_module.arm_sequencer(readout_sequencer_number)
                drive_module.start_sequencer(drive_sequencer_number)
                readout_module.start_sequencer(readout_sequencer_number)

                # wait until all sequencers stop
                for module in [drive_module, readout_module]:
                    for sequencer_number in range(len(module.sequencers)):
                        wait_for_sequencer_stop(module, sequencer_number)

                drive_module.stop_sequencer(drive_sequencer_number)
                readout_module.stop_sequencer(readout_sequencer_number)

                duration = qubit.feedback.port.acquisition_duration
                results = readout_module.get_acquisitions(readout_sequencer_number)

                _is = (
                    np.array(
                        results[ro_pulse.serial]["acquisition"]["bins"]["integration"][
                            "path0"
                        ]
                    )
                    / duration
                )
                _qs = (
                    np.array(
                        results[ro_pulse.serial]["acquisition"]["bins"]["integration"][
                            "path1"
                        ]
                    )
                    / duration
                )
                leakage_data[qubit_id][q, i] = np.sqrt(_is**2 + _qs**2)

        min_index = np.argmin(leakage_data[qubit_id])
        min_coords = np.unravel_index(min_index, leakage_data[qubit_id].shape)

        drive_module.set(
            f"out{drive_port_number}_offset_path0", offset_i_range[min_coords[1]]
        )
        drive_module.set(
            f"out{drive_port_number}_offset_path1", offset_q_range[min_coords[0]]
        )
        ####################################################################################
        # image
        ####################################################################################

        image_data[qubit_id] = np.zeros(
            (len(gain_ratio_range), len(phase_offset_range))
        )
        qubit.drive.lo_frequency = drive_lo_frequency + 2 * drive_intermediate_frequency

        # control_module.print_readable_snapshot(update=True)
        # readout_module.print_readable_snapshot(update=True)

        for g, gain_ratio in enumerate(gain_ratio_range):
            drive_sequencer.set("mixer_corr_gain_ratio", gain_ratio)
            log.info(
                f"qubit {qubit_id} image mitigation progress: {int(g/n_points*100)}%"
            )

            for p, phase_offset in enumerate(phase_offset_range):
                drive_sequencer.set("mixer_corr_phase_offset_degree", phase_offset)

                drive_module.arm_sequencer(drive_sequencer_number)
                readout_sequencer.delete_acquisition_data(all=True)
                readout_module.arm_sequencer(readout_sequencer_number)
                drive_module.start_sequencer(drive_sequencer_number)
                readout_module.start_sequencer(readout_sequencer_number)

                # wait until all sequencers stop
                for module in [drive_module, readout_module]:
                    for sequencer_number in range(len(module.sequencers)):
                        wait_for_sequencer_stop(module, sequencer_number)

                drive_module.stop_sequencer(drive_sequencer_number)
                readout_module.stop_sequencer(readout_sequencer_number)

                duration = qubit.feedback.port.acquisition_duration
                results = readout_module.get_acquisitions(readout_sequencer_number)

                _is = (
                    np.array(
                        results[ro_pulse.serial]["acquisition"]["bins"]["integration"][
                            "path0"
                        ]
                    )
                    / duration
                )
                _qs = (
                    np.array(
                        results[ro_pulse.serial]["acquisition"]["bins"]["integration"][
                            "path1"
                        ]
                    )
                    / duration
                )
                image_data[qubit_id][p, g] = np.sqrt(_is**2 + _qs**2)

        ####################################################################################

        qubit.drive.lo_frequency = drive_lo_frequency
        drive_sequencer.offset_awg_path0(0)  # 0
        drive_sequencer.offset_awg_path1(0)  # 0
        drive_sequencer.nco_freq(drive_intermediate_frequency)  # 0

    for qubit_id in qubits:
        qubit: Qubit = qubits[qubit_id]
        flux_module = qubit.flux.port.module.device
        flux_port_number = qubit.flux.port.port_number
        # flux module
        flux_module.set(f"out{flux_port_number}_offset", value=0)

        # module.print_readable_snapshot(update=True)

    data = MixerCalibrationData(resonator_type=platform.resonator_type)

    for qubit_id in qubits:
        data.register_qubit(
            qubit_id,
            offset_i_range=offset_i_range,
            offset_q_range=offset_q_range,
            gain_ratio_range=gain_ratio_range,
            phase_offset_range=phase_offset_range,
            leakage_data=leakage_data[qubit_id],
            image_data=image_data[qubit_id],
        )
    qubit.drive.port.module._cluster.device.reset()
    return data


def _fit(data: MixerCalibrationData) -> MixerCalibrationResults:
    """
    Post-processing for Mixer Calibration Experiment.
    """

    return MixerCalibrationResults(None, None)


def _plot(data: MixerCalibrationData, fit: MixerCalibrationResults, qubit):
    """Plotting function for MixerCalibration Experiment."""

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "Leakage",
            "Image",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=data.data[qubit][0],
            zmin=np.min(data.data[qubit]),
            zmax=np.max(data.data[qubit]),
            colorscale="Viridis",
            x=data.offset_i_range[qubit],
            y=data.offset_q_range[qubit],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=data.data[qubit][1],
            zmin=np.min(data.data[qubit]),
            zmax=np.max(data.data[qubit]),
            colorscale="Viridis",
            showscale=False,
            x=data.gain_ratio_range[qubit],
            y=data.phase_offset_range[qubit],
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=True,
        xaxis_title="offset_i",
        yaxis_title="offset_q",
        xaxis2_title="gain_ratio",
        yaxis2_title="phase_offset",
        xaxis2_type="log",
        xaxis2_exponentformat="power",
    )

    figures = [fig]
    fitting_report = ""

    return figures, fitting_report


def _update(results: MixerCalibrationResults, platform: Platform, qubit: QubitId):
    # update.bare_resonator_frequency_sweetspot(results.brf[qubit], platform, qubit)
    # update.readout_frequency(results.frequency[qubit], platform, qubit)
    # update.flux_to_bias(results.flux_to_bias[qubit], platform, qubit)
    # update.asymmetry(results.asymmetry[qubit], platform, qubit)
    # update.ratio_sweetspot_qubit_freq_bare_resonator_freq(
    #     results.ssf_brf[qubit], platform, qubit
    # )
    # update.charging_energy(results.ECs[qubit], platform, qubit)
    # update.josephson_energy(results.EJs[qubit], platform, qubit)
    # update.coupling(results.Gs[qubit], platform, qubit)
    pass


mixer_calibration = Routine(_acquisition, _fit, _plot, _update)
"""MixerCalibration Routine object."""


# from __future__ import annotations

# import json
# from typing import TYPE_CHECKING, Callable

# import ipywidgets as widgets
# from ipywidgets import interact
# from qblox_instruments import Cluster, ClusterType
# from qcodes.instrument import find_or_create_instrument

# if TYPE_CHECKING:
#     from qblox_instruments.qcodes_drivers.module import QcmQrm

# cluster_ip = "192.168.0.3"
# cluster_name = "cluster3"

# cluster = find_or_create_instrument(
#     Cluster,
#     recreate=True,
#     name=cluster_name,
#     identifier=cluster_ip,
#     dummy_cfg=(
#         {
#             2: ClusterType.CLUSTER_QCM,
#             4: ClusterType.CLUSTER_QCM,
#             8: ClusterType.CLUSTER_QCM_RF,
#             16: ClusterType.CLUSTER_QRM_RF,
#             18: ClusterType.CLUSTER_QRM_RF,
#         }
#         if cluster_ip is None
#         else None
#     ),
# )


# def get_connected_modules(
#     cluster: Cluster, filter_fn: Callable | None = None
# ) -> dict[int, QcmQrm]:
#     def checked_filter_fn(mod: ClusterType) -> bool:
#         if filter_fn is not None:
#             return filter_fn(mod)
#         return True

#     return {
#         mod.slot_idx: mod
#         for mod in cluster.modules
#         if mod.present() and checked_filter_fn(mod)
#     }


# # RF modules
# modules = get_connected_modules(cluster, lambda mod: mod.is_rf_type)
# modules

# module = modules[8]

# cluster.reset()
# print(cluster.get_system_state())

# module.print_readable_snapshot(update=True)

# # Program sequence we will not use.
# sequence = {"waveforms": {}, "weights": {}, "acquisitions": {}, "program": "stop"}
# with open("sequence.json", "w", encoding="utf-8") as file:
#     json.dump(sequence, file, indent=4)
#     file.close()
# module.sequencer0.sequence(sequence)

# # Program fullscale DC offset on I & Q, turn on NCO and enable modulation.
# module.sequencer0.marker_ovr_en = True
# module.sequencer0.marker_ovr_value = 15
# module.sequencer0.offset_awg_path0(1.0)
# module.sequencer0.offset_awg_path1(1.0)
# module.sequencer0.nco_freq(200e6)
# module.sequencer0.mod_en_awg(True)


# def set_offset_I(offset_I: float) -> None:
#     module.out0_offset_path0(offset_I)
#     module.arm_sequencer(0)
#     module.start_sequencer(0)


# def set_offset_Q(offset_Q: float) -> None:
#     module.out0_offset_path1(offset_Q)
#     module.arm_sequencer(0)
#     module.start_sequencer(0)


# def set_gain_ratio(gain_ratio: float) -> None:
#     module.sequencer0.mixer_corr_gain_ratio(gain_ratio)
#     module.arm_sequencer(0)
#     module.start_sequencer(0)


# def set_phase_offset(phase_offset: float) -> None:
#     module.sequencer0.mixer_corr_phase_offset_degree(phase_offset)
#     module.arm_sequencer(0)
#     module.start_sequencer(0)


# I_bounds = module.out0_offset_path0.vals.valid_values
# interact(
#     set_offset_I,
#     offset_I=widgets.FloatSlider(
#         min=I_bounds[0], max=I_bounds[1], step=0.01, value=0.0, description="Offset I:"
#     ),
# )

# Q_bounds = module.out0_offset_path1.vals.valid_values
# interact(
#     set_offset_Q,
#     offset_Q=widgets.FloatSlider(
#         min=Q_bounds[0], max=Q_bounds[1], step=0.01, value=0.0, description="Offset Q:"
#     ),
# )

# # The gain ratio correction is bounded between 1/2 and 2
# interact(
#     set_gain_ratio,
#     gain_ratio=widgets.FloatLogSlider(
#         min=-1, max=1, step=0.1, value=1.0, base=2, description="Gain ratio:"
#     ),
# )

# ph_bounds = module.sequencer0.mixer_corr_phase_offset_degree.vals.valid_values
# interact(
#     set_phase_offset,
#     phase_offset=widgets.FloatSlider(
#         min=ph_bounds[0],
#         max=ph_bounds[1],
#         step=1.0,
#         value=0.0,
#         description="Phase offset:",
#     ),
# )

# cluster.reset()
# print(cluster.get_system_state())
