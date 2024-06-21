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

        # get qubit readout frequencies, modules and sequencers

        readout_lo_frequency = qubit.readout.lo_frequency
        readout_frequency = qubit.readout_frequency
        readout_intermediate_frequency = readout_frequency - readout_lo_frequency
        readout_module = qubit.readout.port.module.device
        readout_port_number = qubit.readout.port.port_number
        readout_sequencer_number = qubit.readout.port.sequencer_number
        readout_sequencer = readout_module.sequencers[readout_port_number]

        # save current values
        # current_offset_i = readout_module.get(f"out{readout_port_number}_offset_path0")
        # current_offset_q = readout_module.get(f"out{readout_port_number}_offset_path1")
        # current_gain_ratio = readout_sequencer.get(f"mixer_corr_gain_ratio")
        # current_phase_offset = readout_sequencer.get(f"mixer_corr_phase_offset_degree")

        zoom = 5

        # get parameter bounds
        offset_i_bounds = readout_module.__getattr__(
            f"out{readout_port_number}_offset_path0"
        ).vals.valid_values
        offset_q_bounds = readout_module.__getattr__(
            f"out{readout_port_number}_offset_path1"
        ).vals.valid_values
        gain_ratio_bounds = readout_sequencer.__getattr__(
            f"mixer_corr_gain_ratio"
        ).vals.valid_values
        phase_offset_bounds = readout_sequencer.__getattr__(
            f"mixer_corr_phase_offset_degree"
        ).vals.valid_values

        # calculate parameter ranges
        n_points = params.n_points
        offset_i_range = (
            np.linspace(offset_i_bounds[0], offset_i_bounds[1], n_points) / zoom
        )
        offset_q_range = (
            np.linspace(offset_q_bounds[0], offset_q_bounds[1], n_points) / zoom
        )
        # gain_ratio_range = np.linspace(gain_ratio_bounds[0], gain_ratio_bounds[1], n_points)
        gain_ratio_range = 2 ** (np.linspace(-1, 1, n_points))
        phase_offset_range = np.linspace(
            phase_offset_bounds[0], phase_offset_bounds[1], n_points
        )

        # configure modules
        # readout module
        ro_pulse = platform.create_qubit_readout_pulse(qubit_id, start=0)
        ro_pulse.amplitude *= 0.5
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

        # # drive module
        # drive_pulse = platform.create_RX_pulse(qubit_id, start=0)
        # drive_sequencer.offset_awg_path0(1.0)  # 0
        # drive_sequencer.offset_awg_path1(1.0)  # 0
        # drive_sequencer.nco_freq(drive_intermediate_frequency)  # 0

        module = readout_module
        sequencer_number = readout_sequencer_number
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
        qubit.readout.lo_frequency = readout_lo_frequency

        # readout_module.print_readable_snapshot(update=True)

        for i, offset_i in enumerate(offset_i_range):
            readout_module.set(f"out{readout_port_number}_offset_path0", offset_i)
            log.info(
                f"qubit {qubit_id} leakage mitigation progress: {int(i/n_points*100)}%"
            )

            for q, offset_q in enumerate(offset_q_range):
                readout_module.set(f"out{readout_port_number}_offset_path1", offset_q)

                readout_sequencer.delete_acquisition_data(all=True)
                readout_module.arm_sequencer(readout_sequencer_number)
                readout_module.start_sequencer(readout_sequencer_number)

                # wait until all sequencers stop
                module = readout_module
                for sequencer_number in range(len(module.sequencers)):
                    wait_for_sequencer_stop(module, sequencer_number)

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

        readout_module.set(
            f"out{readout_port_number}_offset_path0", offset_i_range[min_coords[1]]
        )
        readout_module.set(
            f"out{readout_port_number}_offset_path1", offset_q_range[min_coords[0]]
        )

        # ####################################################################################
        # # image
        # ####################################################################################

        image_data[qubit_id] = np.ones(
            (len(gain_ratio_range), len(phase_offset_range))
        ) * np.average(leakage_data[qubit_id])
        # qubit.drive.lo_frequency = drive_lo_frequency + 2 * drive_intermediate_frequency

        # # control_module.print_readable_snapshot(update=True)
        # # readout_module.print_readable_snapshot(update=True)

        # for g, gain_ratio in enumerate(gain_ratio_range):
        #     drive_sequencer.set("mixer_corr_gain_ratio", gain_ratio)
        #     log.info(
        #         f"qubit {qubit_id} image mitigation progress: {int(g/n_points*100)}%"
        #     )

        #     for p, phase_offset in enumerate(phase_offset_range):
        #         drive_sequencer.set("mixer_corr_phase_offset_degree", phase_offset)

        #         drive_module.arm_sequencer(drive_sequencer_number)
        #         readout_sequencer.delete_acquisition_data(all=True)
        #         readout_module.arm_sequencer(readout_sequencer_number)
        #         drive_module.start_sequencer(drive_sequencer_number)
        #         readout_module.start_sequencer(readout_sequencer_number)

        #         # wait until all sequencers stop
        #         for module in [drive_module, readout_module]:
        #             for sequencer_number in range(len(module.sequencers)):
        #                 wait_for_sequencer_stop(module, sequencer_number)

        #         drive_module.stop_sequencer(drive_sequencer_number)
        #         readout_module.stop_sequencer(readout_sequencer_number)

        #         duration = qubit.feedback.port.acquisition_duration
        #         results = readout_module.get_acquisitions(readout_sequencer_number)

        #         _is = (
        #             np.array(
        #                 results[ro_pulse.serial]["acquisition"]["bins"]["integration"][
        #                     "path0"
        #                 ]
        #             )
        #             / duration
        #         )
        #         _qs = (
        #             np.array(
        #                 results[ro_pulse.serial]["acquisition"]["bins"]["integration"][
        #                     "path1"
        #                 ]
        #             )
        #             / duration
        #         )
        #         image_data[qubit_id][p, g] = np.sqrt(_is**2 + _qs**2)

        # ####################################################################################

        # qubit.drive.lo_frequency = drive_lo_frequency
        # drive_sequencer.offset_awg_path0(0)  # 0
        # drive_sequencer.offset_awg_path1(0)  # 0
        # drive_sequencer.nco_freq(drive_intermediate_frequency)  # 0

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
    qubit.readout.port.module._cluster.device.reset()
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


resonator_mixer_calibration = Routine(_acquisition, _fit, _plot, _update)
"""MixerCalibration Routine object."""
