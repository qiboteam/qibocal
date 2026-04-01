"""This function performs mixer calibration for QBlox RF modules by running the built-in
LO and sideband calibration routines. To be moved to the qblox-driver.
"""

from dataclasses import dataclass, field

import plotly.graph_objects as go
from qibolab._core.components.channels import AcquisitionChannel, IqChannel
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config import PortAddress
from qibolab._core.instruments.qblox.identifiers import SequencerMap

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

__all__ = ["calibrate_mixers"]


@dataclass
class ModuleCalibrationData:
    """Calibration data for a single QBlox module."""

    module_name: str
    """Module identifier."""
    offset_i: dict[int, float] = field(default_factory=dict)
    """I offset values for each output."""
    offset_q: dict[int, float] = field(default_factory=dict)
    """Q offset values for each output."""
    lo_freq: dict[int, float] = field(default_factory=dict)
    """LO frequencies per output."""
    gain_ratio: dict[int, dict[int, float]] = field(default_factory=dict)
    """Gain ratio corrections per sequencer."""
    phase_offset: dict[int, dict[int, float]] = field(default_factory=dict)
    """Phase offset corrections per sequencer."""
    nco_freq: dict[int, dict[int, float]] = field(default_factory=dict)
    """NCO frequencies per sequencer."""

    @classmethod
    def from_dict(cls, data: "dict | ModuleCalibrationData") -> "ModuleCalibrationData":
        """Create ModuleCalibrationData from a dictionary.

        Args:
            data: Dictionary containing calibration data

        Returns:
            ModuleCalibrationData instance
        """
        if isinstance(data, ModuleCalibrationData):
            return data
        if not isinstance(data, dict):
            raise TypeError(
                f"Cannot create ModuleCalibrationData from type {type(data)}"
            )

        flat_fields = ["offset_i", "offset_q", "lo_freq"]
        nested_fields = ["gain_ratio", "phase_offset", "nco_freq"]

        nested_values = {
            field: {
                int(port): {int(seq): value for seq, value in seq_values.items()}
                for port, seq_values in data.get(field, {}).items()
            }
            for field in nested_fields
        }
        flat_values = {
            field: {int(port): value for port, value in data.get(field, {}).items()}
            for field in flat_fields
        }

        return cls(
            module_name=data["module_name"],
            offset_i=flat_values["offset_i"],
            offset_q=flat_values["offset_q"],
            lo_freq=flat_values["lo_freq"],
            gain_ratio=nested_values["gain_ratio"],
            phase_offset=nested_values["phase_offset"],
            nco_freq=nested_values["nco_freq"],
        )


@dataclass
class CalibrateMixersParameters(Parameters):
    """Calibrate mixers runcard inputs."""


@dataclass
class CalibrateMixersResults(Results):
    """Calibrate mixers outputs."""

    sequencer_map: SequencerMap = field(default_factory=dict)
    """Channel to sequencer assignment."""

    final_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Final calibration values after running calibration."""


@dataclass
class CalibrateMixersData(Data):
    """Calibrate mixers acquisition outputs."""

    sequencer_map: SequencerMap = field(default_factory=dict)
    """Channel to sequencer assignment."""

    initial_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Initial calibration values before running calibration."""

    final_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Final calibration values after running calibration."""


def _get_hardware_calibration(
    cluster: Cluster, seq_map: SequencerMap
) -> dict[str, ModuleCalibrationData]:
    """Read the calibration values from the cluster."""

    modules = cluster.cluster.get_connected_modules(
        lambda mod: mod.is_rf_type
    )  # Make a list of RF modules
    data: dict[str, ModuleCalibrationData] = {}

    # Iterate over seq_map to get channels and their assigned sequencers
    for slot_id, channels in seq_map.items():
        module = modules.get(slot_id)
        if module is None:
            # Skip if the module for this slot is not found (e.g. non-RF module)
            continue
        mod_name = module.short_name
        mod_data = ModuleCalibrationData(mod_name)

        for ch_name, seq_id in channels.items():
            address = PortAddress.from_path(cluster.channels[ch_name].path)
            output = address.ports[0] - 1

            # Only set offset and LO freq once per output (not per sequencer)
            if output not in mod_data.offset_i:
                mod_data.offset_i[output] = getattr(
                    module, f"out{output}_offset_path0"
                )()
                mod_data.offset_q[output] = getattr(
                    module, f"out{output}_offset_path1"
                )()
                mod_data.lo_freq[output] = getattr(
                    module,
                    f"out{output}_lo_freq"
                    if module.is_qcm_type
                    else f"out{output}_in{output}_lo_freq",
                )()

            if output not in mod_data.gain_ratio:
                mod_data.gain_ratio[output] = {}
                mod_data.phase_offset[output] = {}
                mod_data.nco_freq[output] = {}

            # Use the sequencer ID from seq_map
            seq = getattr(module, f"sequencer{seq_id}")
            mod_data.gain_ratio[output][seq_id] = seq.mixer_corr_gain_ratio()
            mod_data.phase_offset[output][seq_id] = seq.mixer_corr_phase_offset_degree()
            mod_data.nco_freq[output][seq_id] = seq.nco_freq()

        data[mod_name] = mod_data

    return data


def _acquisition(
    params: CalibrateMixersParameters,
    platform: CalibrationPlatform,
    targets: list[str],
) -> CalibrateMixersData:
    """Data acquisition for mixer calibration.

    This routine calibrates the IQ mixer offsets, gain ratios, and phase offsets
    for all QBlox RF modules in the platform. This is currently done specifically for
    QBlox electronics but could be included as part of qibolab to make it more general.

    Args:
        params: Input parameters (currently empty)
        platform: Qibolab's calibration platform
        targets: List of target qubits (not used, calibrates all mixers)

    Returns:
        CalibrateMixersData with initial and final calibration values
    """

    data = CalibrateMixersData()

    clusters = [
        instrument
        for instrument in platform.instruments.values()
        if isinstance(instrument, Cluster)
    ]
    assert len(clusters) == 1, (
        "This protocol only works for platforms with exactly one qblox Cluster as controller."
    )
    cluster = clusters[0]
    configs = platform.parameters.configs.copy()

    # Setup one sequencer per channel with a dummy sequence
    # TODO: Optimize by directly using Cluster._channels_by_module to assign
    # one sequencer per output port instead of calling configure(). This would
    # allow sequential calibration of individual channel mixers.
    seq_map, _ = cluster.configure(
        configs=configs,
    )

    modules = cluster.cluster.get_connected_modules(lambda mod: mod.is_rf_type)

    # Read current hardware calibration values.
    initial_calibration = _get_hardware_calibration(cluster, seq_map)

    # Perform calibration
    # seq_map is of shape {module_id: {ch_name: seq_id}}, so the outer loop is over
    # modules
    for channels in seq_map.values():
        for ch_id, seq_id in channels.items():
            address = PortAddress.from_path(cluster.channels[ch_id].path)
            module = modules.get(address.slot)
            if module is None:
                # Skip if the module for this channel is not found (e.g. non-RF module)
                continue
            port = address.ports[0]

            # Run LO calibration
            output = port - 1
            qblox_function = (
                f"out{output}_lo_cal"
                if module.is_qcm_type
                else f"out{output}_in{output}_lo_cal"
            )
            getattr(module, qblox_function)()

            # Run sideband calibration
            sequencer = getattr(module, f"sequencer{seq_id}")
            if all(
                sequencer._get_sequencer_connect_out(i) == "off"
                for i in range(2 if module.is_qcm_type else 1)
            ):
                # If no port is assigned to the sequencer, connect it to the current
                # output
                sequencer.connect_sequencer(f"out{port - 1}")
            sequencer.sideband_cal()

    final_calibration = _get_hardware_calibration(cluster, seq_map)

    data = CalibrateMixersData(
        sequencer_map=dict(seq_map),  # outer defaultdict -> dict for JSON serialization
        initial_calibration=initial_calibration,
        final_calibration=final_calibration,
    )

    return data


def _fit(data: CalibrateMixersData) -> CalibrateMixersResults:
    """Post-processing function for mixer calibration.

    No fitting is performed for this routine.

    Args:
        data: Acquisition data

    Returns:
        Empty results
    """
    return CalibrateMixersResults(
        sequencer_map=data.sequencer_map,
        final_calibration=data.final_calibration,
    )


def _plot(data: CalibrateMixersData, target: QubitId, fit: CalibrateMixersResults):
    """Plotting function for mixer calibration.

    Creates a table showing initial and final calibration values for all modules.

    Args:
        data: Acquisition data with calibration values
        target: Target qubit (not used, shows all modules)
        fit: Fit results (not used)

    Returns:
        Tuple of (list of figures, HTML report)
    """
    figures = []
    # Create a comprehensive table with all calibration data
    table_rows = []

    initial_calibration, final_calibration = (
        data.initial_calibration,
        data.final_calibration,
    )
    for module_key in sorted(initial_calibration):
        initial = initial_calibration[module_key]
        final = final_calibration[module_key]

        # Need this when loading from JSON
        if isinstance(initial, ModuleCalibrationData) or isinstance(
            final, ModuleCalibrationData
        ):
            initial = ModuleCalibrationData.from_dict(initial)
            final = ModuleCalibrationData.from_dict(final)

        # Add module header
        table_rows.append(
            [
                f"<b>{initial.module_name}</b>",
                "<b>Initial</b>",
                "<b>Final</b>",
                "<b>Change</b>",
            ]
        )

        # Add offset data for each output
        for port in initial.offset_i.keys():
            offset_i_change = final.offset_i[port] - initial.offset_i[port]
            offset_q_change = final.offset_q[port] - initial.offset_q[port]

            table_rows.append(
                [
                    f"Out{port} Offset I",
                    f"{initial.offset_i[port]:.6f}",
                    f"{final.offset_i[port]:.6f}",
                    f"{offset_i_change:.6f}",
                ]
            )
            table_rows.append(
                [
                    f"Out{port} Offset Q",
                    f"{initial.offset_q[port]:.6f}",
                    f"{final.offset_q[port]:.6f}",
                    f"{offset_q_change:.6f}",
                ]
            )

            # Add LO frequency
            lo_freq_change = final.lo_freq[port] - initial.lo_freq[port]
            table_rows.append(
                [
                    f"Out{port} LO Freq (Hz)",
                    f"{initial.lo_freq[port]:.0f}",
                    f"{final.lo_freq[port]:.0f}",
                    f"{lo_freq_change:.0f}",
                ]
            )

            # Add sequencer-specific data
            if port in initial.gain_ratio:
                for seq_idx in initial.gain_ratio[port]:
                    gain_change = (
                        final.gain_ratio[port][seq_idx]
                        - initial.gain_ratio[port][seq_idx]
                    )
                    phase_change = (
                        final.phase_offset[port][seq_idx]
                        - initial.phase_offset[port][seq_idx]
                    )
                    nco_change = (
                        final.nco_freq[port][seq_idx] - initial.nco_freq[port][seq_idx]
                    )

                    table_rows.append(
                        [
                            f"  Seq{seq_idx} Gain Ratio",
                            f"{initial.gain_ratio[port][seq_idx]:.6f}",
                            f"{final.gain_ratio[port][seq_idx]:.6f}",
                            f"{gain_change:.6f}",
                        ]
                    )
                    table_rows.append(
                        [
                            f"  Seq{seq_idx} Phase Offset (°)",
                            f"{initial.phase_offset[port][seq_idx]:.4f}",
                            f"{final.phase_offset[port][seq_idx]:.4f}",
                            f"{phase_change:.4f}",
                        ]
                    )
                    table_rows.append(
                        [
                            f"  Seq{seq_idx} NCO Freq (Hz)",
                            f"{initial.nco_freq[port][seq_idx]:.0f}",
                            f"{final.nco_freq[port][seq_idx]:.0f}",
                            f"{nco_change:.0f}",
                        ]
                    )

        # Add separator row
        table_rows.append(["", "", "", ""])

    # Create plotly table
    if table_rows:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=[
                            "<b>Parameter</b>",
                            "<b>Initial</b>",
                            "<b>Final</b>",
                            "<b>Change</b>",
                        ],
                        align="left",
                        fill_color="gray",
                        font_color="white",
                        font=dict(size=12),
                    ),
                    cells=dict(
                        values=list(zip(*table_rows)), align="left", font=dict(size=11)
                    ),
                )
            ]
        )

        fig.update_layout(
            title="QBlox Mixer Calibration Results",
            height=max(400, len(table_rows) * 25),
        )

        figures.append(fig)

    fitting_report = "<h3>Mixer Calibration Complete</h3>"
    fitting_report += f"<p>Calibrated {len(data.initial_calibration)} module(s)</p>"

    return figures, fitting_report


def _update(
    results: CalibrateMixersResults, platform: CalibrationPlatform, qubit: QubitId
):
    """Update platform parameters with final calibration values.

    Args:
        results: Fit results
        platform: Calibration platform to update
        qubit: Qubit identifier (unused)
    """
    final_cal = results.final_calibration

    clusters = [
        instrument
        for instrument in platform.instruments.values()
        if isinstance(instrument, Cluster)
    ]
    assert len(clusters) == 1, "Exactly one cluster is required."
    cluster = clusters[0]

    channels_by_module: dict = (
        cluster._channels_by_module
    )  # _channels_by_module is not intended as public
    for slot, channels in channels_by_module.items():
        for ch_id, address in channels:
            mod_name = cluster._modules[
                slot
            ].short_name  # _modules is not intended as public
            if mod_name not in final_cal:
                continue  # Skip if no calibration data for this module

            ch = cluster.channels[ch_id]
            if isinstance(ch, AcquisitionChannel):
                # The mixer relevant for an acquisition channel is the one associated to
                # the corresponding probe channel
                probe_channel_id = ch.probe
                assert probe_channel_id is not None
                ch = cluster.channels[probe_channel_id]
            cal = final_cal[mod_name]

            # Update platform parameters with new calibration values
            port = address.ports[0] - 1
            seq_id = results.sequencer_map[slot][ch_id]
            assert isinstance(ch, IqChannel)
            platform.update(
                {
                    f"configs.{ch.mixer}.offset_i": cal.offset_i[port],
                    f"configs.{ch.mixer}.offset_q": cal.offset_q[port],
                    f"configs.{ch.mixer}.scale_q": cal.gain_ratio[port][seq_id],
                    f"configs.{ch.mixer}.phase_q": cal.phase_offset[port][seq_id],
                }
            )


calibrate_mixers = Routine(_acquisition, _fit, _plot, _update)
"""Calibrate mixers Routine object."""
