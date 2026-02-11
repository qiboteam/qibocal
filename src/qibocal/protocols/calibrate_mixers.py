from dataclasses import dataclass, field

import plotly.graph_objects as go
from qblox_instruments import Module, Sequencer
from qibolab._core.components.channels import AcquisitionChannel, IqChannel
from qibolab._core.instruments.abstract import Controller
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config import PortAddress
from qibolab._core.instruments.qblox.identifiers import SequencerMap

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

PortAddress.prefix = lambda self, is_qcm: f"out{self.ports[0]-1}" if is_qcm else f"out{self.ports[0]-1}_in{self.ports[0]-1}"
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
    def from_dict(cls, data: dict) -> "ModuleCalibrationData":
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

        int_key_fields = [
            "gain_ratio",
            "phase_offset",
            "lo_freq",
            "nco_freq",
            "offset_i",
            "offset_q",
        ]
        kwargs = {
            field: {
                int(k): {int(kk): vv for kk, vv in v.items()}
                for k, v in data.get(field, {}).items()
            }
            for field in int_key_fields
        }
        kwargs["module_name"] = data["module_name"]

        return cls(**kwargs)


@dataclass
class CalibrateMixersParameters(Parameters):
    """Calibrate mixers runcard inputs."""


@dataclass
class CalibrateMixersResults(Results):
    """Calibrate mixers outputs."""

    final_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Final calibration values after running calibration."""

    # sequencer_map: dict[str, SequencerMap] = field(default_factory=dict)
    # """Map of module slots to channels and assigned sequencers."""


@dataclass
class CalibrateMixersData(Data):
    """Calibrate mixers acquisition outputs."""

    initial_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Initial calibration values before running calibration."""
    final_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Final calibration values after running calibration."""
    # sequencer_map: dict[str, SequencerMap] = field(default_factory=dict)
    # """Map of module slots to channels and assigned sequencers."""


# This is moslty just removing the repeated acquisiton channel information but may handle other instances of repeated mixer
def unique_channels(cluster) -> dict[str, IqChannel]:
    """Get unique channels from a liost of mixers (skipping duplicates from shared channels)."""
    unique_channels = {}
    for ch_name, mixer in cluster._mixers.items():
        if mixer not in unique_channels:
            unique_channels[mixer] = ch_name
    return unique_channels


def _get_hardware_calibration(
    cluster: Cluster, seq_map: SequencerMap
) -> dict[str, ModuleCalibrationData]:
    modules: dict[str, Module] = cluster._cluster.get_connected_modules(
        lambda mod: mod.is_rf_type
    )
    data: dict[str, ModuleCalibrationData] = {}

    # Iterate over seq_map to get channels and their assigned sequencers
    for slot_id, channels in seq_map.items():
        module = modules[slot_id]
        mod_name = module.short_name
        mod_data = data[mod_name] = ModuleCalibrationData(mod_name)

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
            seq: Sequencer = getattr(module, f"sequencer{seq_id}")
            data[mod_name].gain_ratio[output][seq_id] = seq.mixer_corr_gain_ratio()
            data[mod_name].phase_offset[output][seq_id] = (
                seq.mixer_corr_phase_offset_degree()
            )
            data[mod_name].nco_freq[output][seq_id] = seq.nco_freq()

    return data


def _acquisition(
    params: CalibrateMixersParameters,
    platform: CalibrationPlatform,
    targets: list[str],
) -> CalibrateMixersData:
    """Data acquisition for mixer calibration.

    This routine calibrates the IQ mixer offsets, gain ratios, and phase offsets
    for all QBlox RF modules in the platform. This is currently done specifically for
    QBlox electromnics but could be included as part of qibolab to make it more general.

    Args:
        params: Input parameters (currently empty)
        platform: Qibolab's calibration platform
        targets: List of target qubits (not used, calibrates all mixers)

    Returns:
        CalibrateMixersData with initial and final calibration values
    """
    data = CalibrateMixersData()

    instrs = {
        i: instr
        for i, instr in platform.instruments.items()
        if isinstance(instr, Controller)
    }
    assert len(instrs) <= 1, "Only one controller is supported at a time."
    configs = platform.parameters.configs.copy()

    # data.sequencer_map[targets[0]] = {}
    for instr_id, instr in instrs.items():
        cluster: Cluster = instr

        # Setup one sequencer per channel with a dummy sequence
        seq_map, _ = cluster.configure(
            configs=configs,
        )
        # data.sequencer_map[targets[0]] = seq_map
        modules: dict[str, Module] = cluster._cluster.get_connected_modules(
            lambda mod: mod.is_rf_type
        )

        # Read current hardware calibration values (should match those in parameters.json, this works only instr for one at the moment)
        data.initial_calibration[instr_id] = _get_hardware_calibration(cluster, seq_map)

        # Perform calibration
        for channels in seq_map.values():
            for ch_id, seq_id in channels.items():
                address = PortAddress.from_path(cluster.channels[ch_id].path)
                module = modules[address.slot]
                port = address.ports[0]
               
                # Run LO calibration
                getattr(module, f"{address.prefix(module.is_qcm_type)}_lo_cal")()               

                # Run sideband calibration
                sequencer: Sequencer = getattr(module, f"sequencer{seq_id}", None)
                if module.is_qcm_type:
                    if all(
                        sequencer._get_sequencer_connect_out(i) == "off"
                        for i in range(2)
                    ):
                        # If no port is assigned to the sequencer, connect it to the current output
                        getattr(module, f"sequencer{seq_id}").connect_sequencer(
                            f"out{port - 1}"
                        )
                else:  # is qrm_type
                    if sequencer._get_sequencer_connect_out(0) == "off":
                        getattr(module, f"sequencer{seq_id}").connect_sequencer(
                            f"out{port - 1}"
                        )
                getattr(module, f"sequencer{seq_id}").sideband_cal()

        data.final_calibration[instr_id] = _get_hardware_calibration(cluster, seq_map)
    return data


def _fit(data: CalibrateMixersData) -> CalibrateMixersResults:
    """Post-processing function for mixer calibration.

    No fitting is performed for this routine.

    Args:
        data: Acquisition data

    Returns:
        Empty results
    """
    return CalibrateMixersResults(data.final_calibration)


def _plot(data: CalibrateMixersData, target: str, fit: CalibrateMixersResults):
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

    inital_calibration, final_calibration = (
        data.initial_calibration,
        data.final_calibration,
    )
    instrs = list(inital_calibration.keys())

    for module_key in sorted(inital_calibration[instrs[0]].keys()):
        initial = inital_calibration[instrs[0]][module_key]
        final = final_calibration[instrs[0]][module_key]

        # Need this when loading from JSON
        if (
            type(initial) is not ModuleCalibrationData
            or type(final) is not ModuleCalibrationData
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
    fitting_report += (
        f"<p>Calibrated {len(data.initial_calibration[instrs[0]])} module(s)</p>"
    )

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

    cluster: Cluster = list(platform.instruments.values())[
        0
    ]  # Assuming only one controller/instrument
    seq_map, _ = cluster.configure(
        configs=platform.parameters.configs.copy()
    )  # I dont know iof its healthy to reconfigure here, also there is no certainty that the sequencerMap will be the same as during acquisition

    for instr in (
        final_cal.keys()
    ):  # Only one instriument supported at the moment, here for future
        for channels in seq_map.values():
            for ch_id, seq_id in channels.items():
                address = PortAddress.from_path(cluster.channels[ch_id].path)
                module = cluster._cluster.modules[
                    address.slot - 1
                ]  # I dont know why here I need to use -1 here but not before
                port = address.ports[0] - 1
                mod_name = module.short_name
                if mod_name not in final_cal[instr]:
                    continue  # Skip if no calibration data for this module

                ch = cluster.channels[ch_id]
                if type(ch) is AcquisitionChannel:
                    ch = cluster.channels[ch_id.replace("acquisition", "probe")]

                cal = final_cal[instr][mod_name]

                # Update platform parameters with new calibration values
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
