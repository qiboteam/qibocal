from dataclasses import dataclass, field

import plotly.graph_objects as go
from qblox_instruments import Module
from qibolab._core.components.channels import IqChannel
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.instruments.abstract import Controller
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.sequence import Q1Sequence

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

__all__ = ["calibrate_mixers"]

# Add helper methods to qblox_instruments Module
Module.number_of_channels = lambda self: 2 if self.is_qcm_type else 1


@dataclass
class ModuleCalibrationData:
    """Calibration data for a single QBlox module."""

    module_name: str
    """Module identifier."""
    offset_i: list[float]
    """I offset values for each output."""
    offset_q: list[float]
    """Q offset values for each output."""
    gain_ratio: dict[int, list[float]]
    """Gain ratio corrections per output and sequencer."""
    phase_offset: dict[int, list[float]]
    """Phase offset corrections per output and sequencer."""
    lo_freq: dict[int, float]
    """LO frequencies per output."""
    nco_freq: dict[int, list[float]]
    """NCO frequencies per output and sequencer."""

    @classmethod
    def from_dict(cls, data: dict) -> "ModuleCalibrationData":
        """Create ModuleCalibrationData from a dictionary.

        Args:
            data: Dictionary containing calibration data

        Returns:
            ModuleCalibrationData instance
        """
        if type(data) is ModuleCalibrationData:
            return data
        elif not isinstance(data, dict):
            raise TypeError(
                f"Cannot create ModuleCalibrationData from type {type(data)}"
            )

        # Convert string keys to int for nested dicts
        gain_ratio = {int(k): v for k, v in data.get("gain_ratio", {}).items()}
        phase_offset = {int(k): v for k, v in data.get("phase_offset", {}).items()}
        lo_freq = {int(k): v for k, v in data.get("lo_freq", {}).items()}
        nco_freq = {int(k): v for k, v in data.get("nco_freq", {}).items()}

        return cls(
            module_name=data["module_name"],
            offset_i=data["offset_i"],
            offset_q=data["offset_q"],
            gain_ratio=gain_ratio,
            phase_offset=phase_offset,
            lo_freq=lo_freq,
            nco_freq=nco_freq,
        )


@dataclass
class CalibrateMixersParameters(Parameters):
    """Calibrate mixers runcard inputs."""

    pass


@dataclass
class CalibrateMixersResults(Results):
    """Calibrate mixers outputs."""

    final_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Final calibration values after running calibration."""


@dataclass
class CalibrateMixersData(Data):
    """Calibrate mixers acquisition outputs."""

    initial_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Initial calibration values before running calibration."""
    final_calibration: dict[str, ModuleCalibrationData] = field(default_factory=dict)
    """Final calibration values after running calibration."""


def _get_hardware_calibration(module: Module, channels: dict) -> ModuleCalibrationData:
    """Get hardware calibration values from a QBlox module."""

    module_name = module._short_name
    offset_i, offset_q = [], []

    gain_ratio = {output_n: [] for output_n in range(module.number_of_channels())}
    phase_offset = {output_n: [] for output_n in range(module.number_of_channels())}
    lo_freq = {output_n: 0.0 for output_n in range(module.number_of_channels())}
    nco_freq = {output_n: [] for output_n in range(module.number_of_channels())}

    for output_n in range(module.number_of_channels()):
        offset_i.append(getattr(module, f"out{output_n}_offset_path0")())
        offset_q.append(getattr(module, f"out{output_n}_offset_path1")())
        output = (
            f"out{output_n}"
            if module.number_of_channels() > 1
            else f"out{output_n}_in{output_n}"
        )
        lo_freq[output_n] = getattr(module, f"{output}_lo_freq")()

        seq_list = []
        idx_seq = 0
        for _, ch in channels.items():
            path = module_name.lower() + f"/o{output_n + 1}"
            if f"module{ch.path}" == path:
                seq_list.append(f"sequencer{idx_seq}")
                idx_seq += 1

        gain_ratio[output_n] = [0.0] * len(seq_list)
        phase_offset[output_n] = [0.0] * len(seq_list)
        nco_freq[output_n] = [0.0] * len(seq_list)

        for idx_seq, seq in enumerate(seq_list):
            sequencer = getattr(module, seq)
            gain_ratio[output_n][idx_seq] = sequencer.mixer_corr_gain_ratio()
            phase_offset[output_n][idx_seq] = sequencer.mixer_corr_phase_offset_degree()
            nco_freq[output_n][idx_seq] = sequencer.nco_freq()

    return ModuleCalibrationData(
        module_name=module_name,
        offset_i=offset_i,
        offset_q=offset_q,
        gain_ratio=gain_ratio,
        phase_offset=phase_offset,
        lo_freq=lo_freq,
        nco_freq=nco_freq,
    )


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
    configs = platform.parameters.configs.copy()

    for _, instr in instrs.items():
        if not isinstance(instr, Controller):
            continue  # Skip TWPA
        cluster: Cluster = instr

        # Get list of channels that are IqChannel
        channels = {
            chn: ch for chn, ch in cluster.channels.items() if isinstance(ch, IqChannel)
        }

        # Setup one sequencer per channel with a dummy sequence (not working, )
        seqs = cluster.configure(
            configs=configs,
            acquisition=AcquisitionType.RAW,
            sequences={ch: Q1Sequence.empty() for ch in channels.keys()},
        )

        modules: dict[str, Module] = cluster._cluster.get_connected_modules(
            lambda mod: mod.is_rf_type
        )

        # Read initial calibration values
        for module_idx, module in modules.items():
            initial_cal = _get_hardware_calibration(module, channels)
            data.initial_calibration[f"{cluster.name}_{module_idx}"] = initial_cal

        # Perform calibration
        for _, ch in channels.items():
            module_number = int(ch.path.split("/")[0])
            output_number = int(ch.path.split("/")[-1][-1])
            if module_number not in modules:
                continue
            module = modules[module_number]

            # Run LO calibration
            if module.is_qcm_type:
                getattr(module, f"out{output_number - 1}_lo_cal")()
            else:
                getattr(
                    module, f"out{output_number - 1}_in{output_number - 1}_lo_cal"
                )()

        # Read final calibration values
        for module_idx, module in modules.items():
            final_cal = _get_hardware_calibration(module, channels)
            data.final_calibration[f"{cluster.name}_{module_idx}"] = final_cal

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


def _plot(data: CalibrateMixersData, target: str, fit: CalibrateMixersResults = None):
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

    modules = data.initial_calibration.keys()

    for module_key in sorted(data.initial_calibration.keys()):
        initial = data.initial_calibration[module_key]
        final = data.final_calibration[module_key]

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
        for output_n in range(len(initial.offset_i)):
            offset_i_change = final.offset_i[output_n] - initial.offset_i[output_n]
            offset_q_change = final.offset_q[output_n] - initial.offset_q[output_n]

            table_rows.append(
                [
                    f"Out{output_n} Offset I",
                    f"{initial.offset_i[output_n]:.6f}",
                    f"{final.offset_i[output_n]:.6f}",
                    f"{offset_i_change:.6f}",
                ]
            )
            table_rows.append(
                [
                    f"Out{output_n} Offset Q",
                    f"{initial.offset_q[output_n]:.6f}",
                    f"{final.offset_q[output_n]:.6f}",
                    f"{offset_q_change:.6f}",
                ]
            )

            # Add LO frequency
            lo_freq_change = final.lo_freq[output_n] - initial.lo_freq[output_n]
            table_rows.append(
                [
                    f"Out{output_n} LO Freq (Hz)",
                    f"{initial.lo_freq[output_n]:.0f}",
                    f"{final.lo_freq[output_n]:.0f}",
                    f"{lo_freq_change:.0f}",
                ]
            )

            # Add sequencer-specific data
            if output_n in initial.gain_ratio:
                for seq_idx in range(len(initial.gain_ratio[output_n])):
                    gain_change = (
                        final.gain_ratio[output_n][seq_idx]
                        - initial.gain_ratio[output_n][seq_idx]
                    )
                    phase_change = (
                        final.phase_offset[output_n][seq_idx]
                        - initial.phase_offset[output_n][seq_idx]
                    )
                    nco_change = (
                        final.nco_freq[output_n][seq_idx]
                        - initial.nco_freq[output_n][seq_idx]
                    )

                    table_rows.append(
                        [
                            f"  Seq{seq_idx} Gain Ratio",
                            f"{initial.gain_ratio[output_n][seq_idx]:.6f}",
                            f"{final.gain_ratio[output_n][seq_idx]:.6f}",
                            f"{gain_change:.6f}",
                        ]
                    )
                    table_rows.append(
                        [
                            f"  Seq{seq_idx} Phase Offset (°)",
                            f"{initial.phase_offset[output_n][seq_idx]:.4f}",
                            f"{final.phase_offset[output_n][seq_idx]:.4f}",
                            f"{phase_change:.4f}",
                        ]
                    )
                    table_rows.append(
                        [
                            f"  Seq{seq_idx} NCO Freq (Hz)",
                            f"{initial.nco_freq[output_n][seq_idx]:.0f}",
                            f"{final.nco_freq[output_n][seq_idx]:.0f}",
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

    for _, instr in platform.instruments.items():
        if not isinstance(instr, Controller):
            continue  # Skip non-controller instruments (e.g. TWPA)
        cluster: Cluster = instr

        for ch_name, ch in cluster.channels.items():
            if type(ch) is not IqChannel:
                continue
            cal_key = platform.name + "_" + ch.path.split("/")[0]
            if cal_key not in results.final_calibration:
                continue
            else:
                cal = ModuleCalibrationData.from_dict(final_cal[cal_key])
                output = int(ch.path.split("/")[-1][-1]) - 1
                platform.update({f"configs.{ch.mixer}.offset_i": cal.offset_i[output]})
                platform.update({f"configs.{ch.mixer}.offset_q": cal.offset_q[output]})


calibrate_mixers = Routine(_acquisition, _fit, _plot, _update)
"""Calibrate mixers Routine object."""
