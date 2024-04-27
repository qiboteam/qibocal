from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

import ipywidgets as widgets
from ipywidgets import interact
from qblox_instruments import Cluster, ClusterType
from qcodes.instrument import find_or_create_instrument

if TYPE_CHECKING:
    from qblox_instruments.qcodes_drivers.module import QcmQrm


cluster_ip = "192.168.0.3"
cluster_name = "cluster3"

cluster = find_or_create_instrument(
    Cluster,
    recreate=True,
    name=cluster_name,
    identifier=cluster_ip,
    dummy_cfg=(
        {
            2: ClusterType.CLUSTER_QCM,
            4: ClusterType.CLUSTER_QCM,
            8: ClusterType.CLUSTER_QCM_RF,
            16: ClusterType.CLUSTER_QRM_RF,
            18: ClusterType.CLUSTER_QRM_RF,
        }
        if cluster_ip is None
        else None
    ),
)


def get_connected_modules(
    cluster: Cluster, filter_fn: Callable | None = None
) -> dict[int, QcmQrm]:
    def checked_filter_fn(mod: ClusterType) -> bool:
        if filter_fn is not None:
            return filter_fn(mod)
        return True

    return {
        mod.slot_idx: mod
        for mod in cluster.modules
        if mod.present() and checked_filter_fn(mod)
    }


# RF modules
modules = get_connected_modules(cluster, lambda mod: mod.is_rf_type)
modules

module = modules[8]

cluster.reset()
print(cluster.get_system_state())

module.print_readable_snapshot(update=True)

# Program sequence we will not use.
sequence = {"waveforms": {}, "weights": {}, "acquisitions": {}, "program": "stop"}
with open("sequence.json", "w", encoding="utf-8") as file:
    json.dump(sequence, file, indent=4)
    file.close()
module.sequencer0.sequence(sequence)

# Program fullscale DC offset on I & Q, turn on NCO and enable modulation.
module.sequencer0.marker_ovr_en = True
module.sequencer0.marker_ovr_value = 15
module.sequencer0.offset_awg_path0(1.0)
module.sequencer0.offset_awg_path1(1.0)
module.sequencer0.nco_freq(200e6)
module.sequencer0.mod_en_awg(True)


def set_offset_I(offset_I: float) -> None:
    module.out0_offset_path0(offset_I)
    module.arm_sequencer(0)
    module.start_sequencer(0)


def set_offset_Q(offset_Q: float) -> None:
    module.out0_offset_path1(offset_Q)
    module.arm_sequencer(0)
    module.start_sequencer(0)


def set_gain_ratio(gain_ratio: float) -> None:
    module.sequencer0.mixer_corr_gain_ratio(gain_ratio)
    module.arm_sequencer(0)
    module.start_sequencer(0)


def set_phase_offset(phase_offset: float) -> None:
    module.sequencer0.mixer_corr_phase_offset_degree(phase_offset)
    module.arm_sequencer(0)
    module.start_sequencer(0)


I_bounds = module.out0_offset_path0.vals.valid_values
interact(
    set_offset_I,
    offset_I=widgets.FloatSlider(
        min=I_bounds[0], max=I_bounds[1], step=0.01, value=0.0, description="Offset I:"
    ),
)

Q_bounds = module.out0_offset_path1.vals.valid_values
interact(
    set_offset_Q,
    offset_Q=widgets.FloatSlider(
        min=Q_bounds[0], max=Q_bounds[1], step=0.01, value=0.0, description="Offset Q:"
    ),
)

# The gain ratio correction is bounded between 1/2 and 2
interact(
    set_gain_ratio,
    gain_ratio=widgets.FloatLogSlider(
        min=-1, max=1, step=0.1, value=1.0, base=2, description="Gain ratio:"
    ),
)

ph_bounds = module.sequencer0.mixer_corr_phase_offset_degree.vals.valid_values
interact(
    set_phase_offset,
    phase_offset=widgets.FloatSlider(
        min=ph_bounds[0],
        max=ph_bounds[1],
        step=1.0,
        value=0.0,
        description="Phase offset:",
    ),
)

cluster.reset()
print(cluster.get_system_state())
