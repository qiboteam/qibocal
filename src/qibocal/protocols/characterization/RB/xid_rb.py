from dataclasses import dataclass
from typing import Iterable, Union

from qibo.gates import X
from qibo.noise import NoiseModel

from qibocal.auto.operation import Routine
from qibocal.calibrations.niGSC.XIdrb import ModuleFactory as XIdScan
from qibocal.protocols.characterization.RB.result import DecayResult, get_hists_data
from qibocal.protocols.characterization.RB.utils import extract_from_data

from .data import RBData
from .params import RBParameters

NoneType = type(None)


@dataclass
class XIdResult(DecayResult):
    pass


def setup_scan(params: RBParameters) -> Iterable:
    return XIdScan(params.nqubits, params.depths * params.niter, params.qubits)


def execute(
    scan: Iterable,
    nshots: Union[int, NoneType] = None,
    noise_model: Union[NoiseModel, NoneType] = None,
):
    # Execute
    data_list = []
    for c in scan:
        depth = c.depth
        nx = len(c.gates_of_type("x"))
        # nx = c.gate_types[X]
        if noise_model is not None:
            c = noise_model.apply(c)
        samples = c.execute(nshots=nshots).samples()
        data_list.append({"depth": depth, "samples": samples, "nx": nx})
    return data_list


def aggregate(data: RBData):
    def filter(nx_list, samples_list):
        return [
            sum((-1) ** (nx % 2 + s[0]) / 2.0 for s in samples) / len(samples_list[0])
            for nx, samples in zip(nx_list, samples_list)
        ]

    data_agg = data.assign(signal=lambda x: filter(x.nx.to_list(), x.samples.to_list()))
    # Histogram
    hists = get_hists_data(data_agg)
    # Build the result object
    return XIdResult(
        *extract_from_data(data_agg, "signal", "depth", "mean"),
        hists=hists,
        meta_data=data.attrs,
    )


def acquire(params: RBParameters, *args) -> RBData:
    scan = setup_scan(params)
    if params.noise_model:
        from qibocal.calibrations.niGSC.basics import noisemodels

        noise_model = getattr(noisemodels, params.noise_model)(*params.noise_params)
    else:
        noise_model = None
    # Here the platform can be connected
    data = execute(scan, params.nshots, noise_model)
    xid_data = RBData(data)
    xid_data.attrs = params.__dict__
    return xid_data


def extract(data: RBData):
    result = aggregate(data)
    result.fit()
    return result


def plot(data: RBData, result: XIdResult, *args):
    table_str = "".join(
        [f" | {key}: {value}<br>" for key, value in {**result.meta_data}.items()]
    )
    fig = result.plot()
    return [fig], table_str


xid_rb = Routine(acquire, extract, plot)
