# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence
from qibolab.platforms.abstract import AbstractPlatform
from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def qubit_spectroscopy(
    platform: AbstractPlatform,
    qubit,
    fast_start,
    fast_end,
    fast_step,
    precision_start,
    precision_end,
    precision_step,
    software_averages,
    points=10,
):
    platform.reload_settings()
    
    #data = Dataset(quantities={"frequency": "Hz", "attenuation": "dB"})
    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]

    frequency_range = np.arange(fast_start, fast_end, fast_step) + qubit_frequency

    # FIXME: Waiting for Qblox platform to take care of that
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse.frequency
    )

    data = Dataset(name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0:
                yield data
            platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data.add(results)
            count += 1
    yield data

    if platform.resonator_type == "3D":
        qubit_frequency = data.df.frequency[data.df.MSR.index[data.df.MSR.argmin()]].magnitude
        avg_voltage = (
            np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
        )
    else:
        qubit_frequency = data.df.frequency[data.df.MSR.index[data.df.MSR.argmax()]].magnitude
        avg_voltage = (
            np.mean(data.df.MSR.values[: ((fast_end - fast_start) // fast_step)]) * 1e6
        )

    prec_data = Dataset(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    frequency_range = (
        np.arange(precision_start, precision_end, precision_step) + qubit_frequency
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0:
                yield prec_data
            platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            prec_data.add(results)
            count += 1
    yield prec_data

    # Fitting
    # if self.resonator_type == '3D':
    #     f0, BW, Q, peak_voltage = fitting.lorentzian_fit("last", min, "Qubit_spectroscopy")
    #     qubit_freq = int(f0 + qd_pulse.frequency)
    #     # TODO: Fix fitting of minimum values
    # elif self.resonator_type == '2D':
    #     f0, BW, Q, peak_voltage = fitting.lorentzian_fit("last", max, "Qubit_spectroscopy")
    #     qubit_freq = int(f0 + qd_pulse.frequency)

    # # TODO: Estimate avg_voltage correctly
    # print(f"\nQubit Frequency = {qubit_freq}")
    # return qubit_freq, avg_voltage, peak_voltage, dataset


@store
def qubit_spectroscopy_flux(
    platform: AbstractPlatform,
    qubit,
    freq_width,
    freq_step,
    current_max,
    current_min,
    current_step,
    software_averages,
    fluxline=0,
    points=10,
):
    platform.reload_settings()
    

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    data = Dataset(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]
    frequency_range = np.arange(-freq_width, freq_width, freq_step) + qubit_frequency
    current_range = np.arange(current_min, current_max, current_step)

    count = 0
    for _ in range(software_averages):
        for curr in current_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
                platform.qf_port[fluxline].current = curr
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "current[A]": curr,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data


import os
import time
from scipy.optimize import minimize
import adaptive
from adaptive.learner.learner1D import (curvature_loss_function,
                                        uniform_loss,
                                        default_loss)
@store
def qubit_spectroscopy_optimize(
    platform: AbstractPlatform,
    qubit,
    width,
    maxiter = 500,
    points=10,
):

    platform.reload_settings()

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit]["resonator_freq"]
    platform.ro_port[qubit].lo_frequency = resonator_frequency - ro_pulse.frequency

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]


    bounds = (
        qubit_frequency - width,
        qubit_frequency + width
    )

    optimized_sweep_data = Dataset(name=f"data_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    def resonator_probe(freq):
        platform.qd_port[qubit].lo_frequency = freq[1] - qd_pulse.frequency
        msr, phase, i, q =  platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "frequency[Hz]": np.float64(freq[1]),
        }
        optimized_sweep_data.add(results)
        nonlocal count
        if count % points == 0:
            optimized_sweep_data.df.pint.dequantify().to_csv(f"debug/data/qubit_spectroscopy_optimize/data_q{qubit}.csv")
        print(f"probing at frequency {freq[1]}: {msr}")
        count += 1
        return msr
    
    def try_yield():
        yield optimized_sweep_data

    yield optimized_sweep_data


    def goal(l):
        return l.nsamples >= maxiter 
    learner = adaptive.AverageLearner1D(resonator_probe, bounds=bounds, min_error = 500, min_samples = 1, max_samples = 3)

    # runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)
    adaptive.runner.simple(learner, goal=goal)



@store
def qubit_spectroscopy_optimizeLearner1D(
    platform: AbstractPlatform,
    qubit,
    width,
    maxiter = 500,
    points=10,
):

    platform.reload_settings()

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit]["resonator_freq"]
    platform.ro_port[qubit].lo_frequency = resonator_frequency - ro_pulse.frequency

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]


    bounds = (
        qubit_frequency - width,
        qubit_frequency + width
    )

    optimized_sweep_data = Dataset(name=f"data_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    def resonator_probe(freq):
        platform.qd_port[qubit].lo_frequency = freq - qd_pulse.frequency
        # platform.qd_port[qubit].lo_frequency = freq[1] - qd_pulse.frequency
        msr, phase, i, q =  platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "frequency[Hz]": np.float64(freq),
            # "frequency[Hz]": np.float64(freq[1]),
        }
        optimized_sweep_data.add(results)
        nonlocal count
        if count % points == 0:
            optimized_sweep_data.df.pint.dequantify().to_csv(f"debug/data/qubit_spectroscopy_optimize/data_q{qubit}.csv")
        # print(f"probing at frequency {freq}: {msr}")
        # print(f"probing at frequency {freq[1]}: {msr}")
        count += 1
        return msr
    
    def try_yield():
        yield optimized_sweep_data

    yield optimized_sweep_data

    curvature_loss = curvature_loss_function()
    learner = adaptive.Learner1D(resonator_probe, bounds=bounds)
    # learner = adaptive.Learner1D(resonator_probe, bounds=bounds, loss_per_interval=curvature_loss)
    def goal(l):
        return l.npoints >= maxiter 
    # learner = adaptive.AverageLearner1D(resonator_probe, bounds=bounds, min_error = 0.00005, min_samples = 1, max_samples = 3)
    # def goal(l):
    #     return l.nsamples >= maxiter 

    # runner = adaptive.Runner(learner, goal=lambda l: l.loss() < 0.01)
    adaptive.runner.simple(learner, goal=goal)
    print('finished')

    # for freq, msr in learner.data.items():
    #     results = {
    #         "MSR[V]": np.float64(msr),
    #         "i[V]": np.float64(0),
    #         "q[V]": np.float64(0),
    #         "phase[rad]": np.float64(0),
    #         "frequency[Hz]": np.float64(freq),
    #     }
    #     optimized_sweep_data.add(results)
    # yield optimized_sweep_data

def qubit_spectroscopy_optimize_sergi(
    platform: AbstractPlatform,
    qubit,
    width,
    maxiter = 500,
    points=10,
):

    platform.reload_settings()

    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit]["resonator_freq"]
    platform.ro_port[qubit].lo_frequency = resonator_frequency - ro_pulse.frequency

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]



    bounds = [(
        qubit_frequency - width,

        qubit_frequency + width
    )]

    optimized_sweep_data = Dataset(name=f"data_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    def resonator_probe(freq):
        platform.qd_port[qubit].lo_frequency = freq[0] - qd_pulse.frequency
        msr, phase, i, q =  platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "frequency[Hz]": freq[0],
        }
        optimized_sweep_data.add(results)
        print(f"probing at frequency {freq[0]}: {msr}")
        nonlocal count
        count += 1
        return -msr
    
    def status_update(x):
        print(count, x)
        yield optimized_sweep_data

    yield optimized_sweep_data
    res = minimize(resonator_probe, qubit_frequency - width, method='Powell',
                            bounds=bounds,
                            callback=status_update,
                            options={'xtol': 0.0001, 
                            'ftol': 1000, 'maxiter': maxiter, 
                            'maxfev': 500},
                            )
    print(f'Found frequency: {res.x[0]} \n')
    print(f'Function evaluations: {res.nfev}\n')
    yield optimized_sweep_data