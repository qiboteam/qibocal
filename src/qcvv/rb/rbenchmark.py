# -*- coding: utf-8 -*-
import numpy as np
from qcvv.rb.random import measure
from scipy.optimize import curve_fit


class RandomizedBenchmark:
    """ "Implementation of Randomized Benchmarking"""

    def __init__(self, nqubits=1, experiment=None, isHardware=False):
        self.nqubits = nqubits
        self.experiment = experiment
        self.isHardware = isHardware

    def process_data(self, name="Ground State Probability"):
        """Calculate alpha and EPC for one repetition of the experiment"""
        print("[+] Processing data")

        if name == "Ground State Probability":
            self.probs = []
            self.lengths = list(self.experiment.lengths)
            self.lengths.sort()
            if self.isHardware:
                data = [[i.probabilities[0], i.length] for i in self.experiment]
            else:
                data = [
                    [np.count_nonzero(i.samples == 0) / len(i.samples), i.length]
                    for i in self.experiment
                ]
            for j in self.lengths:
                self.probs.append(np.mean([i[0] for i in data if i[1] == j]))

        # Helper function for fitting the single exponential
        def model(x, a0, alpha, b0):
            return a0 * alpha**x + b0

        if self.isHardware:
            popt, _ = curve_fit(
                model,
                np.array(self.lengths),
                np.mean(np.array(self.all_probs), axis=0),
                p0=[1, 0.98, 0],
                maxfev=5000,
            )
        else:
            popt, _ = curve_fit(
                model, self.lengths, self.probs, p0=[0.5, 0.98, 0.5], maxfev=5000
            )

        a0, alpha, b0 = popt
        epc = (2**self.nqubits - 1) * (1 - alpha) / 2**self.nqubits

        return alpha, epc, a0, b0, self.probs

    def __call__(self, repetitions=5):
        """Run n repetitions of the experiment"""
        self.alphas, self.epcs, self.a0s, self.b0s, self.all_probs = [], [], [], [], []
        for i in range(repetitions):
            print(f"Running simulation {i}")
            self.experiment = measure(
                self.experiment, self.experiment.nshots, self.isHardware
            )
            alpha, epc, a0, b0, probs = self.process_data()
            print(f"[+] Alpha = {alpha}, EPC = {epc}")
            self.alphas.append(alpha)
            self.epcs.append(epc)
            self.a0s.append(a0)
            self.b0s.append(b0)
            self.all_probs.append(probs)
