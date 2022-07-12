# -*- coding: utf-8 -*-
from qcvv.rb import RandomizedBenchmark, Experiment, CircuitGenerator
import click
from ._base import command


@command.command("rb")
@click.argument("repetitions", type=int)
@click.option(
    "--lengths", "-l", default=[1, 5, 10, 25, 50, 75, 100, 150, 200], type=list
)
@click.option("--seqs_per_length", "-s", default=10, type=int)
@click.pass_context
def run_rb(
    ctx,
    lengths=[1, 5, 10, 25, 50, 75, 100, 150, 200],
    seqs_per_length=10,
    repetitions=5,
):

    from qibo import gates
    from qibo.noise import PauliError, NoiseModel
    import numpy as np

    nqubits = ctx.obj["nqubits"]
    path = ctx.obj["path"]
    exp = Experiment(nshots=30)
    pauli = PauliError(0, 0.01, 0)

    # define noise model and add quantum errors
    noise = NoiseModel()
    noise.add(pauli, gates.Unitary, 0)
    # noise.add(pauli, gates.Z, 0)
    generator = CircuitGenerator(noiseModel=noise)
    for l in lengths:
        for _ in range(seqs_per_length):
            exp.append(next(generator(length=l)), l)
    rb = RandomizedBenchmark(nqubits=nqubits, experiment=exp)
    # Run experiment for several times
    rb(repetitions=repetitions)

    rb.save_report(path)
    plot(rb, repetitions, path)


def plot(rb, repetitions, path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    def model(x, a0, alpha, b0):
        return a0 * alpha**x + b0

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_title("Randomized Benchmarking 1 qubit")
    ax.scatter(rb.lengths * repetitions, rb.all_probs, label="Data")
    ax.plot(
        rb.lengths,
        model(rb.lengths, np.mean(rb.a0s), np.mean(rb.alphas), np.mean(rb.b0s)),
        "r--",
        label="Fit",
    )
    ax.set_ylabel("Ground state population")
    ax.set_xlabel("Clifford length")
    ax.legend()
    if path is not None:
        plt.savefig(f"{path}/rb_plot.jpg")
    else:
        plt.show()
