from qcvv.rb import RandomizedBenchmark, Experiment, CircuitGenerator
import click


@click.command()
@click.argument("repetitions", type=int)
@click.option("--lengths", "-l", default=[1,5,10,25,50,75,100,150,200], type=list)
@click.option("--seqs_per_length", "-s", default=10, type=int)
def run_rb(nqubits=1, lengths=[1,5,10,25,50,75,100,150,200], seqs_per_length=10, repetitions=5, save=False):

    from qibo import gates
    from qibo.noise import PauliError, ThermalRelaxationError, NoiseModel, ResetError
    import numpy as np
    exp = Experiment(nshots=30)
    pauli = PauliError(0, 0.05, 0)
    # define noise model and add quantum errors
    noise = NoiseModel()
    noise.add(pauli, gates.Unitary, 0)
    # noise.add(pauli, gates.Z, 0)
    generator = CircuitGenerator(noiseModel=noise)
    for l in lengths:
        for _ in range(seqs_per_length):
            exp.append(next(generator(length=l)),l)
    rb = RandomizedBenchmark(experiment=exp)
    #Run experiment for several times
    rb(repetitions=repetitions)
    print(f"alpha = {np.mean(rb.alphas)} +/- {np.std(rb.alphas)}, epc = {np.mean(rb.epcs)} +/- {np.std(rb.epcs)}")
    #Plot results
    plot(rb, repetitions)

def plot(rb, repetitions):
    import matplotlib.pyplot as plt
    import numpy as np 
    def model(x, a0, alpha, b0):
        return a0 * alpha**x + b0

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    ax.set_title("Randomized Benchmarking 1 qubit")
    print(rb.lengths)
    ax.scatter(rb.lengths*repetitions, rb.all_probs, label="Data")
    ax.plot(rb.lengths, model(rb.lengths, np.mean(rb.a0s), np.mean(rb.alphas), np.mean(rb.b0s)),'r--', label="Fit")
    ax.set_ylabel("Ground state population")
    ax.set_xlabel("Clifford length")
    ax.legend()
    plt.show()