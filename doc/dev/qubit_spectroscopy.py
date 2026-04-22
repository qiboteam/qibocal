import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as sig
    from broad_wave import pipe, rect, sample


@app.cell
def _():
    fig, signal = pipe(rect, [sample], slot=(200, 7800), freqs=(4.98, 5.02))
    fig
    return (signal,)


@app.cell
def _(signal):
    def gwind(sigma: float, step: float, length: int) -> np.ndarray:
        return sig.windows.gaussian(length, std=sigma / step)

    length = 10000
    step = np.diff(signal.freqs[:2])[0]
    window = gwind(sigma=1.5e-3, step=step, length=length)

    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(length) * step, window)
    mo.mpl.interactive(plt.gca())
    return (window,)


@app.cell
def _(signal, window):
    fig1, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig1.subplots_adjust(hspace=0)
    power = np.convolve(signal.yf, window, mode="same")
    power /= power.max() / np.pi
    a1.plot(signal.freqs, power)
    a2.plot(signal.freqs, (1 - np.cos(power)) / 2)
    mo.mpl.interactive(fig1)
    return


if __name__ == "__main__":
    app.run()
