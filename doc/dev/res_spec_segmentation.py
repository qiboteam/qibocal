import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage, signal

    return mo, ndimage, np, plt, signal


@app.function
def scale(x):
    min = x.min()
    return (x - min) / (x.max() - min)


@app.cell
def _(np):
    y = np.load("data/res_spec.npz")["0"]["signal"]
    return (y,)


@app.cell
def _(mo, plt, y):
    plt.plot(y)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo, ndimage, np, plt, signal, sigpeaks, y):
    def savgolpeaks(
        y, sgwin=10, sgorder=3, abs_smooth=3, qthresh=0.9, peak_id_cut=0.1, grow=0.05
    ):
        fig, axs = plt.subplots(
            5, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.0}
        )
        for ax in axs:
            ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        axs[0].plot(y)
        peaks, _, idx = sigpeaks(y, peak_id_cut)
        x = peaks[idx]
        axs[0].scatter(x, scale(y)[x], marker="x", color="r")
        dy = abs(signal.savgol_filter(y, sgwin, sgorder, deriv=1))
        axs[1].plot(dy)
        reconnected = signal.medfilt(dy, kernel_size=abs_smooth)
        axs[2].plot(reconnected)
        axs[3].plot(scale(y))
        thresholded = reconnected > np.quantile(reconnected, qthresh)
        labels, nlabels = ndimage.label(thresholded)
        axs[3].plot(labels / nlabels)
        label = labels[peaks[idx]]
        axs[4].plot(scale(y))
        window = labels == label
        axs[4].plot(window)
        axs[4].plot(np.convolve(window, np.ones(int(len(y) * grow)), mode="same") > 0)
        return mo.mpl.interactive(fig)

    savgolpeaks(y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploration
    """)
    return


@app.cell
def _(mo, plt, signal, y):
    def savgol():
        fig, axs = plt.subplots(
            5, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.0}
        )
        axs[0].plot(y)
        for n, ax in enumerate(axs[1:]):
            ax.plot(signal.savgol_filter(y, 2 * (n + 2), 3, deriv=1))
        return mo.mpl.interactive(fig)

    savgol()
    return


@app.cell
def _(mo, np, plt, signal, y):
    def sigpeaks(y, cut=0.01):
        peaks, prominence = signal.find_peaks(scale(-y), prominence=cut)
        idx = np.argmax(prominence["prominences"])
        return peaks, prominence, idx

    def peakplot(ax, y, peaks, prominence, idx):
        ax.plot(y)
        for i, (p, prom, left, right) in enumerate(zip(peaks, *prominence.values())):
            pmax = i == idx
            color = "tab:red" if pmax else "tab:blue"
            ls = "-" if pmax else "--"
            alpha = 0.2 if pmax else 0.1
            ax.axvspan(left, right, color=color, alpha=alpha)
            ax.axvline(p, color=color, linestyle=ls)

    def plotpeaks(y, peaks, prominence, idx):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        peakplot(ax, y, peaks, prominence, idx)
        return mo.mpl.interactive(fig)

    plotpeaks(y, *sigpeaks(y))
    return plotpeaks, sigpeaks


@app.cell
def _(np, plotpeaks, signal, sigpeaks, y):
    def mainidx(y, peaks):
        speaks, _, sidx = sigpeaks(y)
        return np.argmin(abs(speaks[sidx] - np.array(peaks)))

    def derivpeaks(y):
        peaks, prominence = signal.find_peaks(
            scale(-signal.savgol_filter(y, 10, 3, deriv=1)), prominence=0.1
        )
        idx = mainidx(y, peaks)
        return peaks, prominence, idx

    plotpeaks(y, *derivpeaks(y))
    return


if __name__ == "__main__":
    app.run()
