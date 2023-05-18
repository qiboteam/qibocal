import matplotlib.pyplot as plt  # TODO: remove plotting lines
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams["figure.dpi"] = 600
FONTSIZE = 15


def plot_hist(ax, data, color_index, xlabel, xlim, wait):
    # Plots histograms
    ax.hist(
        data,
        bins=100,
        histtype="step",
        density=True,
        color=f"C{color_index}",
    )
    ax.set_xlim(xlim)
    ax.set_xlabel(f"{xlabel}", fontsize=FONTSIZE)
    ax.set_title(f"distribution of the {xlabel} ({wait} ns)", fontsize=FONTSIZE)


data = pd.read_csv("ramsey_raw.csv", skiprows=[1])

waits = data["waits"].unique()
fig, ax = plt.subplots(2, 2, figsize=(14, 7))  # TODO: remove plotting lines

for i, wait in enumerate(waits):
    print(wait)
    # Select data for each wait time
    data_wait = data[data["waits"] == wait]
    i_raw = data_wait["i[V]"]
    q_raw = data_wait["q[V]"]
    msr_raw = data_wait["MSR[V]"]

    msr_raw = np.sort(msr_raw)
    median_index = int(len(msr_raw) / 2)
    median = np.median(msr_raw)
    # Evaluate the error bars as the 68% confidence interval
    low_error = [median - np.percentile(msr_raw[:median_index], 66)]
    high_error = [np.percentile(msr_raw[median_index:], 34) - median]
    # ax[0, 0].scatter([wait] * len(msr_raw), msr_raw, s=1, color=f"C{int(i/10)}")
    ax[0, 0].errorbar(
        [wait],
        median,
        yerr=np.stack((low_error, high_error), axis=0),
        ms=10,
        color=f"C{i}",
        marker="x",
    )
    ax[0, 0].set_xlabel("wait[ns]", fontsize=FONTSIZE)
    ax[0, 0].set_ylabel("MSR[V]", fontsize=FONTSIZE)
    ax[0, 0].set_title("Ramsey", fontsize=FONTSIZE)
    ax[0, 0].set_xlim(-10, np.max(waits))

    plot_hist(ax[1, 0], msr_raw, i, "MSR[V]", (0, 5), wait)
    plot_hist(ax[0, 1], i_raw, i, "I[V]", (-5, 5), wait)
    plot_hist(ax[1, 1], q_raw, i, "Q[V]", (-5, 5), wait)

    plt.tight_layout()
    fig.savefig(f"ramsey_plots/ramsey_{int(i)}.png")
    ax[0, 1].clear()
    ax[1, 0].clear()
    ax[1, 1].clear()
