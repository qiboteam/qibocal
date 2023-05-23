import matplotlib.pyplot as plt  # TODO: remove plotting lines
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import bootstrap

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
    ax.axvline(
        x=np.median(data), linestyle="--", linewidth=2, label="median", color="navy"
    )
    ax.axvline(
        x=np.average(data),
        linestyle="--",
        linewidth=2,
        label="average",
        color="firebrick",
    )


def ramsey_fit(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x / p4)


def fit(times, voltages):
    y_max = np.max(voltages)
    y_min = np.min(voltages)
    y = (voltages - y_min) / (y_max - y_min)
    x_max = np.max(times)
    x_min = np.min(times)
    x = (times - x_min) / (x_max - x_min)

    ft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), x[1] - x[0])
    mags = abs(ft)
    index = np.argmax(mags) if np.argmax(mags) != 0 else np.argmax(mags[1:]) + 1
    f = freqs[index] * 2 * np.pi
    p0 = [
        0.5,
        0.5,
        f,
        0,
        0.5,
    ]
    try:
        popt, pcov = curve_fit(
            ramsey_fit,
            x,
            y,
            p0=p0,
            maxfev=20000,
            bounds=(
                [0, 0, 0, -np.pi, -1e6],
                [1, 1, np.inf, np.pi, 1e6],
            ),
            # sigma=np.array(errors, dtype=float) / (y_max - y_min),
            # absolute_sigma=True
        )
        err_popt = np.sqrt(np.diag(pcov))
        popt = [
            (y_max - y_min) * popt[0] + y_min,
            (y_max - y_min) * popt[1] * np.exp(x_min / ((x_max - x_min) * popt[4])),
            popt[2] / (x_max - x_min),
            popt[3] - x_min * popt[2] / (x_max - x_min),
            popt[4] * (x_max - x_min),
        ]
        t2 = popt[4]
    except RuntimeError:
        t2 = None
    return t2


def plots_distribution_gif(file_data="ramsey_raw.csv"):
    data = pd.read_csv(file_data, skiprows=[1])

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
        # Evaluate the asymetric error bars as the 68% confidence interval
        low_error = [median - np.percentile(msr_raw[:median_index], 100 - 68)]
        high_error = [np.percentile(msr_raw[median_index:], 68) - median]

        # Evaluate the symmetric intervals
        # median_distances = np.unique(abs(msr_raw-median))
        # percentiles = np.array([np.count_nonzero(abs(msr_raw-median)<k)/len(msr_raw) for k in median_distances])
        # confidence_interval = np.max(percentiles[percentiles<0.68])

        # Try the bootstrap
        # msr_res = (msr_raw,)
        # res = bootstrap(msr_res, np.median, confidence_level=0.68, method = "percentile")
        # print(res.confidence_interval)
        # print(abs(median -res.confidence_interval.low), abs(median - res.confidence_interval.high))

        ax[0, 0].errorbar(
            [wait],
            median,
            yerr=np.stack(
                (low_error, high_error), axis=0
            ),  #   yerr = confidence_interval,
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

        ax[1, 0].legend()
        plt.tight_layout()
        fig.savefig(f"ramsey_plots/ramsey_{int(i)}.png")
        ax[0, 1].clear()
        ax[1, 0].clear()
        ax[1, 1].clear()


def t2(data):
    waits = data["waits"].unique()
    voltages = np.array(
        [data[data["waits"] == wait]["MSR[V]"].to_list() for wait in waits]
    )

    rng = np.random.default_rng(seed=0)
    rng.shuffle(voltages, axis=1)
    fig, ax = plt.subplots(1, 1)
    # plt.hist(voltages[0])
    t2s = []
    high = 0
    low = 0
    for i in range(1000):  # (voltages.shape[1]):
        y = voltages[:, i]
        t2s.append(fit(waits, y))
        print(t2s[-1], len(t2s))
        if t2s[-1] is not None:
            fig, ax = plt.subplots(1, 1)
            plt.scatter(waits, y)
            if t2s[-1] > 1e6 and high < 5:
                plt.savefig(f"ramseys_bad_high{high}.png")
                high += 1
            elif t2s[-1] < 1 and low < 5:
                plt.savefig(f"ramseys_bad_low{low}.png")
                low += 1

    print("GGGGGG")
    t2s = t2s[t2s != None]
    plt.hist(t2s, bins=1000, density=True)
    plt.savefig("t2.png")
    return np.median(t2s), np.std(t2s)


def main(file_data="ramsey_raw.csv"):
    data = pd.read_csv(file_data, dtype=float)

    waits = data["waits"].unique()
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    medians = []
    msr_raws = []
    for i, wait in enumerate(waits):
        print(wait)
        # Select data for each wait time
        data_wait = data[data["waits"] == wait]
        msr_raw = data_wait["MSR[V]"]
        msr_raws.append(msr_raw)
        msr_raw = np.sort(msr_raw)
        median_index = int(len(msr_raw) / 2)
        print(msr_raw)
        median = np.median(msr_raw)
        medians.append(median)
        # Evaluate the asymetric error bars as the 68% confidence interval
        low_error = [median - np.percentile(msr_raw[:median_index], 100 - 68)]
        high_error = [np.percentile(msr_raw[median_index:], 68) - median]

        ax.errorbar(
            [wait],
            median,
            yerr=np.stack(
                (low_error, high_error), axis=0
            ),  #   yerr = confidence_interval,
            ms=10,
            # color=f"C{i}",
            marker="x",
        )

    ax.set_xlabel("wait[ns]", fontsize=FONTSIZE)
    ax.set_ylabel("MSR[V]", fontsize=FONTSIZE)
    ax.set_title("Ramsey", fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig(f"ramsey_data.png")
    print(t2(data))


if __name__ == "__main__":
    main()
