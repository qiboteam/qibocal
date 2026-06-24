import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.gridspec import GridSpec
    from pydantic import BaseModel

    MHZ = r"\text{[MHz]}"
    GHZ = r"\text{[GHz]}"


@app.cell
def _():
    delta = mo.ui.slider(
        start=50,
        stop=1000,
        step=5,
        value=200,
        show_value=True,
        debounce=True,
        full_width=True,
        label=rf"$\Delta~{MHZ}$",
    )
    eta_a = mo.ui.slider(
        start=10,
        stop=1000,
        step=5,
        value=330,
        show_value=True,
        debounce=True,
        full_width=True,
        label=rf"$\eta_A~{MHZ}$",
    )
    eta_b = mo.ui.slider(
        start=10,
        stop=1000,
        step=5,
        value=330,
        show_value=True,
        debounce=True,
        full_width=True,
        label=rf"$\eta_B~{MHZ}$",
    )
    g = mo.ui.slider(
        start=0.1,
        stop=10,
        step=0.1,
        value=3.8,
        show_value=True,
        debounce=True,
        full_width=True,
        label=rf"$g~{MHZ}$",
    )
    sliders = mo.vstack([delta, eta_a, eta_b, g])
    sliders
    return delta, eta_a, eta_b, g


@app.cell
def _(delta, eta_a, eta_b, g, plot):
    plot(
        Measurements(
            f01=(5.0, 5.0 + delta.value * 1e-3),
            eta=(-eta_a.value * 1e-3, -eta_b.value * 1e-3),
            g=g.value * 1e-3,
            tpi=(np.nan, 20),
            tpi_cr=(np.nan, 160),
        ),
        # path="interactive.svg"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Case studies

    IBM cancellation paper, https://arxiv.org/abs/1603.04821
    """)
    return


@app.cell
def _(plot):
    plot(
        Measurements(
            f01=(4.914, 5.114),
            eta=(-0.330, -0.330),
            g=3.8e-3,
            tpi=(np.nan, 20),
            tpi_cr=(np.nan, 160),
        ),
        # path="ibm.svg"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Our beloved QPU 167
    """)
    return


@app.cell
def _(plot):
    plot(
        Measurements(
            f01=(4.0978, 4.3416),
            eta=(-0.272, -0.258),
            zz=2e-3,
            tpi=(80, 80),
            api=(0.17, 0.062),
            tpi_cr=(200, 200),
        ),
        # path="qpu167.svg"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Input type
    """)
    return


@app.class_definition
class Measurements(BaseModel):
    f01: tuple[float, float]
    eta: tuple[float, float]
    g: float = np.nan
    zz: float = np.nan
    tpi: tuple[float, float] = (np.nan, np.nan)
    "duration of control pi-pulse"
    api: tuple[float, float] = (1.0, 1.0)
    "amplitude of control pi-pulse"
    tpi_cr: tuple[float, float] = (np.nan, np.nan)
    """duration of cross-resonance pulse

    the first position is intended to be used when the first qubit is used as control
    """

    @property
    def f12(self) -> tuple[float, float]:
        return tuple(f + e for f, e in zip(self.f01, self.eta))

    @property
    def f01a(self) -> np.ndarray:
        return np.array(self.f01)

    @property
    def f12a(self) -> np.ndarray:
        return np.array(self.f12)

    @property
    def etaa(self) -> np.ndarray:
        return np.array(self.eta)

    @property
    def tpia(self) -> np.ndarray:
        return np.array(self.tpi)

    @property
    def apia(self) -> np.ndarray:
        return np.array(self.api)

    @property
    def tpi_cra(self) -> np.ndarray:
        return np.array(self.tpi_cr)

    @property
    def detuning(self) -> np.ndarray:
        f01 = self.f01a
        return f01 - f01[::-1]

    @property
    def cross_detuning(self) -> np.ndarray:
        return self.f12a - self.f01a[::-1]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calculation
    """)
    return


@app.class_definition
class Derived(BaseModel):
    eta: tuple[float, float]
    "anharmonicities"
    detuning: float
    "detuning of f01 frequencies"
    detuning12: float
    "detuning of f12 frequencies"
    cross_detuning: tuple[float, float]
    "detuning of f12 of one qubit from f01 of the other (signed)"
    g: float = np.nan
    "coupling strength"
    zz: float = np.nan
    r"\zeta, but converted as a frequency"
    mu: tuple[float, float] = (np.nan, np.nan)
    "cross resonance couplings, i.e. ZX terms"
    f_rabi: tuple[float, float] = (np.nan, np.nan)
    "Rabi frequencies"
    amp_ratio: tuple[float, float] = (np.nan, np.nan)
    "cross resonance to pi-pulse amplitudes ratio"

    @property
    def detunings(self) -> dict[str, float]:
        return {
            r"\Delta": self.detuning,
            r"\Delta_{12}": self.detuning12,
            r"\tilde{\Delta}_{AB}": self.cross_detuning[0],
            r"\tilde{\Delta}_{BA}": self.cross_detuning[1],
            r"\eta_A": self.eta[0],
            r"\eta_B": self.eta[1],
        }

    @property
    def quantities(self) -> dict[str, tuple[float, float, bool]]:
        """Compute quantities to display.

        Return quantities with reference critical value, and toggle to declare
        whether it is critical above or below.
        """
        qs = {i: i for i in range(0)}
        if not np.isnan(self.g):
            qs[rf"g/(2\pi)~{MHZ}"] = (self.g * 1e3, 1.5, True)
        if not np.isnan(self.zz):
            qs[rf"\zeta/(2\pi)~{MHZ}"] = (self.zz * 1e3, 0.5, False)
        if not np.isnan(self.mu[0]):
            qs[r"\mu_{AB}"] = (self.mu[0], 0.03, True)
        if not np.isnan(self.f_rabi[0]):
            qs[rf"\Omega_A~{MHZ}"] = (self.f_rabi[0] * 1e3, 20, True)
        if not np.isnan(self.amp_ratio[0]):
            qs[r"A_{AB}/A_B"] = (self.amp_ratio[0], 10, False)
        if not np.isnan(self.mu[1]):
            qs[r"\mu_{BA}"] = (self.mu[1], 0.03, True)
        if not np.isnan(self.f_rabi[1]):
            qs[rf"\Omega_B~{MHZ}"] = (self.f_rabi[1] * 1e3, 20, True)
        if not np.isnan(self.amp_ratio[1]):
            qs[r"A_{BA}/A_A"] = (self.amp_ratio[1], 10, False)

        return qs


@app.cell
def _():
    def coupling(meas: Measurements) -> tuple[float, float]:
        if not np.isnan(meas.g):
            g = meas.g
            zz = -2 * g**2 * (1 / meas.cross_detuning[::-1] + 1 / meas.cross_detuning)
            return g, zz[0]
        if not np.isnan(meas.zz):
            zz = meas.zz
            g = np.sqrt(np.abs(zz / (2 * (1 / meas.cross_detuning).sum())))
            return g, zz

        return np.nan, np.nan

    def compute(meas: Measurements) -> Derived:
        g, zz = coupling(meas)
        mu = tuple(-g * meas.etaa / (meas.detuning * meas.cross_detuning))
        return Derived(
            eta=meas.eta,
            detuning=meas.detuning[0],
            detuning12=meas.f12[0] - meas.f12[1],
            cross_detuning=meas.cross_detuning,
            g=g,
            zz=zz,
            mu=mu,
            f_rabi=1 / (meas.tpia * meas.apia),
            amp_ratio=meas.tpia / (meas.tpi_cra * mu),
        )

    return (compute,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plotting
    """)
    return


@app.cell
def _(compute):
    def plot(meas: Measurements, path: str | None = None):
        derived = compute(meas)

        fig = plt.figure(figsize=(10, 5), layout="constrained")
        gs1 = GridSpec(10, 10, left=0.05, right=0.95, wspace=0, figure=fig)
        ax1 = fig.add_subplot(gs1[:6, :7])
        ax2 = fig.add_subplot(gs1[6:, :7])
        ax3 = fig.add_subplot(gs1[:, 7:])

        freqs(ax1, meas.f01, meas.f12)
        detunings(ax2, derived.detunings)
        parameters(ax3, derived)

        fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

        if path is not None:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.5)

        return mo.mpl.interactive(fig)

    return (plot,)


@app.function
def freqs(ax: Axes, f01: tuple[float, float], f12: tuple[float, float]) -> None:
    fs = sorted(f01 + f12)
    ax.set_xticks(fs, labels=[f"{f:.2f}" for f in fs], rotation=30)
    ax.tick_params(
        labeltop=True,
        top=False,
        labelbottom=False,
        bottom=False,
        labelleft=False,
        left=False,
    )
    ax.annotate(
        "[GHz]",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(-30, 5),
        textcoords="offset points",
    )
    ax.margins(0.1, 0)

    ax.annotate(
        "",
        xy=(f01[0], 1),
        xytext=(f01[1], 1),
        arrowprops=dict(arrowstyle="<->"),
    )
    ax.annotate(
        r"$\Delta$",
        xy=((f01[0] + f01[1]) / 2, 1),
        xytext=(0, 5),
        textcoords="offset points",
    )
    ax.annotate(
        "",
        xy=(f12[0], 2),
        xytext=(f12[1], 2),
        arrowprops=dict(arrowstyle="<->"),
    )
    ax.annotate(
        r"$\Delta_{12}$",
        xy=((f12[0] + f12[1]) / 2, 2),
        xytext=(0, 5),
        textcoords="offset points",
    )
    for i, (ge, ef, c) in enumerate(zip(f01, f12, ("0.6", "k"))):
        ax.vlines(ge, 0, 9, c)
        ax.vlines(ef, 0, 9, c, ls="--")
        ax.annotate(
            "",
            xy=(ef, 4 + i),
            xytext=(ge, 4 + i),
            arrowprops=dict(arrowstyle="<->"),
        )
        ax.annotate(
            rf"$\eta_{chr(65 + i)}$",
            xy=((ge + ef) / 2, 4 + i),
            xytext=(0, 5),
            textcoords="offset points",
        )
        j = (i + 1) % 2
        ax.annotate(
            "",
            xy=(ge, 7 + i),
            xytext=(f12[j], 7 + i),
            arrowprops=dict(arrowstyle="<->"),
        )
        ax.annotate(
            rf"$\tilde{{\Delta}}_{{{chr(65 + i)}{chr(65 + j)}}}$",
            xy=((ge + f12[j]) / 2, 7 + i),
            xytext=(0, 5),
            textcoords="offset points",
        )


@app.function
def detunings(ax: Axes, detunings: dict[str, float]) -> None:
    sorted_ = dict(sorted(detunings.items(), key=lambda item: item[1]))
    ax.set_xticks(
        np.abs(list(sorted_.values())),
        labels=[f"${sym}$" for sym in sorted_.keys()],
    )
    ax.tick_params(labelbottom=True, bottom=False, labelleft=False, left=False)
    ax.margins(0)

    dets = np.abs(list(detunings.values()))
    ax.vlines(dets, 0, 0.82, "k")
    ax.fill_between([-0.1e-3, 40e-3], 0, 1, facecolor="red", alpha=0.2)
    ax.annotate(
        r"$40~\text{MHz}$",
        xy=(0.04, 0.88),
        xytext=(-15, 0),
        textcoords="offset points",
    )
    ax.fill_between([0.04, 0.34], 0, 1, facecolor="orange", alpha=0.2)
    ax.annotate(
        r"$300~\text{MHz}$",
        xy=(0.34, 0.88),
        xytext=(-20, 0),
        textcoords="offset points",
    )
    ax.fill_between([-0.1e-3, max(dets) + 0.01], 0, 1, facecolor="red", alpha=0.0)


@app.function
def parameters(ax: Axes, derived: Derived) -> None:
    qs = derived.quantities
    n = len(qs)
    if n > 0:
        values_, limits, increasing = tuple(
            np.array(list(it)) for it in zip(*qs.values())
        )
        values = abs(values_)
    else:
        values, limits, increasing = (np.array([]),) * 3

    ax.set_yticks(np.arange(n), labels=[f"${q}$" for q in qs])
    ax.tick_params(
        labelbottom=False,
        bottom=False,
        labelleft=False,
        left=False,
        labelright=True,
    )
    # ax3.spines[:].set_visible(False)
    ax.margins(0)

    low = -0.5 * max((1 - 1.2 / n), 0.1) if n > 0 else 0
    high = (n - 1 + 1 / n) * 1.07 if n > 0 else 1
    threshold = 0.3

    ax.fill_between([-0.1, threshold], low, high, facecolor="red", alpha=0.15)
    ax.fill_between([-0.1, 1.1], -low, high, facecolor="red", alpha=0.0)
    i = np.arange(n)
    v = (values / limits) ** np.where(increasing, 1, -1) * threshold
    e = v < threshold
    m = v > 1
    ax.scatter(v[e], i[e], c="r", marker="x")
    ax.scatter(v[~m & ~e], i[~m & ~e], c="k")
    ax.scatter(np.ones_like(i[m]), i[m], marker=">", c="0.7")
    for vv, lab, ii in zip(v, values, i):
        ax.annotate(
            f"{lab:#.3g}",
            xy=(min(vv, 1), ii),
            xytext=(-10, 7),
            textcoords="offset points",
        )


if __name__ == "__main__":
    app.run()
