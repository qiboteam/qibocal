import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    from dataclasses import dataclass
    from functools import reduce
    from typing import Any, Callable, Optional, Union

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as sig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Broad wave generation

    This notebook explores the generation of a long pulse capable of exciting a certain known frequency (targeted through *carrier* selection) and further unknown frequencies in a finite neighborhood around that.

    The requisite we are trying to address are the following:
    - **broad spectrum:** we want to obtain a broad enough spectrum, ideally uniformly covering a specified neighborhood of a target frequency
    - **spread in time:** it should not be focused into a small time interval, to avoid effects resulting from the relaxation of the driven system
    - **sufficient power:** we want to deliver a sufficient amount of spectral power to the driven system
    - **realistic:** it has to be possible to implement it as a finite pulse shape with a finite sampling rate

    We also assume that the targeted system can absorb power over a certain interval of frequencies: its frequency response can be considered Lorentzian (or bell-shaped, in general), with a certain peak frequency and a bandwidth.

    What we are trying to achieve with the broad spectrum is specifically to drive multiple copies of the system, of which one is exactly centered at the target frequency.
    Instead, the other copies are supposed to be in a certain neighborhood, and are known to have a similar bandwidth, but their peak frequency is unknown a prior.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Some indicative values are:
    - a target `frequency` of $5~\textrm{GHz}$
    - a common `bandwidth` of $2-5~\textrm{MHz}$
    - for the generation parameters, a `sampling_rate` of $1~\textrm{GSa/s}$ and a maximum duration of $8~\mu\textrm{s}$

    /// admonition | Spectral power amount

    The y-axis on the Fourier transform seems quite useless, since we are lacking a number for direct and immediate comparison.

    However, the relative scale of the peaks in the various transforms has an important impact in terms of the amount of power which is delivered to the driven system. Despite not being that immediate to correctly *integrate* the contributions by eye, the order of magnitude of the y-axis (at least for similar shapes) gives already an idea about the amount of power in a certain region of the spectrum.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Waveforms

    First, we validate our machinery observing a **continuous wave** approximation, made by a single extended rectangular pulse, of our maximal length.
    """)
    return


@app.cell
def _():
    def wave(x: np.ndarray, res: float) -> np.ndarray:
        return np.ones_like(x)

    pipe(wave, freqs=(4.99, 5.01))[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Rectangular

    Still as validation and preparation, we explore the behavior of **rectangular** pulses.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start with a single **short** pulse, which is expected to have the desired property of a broad spectrum.
    """)
    return


@app.function
def rect(
    x: np.ndarray,
    res: float,
    slot: tuple[float, float] = (0.0, float("inf")),
) -> np.ndarray:
    y = np.zeros_like(x)
    b, e = tuple(np.searchsorted(x, slot))
    y[b:e] = 1
    return y


@app.cell
def _():
    pipe(rect, [sample], slot=(200, 300), freqs=(4.98, 5.02))[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And their **repeated** version, to prevent *relaxation*.

    It can be observed how a more continuous frequency shape turns into a series of sharp peaks, because of a periodicity. This is essentially approximating a Fourier series, and it is a limitation for our design, since the generated spacing between frequencies has to be taken into account.
    """)
    return


@app.cell
def _():
    pipe(
        rect,
        [sample, repeat],
        slot=(0, 50),
        reps=50,
        freqs=(4.8, 5.2),
        times=(0, 1000),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Gaussian

    As yet a further step, we explore the effect of continuous (and analytic) pulse shapes, which should focus the spectral components around the carrier frequency.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since it is well known that the transform of a Gaussian is a Gaussian, we start just looking into the impact of the sampling, by comparing it with a smooth short pulse.
    """)
    return


@app.function
def gauss(x: np.ndarray, res: float, sigma: float) -> np.ndarray:
    return sig.windows.gaussian(x.size, sigma / res)


@app.cell
def _():
    pipe(gauss, [cut, pad], length=100, sigma=5, freqs=(4.9, 5.1))[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the next plot, we just zoom in time, and zoom out in the frequency domain, to move into the regions where the comparison with the sampled version will be most visible.
    """)
    return


@app.cell
def _():
    pipe(gauss, [cut, pad], length=100, sigma=5, freqs=(3, 7), times=(20, 80))[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now, we introduce the finite sampling rate effect.
    """)
    return


@app.cell
def _():
    pipe(
        gauss,
        [cut, sample, pad],
        length=100,
        sigma=5,
        freqs=(3, 7),
        times=(20, 80),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Moreover, since one of the initial goals of this exercise was to avoid the decay of the state, we repeat the Gaussian pulse multiple times, to avoid inducing long gaps.

    However, this repetition induces an approximate periodicity, only limited by the finite interval we are working in. The effect of this periodicity in time is to concentrate the frequencies in individual modes, whose width is only controlled by the size of the whole window.

    The spacing in frequency is instead inversely proportional to the spacing in time. Then, we conclude that, if we want to gather the modes closer, we need to space them more in time.
    This is generating a trade-off, since:
    - we want to limit the distancing in the time-domain, to avoid the relaxation of the driven system
    - but we also want to have dense modes, without generating large gaps in the spectrum
    """)
    return


@app.cell
def _():
    pipe(
        gauss,
        [cut, sample, repeat],
        length=100,
        sigma=5,
        reps=100,
        freqs=(4.8, 5.2),
        times=(0, 1000),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And let's zoom in and out (in time and frequency) once more.
    """)
    return


@app.cell
def _():
    pipe(
        gauss,
        [cut, sample, repeat],
        length=100,
        sigma=5,
        reps=100,
        freqs=(3, 7),
        times=(0, 200),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To magnify the effect of one extreme of the trade-off, we look into the Fourier spectrum of extremely dense time scheduling.

    In this limit configuration, the frequencies become even more sparse, which is destroying the density of the spectrum.
    """)
    return


@app.cell
def _():
    pipe(
        gauss,
        [repeat, sample, pad],
        sigma=5,
        reps=300,
        freqs=(4.8, 5.2),
        times=(0, 200),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To recover a larger spectrum, preserving short-enough pulses we try to introduce an idea we will also use later on: we try to break a bit more the periodicity, by sending pulses of different lengths.

    - The shorter pulses will ensure the spectrum broadening
        - a single long Gaussian pulse would be extremely narrow in frequency.
    - The longer pulses will preserve a strong carrier component
    - The increased length of the base unit will reduce the spectrum stride
        - the step of a Fourier series is inverse proportional to the periodicity
    """)
    return


@app.cell
def _():
    def gausses(x: np.ndarray, res: float, sigmas: tuple[float, ...]) -> np.ndarray:
        return np.concatenate(
            [sig.windows.gaussian(x.size // len(sigmas), s / res) for s in sigmas]
        )

    pipe(
        gausses,
        [sample, repeat, pad],
        sigmas=(20, 30, 40),
        reps=10,
        freqs=(4.95, 5.05),
        times=(0, 2000),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exponential

    A pure exponential shape is also analytical, and falling less sharply then a Gaussian (the exponential of a square).
    """)
    return


@app.function
def decay(x: np.ndarray, tau: float, res: float):
    return np.exp(-x / tau)


@app.cell
def _():
    pipe(
        decay,
        [cut, sample, repeat, pad],
        tau=20,
        reps=50,
        length=200,
        freqs=(4, 6),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Despite being a possible option, we will not further investigate in this direction, and move to a different technique.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Inverse Fourier

    Since we are explicitly targeting the spectrum of our sequence, another available tool is to directly design that, and reconstruct our time domain shape through an inverse Fourier transform.
    """)
    return


@app.function
def ifft(
    x: np.ndarray, slot: tuple[float, float], sigma: float, res: float
) -> np.ndarray:
    fs = np.fft.rfftfreq(x.size, x[1] - x[0])
    yf = np.zeros_like(fs)
    b, e = tuple(np.searchsorted(fs, slot))
    # target
    # yf[b:e] = 1
    yf[b:e] = sig.windows.gaussian(e - b, sigma)
    y = np.fft.irfft(yf)
    return y / abs(y).max()


@app.cell
def _():
    pipe(ifft, [sample, pad], slot=(0, 0.5), sigma=400, freqs=(3.5, 6.5))[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    While the spectrum is highly deformed by the finite sampling rate, we are still essentially able to target a very wide and smooth frequency packet. But we clearly paying it in terms of power delivered, and time spacing.

    The next step is then to draw again from our toolbox, and repeat the shape on a smaller interval, tuning a bit its parameters to fit our conditions in both domains.
    """)
    return


@app.cell
def _():
    pipe(
        ifft,
        [cut, sample, repeat, pad],
        slot=(0, 0.15),
        sigma=25,
        length=200,
        reps=25,
        freqs=(4.8, 5.2),
        times=(0, 250),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Piecewise oscillations

    Inspired by the former exercise, we just look for simpler shapes that could deliver similar properties, optimizing a bit more the time slots for power delivery.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First, we start exploring the maximal oscillations we can produce on top of the carrier mode.
    """)
    return


@app.function
def triangular(x: np.ndarray, res: float):
    step = int(1 / res)
    y = np.zeros(x.size // step)
    y[:40:2] = 1
    y[1:40:2] = -1
    return np.repeat(y, step)


@app.cell
def _():
    pipe(triangular, [repeat], reps=200, times=(0, 10), freqs=(4, 6))[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    While preserving a maximal amount power around the shifted frequencies, the is certainly far from the desired outcome.

    We then proceed by reducing a bit the frequency of the oscillation, and restoring portions of basic rectangular pulses, to preserve the central mode.
    """)
    return


@app.function
def multisteps(x: np.ndarray, res: float):
    step = int(1 / res)
    y = np.zeros(x.size // step)
    y[:50] = 1
    y[50:150:4] = 1
    y[52:150:4] = -1
    y[150:300:8] = 1
    y[151:300:8] = y[107:200:8] = 1 / 2
    y[154:300:8] = -1
    y[153:300:8] = y[105:200:8] = -1 / 2
    return np.repeat(y, step)


@app.cell
def _():
    pipe(multisteps, [repeat, pad], reps=30, times=(0, 500), freqs=(4, 6))[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Although the achieved spectrum is already acceptable, it is not optimal for our target parameters. This because its bulk is covering a wide range, $\sim~\pm500~\textrm{MHz}$, whereas the main modes are more spaced than we wished.

    We can tune this result, iterating on the exact same idea, just lowering a bit the frequencies contained in the envelope, to squeeze the spectrum in the desired area.
    """)
    return


@app.function
def multisine(x: np.ndarray, modes: list, sections: list, res: float):
    segments = np.astype(np.array(sections) / res, int)
    limits = np.insert(np.add.accumulate(segments), 0, 0)

    y = np.zeros(x.size)
    for b, e, s, f in zip(limits, limits[1:], segments, modes):
        y[b:e] = np.cos(np.linspace(0, f * 2 * np.pi, s))
    return y


@app.cell
def _():
    pipe(
        multisine,
        [repeat, sample, pad],
        modes=[0, 4, 0.5, 2],
        sections=[20, 120, 120, 120],
        reps=20,
        times=(0, 500),
        freqs=(4.9, 5.1),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now our spectrum is concentrated in a range of $\sim~\pm50~\textrm{MHz}$, and its main mode are $2-3~\textrm{MHz}$ apart, with a quite high spectral density.

    The result is now fitting our original constraints. The last step is just simplifying it.
    """)
    return


@app.function
def chirp(x: np.ndarray, alpha: float, beta: float, interval: float, res: float):
    xp = x % interval / interval
    return np.cos(2 * np.pi * (alpha * xp + beta) * xp)


@app.cell
def _():
    pipe(
        chirp,
        [sample],
        alpha=8,
        beta=0,
        interval=500,
        times=(0, 1000),
        freqs=(4.9, 5.1),
    )[0]
    return


@app.cell
def _():
    def envelope(duration: int, alpha: float, beta: float, interval: int):
        x = np.arange(duration)
        xp = x % interval / interval
        return np.cos(2 * np.pi * (alpha * xp + beta) * xp)

    plt.figure(figsize=(10, 4))
    plt.plot(envelope(8000, 8, 0, 500))
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pseudo-random

    White noise has a flat spectrum. So, that's also a good option.
    """)
    return


@app.function
def white(x: np.ndarray, window: int, res: float):
    return np.convolve(
        np.random.rand(x.size), sig.windows.blackman(window), mode="same"
    )


@app.cell
def _():
    pipe(
        white,
        [sample],
        window=30,
        times=(0, 1000),
        freqs=(4.9, 5.1),
    )[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// Attention | Seed

    Just notice that we did not fix the seed, neither in the cell above, nor in the one below.

    ///
    """)
    return


@app.cell
def _():
    def envelope_white(duration: int, window: int):
        return np.convolve(
            np.random.rand(duration), sig.windows.cosine(window), mode="same"
        )

    plt.figure(figsize=(10, 4))
    plt.plot(envelope_white(8000, 500))
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Library
    """)
    return


@app.function
def window(duration: float, res: float) -> np.ndarray:
    """Generate a time window, with a certain duration and resolution."""
    return np.linspace(0, duration, int(duration / res))


@app.function
def mix(x: np.ndarray, yenv: np.ndarray, carrier: float) -> np.ndarray:
    """Upconvert a base band signal."""
    return np.sin(2 * np.pi * carrier * x) * yenv


@app.function
def transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fourier transform, of signal and domain."""
    xf = np.fft.rfftfreq(x.size, x[1] - x[0])
    yf = np.fft.rfft(y).real
    return xf, yf


@app.function
def plot(
    env: tuple[np.ndarray, np.ndarray],
    t: tuple[np.ndarray, np.ndarray],
    f: tuple[np.ndarray, np.ndarray],
    times: tuple[float, float],
    freqs: tuple[float, float],
) -> plt.Figure:
    """Plot signal in time and frequency domains."""
    fig, (time, fourier) = plt.subplots(1, 2, figsize=(10, 3))

    start, stop = tuple(np.searchsorted(t[0][: t[0].size], times))
    x = t[0][start:stop]
    y = t[1][start:stop]
    time.plot(x, y, x, env[start:stop])

    fstart, fstop = tuple(np.searchsorted(f[0], freqs))
    xf = f[0][fstart:fstop]
    yf = f[1][fstart:fstop]
    fourier.plot(xf, yf)
    fourier.yaxis.get_major_formatter().set_powerlimits((-2, 3))
    return fig


@app.function
def pipe(
    env: Callable[[np.ndarray, float, ...], np.ndarray],
    maps: Optional[list[Callable]] = None,
    res: float = 0.01,
    duration: float = 8000,
    carrier: float = 5,
    times: Union[float, tuple[float, float]] = (0, float("inf")),
    freqs: Union[float, tuple[float, float]] = (0, float("inf")),
    **kwargs,
) -> tuple[plt.Figure, Any]:
    """Pipe multiple transformation maps into a signal, and plot.

    First, the time window is extracted, based on the `res`olution and `duration`.
    Then, it is piped into the transformations, specified in `maps`, which also
    receive the `res` parameter, and all the extra keyword arguments in `kwargs`.

    The resulting envelope is upconverted with the `carrier` frequency, and the
    result is plotted in time and frequency domain, in the `times` and `freqs`
    windows.

    .. note::

      The `kwargs` are passed down to the most external mapping (the last one in
      `maps`), and it is intended that they should be propagated by each
      transformation, possibly consuming only the arguments whish are specific to
      that map.
    """
    if not isinstance(freqs, tuple):
        freqs = (0, freqs)
    if maps is None:
        maps = []

    x = window(res=res, duration=duration)
    yenv = reduce(lambda f, g: g(f), maps, env)(x, res=res, **kwargs)

    y = mix(x, yenv, carrier)
    xf, yf = transform(x, y)
    fig = plot(yenv, (x, y), (xf, yf), times=times, freqs=freqs)
    return mo.mpl.interactive(fig), Signal(times=x, freqs=xf, y=y, yf=yf, envelope=yenv)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Maps

    Common transformations for envelopes
    """)
    return


@app.function
def sample(envelope: Callable) -> Callable:
    """Replace envelope with step function.

    Consumes: `sampling_rate`.
    """

    def step(
        x: np.ndarray, res: float, sampling_rate: float = 1.0, **kwargs
    ) -> np.ndarray:
        step = int(1 / (res * sampling_rate))
        return np.repeat(envelope(x[::step], res=res * step, **kwargs), step)

    return step


@app.function
def cut(envelope: Callable) -> Callable:
    """Cut function into smaller window.

    Consumes: `length`.
    """

    def c(x: np.ndarray, length: int, **kwargs) -> np.ndarray:
        size = np.searchsorted(x, length)
        return envelope(x[:size], **kwargs)

    return c


@app.function
def repeat(envelope: Callable) -> Callable:
    """Repeat function multiple times.

    Consumes: `reps`.
    """

    def rep(x: np.ndarray, reps: int, **kwargs) -> np.ndarray:
        cut = x.size // reps
        return np.tile(envelope(x[:cut], **kwargs), reps)

    return rep


@app.function
def pad(envelope: Callable) -> Callable:
    """Pad function with zeros."""

    def p(x: np.ndarray, **kwargs) -> np.ndarray:
        base = envelope(x, **kwargs)
        return np.pad(base, (0, x.size - base.size))

    return p


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Data
    """)
    return


@app.class_definition
@dataclass
class Signal:
    times: np.ndarray
    freqs: np.ndarray
    y: np.ndarray
    yf: np.ndarray
    envelope: np.ndarray


if __name__ == "__main__":
    app.run()
