import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy

    ground = np.diag([1.0, 0])

    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    paulis = np.array([sx, sy, sz])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiments
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Ramsey
    """)
    return


@app.function
def ramsey(omegax, omegaz, thetas):
    rx90 = scipy.linalg.expm(1j * (omegax * sx + omegaz * sz))
    sweep = scipy.linalg.expm(1j * thetas[:, None, None] * sz)
    u = np.einsum("ij,ajk,km->aim", rx90, sweep, rx90)
    return np.einsum("aji,jk,akm->aim", np.conj(u), ground, u)


@app.cell
def _():
    omegax = mo.ui.slider(
        0, 2, 0.01, value=1, full_width=True, label=r"$\omega_X$", show_value=True
    )
    omegaz = mo.ui.slider(
        0, 2, 0.01, value=1, full_width=True, label=r"$\omega_Z$", show_value=True
    )
    n = mo.ui.slider(10, 100, 1, value=50, full_width=True, label="n", show_value=True)
    mo.vstack([omegax, omegaz, n])
    return n, omegax, omegaz


@app.cell
def _(n, omegax, omegaz):
    traj = ramsey(omegax.value, omegaz.value, np.linspace(0, 2 * np.pi, n.value))
    sphere(expectations(traj))
    mo.mpl.interactive(plt.gcf()).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Ramsey flux
    """)
    return


@app.function
def ramsey_flux(deltas):
    rx90 = scipy.linalg.expm(1j * (np.pi / 4 * sx))
    sweep = scipy.linalg.expm(1j * deltas[:, None, None] * sz)
    # u = np.einsum("ij,ajk,km->aim", rx90, sweep, rx90)
    u = np.einsum("ij,ajk->aik", rx90, sweep)
    return np.einsum("aji,jk,akm->aim", np.conj(u), ground, u)


@app.cell
def _():
    deltamax = mo.ui.slider(
        0,
        400,
        0.1,
        value=30,
        full_width=True,
        label=r"$\delta_\mathrm{max}$",
        show_value=True,
    )
    nflux = mo.ui.slider(
        10, 100, 1, value=50, full_width=True, label="n", show_value=True
    )
    mo.vstack([deltamax, nflux])
    return deltamax, nflux


@app.cell
def _(deltamax, nflux):
    trajflux = ramsey_flux(deltamax.value / 2 * np.linspace(0, 1, nflux.value) ** 2)
    sphere(expectations(trajflux))
    mo.mpl.interactive(plt.gcf()).center()
    return (trajflux,)


@app.cell
def _(trajflux):
    x, y, z = expectations(trajflux)
    return x, y


@app.cell
def _(x, y):
    plt.plot(x)
    plt.plot(y)
    return


@app.cell
def _(nflux, x, y):
    phase = np.unwrap(np.angle(x + 1j * y))
    # normalization required to avoid problems with arccos
    phase -= phase[0]
    plt.figure()
    plt.plot(np.linspace(0, 100, nflux.value), np.angle(x + 1j * y))
    plt.figure()
    plt.plot(np.linspace(0, 100, nflux.value), phase)
    plt.figure()
    f = 5
    det = phase / 1 / 2 / np.pi
    # to make sure that flux is invertible
    det[np.abs(det) < 1e-3] = 0
    # from inversion of flux dependence formula assuming negligible Ec and asymmetry
    derived_flux = 1 / np.pi * np.arccos(((f + det) / f) ** 2)
    plt.plot(derived_flux)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Utils
    """)
    return


@app.cell
def _():
    # Pauli validation

    np.testing.assert_allclose(sx @ sy, 1j * sz)
    np.testing.assert_allclose(sy @ sz, 1j * sx)
    np.testing.assert_allclose(sz @ sx, 1j * sy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sphere
    """)
    return


@app.function
def expectations(density_matrix):
    return np.einsum("bij,aji->ba", paulis, density_matrix).real


@app.function
def sphere(exps):
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    # Import data
    xx, yy, zz = exps

    # Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="c", alpha=0.3, linewidth=0)

    ax.scatter(xx, yy, zz, color="k", s=20)
    ax.scatter(0, 0, 1, color="r", s=20)
    ax.plot([0, xx[0]], [0, yy[0]], [1, zz[0]], color="r")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()


if __name__ == "__main__":
    app.run()
