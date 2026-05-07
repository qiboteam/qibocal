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


@app.cell
def _():
    alphax = mo.ui.slider(
        0, 2, 0.01, value=1, full_width=True, label=r"$\omega_X$", show_value=True
    )
    alphaz = mo.ui.slider(
        0, 2, 0.01, value=1, full_width=True, label=r"$\omega_Z$", show_value=True
    )
    n = mo.ui.slider(10, 100, 1, value=50, full_width=True, label="n", show_value=True)
    mo.vstack([alphax, alphaz, n])
    return alphax, alphaz, n


@app.cell
def _(alphax, alphaz, n):
    traj = ramsey(alphax.value, alphaz.value, np.linspace(0, 2 * np.pi, n.value))
    sphere(expectations(traj))
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pauli
    """)
    return


@app.cell
def _():
    np.testing.assert_allclose(sx @ sy, 1j * sz)
    np.testing.assert_allclose(sy @ sz, 1j * sx)
    np.testing.assert_allclose(sz @ sx, 1j * sy)
    return


@app.function
def expectations(density_matrix):
    return np.einsum("bij,aji->ba", paulis, density_matrix).real


@app.function
def ramsey(alpha, alphaz, thetas):
    rx90 = scipy.linalg.expm(1j * (alpha * sx + alphaz * sz))
    sweep = scipy.linalg.expm(1j * thetas[:, None, None] * sz)
    u = np.einsum("ij,ajk,km->aim", rx90, sweep, rx90)
    return np.einsum("aji,jk,akm->aim", np.conj(u), ground, u)


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
