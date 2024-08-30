"""Auxiliary functions to run CHSH protocol."""

READOUT_BASIS = ["ZZ", "ZX", "XZ", "XX"]


def compute_chsh(frequencies, i, bell_state):
    """Computes the chsh inequality out of the frequencies of the 4 circuits executed."""

    zz = frequencies["ZZ"]
    zx = frequencies["ZX"]
    xz = frequencies["XZ"]
    xx = frequencies["XX"]

    nshots = zz["11"][i] + zz["00"][i] + zz["10"][i] + zz["01"][i]

    result_zz = zz["11"][i] + zz["00"][i] - zz["10"][i] - zz["01"][i]
    result_zx = zx["11"][i] + zx["00"][i] - zx["10"][i] - zx["01"][i]
    result_xz = xz["11"][i] + xz["00"][i] - xz["10"][i] - xz["01"][i]
    result_xx = xx["11"][i] + xx["00"][i] - xx["10"][i] - xx["01"][i]
    if bell_state % 2 == 0:
        return (result_zz + result_xz - result_zx + result_xx) / nshots
    return (result_zz + result_xz + result_zx - result_xx) / nshots
