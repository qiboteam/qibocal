"""
Core routines for circuit-level 2q Clifford RB / CZ-interleaved RB simulation (numpy only).

Noise model (gate-independent, exact):
  reference Clifford C  ->  D_aR o U_ZZ(thR) after C
  target CZ             ->  D_aG o U_ZZ(+/-thG) after CZ
  recovery              ->  ideal
  U_ZZ(th) = exp(-i th ZZ/2),  D_a(rho) = a rho + (1-a) I/4.
  ZZ commutes with CZ and D_a commutes with every unitary, so an interleaved block
  (C_i, Lam_R, CZ, Lam_G) = D_{aR aG} o U_ZZ(thR +/- thG) o (CZ C_i).

Toggling-frame shortcut (exact for this model): with cumulative frames T_k = C_k...C_1 (iid uniform
Cliffords since (C_1..C_m) -> (T_1..T_m) is a bijection of G^m), the ideal-recovery output is
  rho = D_{a^m}( V_m ... V_1 |00><00| V_1^+ ... V_m^+ ),  V_k = T_k^+ U_ZZ(th) T_k = exp(-i th s_k P_k / 2),
with (s_k, P_k) iid uniform over the 30 signed non-identity 2q Paulis.
Basis index i = 2*b1 + b2 (b1 = first qubit).
"""
import math

import numpy as np

# ---------------------------------------------------------------- Paulis
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
P1 = [_I, _X, _Y, _Z]
PAULI = np.array([np.kron(P1[i], P1[j]) for i in range(4) for j in range(4)])  # index 4*i+j
PAULI_LABEL = [a + b for a in "IXYZ" for b in "IXYZ"]
ZZ_IDX = 15
PERM = np.zeros((16, 4), dtype=np.int64)   # (P psi)[r] = PH[p, r] * psi[PERM[p, r]]
PH = np.zeros((16, 4), dtype=complex)
for _p in range(16):
    for _r in range(4):
        _c = int(np.argmax(np.abs(PAULI[_p, _r])))
        PERM[_p, _r] = _c
        PH[_p, _r] = PAULI[_p, _r, _c]
ZZ_DIAG = np.real(np.diag(PAULI[ZZ_IDX]))   # [1,-1,-1,1]

# per-shot indicator weights over outcomes (00,01,10,11)
EST_W = {"P00": np.array([1.0, 0, 0, 0]), "q1": np.array([1.0, 1, 0, 0]),
         "H": np.array([1.0, 0.5, 0.5, 0])}
EST_NAMES = ("P00", "q1", "H", "min")
EST_AB = {"P00": (0.75, 0.25), "q1": (0.5, 0.5), "H": (0.5, 0.5), "min": (0.5, 0.5)}


# ---------------------------------------------------------------- noise parametrisation
def pu(theta):
    """Depolarizing parameter of the twirled U_ZZ(theta): (16 cos^2(theta/2) - 1)/15."""
    return (16 * math.cos(theta / 2) ** 2 - 1) / 15


def p_of_F(F):
    return (4 * F - 1) / 3


def noise_params(F, kind):
    """Return (a, theta>=0) with Favg(D_a o U_ZZ(theta)) = F exactly."""
    pF = p_of_F(F)
    if kind == "depol":
        return pF, 0.0
    if kind == "coherent":
        return 1.0, 2 * math.asin(math.sqrt(5 * (1 - F) / 4))
    if kind == "mixed":
        s = math.sqrt(pF)                          # a = pu(theta) = sqrt(pF)
        return s, 2 * math.acos(math.sqrt((15 * s + 1) / 16))
    raise ValueError(kind)


def favg(a, theta):
    return (1 + 3 * a * pu(theta)) / 4


# ---------------------------------------------------------------- shortcut simulator
def simulate_shortcut(rng, n, m, theta, a, chunk=60000):
    """Exact output probabilities (n,4) of n independent RB circuits of length m (toggling frame)."""
    out = np.empty((n, 4))
    am = a ** m
    for s0 in range(0, n, chunk):
        nn = min(chunk, n - s0)
        if theta == 0.0 or m == 0:
            pr = np.zeros((nn, 4)); pr[:, 0] = 1.0
        else:
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            psi = np.zeros((nn, 4), dtype=complex); psi[:, 0] = 1.0
            rows = np.arange(nn)[:, None]
            for _ in range(m):
                idx = rng.integers(1, 16, nn)
                sg = (2 * rng.integers(0, 2, nn) - 1).astype(float)
                ppsi = PH[idx] * psi[rows, PERM[idx]]
                psi = c * psi - 1j * (s * sg)[:, None] * ppsi
            pr = psi.real ** 2 + psi.imag ** 2
        out[s0:s0 + nn] = am * pr + (1 - am) / 4
    # remove ~1e-16 norm drift so multinomial accepts the probabilities
    np.clip(out, 0.0, 1.0, out=out)
    out /= out.sum(1, keepdims=True)
    return out


def apply_signed_paulis(idx, sgn, theta):
    """Deterministic toggling-frame state for given Pauli index/sign sequences (n,m). Returns |psi|^2 (n,4)."""
    n, m = idx.shape
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    psi = np.zeros((n, 4), dtype=complex); psi[:, 0] = 1.0
    rows = np.arange(n)[:, None]
    for k in range(m):
        ppsi = PH[idx[:, k]] * psi[rows, PERM[idx[:, k]]]
        psi = c * psi - 1j * (s * sgn[:, k])[:, None] * ppsi
    return psi.real ** 2 + psi.imag ** 2


# ---------------------------------------------------------------- Clifford group (for validation)
def _canon_key(U):
    f = U.ravel()
    k = int(np.argmax(np.abs(f) > 1e-9))
    V = U * (abs(f[k]) / f[k])
    # entries of 2q Cliffords (phase-fixed) lie in {0, +-1, +-i, ...}/2^(j/2): snap to kill drift
    Vs = np.round(V.real * 2, 6) / 2 + 1j * np.round(V.imag * 2, 6) / 2
    for val in (0.5, 1 / math.sqrt(2), 1.0):
        for part in ("real", "imag"):
            arr = getattr(Vs, part)
            msk = np.abs(np.abs(arr) - val) < 1e-5
            arr = np.where(msk, np.sign(arr) * val, arr)
            Vs = arr + 1j * Vs.imag if part == "real" else Vs.real + 1j * arr
    key = (np.round(Vs, 6) + 0.0).tobytes()
    return key, Vs


def clifford_group_2q():
    """All 11520 two-qubit Cliffords (mod global phase) by BFS over {H1,H2,S1,S2,CZ}."""
    H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    S = np.diag([1, 1j])
    CZ = np.diag([1, 1, 1, -1]).astype(complex)
    gens = [np.kron(H, _I), np.kron(_I, H), np.kron(S, _I), np.kron(_I, S), CZ]
    k0, V0 = _canon_key(np.eye(4, dtype=complex))
    seen = {k0: 0}
    mats = [V0]
    frontier = [V0]
    while frontier:
        nxt = []
        for U in frontier:
            for g in gens:
                k, V = _canon_key(g @ U)
                if k not in seen:
                    seen[k] = len(mats)
                    mats.append(V)
                    nxt.append(V)
        frontier = nxt
    return np.array(mats), CZ


def signed_pauli_of(M):
    """Identify M = s * PAULI[j]; return (j, s)."""
    ov = np.einsum("pij,ji->p", PAULI, M) / 4      # Tr(P_j M)/4
    j = int(np.argmax(np.abs(ov)))
    if abs(abs(ov[j]) - 1) > 1e-6 or abs(ov[j].imag) > 1e-6:
        raise RuntimeError("not a signed Pauli")
    return j, int(round(ov[j].real))


# ---------------------------------------------------------------- direct density-matrix simulator (validation)
def _uzz(theta):
    return np.diag(np.exp(-1j * theta / 2 * ZZ_DIAG))


def simulate_direct(cliffs, seq, aR, thR, interleave=None):
    """Direct density-matrix simulation.
    cliffs: (G,4,4); seq: (n,m) Clifford indices.
    interleave: None (standard RB) or (CZ, aG, thG_signed) -> C_i, Lam_R, CZ, Lam_G per block.
    Returns output probabilities (n,4) and the per-step cumulative frames' signed Paulis (idx,sgn)."""
    n, m = seq.shape
    rho = np.zeros((n, 4, 4), dtype=complex); rho[:, 0, 0] = 1
    T = np.tile(np.eye(4, dtype=complex), (n, 1, 1))
    UR = _uzz(thR)
    I4 = np.eye(4) / 4
    idx = np.zeros((n, m), dtype=np.int64); sgn = np.zeros((n, m))
    ZZm = PAULI[ZZ_IDX]
    for k in range(m):
        C = cliffs[seq[:, k]]
        rho = C @ rho @ np.conj(np.transpose(C, (0, 2, 1)))
        rho = UR @ rho @ UR.conj().T
        rho = aR * rho + (1 - aR) * I4
        T = C @ T
        if interleave is not None:
            CZ, aG, thG = interleave
            UG = _uzz(thG)
            rho = CZ @ rho @ CZ.conj().T
            rho = UG @ rho @ UG.conj().T
            rho = aG * rho + (1 - aG) * I4
            T = CZ @ T
        Mk = np.conj(np.transpose(T, (0, 2, 1))) @ ZZm @ T
        for i in range(n):
            idx[i, k], sgn[i, k] = signed_pauli_of(Mk[i])
    Td = np.conj(np.transpose(T, (0, 2, 1)))
    rho = Td @ rho @ T
    return np.real(np.einsum("nii->ni", rho)), idx, sgn


# ---------------------------------------------------------------- estimators
def pooled_estimators(counts):
    """counts (...,4) pooled over shots -> dict of estimator arrays."""
    N = counts.sum(-1)
    f = counts / N[..., None]
    q1 = f[..., 0] + f[..., 1]
    q2 = f[..., 0] + f[..., 2]
    return {"P00": f[..., 0], "q1": q1, "H": (q1 + q2) / 2, "min": np.minimum(q1, q2)}


# ---------------------------------------------------------------- batched weighted LM fit:  y = A exp(-lam m) + B
def _model(th, x):
    return th[:, 0:1] * np.exp(-th[:, 2:3] * x) + th[:, 1:2]


def _jac(th, x):
    E = np.exp(-th[:, 2:3] * x)
    return np.stack([E, np.ones_like(E), -th[:, 0:1] * x * E], axis=-1)


def fit_wls(Y, x, V, x0, maxit=400):
    """3-parameter WLS (A, B, lam all free; lam unconstrained in sign).
    Y (R,n); x (n,) depths; V (n,) known variances; x0 (3,) or (R,3).
    Returns theta (R,3), converged (R,) bool. Converged := finite and Gauss-Newton decrement < 1e-10."""
    Y = np.atleast_2d(Y)
    R = Y.shape[0]
    x = np.asarray(x, float)
    w = 1 / np.sqrt(np.asarray(V, float))
    th = np.array(np.broadcast_to(np.asarray(x0, float), (R, 3)))
    mu = np.full(R, 1e-3)

    def cost(t):
        with np.errstate(over="ignore", invalid="ignore"):
            r = (_model(t, x) - Y) * w
            c = np.sum(r * r, axis=1)
        c[~np.isfinite(c)] = np.inf
        return c

    C = cost(th)
    eye = np.eye(3)
    for _ in range(maxit):
        with np.errstate(over="ignore", invalid="ignore"):
            r = (_model(th, x) - Y) * w
            J = _jac(th, x) * w[None, :, None]
        g = np.einsum("rti,rt->ri", J, r)
        Hm = np.einsum("rti,rtj->rij", J, J)
        D = np.diagonal(Hm, axis1=1, axis2=2)
        Hd = Hm + mu[:, None, None] * (eye * np.maximum(D, 1e-30)[:, :, None])
        bad = ~np.isfinite(Hd).all((1, 2)) | ~np.isfinite(g).all(1)
        Hd[bad] = eye; g[bad] = 0
        try:
            step = -np.linalg.solve(Hd, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = -np.array([np.linalg.lstsq(Hd[i], g[i], rcond=None)[0] for i in range(R)])
        trial = th + step
        Ct = cost(trial)
        acc = Ct < C
        th[acc], C[acc] = trial[acc], Ct[acc]
        mu = np.clip(np.where(acc, mu / 3, mu * 4), 1e-12, 1e12)
        if np.all(np.abs(step) <= 1e-12 * (1 + np.abs(th))):
            break
    with np.errstate(over="ignore", invalid="ignore"):
        r = (_model(th, x) - Y) * w
        J = _jac(th, x) * w[None, :, None]
        g = np.einsum("rti,rt->ri", J, r)
        Hm = np.einsum("rti,rtj->rij", J, J)
    dec = np.full(R, np.inf)
    ok = np.isfinite(Hm).all((1, 2)) & np.isfinite(g).all(1)
    for i in np.where(ok)[0]:
        try:
            dec[i] = g[i] @ np.linalg.solve(Hm[i], g[i])
        except np.linalg.LinAlgError:
            pass
    conv = np.isfinite(th).all(1) & (dec < 1e-10)
    return th, conv, C
