"""Protocol-level model of polarization-encoded decoy-state MDI-QKD.

* analytic gains / QBERs of a linear-optics Bell-state measurement (BSM) with
  threshold detectors (Ma & Razavi 2012; Xu, Curty, Qi & Lo 2013)
* slot-level Monte-Carlo of the same BSM (used to validate the analytic model)
* decoy-state estimation of Y11 and e11 by linear programming (3 intensities
  per user), asymptotic and with Gaussian statistical fluctuations
* secret-key rate (Lo, Curty & Qi 2012)
"""
import numpy as np
from scipy.special import i0e, gammaln
from scipy.optimize import linprog


def h2(x):
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def I0(x):
    return i0e(x) * np.exp(x)


# ----------------------------------------------------------------- analytic gains
def gains(mu_a, mu_b, eta_a, eta_b, pd, ed):
    """Overall gains and QBERs of the |psi-> projection for Z and X bases.
    eta_* include channel transmittance and detection efficiency; pd is the
    spurious-click probability per detector per time slot."""
    mp = eta_a * mu_a + eta_b * mu_b
    x = np.sqrt(eta_a * mu_a * eta_b * mu_b) / 2.0
    y = (1 - pd) * np.exp(-mp / 4.0)
    QC = 2 * (1 - pd) ** 2 * np.exp(-mp / 2) * (1 - (1 - pd) * np.exp(-eta_a * mu_a / 2)) \
        * (1 - (1 - pd) * np.exp(-eta_b * mu_b / 2))
    QE = 2 * pd * (1 - pd) ** 2 * np.exp(-mp / 2) * (I0(2 * x) - (1 - pd) * np.exp(-mp / 2))
    QZ = QC + QE
    EQZ = ed * QC + (1 - ed) * QE
    QX = 2 * y ** 2 * (1 + 2 * y ** 2 - 4 * y * I0(x) + I0(2 * x))
    EQX = 0.5 * QX - 2 * (0.5 - ed) * y ** 2 * (I0(2 * x) - 1)
    return QZ, EQZ, QX, EQX


def y11_exact(eta_a, eta_b, pd, ed):
    """Single-photon-pair yield and X-basis error (infinite-decoy limit)."""
    Y11 = (1 - pd) ** 2 * (eta_a * eta_b / 2 + (2 * eta_a + 2 * eta_b - 3 * eta_a * eta_b) * pd
                           + 4 * (1 - eta_a) * (1 - eta_b) * pd ** 2)
    eY = Y11 / 2 - (0.5 - ed) * (1 - pd) ** 2 * eta_a * eta_b / 2
    return Y11, eY / Y11


# ----------------------------------------------------------------- slot-level MC
def mc_bsm(mu, eta_a, eta_b, pd, basis, n_slots, rng, ed_angle=0.0, batch=2_000_000):
    """Monte-Carlo of phase-randomized weak coherent pulses entering a
    50:50 BS + two PBS (4 threshold detectors). Returns (gain, QBER) of the
    |psi-> announcement in the given basis ('Z' or 'X')."""
    succ = err = 0
    done = 0
    while done < n_slots:
        b = min(batch, n_slots - done); done += b
        ba = rng.integers(0, 2, b); bb = rng.integers(0, 2, b)
        if basis == "Z":
            pa = np.stack([1 - ba, ba], 1).astype(float); pb = np.stack([1 - bb, bb], 1).astype(float)
        else:
            s = 1 / np.sqrt(2)
            pa = np.stack([np.full(b, s), np.where(ba == 0, s, -s)], 1)
            pb = np.stack([np.full(b, s), np.where(bb == 0, s, -s)], 1)
        th = ed_angle  # polarization misalignment of Bob's photons (rotation)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pb = pb @ rot.T
        phi = rng.uniform(0, 2 * np.pi, b)  # relative phase of the two pulses
        aa = np.sqrt(eta_a * mu) * pa
        ab = np.sqrt(eta_b * mu) * pb * np.exp(1j * phi)[:, None]
        o1 = (aa + ab) / np.sqrt(2); o2 = (aa - ab) / np.sqrt(2)
        m = np.concatenate([np.abs(o1) ** 2, np.abs(o2) ** 2], 1)  # 1H 1V 2H 2V
        click = rng.random(m.shape) < 1 - (1 - pd) * np.exp(-m)
        c1H, c1V, c2H, c2V = click.T
        psi_m = ((c1H & c2V & ~c1V & ~c2H) | (c1V & c2H & ~c1H & ~c2V))
        psi_p = ((c1H & c1V & ~c2H & ~c2V) | (c2H & c2V & ~c1H & ~c1V))
        # Z basis: both projections imply anti-correlated bits.
        # X basis: psi- implies anti-correlated, psi+ correlated bits.
        if basis == "Z":
            wrong = (psi_m | psi_p) & (ba == bb)
        else:
            wrong = (psi_m & (ba == bb)) | (psi_p & (ba != bb))
        ok = psi_m | psi_p
        succ += ok.sum(); err += wrong.sum()
    return succ / n_slots, (err / succ if succ else np.nan)


# ----------------------------------------------------------------- decoy LP
def _poisson(mu, n):
    k = np.arange(n + 1)
    return np.exp(-mu + k * np.log(max(mu, 1e-300)) - gammaln(k + 1)) if mu > 0 else (k == 0).astype(float)


def decoy_bounds(ints, Q, EQ_x, QX, M=6, fluct=None):
    """Lower bound on Y11 (Z basis), lower bound on Y11 (X) and upper bound on
    e11 Y11 (X) from gains for all intensity pairs.
    ints: list of intensities (same set for Alice and Bob)
    Q[i][j], QX[i][j], EQ_x[i][j]: observed Z gain, X gain, X error-gain.
    fluct: optional (Nz[i][j], Nx[i][j], n_sigma) for Gaussian fluctuations."""
    P = [_poisson(m, M) for m in ints]
    idx = [(n, k) for n in range(M + 1) for k in range(M + 1)]
    nv = len(idx); j11 = idx.index((1, 1))

    def solve(obs, Nobs, sense):
        A, bu, bl = [], [], []
        for i, a in enumerate(ints):
            for j, b in enumerate(ints):
                row = np.array([P[i][n] * P[j][k] for n, k in idx])
                tail = 1 - row.sum()
                q = obs[i][j]
                dq = 0.0
                if fluct is not None:
                    dq = fluct[2] * np.sqrt(max(q, 1e-300) / Nobs[i][j])
                A.append(row); bu.append(q + dq); bl.append(max(q - dq - tail, 0.0))
        A = np.array(A)
        A_ub = np.vstack([A, -A]); b_ub = np.concatenate([bu, -np.array(bl)])
        c = np.zeros(nv); c[j11] = 1.0 if sense == "min" else -1.0
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * nv, method="highs")
        if not r.success:
            return np.nan
        return r.x[j11]

    nz = fluct[0] if fluct else None; nx = fluct[1] if fluct else None
    Y11z = solve(Q, nz, "min")
    Y11x = solve(QX, nx, "min")
    EY11x = solve(EQ_x, nx, "max")
    return Y11z, Y11x, EY11x


def key_rate(L, det, mu, nu, f_ec=1.16, ed=0.015, alpha=0.2, mode="lp", fluct=None):
    """Secret key per pulse (Z-basis signal-signal key, asymptotic basis sifting).
    det: callable(L, mu, nu) -> (eta_eff, pd_eff) giving the effective detector
         efficiency (incl. window/dead-time losses) and spurious-click probability.
    mode: 'inf' (infinite decoy) or 'lp' (vacuum + weak + signal decoys)."""
    eta_ch = 10 ** (-alpha * (L / 2) / 10)
    eta_d, pd = det(L, mu, nu)
    ea = eb = eta_ch * eta_d
    QZ, EQZ, _, _ = gains(mu, mu, ea, eb, pd, ed)
    EZ = EQZ / QZ
    if mode == "inf":
        Y11, e11 = y11_exact(ea, eb, pd, ed)
    else:
        ints = [mu, nu, 0.0]
        Qz = [[gains(a, b, ea, eb, pd, ed)[0] for b in ints] for a in ints]
        G = [[gains(a, b, ea, eb, pd, ed) for b in ints] for a in ints]
        Qx = [[g[2] for g in row] for row in G]
        EQx = [[g[3] for g in row] for row in G]
        fl = None
        if fluct is not None:
            N, pz, pint, ns = fluct  # total pulses, P(Z basis), intensity probs, n_sigma
            Nz = [[N * pz ** 2 * pint[i] * pint[j] for j in range(3)] for i in range(3)]
            Nx = [[N * (1 - pz) ** 2 * pint[i] * pint[j] for j in range(3)] for i in range(3)]
            fl = (Nz, Nx, ns)
        Y11z, Y11x, EY = decoy_bounds(ints, Qz, EQx, Qx, fluct=fl)
        if not (Y11x > 0) or not (Y11z > 0):
            return 0.0, EZ, np.nan
        Y11, e11 = Y11z, min(EY / Y11x, 0.5)
        if fluct is not None:
            # sifting factor: key bits only from Z-basis signal-signal pulses
            N, pz, pint, ns = fluct
            QZs = QZ - ns * np.sqrt(QZ / (N * pz ** 2 * pint[0] ** 2))
            R = pz ** 2 * pint[0] ** 2 * (mu * mu * np.exp(-2 * mu) * Y11 * (1 - h2(e11)) - QZ * f_ec * h2(EZ))
            return max(R, 0.0), EZ, e11
    R = mu * mu * np.exp(-2 * mu) * Y11 * (1 - h2(e11)) - QZ * f_ec * h2(EZ)
    return max(R, 0.0), EZ, e11
