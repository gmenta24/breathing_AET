"""Noise-only LISA inference in the frequency (FD) and WDM time-frequency domains.

Companion code of ``paper/main.tex``.  Everything needed by the notebook
``aet_wdm_analyses.ipynb`` lives here:

* the two-parameter instrumental-noise model (test-mass acceleration ``A`` and
  OMS ``P`` amplitudes) of the single links;
* the closed-form TDI-2 covariance of ``X, Y, Z`` for six arbitrary (frozen) link
  delays -- the formulas of ``main.tex`` generalised to TDI 2 and to unequal
  up/down links -- and an independent brute-force evaluation from the pytdi
  delay polynomials used to validate it;
* the noise simulation: link noises drawn from the PSDs in the frequency domain,
  assembled into the intermediary variables ``eta_ij`` and combined into TDI-2
  ``X2, Y2, Z2`` by pytdi with delays from lisaorbits;
* coarse-grained likelihoods: complex Wishart on frequency bins (FD) and real
  Wishart on time-frequency cells (WDM), with the model surfaces precomputed
  because the covariance is linear in ``A^2`` and ``P^2``;
* a jexplore driver and small posterior helpers.

Conventions
-----------
* Links are ordered as in lisaorbits, ``LINKS = (12, 23, 31, 13, 32, 21)``;
  ``d_ij`` is the light travel time of the beam received on spacecraft ``i``
  from ``j``.  Arm ``k`` is the one opposite spacecraft ``k``: arm 3 = links
  12/21, arm 1 = links 23/32, arm 2 = links 31/13.
* Measurements are fractional-frequency time series.  The one-sided PSD of the
  OMS noise of every link is ``P^2 S_oms(f)`` and that of every test mass is
  ``A^2 S_tm(f)``; ``eta_ij = N_ij + a_ij + D_ij a_ji`` (main.tex eq. link).
* FD normalisation: for data ``x`` with window ``w`` and ``X = rfft(w x)``,
  ``I_k = 2 dt X_k X_k^dagger / sum(w^2)`` has expectation equal to the one-sided
  cross-spectral matrix (time-averaged with weight ``w^2`` if non-stationary).
* WDM normalisation (wdm_transform, discrete-FFT convention): for interior bands
  ``E[w_nm w_nm^T] = N/(2 dt) sum_j Phi(j)^2 Re C(f_{q_m + j}, t_n)``,
  ``q_m = m nt/2``, with ``Phi`` the Meyer window in DFT-bin units
  (``sum_j Phi^2 = 1``).  We store ``y_nm = 2 dt w w^T / N``, so that ``E[y]`` is
  the window-averaged real part of the cross-spectral matrix.
* Channel basis for all 3x3 algebra: ``A = (Z-X)/sqrt2``, ``E = (X-2Y+Z)/sqrt6``,
  ``T = (X+Y+Z)/sqrt3`` (main.tex).  It is only a fixed rotation, used because
  it isolates the tiny null-channel eigenvalue and keeps the closed-form 3x3
  algebra well conditioned.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

C_LIGHT = 299792458.0
L_NOMINAL = 2.5e9
YEAR = 365.25 * 86400.0
LINKS = ("12", "23", "31", "13", "32", "21")
A_TRUE = 3.0     # test-mass acceleration noise, units of 1e-15 m s^-2 Hz^-1/2
P_TRUE = 15.0    # OMS noise, units of 1e-12 m Hz^-1/2
PARAM_NAMES = ("A", "P")
PARAM_LABELS = (r"$A\ [10^{-15}\,\mathrm{m\,s^{-2}\,Hz^{-1/2}}]$",
                r"$P\ [10^{-12}\,\mathrm{m\,Hz^{-1/2}}]$")

#: XYZ -> AET rotation (rows A, E, T), orthogonal.
R_AET = np.array([[-1.0, 0.0, 1.0],
                  [1.0, -2.0, 1.0],
                  [1.0, 1.0, 1.0]]) / np.array([[np.sqrt(2)], [np.sqrt(6)], [np.sqrt(3)]])


# ---------------------------------------------------------------------------
# Single-link noise PSDs (fractional frequency, one-sided)
# ---------------------------------------------------------------------------

def psd_tm_unit(f, xp=np):
    """Test-mass acceleration noise of one link for ``A = 1`` [1/Hz]."""
    return ((1e-15 / (2 * xp.pi * f * C_LIGHT)) ** 2
            * (1 + (4e-4 / f) ** 2) * (1 + (f / 8e-3) ** 4))


def psd_oms_unit(f, xp=np):
    """Optical-metrology-system noise of one link for ``P = 1`` [1/Hz]."""
    return (1e-12 * 2 * xp.pi * f / C_LIGHT) ** 2 * (1 + (2e-3 / f) ** 4)


# ---------------------------------------------------------------------------
# Closed-form TDI-2 covariance for frozen delays
# ---------------------------------------------------------------------------

def arm_quantities(d, xp=np):
    """Mean arm light times ``Lbar_k`` and half link asymmetries ``delta_k``.

    ``d`` has shape ``(..., 6)`` in LINKS order.  Returns two ``(..., 3)`` arrays
    indexed by arm ``k = 1, 2, 3`` (positions 0, 1, 2).
    """
    d12, d23, d31, d13, d32, d21 = (d[..., i] for i in range(6))
    lbar = xp.stack([0.5 * (d23 + d32), 0.5 * (d31 + d13), 0.5 * (d12 + d21)], axis=-1)
    delta = xp.stack([0.5 * (d23 - d32), 0.5 * (d31 - d13), 0.5 * (d12 - d21)], axis=-1)
    return lbar, delta


def tdi2_geometry(f, d, xp=np):
    """Geometric factors of the TDI-2 ``XYZ`` covariance for frozen delays.

    Returns ``(G_tm, G_oms)``, complex arrays of shape ``f.shape + (3, 3)``, such
    that ``C_XYZ(f) = A^2 S_tm(f) G_tm + P^2 S_oms(f) G_oms``.  ``d`` is a single
    set of six delays (shape ``(6,)``).

    TDI-1 part (main.tex Sec. "Exact covariance", generalised to unequal up/down
    links): with ``theta_k = w Lbar_k``, ``s_k = sin theta_k``, ``c_k = cos theta_k``

        S_XX = 8 { s3^2 [2(1+c2^2) Sa + Sn] + s2^2 [2(1+c3^2) Sa + Sn] }
        S_XY = -8 s1 s2 c3 (4 Sa + Sn) exp[-i w (Lbar2 - Lbar1 + delta3)]

    and cyclically.  TDI 2 multiplies channel ``X`` by ``1 - D12 D21 D13 D31``,
    i.e. ``C2 = G C1 G^dagger`` with ``G_X = 2 i sin(w(Lbar2+Lbar3)) e^{-i w
    (Lbar2+Lbar3)}``, which doubles the delay phases of the cross-spectra.
    """
    f = xp.asarray(f)
    w = 2 * xp.pi * f[..., None]                       # (..., 1)
    lbar, delta = arm_quantities(xp.asarray(d), xp)    # (3,), (3,)
    th = w * lbar                                      # (..., 3)
    s, c = xp.sin(th), xp.cos(th)
    s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2]
    c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
    L1, L2, L3 = lbar[0], lbar[1], lbar[2]
    wf = w[..., 0]

    # TDI-1 diagonal coefficients of Sa and Sn
    dtm = xp.stack([16 * (s3**2 * (1 + c2**2) + s2**2 * (1 + c3**2)),
                    16 * (s1**2 * (1 + c3**2) + s3**2 * (1 + c1**2)),
                    16 * (s2**2 * (1 + c1**2) + s1**2 * (1 + c2**2))], axis=-1)
    doms = xp.stack([8 * (s3**2 + s2**2), 8 * (s1**2 + s3**2), 8 * (s2**2 + s1**2)], axis=-1)
    # TDI-1 off-diagonal (common factor 4 Sa + Sn)
    oxy = -8 * s1 * s2 * c3 * xp.exp(-1j * wf * (L2 - L1 + delta[2]))
    oyz = -8 * s2 * s3 * c1 * xp.exp(-1j * wf * (L3 - L2 + delta[0]))
    ozx = -8 * s3 * s1 * c2 * xp.exp(-1j * wf * (L1 - L3 + delta[1]))

    # TDI-2 factors G_P
    gx = 1 - xp.exp(-2j * wf * (L2 + L3))
    gy = 1 - xp.exp(-2j * wf * (L3 + L1))
    gz = 1 - xp.exp(-2j * wf * (L1 + L2))
    g = xp.stack([gx, gy, gz], axis=-1)                # (..., 3)
    gg = g[..., :, None] * xp.conj(g[..., None, :])    # (..., 3, 3)

    zero = xp.zeros_like(oxy)
    off = xp.stack([xp.stack([zero, oxy, xp.conj(ozx)], -1),
                    xp.stack([xp.conj(oxy), zero, oyz], -1),
                    xp.stack([ozx, xp.conj(oyz), zero], -1)], -2)
    eye = xp.eye(3)
    g_tm = gg * (dtm[..., :, None] * eye + 4 * off)
    g_oms = gg * (doms[..., :, None] * eye + off)
    return g_tm, g_oms


def tdi2_unit_cov(f, d, xp=np):
    """Unit-amplitude TDI-2 XYZ covariances ``(U_tm, U_oms)``: ``C = A^2 U_tm + P^2 U_oms``."""
    g_tm, g_oms = tdi2_geometry(f, d, xp)
    f = xp.asarray(f)
    return (psd_tm_unit(f, xp)[..., None, None] * g_tm,
            psd_oms_unit(f, xp)[..., None, None] * g_oms)


def equal_arm_aet_psd(f, L, xp=np):
    """Unit-amplitude diagonal TDI-2 ``A, E, T`` PSDs for six equal constant delays ``L``.

    Returns ``(u_tm, u_oms)`` of shape ``f.shape + (3,)`` (channels A, E, T):
    main.tex ``S_A = S_E = D - O``, ``S_T = D + 2 O`` times ``4 sin^2(2 w L)``.
    """
    th = 2 * xp.pi * f * L
    s, c = xp.sin(th), xp.cos(th)
    g2 = 4 * xp.sin(2 * th) ** 2
    ae_tm = g2 * 32 * s**2 * (1 + c + c**2)
    ae_oms = g2 * 8 * s**2 * (2 + c)
    t_tm = g2 * 32 * s**2 * (1 - c) ** 2
    t_oms = g2 * 16 * s**2 * (1 - c)
    stm, soms = psd_tm_unit(f, xp), psd_oms_unit(f, xp)
    u_tm = xp.stack([ae_tm, ae_tm, t_tm], -1) * stm[..., None]
    u_oms = xp.stack([ae_oms, ae_oms, t_oms], -1) * soms[..., None]
    return u_tm, u_oms


def to_aet(m):
    """Rotate ``(..., 3, 3)`` XYZ matrices (or ``(3, ...)`` data) to the AET basis."""
    m = np.asarray(m)
    return np.einsum("ai,...ij,bj->...ab", R_AET, m, R_AET)


# ---------------------------------------------------------------------------
# Brute-force transfer matrix from the pytdi delay polynomials (validation)
# ---------------------------------------------------------------------------

def pytdi_transfer_matrix(f, d, gen=2):
    """``M(f)``, shape ``(len(f), 3, 12)``: TDI output ``(X, Y, Z)`` per unit
    fractional-frequency noise ``(N_12..N_21, a_12..a_21)`` for constant delays.

    Obtained by replacing every delay operator ``D_ij`` of the pytdi Michelson
    combinations by ``exp(-2 pi i f d_ij)`` and using ``eta_ij = N_ij + a_ij +
    D_ij a_ji``.  Independent of :func:`tdi2_geometry`, hence a check of it.
    """
    from pytdi import michelson
    combs = {1: (michelson.X1_ETA, michelson.Y1_ETA, michelson.Z1_ETA),
             2: (michelson.X2_ETA, michelson.Y2_ETA, michelson.Z2_ETA)}[gen]
    f = np.asarray(f, float)
    dd = dict(zip(LINKS, np.asarray(d, float)))
    z = {l: np.exp(-2j * np.pi * f * dd[l]) for l in LINKS}
    M = np.zeros((f.size, 3, 12), complex)
    for c, comb in enumerate(combs):
        for meas, poly in comb.components.items():
            ij = meas.split("_")[1]
            tf = np.zeros(f.size, complex)
            for fac, ops in poly:
                tau = sum(dd[op.split("_")[1]] for op in ops) if ops else 0.0
                if any(op.startswith("A") for op in ops):
                    raise NotImplementedError("advancements not expected here")
                tf += fac * np.exp(-2j * np.pi * f * tau)
            li = LINKS.index(ij)
            ji = LINKS.index(ij[::-1])
            M[:, c, li] += tf                 # OMS N_ij
            M[:, c, 6 + li] += tf             # local test mass a_ij
            M[:, c, 6 + ji] += tf * z[ij]     # remote test mass a_ji, delayed by d_ij
    return M


# ---------------------------------------------------------------------------
# Noise simulation: link noises from the PSDs -> eta_ij -> pytdi TDI 2
# ---------------------------------------------------------------------------

def draw_colored(rng, n, dt, psd_fn, size=1):
    """``size`` independent real series of length ``n`` with one-sided PSD ``psd_fn``,
    drawn on the rfft grid (hence periodic with period ``n``)."""
    f = np.fft.rfftfreq(n, dt)
    amp = np.zeros_like(f)
    amp[1:] = np.sqrt(psd_fn(f[1:]) * n / (2 * dt))
    z = (rng.standard_normal((size, f.size)) + 1j * rng.standard_normal((size, f.size))) / np.sqrt(2)
    xf = amp * z
    if n % 2 == 0:
        xf[:, -1] = xf[:, -1].real * np.sqrt(2)
    return np.fft.irfft(xf, n=n, axis=-1)


def simulate_tdi2(rng, n, dt, A, P, delays, *, periodic=False, margin=128, order=31):
    """Simulate ``(3, n)`` TDI-2 ``X2, Y2, Z2`` of TM + OMS noise with pytdi.

    Parameters
    ----------
    delays
        Either a length-6 sequence of constant delays [s] (LINKS order) or a
        callable ``t -> (len(t), 6)`` array (e.g. ``orbits.compute_ltt``) giving
        the time-varying light travel times at sample times ``t = k dt`` (``k``
        counted from the first returned sample).
    periodic
        Constant delays only.  Link noises are drawn with period ``n`` and
        extended periodically, so the output is exactly the circular TDI of a
        stationary process (no edge effects, exact DFT diagonalisation).
    """
    from pytdi import michelson, dsp
    fs = 1.0 / dt
    const = not callable(delays)
    if periodic and not const:
        raise ValueError("periodic simulation requires constant delays")
    ntot = n + 2 * margin
    if periodic:
        base = np.concatenate([draw_colored(rng, n, dt, lambda f: psd_oms_unit(f) * P**2, 6),
                               draw_colored(rng, n, dt, lambda f: psd_tm_unit(f) * A**2, 6)])
        noise = np.concatenate([base[:, -margin:], base, base[:, :margin]], axis=1)
    else:
        noise = np.concatenate([draw_colored(rng, ntot, dt, lambda f: psd_oms_unit(f) * P**2, 6),
                                draw_colored(rng, ntot, dt, lambda f: psd_tm_unit(f) * A**2, 6)])
    if const:
        dmap = {f"d_{l}": float(v) for l, v in zip(LINKS, delays)}
    else:
        t = (np.arange(ntot) - margin) * dt
        ltt = np.asarray(delays(t))
        dmap = {f"d_{l}": ltt[:, i] for i, l in enumerate(LINKS)}
    eta = {}
    for i, l in enumerate(LINKS):
        j = LINKS.index(l[::-1])
        remote = dsp.timeshift(noise[6 + j], -dmap[f"d_{l}"] * fs, order)
        eta[f"eta_{l}"] = noise[i] + noise[6 + i] + remote
    out = []
    for comb in (michelson.X2_ETA, michelson.Y2_ETA, michelson.Z2_ETA):
        built = comb.build(dmap, fs)
        out.append(np.asarray(built(eta, order=order, unit="phase"))[margin:margin + n])
    return np.stack(out)


# ---------------------------------------------------------------------------
# Grids and cells
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """Sampling and WDM tiling: ``N = nt * nf`` samples of cadence ``dt``."""

    dt: float = 10.0
    nt: int = 3072
    nf: int = 1024

    @property
    def n(self):
        return self.nt * self.nf

    @property
    def T(self):
        return self.n * self.dt

    @property
    def delta_t(self):
        """WDM time-bin width ``nf dt``."""
        return self.nf * self.dt

    @property
    def delta_f(self):
        """WDM band spacing ``1/(2 nf dt)``."""
        return 1.0 / (2.0 * self.nf * self.dt)

    @property
    def df(self):
        """FD bin width ``1/(N dt)``."""
        return 1.0 / (self.n * self.dt)


def log_edges(m_lo, m_hi, ncell):
    """Strictly increasing integer edges covering bands ``m_lo..m_hi`` (inclusive),
    roughly log-spaced, at least one band per cell."""
    e = np.unique(np.round(np.geomspace(m_lo, m_hi + 1, ncell + 1)).astype(int))
    e[0], e[-1] = m_lo, m_hi + 1
    return np.unique(e)


@dataclass
class Cells:
    """Common frequency cells for FD and WDM, plus the WDM time cells.

    ``medges``: band-index edges (cell ``c`` holds WDM bands ``medges[c] ..
    medges[c+1]-1``).  The matching FD bin ``c`` holds the DFT bins whose
    frequency lies in ``[f_{m_lo} - df/2, f_{m_hi} + df/2)`` with ``df`` the WDM
    band spacing, i.e. exactly the frequency span of the WDM bands.
    ``tedges``: WDM time-bin edges (cells of whole WDM time bins).
    """

    grid: Grid
    medges: np.ndarray
    tedges: np.ndarray

    @property
    def nfcell(self):
        return len(self.medges) - 1

    @property
    def ntcell(self):
        return len(self.tedges) - 1

    def kedges(self):
        """DFT-bin edges of the FD coarse bins (``q_m = m nt/2`` is the centre of band m)."""
        half = self.grid.nt // 2
        return self.medges * half - half // 2

    def fcentres(self):
        """Geometric-mean frequency of each frequency cell [Hz]."""
        m = self.medges.astype(float)
        return np.sqrt(m[:-1] * (m[1:] - 1)) * self.grid.delta_f

    def fd_dof(self):
        return np.diff(self.kedges()).astype(float)

    def wdm_dof(self):
        return np.outer(np.diff(self.tedges), np.diff(self.medges)).astype(float)


# ---------------------------------------------------------------------------
# FD data products and model surfaces
# ---------------------------------------------------------------------------

def tukey(n, taper):
    """Tukey window with ``taper`` samples of cosine roll-off at each end."""
    w = np.ones(n)
    if taper > 0:
        x = np.arange(taper) / taper
        ramp = 0.5 * (1 - np.cos(np.pi * x))
        w[:taper] = ramp
        w[-taper:] = ramp[::-1]
    return w


def fd_periodograms(xyz, grid, cells, window=None):
    """Coarse-grained FD periodogram matrices in the AET basis.

    Returns ``(Ibar, K)``: ``(nfcell, 3, 3)`` complex averages of
    ``I_k = 2 dt X_k X_k^dagger / sum(w^2)`` and the number of DFT bins per cell.
    """
    w = np.ones(grid.n) if window is None else window
    aet = np.einsum("ai,in->an", R_AET, xyz)
    X = np.fft.rfft(aet * w, axis=-1)                     # (3, n/2+1)
    norm = 2 * grid.dt / np.sum(w**2)
    ke = cells.kedges()
    out = np.zeros((cells.nfcell, 3, 3), complex)
    for c in range(cells.nfcell):
        blk = X[:, ke[c]:ke[c + 1]]
        out[c] = norm * (blk @ blk.conj().T) / blk.shape[1]
    return out, np.diff(ke).astype(float)


def fd_model_static(cells, d):
    """FD model surfaces (AET basis) for constant delays: coarse-bin averages of the
    exact covariance on the DFT grid.  Returns ``(U_tm, U_oms)`` ``(nfcell, 3, 3)``."""
    grid = cells.grid
    ke = cells.kedges()
    f = np.arange(ke[0], ke[-1]) * grid.df
    utm, uoms = tdi2_unit_cov(f, d)
    utm, uoms = to_aet(utm), to_aet(uoms)
    idx = ke - ke[0]
    avg = lambda u: np.add.reduceat(u, idx[:-1], axis=0) / np.diff(idx)[:, None, None]
    return avg(utm), avg(uoms)


def time_hat_weights(tnodes, tsamp, wsamp2):
    """Weights ``omega_i`` such that ``sum_k w_k^2 g(t_k) ~= sum_i omega_i g(t_i)`` for a
    function ``g`` linearly interpolated between the nodes ``tnodes``."""
    pos = np.interp(tsamp, tnodes, np.arange(tnodes.size))
    i0 = np.clip(np.floor(pos).astype(int), 0, tnodes.size - 2)
    fr = pos - i0
    om = np.zeros(tnodes.size, dtype=np.result_type(wsamp2, float))   # complex for cyclic weights
    np.add.at(om, i0, wsamp2 * (1 - fr))
    np.add.at(om, i0 + 1, wsamp2 * fr)
    return om


def fd_model_timeavg(cells, ltt_fn, tnodes, window=None, chunk=None):
    """FD model surfaces (AET) for time-varying delays: ``sum_t w_t^2 C(f, t) / sum w^2``
    evaluated on the DFT grid and averaged over each coarse bin.

    ``ltt_fn(t)`` returns the ``(len(t), 6)`` delays; the covariance is evaluated at
    the time nodes ``tnodes`` and integrated with piecewise-linear weights.
    """
    grid = cells.grid
    w = np.ones(grid.n) if window is None else window
    tsamp = np.arange(grid.n) * grid.dt
    om = time_hat_weights(tnodes, tsamp, w**2)
    om = om / om.sum()
    dnodes = np.asarray(ltt_fn(tnodes))
    ke = cells.kedges()
    f = jnp.asarray(np.arange(ke[0], ke[-1]) * grid.df)
    acc = jax.jit(lambda a, b, u, v, o: (a + o * u, b + o * v))
    acc_tm = jnp.zeros((f.size, 3, 3), complex)
    acc_oms = jnp.zeros((f.size, 3, 3), complex)
    for i in range(tnodes.size):
        if om[i] == 0:
            continue
        utm, uoms = unit_cov_aet_jax(f, jnp.asarray(dnodes[i]))
        acc_tm, acc_oms = acc(acc_tm, acc_oms, utm, uoms, om[i])
    idx = ke - ke[0]
    avg = lambda u: np.add.reduceat(np.asarray(u), idx[:-1], axis=0) / np.diff(idx)[:, None, None]
    return avg(acc_tm), avg(acc_oms)


def fd_model_equal_arm(cells, L):
    """Standard equal-arm diagonal AET model surfaces ``(nfcell, 3)``."""
    grid = cells.grid
    ke = cells.kedges()
    f = np.arange(ke[0], ke[-1]) * grid.df
    utm, uoms = equal_arm_aet_psd(f, L)
    idx = ke - ke[0]
    avg = lambda u: np.add.reduceat(u, idx[:-1], axis=0) / np.diff(idx)[:, None]
    return avg(utm), avg(uoms)


# ---------------------------------------------------------------------------
# WDM data products and model surfaces
# ---------------------------------------------------------------------------

def wdm_transform(xyz, grid):
    """WDM coefficients ``(3, nt, nf+1)`` of the AET channels (wdm_transform, numpy)."""
    from wdm_transform import TimeSeries
    aet = np.einsum("ai,in->an", R_AET, xyz)
    return np.asarray(TimeSeries(aet, dt=grid.dt).to_wdm(nt=grid.nt).coeffs)


def wdm_cells(coeffs, cells):
    """Cell averages ``(ntcell, nfcell, 3, 3)`` of ``y = 2 dt w w^T / N`` and dof."""
    grid = cells.grid
    te, me = cells.tedges, cells.medges
    w = coeffs[:, te[0]:te[-1], me[0]:me[-1]]
    y = np.einsum("inm,jnm->nmij", w, w) * (2 * grid.dt / grid.n)
    s = np.add.reduceat(np.add.reduceat(y, te[:-1] - te[0], axis=0), me[:-1] - me[0], axis=1)
    dof = cells.wdm_dof()
    return s / dof[..., None, None], dof


def meyer_window_sq(nt):
    """``Phi(j)^2`` for ``|j| <= nt/3`` (DFT-bin units), summing to 1."""
    from wdm_transform.windows import phi_unit
    from wdm_transform.backends import get_backend
    j = np.arange(-(nt // 3), nt // 3 + 1)
    phi = np.asarray(phi_unit(get_backend("numpy"), j / (nt / 2), 1 / 3, 1.0)) * np.sqrt(2 / nt)
    return j, phi**2


def wdm_band_average(u_fine, lo, cells):
    """Window-average ``(nfine, 3, 3)`` spectra sampled on DFT bins ``lo, lo+1, ...``
    onto the WDM bands ``medges[0] .. medges[-1]-1``: ``sum_j Phi(j)^2 u(q_m + j)``.
    Returns the real part, shape ``(nband, 3, 3)``."""
    grid = cells.grid
    j, phi2 = meyer_window_sq(grid.nt)
    m = np.arange(cells.medges[0], cells.medges[-1])
    q = m * (grid.nt // 2)
    out = np.zeros((m.size, 3, 3))
    ur = np.real(u_fine)
    for k, qq in enumerate(q):
        out[k] = np.tensordot(phi2, ur[qq + j - lo], axes=(0, 0))
    return out


def wdm_fine_range(cells):
    """DFT-bin range ``[lo, hi)`` needed by :func:`wdm_band_average`."""
    grid = cells.grid
    L = grid.nt // 3
    return cells.medges[0] * (grid.nt // 2) - L, (cells.medges[-1] - 1) * (grid.nt // 2) + L + 1


def wdm_model_static(cells, d):
    """WDM model surfaces (AET, real) for constant delays: ``(nfcell, 3, 3)`` each,
    identical for every time cell."""
    grid = cells.grid
    lo, hi = wdm_fine_range(cells)
    f = np.arange(lo, hi) * grid.df
    utm, uoms = tdi2_unit_cov(f, d)
    btm = wdm_band_average(to_aet(utm), lo, cells)
    boms = wdm_band_average(to_aet(uoms), lo, cells)
    idx = cells.medges - cells.medges[0]
    avg = lambda u: np.add.reduceat(u, idx[:-1], axis=0) / np.diff(idx)[:, None, None]
    return avg(btm), avg(boms)


def wdm_model_equal_arm(cells, L):
    """Equal-arm diagonal AET WDM model ``(nfcell, 3)`` (window-averaged)."""
    grid = cells.grid
    lo, hi = wdm_fine_range(cells)
    f = np.arange(lo, hi) * grid.df
    f = np.where(f > 0, f, grid.df)
    utm, uoms = equal_arm_aet_psd(f, L)
    j, phi2 = meyer_window_sq(grid.nt)
    m = np.arange(cells.medges[0], cells.medges[-1])
    q = m * (grid.nt // 2)
    btm = np.stack([phi2 @ utm[qq + j - lo] for qq in q])
    boms = np.stack([phi2 @ uoms[qq + j - lo] for qq in q])
    idx = cells.medges - cells.medges[0]
    avg = lambda u: np.add.reduceat(u, idx[:-1], axis=0) / np.diff(idx)[:, None]
    return avg(btm), avg(boms)


@jax.jit
def unit_cov_aet_jax(f, d):
    """JAX: unit TM and OMS covariances on frequencies ``f`` for delays ``d``, AET basis."""
    utm, uoms = tdi2_unit_cov(f, d, xp=jnp)
    R = jnp.asarray(R_AET)
    rot = lambda m: jnp.einsum("ai,...ij,bj->...ab", R, m, R)
    return rot(utm), rot(uoms)


def wdm_band_nodes(cells, ltt_fn, tnodes):
    """Window-averaged (real) unit covariances on every WDM band of the analysis
    range, at each time node: two arrays ``(len(tnodes), nband, 3, 3)``."""
    grid = cells.grid
    lo, hi = wdm_fine_range(cells)
    f = jnp.asarray(np.arange(lo, hi) * grid.df)
    j, phi2 = meyer_window_sq(grid.nt)
    m = np.arange(cells.medges[0], cells.medges[-1])
    idx = jnp.asarray(m[:, None] * (grid.nt // 2) + j[None, :] - lo)
    phi2 = jnp.asarray(phi2)

    @jax.jit
    def one(d):
        utm, uoms = unit_cov_aet_jax(f, d)
        avg = lambda u: jnp.einsum("j,mjab->mab", phi2, jnp.real(u)[idx])
        return avg(utm), avg(uoms)

    dnodes = np.asarray(ltt_fn(tnodes))
    out = [one(jnp.asarray(dn)) for dn in dnodes]
    return (np.stack([np.asarray(o[0]) for o in out]),
            np.stack([np.asarray(o[1]) for o in out]))


def wdm_model_local(cells, ltt_fn, tnodes, t_offset=0.0, nodes=None):
    """Time-local WDM model surfaces (AET, real): ``(ntcell, nfcell, 3, 3)`` each.

    At each time node the frozen-arm covariance (closed form) is window-averaged
    onto the WDM bands; the result is interpolated in time (cubic spline) to the
    WDM bin times ``t_n = t_offset + n nf dt`` and averaged over the time cells.
    ``nodes`` may pass precomputed :func:`wdm_band_nodes` output.
    """
    from scipy.interpolate import CubicSpline
    grid = cells.grid
    btm, boms = wdm_band_nodes(cells, ltt_fn, tnodes) if nodes is None else nodes
    tn = t_offset + np.arange(cells.tedges[0], cells.tedges[-1]) * grid.delta_t
    fine_tm = CubicSpline(tnodes, btm, axis=0)(tn)
    fine_oms = CubicSpline(tnodes, boms, axis=0)(tn)
    ti = cells.tedges - cells.tedges[0]
    mi = cells.medges - cells.medges[0]

    def avg(u):
        s = np.add.reduceat(np.add.reduceat(u, ti[:-1], axis=0), mi[:-1], axis=1)
        return s / cells.wdm_dof()[..., None, None]
    return avg(fine_tm), avg(fine_oms)


# ---------------------------------------------------------------------------
# Likelihoods (JAX)
# ---------------------------------------------------------------------------

def _herm3_logdet_trace(c, y):
    """``log det C`` and ``tr(C^{-1} Y)`` for batches of Hermitian (or real symmetric)
    3x3 matrices, by cofactor expansion (much faster than batched LAPACK)."""
    a, b, d = c[..., 0, 0].real, c[..., 0, 1], c[..., 0, 2]
    e, g = c[..., 1, 1].real, c[..., 1, 2]
    h = c[..., 2, 2].real
    # adjugate of [[a, b, d], [b*, e, g], [d*, g*, h]]
    adj00 = e * h - (g * jnp.conj(g)).real
    adj11 = a * h - (d * jnp.conj(d)).real
    adj22 = a * e - (b * jnp.conj(b)).real
    adj01 = d * jnp.conj(g) - b * h          # (C^{-1})_{01} * det
    adj02 = b * g - d * e
    adj12 = d * jnp.conj(b) - a * g
    det = (a * adj00 + b * jnp.conj(adj01) + d * jnp.conj(adj02)).real
    # tr(adj(C) Y) with adj Hermitian
    tr = (adj00 * y[..., 0, 0].real + adj11 * y[..., 1, 1].real + adj22 * y[..., 2, 2].real
          + 2 * (adj01 * y[..., 1, 0]).real + 2 * (adj02 * y[..., 2, 0]).real
          + 2 * (adj12 * y[..., 2, 1]).real)
    return jnp.log(det), tr / det


def _cell_scale(u_tm, u_oms, theta_ref=(A_TRUE, P_TRUE)):
    """Per-cell positive scale (mean diagonal of the reference covariance).  Dividing
    data and model by it leaves the likelihood unchanged up to a constant but keeps
    determinants O(1): with entries ~1e-43 and a null channel 1e-7 smaller,
    ``det C ~ 1e-136`` and its square underflows in second derivatives."""
    c = theta_ref[0] ** 2 * np.asarray(u_tm) + theta_ref[1] ** 2 * np.asarray(u_oms)
    if c.ndim >= 2 and c.shape[-1] == c.shape[-2] == 3:
        return np.real(np.trace(c, axis1=-2, axis2=-1)) / 3.0
    return np.mean(np.real(c), axis=-1)


def make_loglik_full(y, dof, u_tm, u_oms, complex_data):
    """Wishart log-likelihood with full 3x3 covariance ``A^2 U_tm + P^2 U_oms``.

    ``complex_data=True``: complex Wishart (FD), weight ``dof``;
    ``False``: real Wishart (WDM), weight ``dof/2``.  Data and model are rescaled
    cell by cell (:func:`_cell_scale`); the returned value includes the constant.
    """
    sc = _cell_scale(u_tm, u_oms)
    sc = np.broadcast_to(sc, np.broadcast_shapes(sc.shape, np.shape(dof)))
    y = jnp.asarray(np.asarray(y) / sc[..., None, None])
    u_tm = jnp.asarray(np.asarray(u_tm) / np.broadcast_to(sc, np.shape(u_tm)[:-2])[..., None, None])
    u_oms = jnp.asarray(np.asarray(u_oms) / np.broadcast_to(sc, np.shape(u_oms)[:-2])[..., None, None])
    wgt_np = np.asarray(dof) if complex_data else 0.5 * np.asarray(dof)
    const = -float(np.sum(wgt_np * 3 * np.log(sc)))
    wgt = jnp.asarray(wgt_np)

    def loglik(theta):
        c = theta[0] ** 2 * u_tm + theta[1] ** 2 * u_oms
        ld, tr = _herm3_logdet_trace(c, y)
        return -jnp.sum(wgt * (ld + tr)) + const
    return jax.jit(loglik)


def make_loglik_diag(y, dof, u_tm, u_oms, complex_data, channels=(0, 1, 2)):
    """Wishart log-likelihood ignoring channel correlations (diagonal AET model).

    ``y``: ``(..., 3, 3)`` data matrices (only the diagonal is used);
    ``u_tm, u_oms``: ``(..., 3)`` model diagonals.
    """
    ch = np.asarray(channels)
    sc = A_TRUE**2 * np.asarray(u_tm)[..., ch] + P_TRUE**2 * np.asarray(u_oms)[..., ch]
    yd_np = np.real(np.diagonal(y, axis1=-2, axis2=-1))[..., ch]
    sc = np.broadcast_to(sc, np.broadcast_shapes(sc.shape, yd_np.shape))
    yd = jnp.asarray(yd_np / sc)
    u_tm = jnp.asarray(np.asarray(u_tm)[..., ch] / sc)
    u_oms = jnp.asarray(np.asarray(u_oms)[..., ch] / sc)
    wgt_np = np.broadcast_to((np.asarray(dof) if complex_data else 0.5 * np.asarray(dof))[..., None], sc.shape)
    const = -float(np.sum(wgt_np * np.log(sc)))
    wgt = jnp.asarray(wgt_np)

    def loglik(theta):
        c = theta[0] ** 2 * u_tm + theta[1] ** 2 * u_oms
        return -jnp.sum(wgt * (jnp.log(c) + yd / c)) + const
    return jax.jit(loglik)


def make_loglik_full_lapack(y, dof, u_tm, u_oms, complex_data):
    """Same as :func:`make_loglik_full` but with numerical (LAPACK) ``slogdet`` and
    ``solve`` instead of the closed-form cofactors: the reference "invert the 3x3
    covariance numerically at every pixel" implementation, for timing only."""
    sc = _cell_scale(u_tm, u_oms)
    sc = np.broadcast_to(sc, np.broadcast_shapes(sc.shape, np.shape(dof)))
    y = jnp.asarray(np.asarray(y) / sc[..., None, None])
    u_tm = jnp.asarray(np.asarray(u_tm) / sc[..., None, None])
    u_oms = jnp.asarray(np.asarray(u_oms) / sc[..., None, None])
    wgt_np = np.asarray(dof) if complex_data else 0.5 * np.asarray(dof)
    const = -float(np.sum(wgt_np * 3 * np.log(sc)))
    wgt = jnp.asarray(wgt_np)

    def loglik(theta):
        c = theta[0] ** 2 * u_tm + theta[1] ** 2 * u_oms
        ld = jnp.linalg.slogdet(c)[1]
        tr = jnp.real(jnp.trace(jnp.linalg.solve(c, y), axis1=-2, axis2=-1))
        return -jnp.sum(wgt * (ld + tr)) + const
    return jax.jit(loglik)


# ---------------------------------------------------------------------------
# Diagonal form: simultaneous diagonalisation of the TM and OMS covariances
# ---------------------------------------------------------------------------
#
# For any frozen delays (TDI 1 or 2, unequal up/down links) the TM and OMS parts of
# the closed form share their off-diagonal elements up to a factor 4 and differ only
# on the diagonal:  G_tm = 4 K - 32 diag(Delta),  G_oms = K,  Delta_X = |G_X|^2 s2^2 s3^2
# (and cyclically), so that  C = (4 Sa + Sn) K - 32 Sa diag(Delta).  With
# D = diag(sqrt(Delta)) and B = D^-1 K D^-1 / 8 = sum_j x_j u_j u_j^T, the basis
# v_j = D^-1 u_j / sqrt(8 x_j) gives  v_j^T C v_k = delta_jk (Sn + kappa_j Sa),
# kappa_j = 4 (1 - 1/x_j): a basis independent of A, P and of the PSD shapes.

def tdi2_delta(f, d, xp=np):
    """Diagonal TM deficit ``Delta`` ``(..., 3)`` of :func:`tdi2_geometry`:
    ``G_tm = 4 G_oms - 32 diag(Delta)``, ``Delta_X = |G_X|^2 sin^2(w L2) sin^2(w L3)``."""
    f = xp.asarray(f)
    w = 2 * xp.pi * f[..., None]
    lbar, _ = arm_quantities(xp.asarray(d), xp)
    s2 = xp.sin(w * lbar) ** 2                                   # (..., 3) arms 1, 2, 3
    lsum = xp.stack([lbar[1] + lbar[2], lbar[2] + lbar[0], lbar[0] + lbar[1]])
    g2 = 4 * xp.sin(w * lsum) ** 2                               # |G_X|^2, |G_Y|^2, |G_Z|^2
    return g2 * xp.stack([s2[..., 1] * s2[..., 2], s2[..., 2] * s2[..., 0],
                          s2[..., 0] * s2[..., 1]], axis=-1)


def sym3_eigh(b, xp=np):
    """Closed-form eigen-decomposition of batches of real symmetric 3x3 matrices.

    Eigenvalues (ascending) from the invariants ``I1 = tr B``, ``I2`` (sum of the
    principal 2x2 minors), ``I3 = det B`` by the trigonometric solution of
    ``x^3 - I1 x^2 + I2 x - I3 = 0``; the smallest one as ``I3 / (x2 x3)``, which
    avoids the cancellation when it is much smaller than the others (null channel).
    Eigenvectors: for the eigenvalue farthest from the other two, the largest column
    of the rank-one adjugate ``adj(B - x I)``; for the other two, the 2x2 rotation
    that diagonalises ``B`` in the orthogonal plane (robust to degeneracies, e.g.
    A and E for equal arms).
    """
    b00, b11, b22 = b[..., 0, 0], b[..., 1, 1], b[..., 2, 2]
    b01, b02, b12 = b[..., 0, 1], b[..., 0, 2], b[..., 1, 2]
    i1 = b00 + b11 + b22
    i2 = b00 * b11 + b11 * b22 + b22 * b00 - b01**2 - b02**2 - b12**2
    i3 = (b00 * (b11 * b22 - b12**2) - b01 * (b01 * b22 - b12 * b02)
          + b02 * (b01 * b12 - b11 * b02))
    pi = xp.maximum(i1**2 - 3 * i2, 0.0)
    cphi = (2 * i1**3 - 9 * i1 * i2 + 27 * i3) / (2 * pi**1.5 + 1e-300)
    phi = xp.arccos(xp.clip(cphi, -1.0, 1.0)) / 3
    x3 = i1 / 3 + 2 / 3 * xp.sqrt(pi) * xp.cos(phi)
    x2 = i1 / 3 + 2 / 3 * xp.sqrt(pi) * xp.cos(phi - 2 * xp.pi / 3)
    x1 = i3 / (x2 * x3)
    x = xp.stack([x1, x2, x3], axis=-1)

    unit = lambda v: v / xp.sqrt(xp.sum(v**2, axis=-1, keepdims=True))
    cross = lambda p, q: xp.stack([p[..., 1] * q[..., 2] - p[..., 2] * q[..., 1],
                                   p[..., 2] * q[..., 0] - p[..., 0] * q[..., 2],
                                   p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]], -1)
    quad = lambda p, q: xp.einsum("...i,...ij,...j->...", p, b, q)

    iso_low = (x2 - x1) > (x3 - x2)                               # isolated eigenvalue: x1, else x3
    xi = xp.where(iso_low, x1, x3)
    a00, a11, a22 = b00 - xi, b11 - xi, b22 - xi
    adj = xp.stack([xp.stack([a11 * a22 - b12**2, b02 * b12 - b01 * a22, b01 * b12 - b02 * a11], -1),
                    xp.stack([b02 * b12 - b01 * a22, a00 * a22 - b02**2, b01 * b02 - a00 * b12], -1),
                    xp.stack([b01 * b12 - b02 * a11, b01 * b02 - a00 * b12, a00 * a11 - b01**2], -1)],
                   -1)                                            # columns of adj(B - x I)
    k = xp.argmax(xp.sum(adj**2, axis=-2), axis=-1)
    ui = unit(xp.take_along_axis(adj, k[..., None, None], axis=-1)[..., 0])
    # orthonormal pair (p, q) spanning the plane orthogonal to ui, then a 2x2 rotation
    ek = (xp.arange(3) == xp.argmin(xp.abs(ui), axis=-1)[..., None]).astype(ui.dtype)
    p = unit(cross(ui, ek))
    q = cross(ui, p)
    ang = 0.5 * xp.arctan2(2 * quad(p, q), quad(p, p) - quad(q, q))
    ca, sa = xp.cos(ang)[..., None], xp.sin(ang)[..., None]
    w_hi, w_lo = ca * p + sa * q, ca * q - sa * p                 # larger, smaller eigenvalue
    il = iso_low[..., None]
    u = xp.stack([xp.where(il, ui, w_lo), xp.where(il, w_lo, w_hi), xp.where(il, w_hi, ui)],
                 axis=-1)                                         # columns = eigenvectors
    return x, u


def kappa_closed_form(f, d, xp=np):
    """``kappa_j`` (ascending) of the complex covariance (FD, TDI 1 or 2) from the cubic
    of main.tex: roots ``zeta`` of ``z^3 - 2 e1 z^2 + (e1 + 3 e2) z - 2 [e2 + e3 (1 -
    c1 c2 c3 cos(w Delta_S / 2))]``, with ``e_k`` the elementary symmetric polynomials
    of ``1/sin^2(w Lbar_k)``, and ``kappa = 4 (1 - 1/zeta)``."""
    f = xp.asarray(f)
    w = 2 * xp.pi * f[..., None]
    lbar, _ = arm_quantities(xp.asarray(d), xp)
    th = w * lbar
    a, c = 1 / xp.sin(th) ** 2, xp.cos(th)
    e1 = a[..., 0] + a[..., 1] + a[..., 2]
    e2 = a[..., 0] * a[..., 1] + a[..., 1] * a[..., 2] + a[..., 2] * a[..., 0]
    e3 = a[..., 0] * a[..., 1] * a[..., 2]
    d12, d23, d31, d13, d32, d21 = (d[..., i] for i in range(6))
    sagnac = d12 + d23 + d31 - d13 - d32 - d21
    i1, i2 = 2 * e1, e1 + 3 * e2
    i3 = 2 * (e2 + e3 * (1 - c[..., 0] * c[..., 1] * c[..., 2] * xp.cos(w[..., 0] * sagnac / 2)))
    pi = i1**2 - 3 * i2
    phi = xp.arccos(xp.clip((2 * i1**3 - 9 * i1 * i2 + 27 * i3) / (2 * pi**1.5), -1.0, 1.0)) / 3
    z3 = i1 / 3 + 2 / 3 * xp.sqrt(pi) * xp.cos(phi)
    z2 = i1 / 3 + 2 / 3 * xp.sqrt(pi) * xp.cos(phi - 2 * xp.pi / 3)
    z1 = i3 / (z2 * z3)
    return 4 * (1 - 1 / xp.stack([z1, z2, z3], axis=-1))


def analytic_basis(f, d, xp=np):
    """Parameter-independent basis that diagonalises ``Re C`` of :func:`tdi2_geometry`.

    Returns ``(v, kappa)``: ``v`` ``(..., 3, 3)`` with columns ``v_j`` in the XYZ
    basis and ``kappa`` ``(..., 3)`` (ascending), such that for every ``A, P``
    ``v^T Re C v = diag(P^2 S_oms + kappa A^2 S_tm)``.  ``kappa_1`` is the null
    channel (``~ (w L)^2`` at low frequency).
    """
    _, k = tdi2_geometry(f, d, xp)
    k = xp.real(k)
    dsq = xp.sqrt(tdi2_delta(f, d, xp))                          # D = sqrt(Delta) > 0
    b = k / (8 * dsq[..., :, None] * dsq[..., None, :])
    x, u = sym3_eigh(b, xp)
    v = u / dsq[..., :, None] / xp.sqrt(8 * x)[..., None, :]
    return v, 4 * (1 - 1 / x)


def simdiag(u_tm, u_oms):
    """Exact simultaneous diagonalisation of batches of real symmetric matrices
    (``u_oms`` positive definite), e.g. window-averaged WDM surfaces.

    Returns ``(v, alpha)`` with ``v^T u_oms v = I`` and ``v^T u_tm v = diag(alpha)``,
    by Cholesky whitening of ``u_oms`` and :func:`sym3_eigh`; any linear two-component
    model ``A^2 u_tm + P^2 u_oms`` is then diagonal in the fixed basis ``v``.
    """
    u_tm, u_oms = np.asarray(u_tm), np.asarray(u_oms)
    sc = np.trace(u_oms, axis1=-2, axis2=-1)[..., None, None] / 3
    lc = np.linalg.cholesky(u_oms / sc)
    li = np.linalg.inv(lc)
    m = li @ (u_tm / sc) @ np.swapaxes(li, -1, -2)
    alpha, q = sym3_eigh(0.5 * (m + np.swapaxes(m, -1, -2)))
    return np.swapaxes(li, -1, -2) @ q / np.sqrt(sc), alpha


def project_diag(y, v):
    """``diag(v^T y v)`` for batches: the data of the diagonal-form likelihood."""
    return np.einsum("...ij,...ik,...jk->...k", np.asarray(y), v, v)


def make_loglik_eig(yd, dof, alpha, beta, complex_data, const=0.0):
    """Wishart log-likelihood in the diagonal form, ``lam_j = A^2 alpha_j + P^2 beta_j``:
    ``-sum wgt sum_j [log lam_j + yd_j / lam_j] + const``.

    ``yd`` ``(..., 3)``: diagonals of the projected data :func:`project_diag`;
    ``beta=None`` means ``beta_j = 1`` (basis from :func:`simdiag`).  A given ``beta``
    is divided out of ``alpha`` and ``yd`` (its logarithm goes into the constant), so
    that ``lam = A^2 alpha + P^2`` is O(1) and its derivatives do not underflow.  One
    logarithm and one division per pixel: ``sum_j log lam_j = log prod_j lam_j``.
    """
    wgt_np = np.asarray(dof) if complex_data else 0.5 * np.asarray(dof)
    if beta is not None:
        beta = np.asarray(beta)
        alpha, yd = np.asarray(alpha) / beta, np.asarray(yd) / beta
        const = const - float(np.sum(wgt_np[..., None] * np.log(beta)))
    yd = jnp.asarray(yd)
    alpha = jnp.asarray(alpha)
    wgt = jnp.asarray(wgt_np)

    def loglik(theta):
        lam = theta[0] ** 2 * alpha + theta[1] ** 2
        l0, l1, l2 = lam[..., 0], lam[..., 1], lam[..., 2]
        p01 = l0 * l1
        prod = p01 * l2
        num = yd[..., 0] * l1 * l2 + yd[..., 1] * l0 * l2 + yd[..., 2] * p01
        return -jnp.sum(wgt * (jnp.log(prod) + num / prod)) + const
    return jax.jit(loglik)


def time_loglik(loglik, theta, nwalker=32, nrep=50):
    """Mean wall time [s] of one vmapped evaluation over ``nwalker`` points."""
    f = jax.jit(jax.vmap(loglik))
    th = jnp.asarray(theta)[None, :] * (1 + 1e-4 * jnp.arange(nwalker)[:, None])
    f(th).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(nrep):
        f(th).block_until_ready()
    return (time.perf_counter() - t0) / nrep


# ---------------------------------------------------------------------------
# Optimisation and sampling
# ---------------------------------------------------------------------------

def find_map(loglik, x0):
    """Maximum-likelihood point and Laplace covariance (flat prior), by
    trust-region Newton iterations with exact JAX gradient and Hessian."""
    from scipy.optimize import minimize
    nll = jax.jit(lambda p: -loglik(p))
    grad = jax.jit(jax.grad(lambda p: -loglik(p)))
    hessf = jax.jit(jax.hessian(lambda p: -loglik(p)))
    res = minimize(lambda p: float(nll(jnp.asarray(p))), np.asarray(x0, float),
                   jac=lambda p: np.asarray(grad(jnp.asarray(p))),
                   hess=lambda p: np.asarray(hessf(jnp.asarray(p))),
                   method="trust-exact", options=dict(gtol=1e-6, maxiter=500))
    hess = np.asarray(hessf(jnp.asarray(res.x)))
    return res.x, np.linalg.inv(hess)


def run_mcmc(loglik, lo, hi, x0, cov0, nwalker=32, niters=3000, burn=1000, seed=1,
             spread=3.0):
    """jexplore ensemble sampler (stretch + differential evolution, one temperature).

    Walkers start in a Gaussian ball of ``spread`` Laplace sigmas around ``x0``.
    Returns ``(samples, info)`` with ``samples`` of shape ``(nsamp, dim)``.
    """
    from jexplore.backends import DefaultBackend
    from jexplore.sampler import JaxSampler, Steps
    from jexplore.sampling import Box, SamplingMH
    from jexplore.steps import DEStep, Stretch

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    dim = lo.size
    lvol = float(np.sum(np.log(hi - lo)))
    jlo, jhi = jnp.asarray(lo), jnp.asarray(hi)

    def logprior(p):
        return jnp.where(jnp.all((p > jlo) & (p < jhi)), -lvol, -jnp.inf)

    sampling = SamplingMH(nwalker=nwalker, temps=jnp.asarray([1.0]), loglik=loglik,
                          logprior=jax.jit(logprior),
                          space=Box(box=[[float(a), float(b)] for a, b in zip(lo, hi)]))
    steps = Steps([{Stretch(permute=True).builder: 1.0, DEStep(permute=True).builder: 1.0}])
    outdir = tempfile.mkdtemp(prefix="jexplore_aet_")
    sampler = JaxSampler(sampling, steps, DefaultBackend(burn=0, outdir=outdir))
    rng = np.random.default_rng(seed)
    p0 = rng.multivariate_normal(np.asarray(x0, float), spread**2 * np.asarray(cov0), size=nwalker)
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    t0 = time.perf_counter()
    sampler.run(sampling.get_epoch(jnp.asarray(p0)), niters=niters, nepoch=1, seed=seed)
    out = sampler.backend.get_samples(burn=burn, pars=["p", "ll"])
    wall = time.perf_counter() - t0
    chains = np.asarray(out["p"])                        # (nwalker, dim, nkept)
    samples = np.transpose(chains, (2, 0, 1)).reshape(-1, dim)
    st = sampler.backend.get_stats()["epoch_0"]
    acc = float(np.sum(st["stats"]) / (nwalker * np.sum(st["counts"])))
    try:
        import emcee
        acl = float(np.max(emcee.autocorr.integrated_time(np.transpose(chains, (2, 0, 1)), quiet=True)))
    except Exception:  # diagnostic is optional
        acl = float("nan")
    for fn in os.listdir(outdir):
        os.remove(os.path.join(outdir, fn))
    os.rmdir(outdir)
    return samples, dict(wall=wall, acceptance=acc, acl=acl, chains=chains)


def summary(samples, truth=(A_TRUE, P_TRUE)):
    """Posterior mean, std and bias in sigma for each parameter."""
    m, s = samples.mean(0), samples.std(0, ddof=1)
    t = np.asarray(truth)
    return dict(mean=m, std=s, bias_sigma=(m - t) / s, rel_bias=(m - t) / t, rel_std=s / t)


# ---------------------------------------------------------------------------
# Non-stationarity diagnostics: exact FD covariance and WDM atom spread
# ---------------------------------------------------------------------------

def cyclic_weights(tnodes, grid, window, qs):
    """Complex time weights ``omega_i^(q) = sum_n (w_n^2/sum w^2) hat_i(t_n) e^{-2 pi i q n/N}``
    for the lags ``qs``: shape ``(len(qs), len(tnodes))``."""
    n = np.arange(grid.n)
    w2 = window**2 / np.sum(window**2)
    tsamp = n * grid.dt
    return np.stack([time_hat_weights(tnodes, tsamp, w2 * np.exp(-2j * np.pi * q * n / grid.n))
                     for q in qs])


def fd_cyclic_spectrum(f, dnodes, omq):
    """``C_q(f) = sum_i omega_i^(q) U(f, t_i)`` for unit TM and OMS covariances (AET):
    two arrays ``(len(q), len(f), 3, 3)``.  ``C_q`` is the covariance between DFT
    bins ``k`` and ``k - q`` at mean frequency ``f`` (locally stationary limit)."""
    f = jnp.asarray(f)
    out_tm = jnp.zeros((omq.shape[0], f.size, 3, 3), complex)
    out_oms = jnp.zeros_like(out_tm)
    om = jnp.asarray(omq)
    for i in range(dnodes.shape[0]):
        utm, uoms = unit_cov_aet_jax(f, jnp.asarray(dnodes[i]))
        out_tm = out_tm + om[:, i, None, None, None] * utm[None]
        out_oms = out_oms + om[:, i, None, None, None] * uoms[None]
    return np.asarray(out_tm), np.asarray(out_oms)


def fd_dense_cov(k0, K, grid, dnodes, tnodes, window, A, P, Q):
    """Dense ``(3K, 3K)`` covariance of the normalised DFT vector ``x_k`` (AET) for
    DFT bins ``k0 .. k0+K-1``, including the correlations between bins induced by
    the time-varying arms (lags ``|k-k'| <= Q``)."""
    qs = np.arange(-Q, Q + 1)
    omq = cyclic_weights(tnodes, grid, window, qs)
    kk = np.arange(k0, k0 + K)
    # C_q evaluated at the mean frequency of the pair (k, k-q): f_k - q df/2
    fmid = (kk[None, :] - 0.5 * qs[:, None]) * grid.df
    S = np.zeros((3 * K, 3 * K), complex)
    for iq, q in enumerate(qs):
        ctm, coms = fd_cyclic_spectrum(fmid[iq], dnodes, omq[iq:iq + 1])
        c = A**2 * ctm[0] + P**2 * coms[0]                # (K, 3, 3)
        for k in range(K):
            kp = k - q
            if 0 <= kp < K:
                S[3 * k:3 * k + 3, 3 * kp:3 * kp + 3] = c[k]
    return 0.5 * (S + S.conj().T)


def wdm_atom_time_spread(nt=64, nf=64):
    """RMS duration of an interior WDM atom in units of the bin width ``nf dt``."""
    from wdm_transform import WDM
    c = np.zeros((nt, nf + 1))
    c[nt // 2, nf // 2] = 1.0
    x = np.asarray(WDM(coeffs=c, dt=1.0).to_time_series().data)
    t = np.arange(x.size) / nf
    p = x**2 / np.sum(x**2)
    mu = np.sum(t * p)
    return float(np.sqrt(np.sum((t - mu) ** 2 * p)))


def fd_sandwich(cells, dnodes, tnodes, window, theta=(A_TRUE, P_TRUE), qmax=12, stride=64):
    """Calibration of the block-diagonal (time-averaged) FD likelihood.

    For Gaussian data whose true covariance couples DFT bins through the cyclic
    spectrum ``C_q`` (:func:`fd_cyclic_spectrum`), the maximum-likelihood estimator of
    the block-diagonal model has covariance ``H^-1 J H^-1`` (sandwich) with
    ``H_ij = sum_k tr(C^-1 d_i C C^-1 d_j C)`` (the model's Fisher matrix) and
    ``J_ij = sum_k sum_q tr(C^-1 d_i C C^-1 C_q C^-1 d_j C C^-1 C_q^dagger)``.
    Sums over the band use every ``stride``-th DFT bin with weight ``stride``.
    Returns ``(H, J, sandwich_cov)``.
    """
    grid = cells.grid
    ke = cells.kedges()
    kk = np.arange(ke[0], ke[-1], stride)
    qs = np.arange(-qmax, qmax + 1)
    omq = cyclic_weights(tnodes, grid, window, qs)
    ctm, coms = fd_cyclic_spectrum(kk * grid.df, dnodes, omq)      # (q, f, 3, 3)
    A, P = theta
    Cq = A**2 * ctm + P**2 * coms
    i0 = qmax
    C0 = Cq[i0]
    sc = np.real(np.trace(C0, axis1=-2, axis2=-1))[:, None, None] / 3     # rescale for conditioning
    Ci = np.linalg.inv(C0 / sc)
    dC = [2 * A * ctm[i0] / sc, 2 * P * coms[i0] / sc]
    Cqs = Cq / sc[None]
    Ai = [Ci @ d @ Ci for d in dC]
    H = np.zeros((2, 2))
    J = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            H[a, b] = stride * np.real(np.einsum("fij,fjk,fkl,fli->", Ci, dC[a], Ci, dC[b]))
            # tr(A_a C_q A_b C_q^dagger), summed over lags and frequencies
            J[a, b] = stride * np.real(np.einsum("qfij,fjk,qflk,fli->", Cqs, Ai[b], np.conj(Cqs), Ai[a]))
    Hi = np.linalg.inv(H)
    return H, J, Hi @ J @ Hi
