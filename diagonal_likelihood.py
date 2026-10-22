# %% [markdown]
# # The WDM likelihood in diagonal form
#
# Companion of `paper/main.tex` (Sec. "Analytic inverse and diagonal likelihood").  For any frozen
# delays the TM and OMS parts of the closed-form TDI covariance share their off-diagonal elements
# up to a factor 4,
#
# $$\boldsymbol C=(4S_{\rm a}+S_{\rm n})\,\boldsymbol K-32\,S_{\rm a}\,\boldsymbol\Delta,\qquad
# \Delta_X=|G_X|^2\sin^2\theta_2\sin^2\theta_3\ \text{(and cyclically)},$$
#
# so a single basis $\boldsymbol v_j=\boldsymbol\Delta^{-1/2}\boldsymbol u_j/\sqrt{8x_j}$, with $x_j$, $\boldsymbol u_j$ the
# eigen-pairs of the purely geometric $\boldsymbol B=\boldsymbol\Delta^{-1/2}\boldsymbol K\boldsymbol\Delta^{-1/2}/8$, diagonalises
# $\boldsymbol C$ for **every** $(A,P)$: $\boldsymbol v_j^\top\boldsymbol C\boldsymbol v_k=\delta_{jk}\,(S_{\rm n}+\kappa_jS_{\rm a})$,
# $\kappa_j=4(1-1/x_j)$.  The WDM likelihood then becomes a sum of one-dimensional Gaussians in a basis
# fixed once and for all, with no $3\times3$ algebra at each evaluation.
#
# This notebook (ESA-orbit data of Analysis 2 of `aet_wdm_analyses.ipynb`):
# 1. checks the identity and the closed-form eigenvalues against the `pytdi` delay polynomials;
# 2. on the 48×76 time–frequency cells, compares the diagonal likelihood with the $3\times3$ one;
# 3. on every WDM pixel ($1.56\times10^6$ per channel), compares the $3\times3$ likelihood (closed-form
#    cofactors, and LAPACK) with the diagonal one, in the exact pixel basis and in the analytic basis
#    evaluated at $(t_n,f_m)$: values, residual correlations, cost and posteriors.

# %%
import json
import os
import time
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
from scipy.interpolate import CubicSpline

import aet_wdm as aw

warnings.filterwarnings("ignore", message=".*preferred interpolation.*")
from lisaorbits import OEMOrbits  # noqa: E402

HERE = os.path.abspath(os.getcwd())
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
COL = {"wdm": "#2a78d6", "fd": "#eb6834", "ctl": "#1baf7a", "ink": "#52514e", "grid": "#e1e0d9"}
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 10,
    "axes.labelsize": 10, "legend.fontsize": 8.5, "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8, "xtick.color": COL["ink"], "ytick.color": COL["ink"],
    "axes.labelcolor": "#0b0b0b", "axes.grid": True, "grid.color": COL["grid"],
    "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
})
print("jax", jax.__version__, jax.devices(), " CPU threads", os.cpu_count())

# same grid, cells, orbits and data as Analysis 2 of aet_wdm_analyses.ipynb
grid = aw.Grid(dt=10.0, nt=3072, nf=1024)
M_LO, M_HI, N_FCELL, N_TCELL, EDGE = 2, 512, 100, 48, 12
medges = aw.log_edges(M_LO, M_HI, N_FCELL)
THETA_TRUE = np.array([aw.A_TRUE, aw.P_TRUE])
PRIOR_LO, PRIOR_HI = np.array([0.1, 0.1]), np.array([100.0, 100.0])
orb = OEMOrbits.from_included("esa-trailing")
T_START = orb.t_init + 10 * 86400.0
ltt_esa = lambda t: orb.compute_ltt(T_START + np.asarray(t))
for fn in ("data_esa_orbits.npz", "surfaces_esa.npz"):
    if not os.path.exists(os.path.join(DATA, fn)):
        raise FileNotFoundError(f"data/{fn} missing: run aet_wdm_analyses.ipynb first")
xyz2 = np.load(os.path.join(DATA, "data_esa_orbits.npz"))["xyz"]
surf = dict(np.load(os.path.join(DATA, "surfaces_esa.npz")))
cells2 = aw.Cells(grid, medges, np.round(np.linspace(EDGE, grid.nt - EDGE, N_TCELL + 1)).astype(int))
window = aw.tukey(grid.n, int(86400 / grid.dt))
tnodes = np.linspace(-2 * 86400, grid.T + 2 * 86400, 151)
W2 = aw.wdm_transform(xyz2 * window, grid)                         # (3, nt, nf+1), AET channels
Y2, D2 = aw.wdm_cells(W2, cells2)
surf["wdm_tm"], surf["wdm_oms"] = aw.wdm_model_local(cells2, ltt_esa, tnodes,
                                                     nodes=(surf["nodes_tm"], surf["nodes_oms"]))
OUT = {}

# %% [markdown]
# ## 1. The identity and the closed-form eigenvalues
#
# Brute-force covariance from the `pytdi` TDI-2 delay polynomials (independent of the closed form),
# split into its TM and OMS parts: $\boldsymbol C_{\rm tm}-4\boldsymbol C_{\rm oms}$ must be diagonal and equal to
# $-32\boldsymbol\Delta$.  The generalized eigenvalues of the pair $(\boldsymbol C_{\rm tm},\boldsymbol C_{\rm oms})$ (unit PSDs
# stripped) must be the $\kappa_j$ of the closed form.

# %%
fchk = np.geomspace(1e-4, 0.025, 300)
chk = {}
for name, d in [("equal arms", np.full(6, aw.L_NOMINAL / aw.C_LIGHT)),
                ("ESA, t = 0", ltt_esa(np.array([0.0]))[0]),
                ("ESA, t = T/2", ltt_esa(np.array([grid.T / 2]))[0])]:
    F = aw.pytdi_transfer_matrix(fchk, d)
    c_oms = np.einsum("fci,fdi->fcd", F[..., :6], F[..., :6].conj())
    c_tm = np.einsum("fci,fdi->fcd", F[..., 6:], F[..., 6:].conj())
    dg = np.abs(np.einsum("fii->fi", c_tm))
    r = c_tm - 4 * c_oms + 32 * aw.tdi2_delta(fchk, d)[:, :, None] * np.eye(3)
    err_id = np.max(np.abs(r) / np.sqrt(dg[:, :, None] * dg[:, None, :]))
    li = np.linalg.inv(np.linalg.cholesky(c_oms))
    kap_num = np.linalg.eigvalsh(li @ c_tm @ np.conj(np.swapaxes(li, -1, -2)))
    err_k = np.max(np.abs(aw.kappa_closed_form(fchk, d) / kap_num - 1))
    # real part (WDM, channels not delayed): analytic basis vs Re of the pytdi covariance
    li_r = np.linalg.inv(np.linalg.cholesky(c_oms.real))
    kap_num_r = np.linalg.eigvalsh(li_r @ c_tm.real @ np.swapaxes(li_r, -1, -2))
    _, kap_r = aw.analytic_basis(fchk, d)
    err_kr = np.max(np.abs(kap_r / kap_num_r - 1))
    lift = kap_r[:, 0] / kap_num[:, 0]
    chk[name] = dict(identity=float(err_id), kappa_complex=float(err_k), kappa_real=float(err_kr),
                     null_lift_real_over_complex_at_0p1mHz=float(lift[0]))
    print(f"{name:13s}: max |C_tm - 4 C_oms + 32 Delta| / sqrt(C_ii C_jj) = {err_id:.1e};"
          f"  max |kappa/kappa_pytdi - 1| complex (cubic) {err_k:.1e}, real (basis) {err_kr:.1e};"
          f"  kappa_1(Re C)/kappa_1(C) at 0.1, 1, 10 mHz = {np.interp([1e-4, 1e-3, 1e-2], fchk, lift).round(3)}")
OUT["checks"] = chk

# generalized eigenvalues kappa_j(f) and the null channel
fk = np.geomspace(1e-4, 0.028, 600)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
th = 2 * np.pi * fk * aw.L_NOMINAL / aw.C_LIGHT
ax[0].loglog(fk * 1e3, 2 * (1 - np.cos(th)), color=COL["ink"], lw=0.9, ls=":", label=r"equal arms, $4\sin^2(\theta/2)$")
ax[0].loglog(fk * 1e3, 4 * (1 + np.cos(th) + np.cos(th) ** 2) / (2 + np.cos(th)), color=COL["ink"], lw=0.9, ls="--",
             label=r"equal arms, $4(1+c+c^2)/(2+c)$")
d_mid = ltt_esa(np.array([grid.T / 2]))[0]
_, kap = aw.analytic_basis(fk, d_mid)
for j, c in zip(range(3), (COL["wdm"], COL["fd"], COL["ctl"])):
    ax[0].loglog(fk * 1e3, kap[:, j], color=c, lw=1.2, label=fr"ESA, $\kappa_{j+1}$")
ax[0].set_xlabel("f [mHz]"); ax[0].set_ylabel(r"$\kappa_j$ (TM-to-OMS weight)"); ax[0].legend(fontsize=7.5)
# null channel: D^{-1}(1,1,1) with the TDI-2 delays 2 Lbar_I, vs (X+Y+Z)/sqrt3
fn_ = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2])
tt = np.linspace(0, grid.T, 60)
rat_T, rat_D = [], []
for t in tt:
    d = ltt_esa(np.array([t]))[0]
    utm, uoms = aw.tdi2_unit_cov(fn_, d)
    C = aw.A_TRUE**2 * utm + aw.P_TRUE**2 * uoms
    lbar, _ = aw.arm_quantities(d)
    ua, uo = aw.equal_arm_aet_psd(fn_, lbar.mean())
    stt0 = aw.A_TRUE**2 * ua[:, 2] + aw.P_TRUE**2 * uo[:, 2]
    vt = np.full((fn_.size, 3), 1 / np.sqrt(3), complex)
    vd = np.exp(2j * np.pi * fn_[:, None] * 2 * lbar) / np.sqrt(aw.tdi2_delta(fn_, d))
    vd /= np.linalg.norm(vd, axis=-1, keepdims=True)
    q = lambda v: np.real(np.einsum("fi,fij,fj->f", v.conj(), C, v))
    rat_T.append(q(vt) / stt0); rat_D.append(q(vd) / stt0)
rat_T, rat_D = np.array(rat_T), np.array(rat_D)
ax[1].loglog(fn_ * 1e3, rat_T.max(0), "o-", color=COL["fd"], label=r"$(X+Y+Z)/\sqrt{3}$, max over the year")
ax[1].loglog(fn_ * 1e3, rat_T.mean(0), "o--", color=COL["fd"], lw=1.0, label=r"$(X+Y+Z)/\sqrt{3}$, yearly mean")
ax[1].loglog(fn_ * 1e3, rat_D.max(0), "s-", color=COL["wdm"], label=r"$\sum_I I/D_I$ (delayed), max")
ax[1].set_xlabel("f [mHz]"); ax[1].set_ylabel(r"$S/S_{TT,0}$"); ax[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(RES, "diag_kappa_nullchannel.pdf")); plt.show()
OUT["null_channel"] = dict(f_mHz=(fn_ * 1e3).tolist(), xyz_sum_max=rat_T.max(0).tolist(),
                           xyz_sum_mean=rat_T.mean(0).tolist(), d_inverse_sum_max=rat_D.max(0).tolist(),
                           d_inverse_sum_min=rat_D.min(0).tolist())
print("S/S_TT0 of (X+Y+Z)/sqrt3 (max over year):", rat_T.max(0).round(3))
print("S/S_TT0 of sum_I I/D_I   (min..max)     :", rat_D.min(0).round(4), rat_D.max(0).round(4))

# %% [markdown]
# ## 2. Cells (48 time × 76 frequency cells, real Wishart)
#
# The exact basis of each cell, $\boldsymbol v^\top\boldsymbol U_{\rm oms}\boldsymbol v=\mathbb I$,
# $\boldsymbol v^\top\boldsymbol U_{\rm tm}\boldsymbol v={\rm diag}(\alpha_j)$, turns the Wishart likelihood into
# $-\tfrac12\sum_pK_p\sum_j[\ln(A^2\alpha_{pj}+P^2)+\hat y_{pj}/(A^2\alpha_{pj}+P^2)]$ with
# $\hat y_{pj}=(\boldsymbol v^\top\bar{\boldsymbol Y}_p\boldsymbol v)_{jj}$, identical to the $3\times3$ form up to the
# constant $\sum_pK_p\ln|\det\boldsymbol v_p|$.

# %%
def fit(name, loglik, store, x0=THETA_TRUE, niters=3000, burn=1000, nwalker=32, seed=7):
    t0 = time.perf_counter()
    xmap, cov = aw.find_map(loglik, x0)
    t_map = time.perf_counter() - t0
    samples, info = aw.run_mcmc(loglik, PRIOR_LO, PRIOR_HI, xmap, cov, nwalker=nwalker,
                                niters=niters, burn=burn, seed=seed)
    s = aw.summary(samples)
    store[name] = dict(mean=s["mean"].tolist(), std=s["std"].tolist(), bias_sigma=s["bias_sigma"].tolist(),
                       map=xmap.tolist(), laplace_std=np.sqrt(np.diag(cov)).tolist(), t_map_s=t_map,
                       t_mcmc_s=info["wall"], acceptance=info["acceptance"], acl=info["acl"])
    print(f"{name:34s} A = {s['mean'][0]:.5f} ± {s['std'][0]:.5f}  P = {s['mean'][1]:.5f} ± {s['std'][1]:.5f}"
          f"  bias/σ = ({s['bias_sigma'][0]:+5.2f}, {s['bias_sigma'][1]:+5.2f})"
          f"  MAP {t_map:5.1f} s, MCMC {info['wall']:6.1f} s, acc {info['acceptance']:.2f}")
    return samples


def compare_values(ll_a, ll_b, label, n=6, seed=0):
    """ll_a - ll_b at random points around the injection: must be a constant."""
    rng = np.random.default_rng(seed)
    th = THETA_TRUE * (1 + 0.01 * rng.standard_normal((n, 2)))
    diff = np.array([float(ll_a(jnp.asarray(t))) - float(ll_b(jnp.asarray(t))) for t in th])
    va = np.array([float(ll_a(jnp.asarray(t))) for t in th])
    spread = np.ptp(diff)
    print(f"{label}: ln L_a - ln L_b = {diff.mean():.6e}, spread over {n} points {spread:.1e}"
          f" (ln L range {np.ptp(va):.1e})")
    return float(spread)


t0 = time.perf_counter()
v_c, al_c = aw.simdiag(surf["wdm_tm"], surf["wdm_oms"])
yd_c = aw.project_diag(Y2, v_c)
t_pre_cells = time.perf_counter() - t0
const_c = float(np.sum(0.5 * D2 * 2 * np.log(np.abs(np.linalg.det(v_c)))))
ll_c_full = aw.make_loglik_full(Y2, D2, surf["wdm_tm"], surf["wdm_oms"], complex_data=False)
ll_c_diag = aw.make_loglik_eig(yd_c, D2, al_c, None, complex_data=False, const=const_c)
OUT["cells"] = dict(n_cells=int(D2.size), precompute_s=t_pre_cells,
                    value_spread=compare_values(ll_c_full, ll_c_diag, "cells: 3x3 vs diagonal"))
POST = {}
s_c_full = fit("cells, 3x3 (closed-form cofactors)", ll_c_full, POST)
s_c_diag = fit("cells, diagonal form", ll_c_diag, POST)

# %% [markdown]
# ## 3. Every pixel
#
# Pixel model: the window-averaged closed form at the 151 orbit epochs, interpolated (cubic spline)
# to every WDM time bin $t_n$, as in `aw.wdm_model_local` before the cell average: $3048\times511$ pixels
# in each of the three channels.  Two bases:
# * **exact**: `aw.simdiag` of the pixel surfaces (window-averaged over the band);
# * **analytic**: `aw.analytic_basis`, the closed form at the pixel centre $(t_n,f_m)$, without any band
#   average; its residual correlations measure what the band average does to the geometric basis.

# %%
te, me = cells2.tedges, cells2.medges
nt_in, nb = te[-1] - te[0], me[-1] - me[0]
tn = np.arange(te[0], te[-1]) * grid.delta_t
fm = np.arange(me[0], me[-1]) * grid.delta_f
pix_tm = CubicSpline(tnodes, surf["nodes_tm"], axis=0)(tn)       # (nt_in, nb, 3, 3), AET
pix_oms = CubicSpline(tnodes, surf["nodes_oms"], axis=0)(tn)
w_pix = np.moveaxis(W2[:, te[0]:te[-1], me[0]:me[-1]], 0, -1)     # (nt_in, nb, 3)
norm = 2 * grid.dt / grid.n
y_pix = norm * w_pix[..., :, None] * w_pix[..., None, :]
dof_pix = np.ones((nt_in, nb))
print(f"{nt_in} x {nb} = {nt_in * nb} pixels per channel")

# exact basis
t0 = time.perf_counter()
v_e, al_e = aw.simdiag(pix_tm, pix_oms)
t_simdiag = time.perf_counter() - t0
t0 = time.perf_counter()
yd_e = norm * np.einsum("...i,...ij->...j", w_pix, v_e) ** 2
t_proj = time.perf_counter() - t0
const_e = float(np.sum(np.log(np.abs(np.linalg.det(v_e)))))

# analytic basis at (t_n, f_m): closed form in jax, rotated to AET (data are A, E, T)
basis_t = jax.jit(jax.vmap(lambda d: aw.analytic_basis(jnp.asarray(fm), d, xp=jnp)))
d_n = jnp.asarray(ltt_esa(tn))
basis_t(d_n[:2])[0].block_until_ready()
t0 = time.perf_counter()
v_g_xyz, kap_g = basis_t(d_n)
v_g_xyz.block_until_ready()
t_analytic = time.perf_counter() - t0
v_g = np.einsum("ai,...ij->...aj", aw.R_AET, np.asarray(v_g_xyz))
al_g = np.einsum("...ij,...ik,...jk->...k", pix_tm, v_g, v_g)
be_g = np.einsum("...ij,...ik,...jk->...k", pix_oms, v_g, v_g)
yd_g = norm * np.einsum("...i,...ij->...j", w_pix, v_g) ** 2
# fully explicit eigenvalues, main.tex Eq. (Ldiag): window-averaged unit PSDs times kappa_j(f_m, t_n)
jw, phi2 = aw.meyer_window_sq(grid.nt)
fl = (np.arange(me[0], me[-1])[:, None] * (grid.nt // 2) + jw[None, :]) * grid.df
sbar_a = (phi2 * aw.psd_tm_unit(fl)).sum(-1)
sbar_n = (phi2 * aw.psd_oms_unit(fl)).sum(-1)
al_x = np.asarray(kap_g) * sbar_a[None, :, None]
be_x = np.broadcast_to(sbar_n[None, :, None], al_x.shape)
print("explicit vs projected eigenvalues: max |kappa Sbar_a / (v^T U_tm v) - 1| =",
      np.abs(al_x / al_g - 1).max().round(6), "  max |Sbar_n / (v^T U_oms v) - 1| =", np.abs(be_x / be_g - 1).max().round(6))
const_g = float(np.sum(np.log(np.abs(np.linalg.det(v_g)))))
c_ref = aw.A_TRUE**2 * pix_tm + aw.P_TRUE**2 * pix_oms
cg = np.einsum("...ij,...ik,...jl->...kl", c_ref, v_g, v_g)
dgg = np.einsum("...ii->...i", cg)
rho_g = np.abs(cg / np.sqrt(dgg[..., :, None] * dgg[..., None, :]))[..., [0, 0, 1], [1, 2, 2]]
print(f"precomputation: exact basis (Cholesky + closed-form eigh, numpy) {t_simdiag:.2f} s,"
      f" projection {t_proj:.2f} s; analytic basis (jax) {t_analytic:.2f} s")
print("analytic basis: residual |rho_jk| of the pixel covariance: max", rho_g.max(0).max(0).round(8),
      " median", np.median(rho_g.reshape(-1, 3), 0))
for flo, fhi in [(0, 1e-3), (1e-3, 1e-2), (1e-2, 1)]:
    sel = (fm >= flo) & (fm < fhi)
    print(f"   {flo*1e3:5.1f}-{min(fhi,0.0256)*1e3:5.1f} mHz: max |rho| = {rho_g[:, sel].max():.2e}")
OUT["pixels"] = dict(n_pixels=int(nt_in * nb), precompute_exact_basis_s=t_simdiag, precompute_projection_s=t_proj,
                     precompute_analytic_basis_s=t_analytic, analytic_rho_max=rho_g.max(0).max(0).tolist(),
                     analytic_rho_median=np.median(rho_g.reshape(-1, 3), 0).tolist())

# %%
ll_p_full = aw.make_loglik_full(y_pix, dof_pix, pix_tm, pix_oms, complex_data=False)
ll_p_lapack = aw.make_loglik_full_lapack(y_pix, dof_pix, pix_tm, pix_oms, complex_data=False)
ll_p_diag = aw.make_loglik_eig(yd_e, dof_pix, al_e, None, complex_data=False, const=const_e)
ll_p_anal = aw.make_loglik_eig(yd_g, dof_pix, al_g, be_g, complex_data=False, const=const_g)
ll_p_expl = aw.make_loglik_eig(yd_g, dof_pix, al_x, be_x, complex_data=False, const=const_g)
ll_p_diag3 = aw.make_loglik_diag(np.einsum("...j,jk->...jk", yd_e, np.eye(3)), dof_pix, al_e,
                                 np.ones_like(al_e), complex_data=False)
OUT["pixels"]["value_spread_full_vs_diag"] = compare_values(ll_p_full, ll_p_diag, "pixels: 3x3 vs diagonal (exact)")
OUT["pixels"]["value_spread_full_vs_lapack"] = compare_values(ll_p_full, ll_p_lapack, "pixels: 3x3 vs LAPACK     ")
OUT["pixels"]["value_spread_full_vs_analytic"] = compare_values(ll_p_full, ll_p_anal, "pixels: 3x3 vs analytic   ")
OUT["pixels"]["value_spread_full_vs_explicit"] = compare_values(ll_p_full, ll_p_expl, "pixels: 3x3 vs explicit   ")

# %% [markdown]
# ### Cost of one likelihood evaluation (32 walkers, vmapped, CPU)

# %%
TIM = {}
for name, ll, nrep in [("cells, 3x3 cofactors", ll_c_full, 200), ("cells, diagonal", ll_c_diag, 200),
                       ("pixels, 3x3 cofactors", ll_p_full, 10), ("pixels, 3x3 LAPACK", ll_p_lapack, 1),
                       ("pixels, diagonal (3 logs)", ll_p_diag3, 10), ("pixels, diagonal (1 log)", ll_p_diag, 10),
                       ("pixels, diagonal analytic basis", ll_p_anal, 10)]:
    TIM[name] = 1e3 * aw.time_loglik(ll, THETA_TRUE, nwalker=32, nrep=nrep)
    print(f"{name:34s} {TIM[name]:10.3f} ms / 32 walkers")
# value + gradient (what a gradient-based sampler would need)
for name, ll in [("pixels, 3x3 cofactors", ll_p_full), ("pixels, diagonal (1 log)", ll_p_diag)]:
    g = jax.jit(jax.vmap(jax.value_and_grad(ll)))
    th = jnp.asarray(THETA_TRUE)[None, :] * (1 + 1e-4 * jnp.arange(32)[:, None])
    jax.block_until_ready(g(th))
    t0 = time.perf_counter()
    for _ in range(5):
        jax.block_until_ready(g(th))
    TIM[name + " +grad"] = 1e3 * (time.perf_counter() - t0) / 5
    print(f"{name + ' +grad':34s} {TIM[name + ' +grad']:10.3f} ms / 32 walkers")
OUT["timing_ms_per_32_walkers"] = TIM

# %% [markdown]
# ### Posteriors on every pixel

# %%
s_p_full = fit("pixels, 3x3 (closed-form cofactors)", ll_p_full, POST)
s_p_diag = fit("pixels, diagonal form (exact basis)", ll_p_diag, POST)
s_p_anal = fit("pixels, diagonal form (analytic basis)", ll_p_anal, POST)
s_p_expl = fit("pixels, explicit Eq. (Ldiag)", ll_p_expl, POST)
OUT["posteriors"] = POST
OUT["grid"] = dict(dt=grid.dt, nt=grid.nt, nf=grid.nf, n_fcell=len(medges) - 1, n_tcell=N_TCELL,
                   cpu_threads=os.cpu_count(), jax=jax.__version__)
with open(os.path.join(RES, "diagonal_likelihood.json"), "w") as fh:
    json.dump(OUT, fh, indent=1)
np.savez(os.path.join(RES, "diag_chains.npz"), cells_full=s_c_full, cells_diag=s_c_diag, pixels_full=s_p_full,
         pixels_diag=s_p_diag, pixels_analytic=s_p_anal, pixels_explicit=s_p_expl)

# %%
def overlay(runs, fname, title):
    lv = (1 - np.exp(-0.5), 1 - np.exp(-2.0))
    allr = np.concatenate([r[0] for r in runs])
    lo, hi = np.percentile(allr, 0.05, axis=0), np.percentile(allr, 99.95, axis=0)
    pad = 0.08 * (hi - lo)
    rng_ = [(lo[i] - pad[i], hi[i] + pad[i]) for i in range(2)]
    fig = None
    for s, lab, c, ls in runs:
        fig = corner.corner(s, fig=fig, color=c, bins=40, smooth=1.0, smooth1d=1.0,
                            weights=np.full(len(s), 1.0 / len(s)), levels=lv, plot_datapoints=False,
                            plot_density=False, fill_contours=False, range=rng_,
                            hist_kwargs=dict(lw=1.4, ls=ls), contour_kwargs=dict(linewidths=1.4, linestyles=ls),
                            labels=aw.PARAM_LABELS, label_kwargs=dict(fontsize=10))
    corner.overplot_lines(fig, THETA_TRUE, color="#0b0b0b", lw=0.9, ls=":")
    handles = [mpl.lines.Line2D([], [], color=c, ls=ls, lw=1.6, label=lab) for _, lab, c, ls in runs]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=9)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    fig.set_size_inches(5.6, 5.2)
    fig.savefig(os.path.join(RES, fname), bbox_inches="tight")
    plt.show()


overlay([(s_c_full, "cells, 3x3", COL["ink"], ":"),
         (s_p_full, "pixels, 3x3", COL["fd"], "--"),
         (s_p_diag, "pixels, diagonal (exact basis)", COL["wdm"], "-"),
         (s_p_anal, "pixels, diagonal (analytic basis)", COL["ctl"], "-."),
         (s_p_expl, "pixels, explicit Eq. (Ldiag)", "#8c5bd6", "--")],
        "diag_corner.pdf", "ESA orbits: 3x3 vs diagonal WDM likelihood")

# %%
print(json.dumps(OUT["timing_ms_per_32_walkers"], indent=1))
